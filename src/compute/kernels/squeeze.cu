// CUDA C versions of the WGSL kernels in ../shaders. Compiled once to PTX with
//
//     nvcc -arch=compute_75 -ptx -O3 -o squeeze.ptx squeeze.cu
//
// and loaded through the driver JIT at runtime (no CUDA toolkit needed on the target, only
// the driver). compute_75 (Turing, T4) PTX runs on every newer GPU. Keep the semantics
// identical to the WGSL shaders: the CPU backend is the reference for both.

#include <cstdint>

#define TILE 16
#define DMAX 16

// ---------------------------------------------------------------------------------------
// Tiled pairwise reductions: out[(ai)*out_stride + bi] = FINAL(sum_c ACC(a[ai,c], b[bi,c]))
// ---------------------------------------------------------------------------------------

template <int VARIANT>
__device__ __forceinline__ float acc_op(float acc, float x, float y) {
    if (VARIANT == 0 || VARIANT == 1) { float v = x - y; return acc + v * v; }   // sq / euclid
    if (VARIANT == 2) { return acc + fabsf(x - y); }                               // manhattan
    return acc + x * y;                                                            // dot / cosine
}

template <int VARIANT>
__device__ __forceinline__ float final_op(float acc) {
    if (VARIANT == 1) return sqrtf(acc);
    if (VARIANT == 4) return 1.0f - acc;
    return acc;
}

template <int VARIANT>
__device__ void dist_tile_impl(const float* __restrict__ a, const float* __restrict__ b,
                               float* __restrict__ out, uint32_t a0, uint32_t a_rows,
                               uint32_t b0, uint32_t b_rows, uint32_t d, uint32_t out_stride) {
    __shared__ float a_tile[TILE * TILE];
    __shared__ float b_tile[TILE * TILE];
    const uint32_t ty = threadIdx.y, tx = threadIdx.x;
    const uint32_t ai = blockIdx.y * TILE + ty;
    const uint32_t bi = blockIdx.x * TILE + tx;
    const bool a_ok = ai < a_rows, b_ok = bi < b_rows;
    const size_t a_row = (size_t)(a0 + ai) * d;
    const size_t b_row = (size_t)(b0 + bi) * d;
    float acc = 0.0f;
    for (uint32_t c0 = 0; c0 < d; c0 += TILE) {
        const uint32_t ca = c0 + tx, cb = c0 + ty;
        a_tile[ty * TILE + tx] = (a_ok && ca < d) ? a[a_row + ca] : 0.0f;
        b_tile[tx * TILE + ty] = (b_ok && cb < d) ? b[b_row + cb] : 0.0f;
        __syncthreads();
        const uint32_t cmax = min((uint32_t)TILE, d - c0);
        for (uint32_t c = 0; c < cmax; ++c) {
            acc = acc_op<VARIANT>(acc, a_tile[ty * TILE + c], b_tile[tx * TILE + c]);
        }
        __syncthreads();
    }
    if (a_ok && b_ok) {
        out[(size_t)ai * out_stride + bi] = final_op<VARIANT>(acc);
    }
}

#define DIST_KERNEL(NAME, V)                                                                   \
    extern "C" __global__ void NAME(const float* a, const float* b, float* out, uint32_t a0,   \
                                    uint32_t a_rows, uint32_t b0, uint32_t b_rows, uint32_t d, \
                                    uint32_t out_stride) {                                     \
        dist_tile_impl<V>(a, b, out, a0, a_rows, b0, b_rows, d, out_stride);                  \
    }

DIST_KERNEL(dist_sqeuclidean, 0)
DIST_KERNEL(dist_euclidean, 1)
DIST_KERNEL(dist_manhattan, 2)
DIST_KERNEL(dist_dot, 3)
DIST_KERNEL(dist_cosine, 4)

// ---------------------------------------------------------------------------------------
// Top-k merge: one thread per query row; (distance, index) ordering as the CPU reference.
// ---------------------------------------------------------------------------------------

__device__ __forceinline__ bool before(float d, uint32_t i, float d2, uint32_t i2) {
    return d < d2 || (d == d2 && i < i2);
}

extern "C" __global__ void topk_merge(const float* __restrict__ tile, uint32_t* best_idx,
                                      float* best_dist, uint32_t q0, uint32_t q_rows,
                                      uint32_t b0, uint32_t b_cols, uint32_t k,
                                      uint32_t tile_stride) {
    const uint32_t q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= q_rows) return;
    const size_t base = (size_t)(q0 + q) * k;
    const size_t last = base + k - 1;
    float thr_d = best_dist[last];
    uint32_t thr_i = best_idx[last];
    const size_t row = (size_t)q * tile_stride;
    for (uint32_t c = 0; c < b_cols; ++c) {
        const float d = tile[row + c];
        const uint32_t i = b0 + c;
        if (!before(d, i, thr_d, thr_i)) continue;
        size_t pos = last;
        while (pos > base) {
            const float pd = best_dist[pos - 1];
            const uint32_t pi = best_idx[pos - 1];
            if (before(d, i, pd, pi)) {
                best_dist[pos] = pd;
                best_idx[pos] = pi;
                --pos;
            } else {
                break;
            }
        }
        best_dist[pos] = d;
        best_idx[pos] = i;
        thr_d = best_dist[last];
        thr_i = best_idx[last];
    }
}

// ---------------------------------------------------------------------------------------
// PaCMAP gradient over a packed CSR: csr = offsets[n+1] | nbr[nbr_off..] | tag[tag_off..]
// ---------------------------------------------------------------------------------------

extern "C" __global__ void pacmap_grad(const float* __restrict__ y, const uint32_t* __restrict__ csr,
                                       float* __restrict__ grad, uint32_t n, uint32_t dim,
                                       float w_near, float w_mn, float w_fp, uint32_t nbr_off,
                                       uint32_t tag_off) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float yi[DMAX], g[DMAX];
    for (uint32_t c = 0; c < dim; ++c) { yi[c] = y[(size_t)i * dim + c]; g[c] = 0.0f; }
    const uint32_t e0 = csr[i], e1 = csr[i + 1];
    for (uint32_t e = e0; e < e1; ++e) {
        const uint32_t j = csr[nbr_off + e];
        float diff[DMAX];
        float d2 = 0.0f;
        for (uint32_t c = 0; c < dim; ++c) {
            const float v = yi[c] - y[(size_t)j * dim + c];
            diff[c] = v;
            d2 += v * v;
        }
        const uint32_t t = csr[tag_off + e];
        float coeff;
        if (t == 0u) { const float s = 10.0f + d2; coeff = w_near * 20.0f / (s * s); }
        else if (t == 1u) { const float s = 10000.0f + d2; coeff = w_mn * 20000.0f / (s * s); }
        else { const float s = 1.0f + d2; coeff = -(w_fp * 2.0f / (s * s)); }
        for (uint32_t c = 0; c < dim; ++c) g[c] += coeff * diff[c];
    }
    for (uint32_t c = 0; c < dim; ++c) grad[(size_t)i * dim + c] = g[c];
}

// ---------------------------------------------------------------------------------------
// TriMap gradient over packed arrays:
// packed = offsets | nbr (triplet idx) | tag (role) | triplets (i,j,k) | weights (f32 bits)
// ---------------------------------------------------------------------------------------

extern "C" __global__ void trimap_grad(const float* __restrict__ y, const uint32_t* __restrict__ packed,
                                       float* __restrict__ grad, uint32_t n, uint32_t dim,
                                       float scale, uint32_t nbr_off, uint32_t tag_off,
                                       uint32_t trip_off, uint32_t w_off) {
    const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    float g[DMAX];
    for (uint32_t c = 0; c < dim; ++c) g[c] = 0.0f;
    const uint32_t e0 = packed[p], e1 = packed[p + 1];
    for (uint32_t e = e0; e < e1; ++e) {
        const uint32_t t = packed[nbr_off + e];
        const uint32_t i = packed[trip_off + 3 * t];
        const uint32_t j = packed[trip_off + 3 * t + 1];
        const uint32_t k = packed[trip_off + 3 * t + 2];
        float dij[DMAX], dik[DMAX];
        float d_ij = 0.0f, d_ik = 0.0f;
        for (uint32_t c = 0; c < dim; ++c) {
            const float yi = y[(size_t)i * dim + c];
            const float a = yi - y[(size_t)j * dim + c];
            const float b = yi - y[(size_t)k * dim + c];
            dij[c] = a; dik[c] = b;
            d_ij += a * a; d_ik += b * b;
        }
        if (d_ij - d_ik + 1.0f <= 0.0f) continue;
        const float sw = scale * __uint_as_float(packed[w_off + t]);
        const uint32_t role = packed[tag_off + e];
        for (uint32_t c = 0; c < dim; ++c) {
            if (role == 0u) g[c] += sw * (dij[c] - dik[c]);
            else if (role == 1u) g[c] -= sw * dij[c];
            else g[c] += sw * dik[c];
        }
    }
    for (uint32_t c = 0; c < dim; ++c) grad[(size_t)p * dim + c] = g[c];
}

// ---------------------------------------------------------------------------------------
// Exact t-SNE: scratch = rowsum[n] | z[1] | grad[n*dim]
// ---------------------------------------------------------------------------------------

extern "C" __global__ void tsne_rowsum(const float* __restrict__ y, float* __restrict__ scratch,
                                       uint32_t n, uint32_t dim) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float yi[DMAX];
    for (uint32_t c = 0; c < dim; ++c) yi[c] = y[(size_t)i * dim + c];
    float s = 0.0f;
    for (uint32_t j = 0; j < n; ++j) {
        if (j == i) continue;
        float d2 = 0.0f;
        for (uint32_t c = 0; c < dim; ++c) { const float v = yi[c] - y[(size_t)j * dim + c]; d2 += v * v; }
        s += 1.0f / (1.0f + d2);
    }
    scratch[i] = s;
}

extern "C" __global__ void tsne_reduce(float* __restrict__ scratch, uint32_t n) {
    __shared__ float partial[256];
    const uint32_t t = threadIdx.x;
    float s = 0.0f;
    for (uint32_t i = t; i < n; i += 256) s += scratch[i];
    partial[t] = s;
    __syncthreads();
    for (uint32_t stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) partial[t] += partial[t + stride];
        __syncthreads();
    }
    if (t == 0) scratch[n] = partial[0];
}

extern "C" __global__ void tsne_grad(const float* __restrict__ y, const float* __restrict__ p,
                                     float* __restrict__ scratch, uint32_t n, uint32_t dim,
                                     float exaggeration) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float zz = scratch[n];
    const float inv_z = zz > 0.0f ? 1.0f / zz : 0.0f;
    float yi[DMAX], g[DMAX];
    for (uint32_t c = 0; c < dim; ++c) { yi[c] = y[(size_t)i * dim + c]; g[c] = 0.0f; }
    const size_t prow = (size_t)i * n;
    for (uint32_t j = 0; j < n; ++j) {
        if (j == i) continue;
        float diff[DMAX];
        float d2 = 0.0f;
        for (uint32_t c = 0; c < dim; ++c) { const float v = yi[c] - y[(size_t)j * dim + c]; diff[c] = v; d2 += v * v; }
        const float kij = 1.0f / (1.0f + d2);
        const float qij = fmaxf(kij * inv_z, 1e-12f);
        const float mult = 4.0f * (exaggeration * p[prow + j] - qij) * kij;
        for (uint32_t c = 0; c < dim; ++c) g[c] += mult * diff[c];
    }
    const size_t gbase = (size_t)n + 1 + (size_t)i * dim;
    for (uint32_t c = 0; c < dim; ++c) scratch[gbase + c] = g[c];
}
