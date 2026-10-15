// Exact t-SNE gradient in three passes sharing one bind group:
//   rowsum : scratch[i] = Σ_{j≠i} 1/(1+d²_ij)                       (i < n)
//   reduce : scratch[n] = Σ_i scratch[i]                             (single workgroup)
//   grad   : scratch[n+1 + i*dim + c] = 4 Σ_{j≠i} (ex·p_ij − max(k_ij/z, 1e-12)) k_ij (y_i − y_j)_c
//
// `scratch` packs rowsum, z and grad so the kernel needs only three storage bindings.

struct Params {
    n: u32,
    dim: u32,
    exaggeration: f32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read> p: array<f32>;
@group(0) @binding(2) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const DMAX: u32 = 16u;

@compute @workgroup_size(64, 1, 1)
fn rowsum_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.n;
    if (i >= n) {
        return;
    }
    let dim = params.dim;
    var yi: array<f32, DMAX>;
    for (var c: u32 = 0u; c < dim; c = c + 1u) {
        yi[c] = y[i * dim + c];
    }
    var s: f32 = 0.0;
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        if (j == i) {
            continue;
        }
        var d2: f32 = 0.0;
        for (var c: u32 = 0u; c < dim; c = c + 1u) {
            let v = yi[c] - y[j * dim + c];
            d2 = d2 + v * v;
        }
        s = s + 1.0 / (1.0 + d2);
    }
    scratch[i] = s;
}

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn reduce_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    var s: f32 = 0.0;
    for (var i: u32 = t; i < params.n; i = i + 256u) {
        s = s + scratch[i];
    }
    partial[t] = s;
    workgroupBarrier();
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (t < stride) {
            partial[t] = partial[t] + partial[t + stride];
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        scratch[params.n] = partial[0];
    }
}

@compute @workgroup_size(64, 1, 1)
fn grad_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.n;
    if (i >= n) {
        return;
    }
    let dim = params.dim;
    let ex = params.exaggeration;
    let zz = scratch[n];
    var inv_z: f32 = 0.0;
    if (zz > 0.0) {
        inv_z = 1.0 / zz;
    }
    var yi: array<f32, DMAX>;
    var g: array<f32, DMAX>;
    for (var c: u32 = 0u; c < dim; c = c + 1u) {
        yi[c] = y[i * dim + c];
        g[c] = 0.0;
    }
    let prow = i * n;
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        if (j == i) {
            continue;
        }
        var diff: array<f32, DMAX>;
        var d2: f32 = 0.0;
        for (var c: u32 = 0u; c < dim; c = c + 1u) {
            let v = yi[c] - y[j * dim + c];
            diff[c] = v;
            d2 = d2 + v * v;
        }
        let kij = 1.0 / (1.0 + d2);
        let qij = max(kij * inv_z, 1e-12);
        let mult = 4.0 * (ex * p[prow + j] - qij) * kij;
        for (var c: u32 = 0u; c < dim; c = c + 1u) {
            g[c] = g[c] + mult * diff[c];
        }
    }
    let gbase = n + 1u + i * dim;
    for (var c: u32 = 0u; c < dim; c = c + 1u) {
        scratch[gbase + c] = g[c];
    }
}
