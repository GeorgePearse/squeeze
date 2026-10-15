// PaCMAP gradient: one invocation per point, gathering over its CSR adjacency.
// tag 0 = near (attract), 1 = mid-near (attract), 2 = far (repel).
//
// `csr` packs three u32 arrays so the kernel needs only three storage bindings (some
// software Vulkan drivers allow no more): offsets[0..n+1], nbr[nbr_off..], tag[tag_off..].

struct Params {
    n: u32,
    dim: u32,
    w_near: f32,
    w_mn: f32,
    w_fp: f32,
    nbr_off: u32,
    tag_off: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read> csr: array<u32>;
@group(0) @binding(2) var<storage, read_write> grad: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const DMAX: u32 = 16u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }
    let dim = params.dim;
    var yi: array<f32, DMAX>;
    var g: array<f32, DMAX>;
    for (var c: u32 = 0u; c < dim; c = c + 1u) {
        yi[c] = y[i * dim + c];
        g[c] = 0.0;
    }
    let e0 = csr[i];
    let e1 = csr[i + 1u];
    for (var e: u32 = e0; e < e1; e = e + 1u) {
        let j = csr[params.nbr_off + e];
        var diff: array<f32, DMAX>;
        var d2: f32 = 0.0;
        for (var c: u32 = 0u; c < dim; c = c + 1u) {
            let v = yi[c] - y[j * dim + c];
            diff[c] = v;
            d2 = d2 + v * v;
        }
        var coeff: f32;
        let t = csr[params.tag_off + e];
        if (t == 0u) {
            let s = 10.0 + d2;
            coeff = params.w_near * 20.0 / (s * s);
        } else if (t == 1u) {
            let s = 10000.0 + d2;
            coeff = params.w_mn * 20000.0 / (s * s);
        } else {
            let s = 1.0 + d2;
            coeff = -(params.w_fp * 2.0 / (s * s));
        }
        for (var c: u32 = 0u; c < dim; c = c + 1u) {
            g[c] = g[c] + coeff * diff[c];
        }
    }
    for (var c: u32 = 0u; c < dim; c = c + 1u) {
        grad[i * dim + c] = g[c];
    }
}
