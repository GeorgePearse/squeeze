// TriMap gradient: one invocation per point, gathering over the triplets it takes part in.
//
// `packed` holds, as u32: offsets[0..n+1], nbr[nbr_off..] (triplet index per CSR entry),
// tag[tag_off..] (role: 0 anchor, 1 positive, 2 negative), triplets[trip_off..] as flat
// (i, j, k) triples, and weights[w_off..] as f32 bit patterns.

struct Params {
    n: u32,
    dim: u32,
    scale: f32,
    nbr_off: u32,
    tag_off: u32,
    trip_off: u32,
    w_off: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read> packed: array<u32>;
@group(0) @binding(2) var<storage, read_write> grad: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const DMAX: u32 = 16u;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= params.n) {
        return;
    }
    let dim = params.dim;
    var g: array<f32, DMAX>;
    for (var c: u32 = 0u; c < dim; c = c + 1u) {
        g[c] = 0.0;
    }
    let e0 = packed[p];
    let e1 = packed[p + 1u];
    for (var e: u32 = e0; e < e1; e = e + 1u) {
        let t = packed[params.nbr_off + e];
        let i = packed[params.trip_off + 3u * t];
        let j = packed[params.trip_off + 3u * t + 1u];
        let k = packed[params.trip_off + 3u * t + 2u];
        var dij: array<f32, DMAX>;
        var dik: array<f32, DMAX>;
        var d_ij: f32 = 0.0;
        var d_ik: f32 = 0.0;
        for (var c: u32 = 0u; c < dim; c = c + 1u) {
            let yi = y[i * dim + c];
            let a = yi - y[j * dim + c];
            let b = yi - y[k * dim + c];
            dij[c] = a;
            dik[c] = b;
            d_ij = d_ij + a * a;
            d_ik = d_ik + b * b;
        }
        if (d_ij - d_ik + 1.0 <= 0.0) {
            continue;
        }
        let sw = params.scale * bitcast<f32>(packed[params.w_off + t]);
        let role = packed[params.tag_off + e];
        for (var c: u32 = 0u; c < dim; c = c + 1u) {
            if (role == 0u) {
                g[c] = g[c] + sw * (dij[c] - dik[c]);
            } else if (role == 1u) {
                g[c] = g[c] - sw * dij[c];
            } else {
                g[c] = g[c] + sw * dik[c];
            }
        }
    }
    for (var c: u32 = 0u; c < dim; c = c + 1u) {
        grad[p * dim + c] = g[c];
    }
}
