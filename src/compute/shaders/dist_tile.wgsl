// Tiled pairwise reduction between a block of `a` rows and a block of `b` rows.
//
// out[(ai - a0) * out_stride + (bi - b0)] = FINAL(sum_c ACC(a[ai, c], b[bi, c]))
//
// The reduction over the feature dimension is tiled through workgroup memory (16 features at
// a time, 16 x 16 outputs per workgroup). `//ACC//` and `//FINAL//` are replaced by the host
// to produce the squared-euclidean, euclidean, manhattan, dot and cosine variants.

struct Params {
    a0: u32,          // first row of `a` in this tile
    a_rows: u32,      // rows of `a` in this tile
    b0: u32,          // first row of `b` in this tile
    b_rows: u32,      // rows of `b` in this tile
    d: u32,           // feature dimension
    out_stride: u32,  // row stride of `out` (>= b_rows)
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 16u;

var<workgroup> a_tile: array<f32, 256>;
var<workgroup> b_tile: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let ty = lid.y;   // local row (a)
    let tx = lid.x;   // local column (b)
    let ai = wid.y * TILE + ty;   // local index into this tile's `a` block
    let bi = wid.x * TILE + tx;   // local index into this tile's `b` block
    let a_ok = ai < params.a_rows;
    let b_ok = bi < params.b_rows;
    let a_row = (params.a0 + ai) * params.d;
    let b_row = (params.b0 + bi) * params.d;

    var acc: f32 = 0.0;
    let d = params.d;
    for (var c0: u32 = 0u; c0 < d; c0 = c0 + TILE) {
        // Cooperative loads: thread (ty, tx) loads feature c0+tx of a-row ty and feature
        // c0+ty of b-row tx.
        let ca = c0 + tx;
        let cb = c0 + ty;
        var av: f32 = 0.0;
        if (a_ok && ca < d) {
            av = a[a_row + ca];
        }
        var bv: f32 = 0.0;
        if (b_ok && cb < d) {
            bv = b[b_row + cb];
        }
        a_tile[ty * TILE + tx] = av;
        b_tile[tx * TILE + ty] = bv;
        workgroupBarrier();
        let cmax = min(TILE, d - c0);
        for (var c: u32 = 0u; c < cmax; c = c + 1u) {
            let x = a_tile[ty * TILE + c];
            let y = b_tile[tx * TILE + c];
            //ACC//
        }
        workgroupBarrier();
    }
    if (a_ok && b_ok) {
        //FINAL//
        out[ai * params.out_stride + bi] = result;
    }
}
