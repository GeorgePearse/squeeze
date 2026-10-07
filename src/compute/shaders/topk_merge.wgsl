// Merge one tile of distances into the running top-k lists of its query rows.
//
// One invocation per query row of the tile. `best_idx` / `best_dist` hold k entries per
// query (global query index), sorted ascending by (distance, index); unfilled slots are
// (0xffffffff, +inf). Ordering by (distance, index) matches the CPU reference exactly, so
// results are identical up to floating-point rounding of the distances themselves.

struct Params {
    q0: u32,          // first global query row of this tile
    q_rows: u32,      // query rows in this tile
    b0: u32,          // global index of the first data row of this tile
    b_cols: u32,      // data rows (columns of `tile`) in this tile
    k: u32,
    tile_stride: u32, // row stride of `tile`
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> tile: array<f32>;
@group(0) @binding(1) var<storage, read_write> best_idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> best_dist: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn before(d: f32, i: u32, d2: f32, i2: u32) -> bool {
    return d < d2 || (d == d2 && i < i2);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if (q >= params.q_rows) {
        return;
    }
    let k = params.k;
    let base = (params.q0 + q) * k;
    let last = base + k - 1u;
    var thr_d = best_dist[last];
    var thr_i = best_idx[last];
    let row = q * params.tile_stride;
    for (var c: u32 = 0u; c < params.b_cols; c = c + 1u) {
        let d = tile[row + c];
        let i = params.b0 + c;
        if (!before(d, i, thr_d, thr_i)) {
            continue;
        }
        // Insertion: shift the tail down one slot until the right position is found.
        var pos = last;
        loop {
            if (pos == base) {
                break;
            }
            let pd = best_dist[pos - 1u];
            let pi = best_idx[pos - 1u];
            if (before(d, i, pd, pi)) {
                best_dist[pos] = pd;
                best_idx[pos] = pi;
                pos = pos - 1u;
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
