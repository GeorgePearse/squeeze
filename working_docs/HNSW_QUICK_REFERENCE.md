# UMAP HNSW-RS Quick Reference Guide

## File Locations Summary

| Component | Path | Lines | Purpose |
|-----------|------|-------|---------|
| **Rust Core** | `/src/lib.rs` | 10 | PyO3 module definition |
| | `/src/hnsw_index.rs` | 382 | Brute-force k-NN search implementation |
| | `/src/metrics.rs` | 171 | Distance metric implementations |
| **Python Wrapper** | `/umap/hnsw_wrapper.py` | 278 | PyNNDescent-compatible API wrapper |
| **Integration** | `/umap/umap_.py` | varies | Backend selection logic |
| **Build Config** | `/Cargo.toml` | 31 | Rust build configuration |
| | `/pyproject.toml` | varies | Python build configuration |
| **Documentation** | `/doc/development_roadmap.md` | - | Architecture & roadmap |
| | `/doc/benchmarking.md` | - | Performance documentation |
| | `/HNSW_ARCHITECTURE_SUMMARY.md` | 20K | Detailed analysis (this repo) |

---

## Key Classes and Methods

### Rust Side

```rust
// src/hnsw_index.rs
pub struct HnswIndex {
    data: Vec<Vec<f32>>,
    n_neighbors: usize,
    metric: String,
    is_angular: bool,
    neighbor_graph_cache: Option<(Vec<Vec<i64>>, Vec<Vec<f32>>)>,
}

impl HnswIndex {
    fn new(data, n_neighbors, metric, m, ef_construction) -> Self
    fn query(queries, k, ef) -> (PyArray<i64>, PyArray<f32>)
    fn neighbor_graph() -> (PyArray<i64>, PyArray<f32>)
    fn prepare() -> None
    fn update(new_data) -> None
    fn compute_distance(a, b) -> f32
}
```

### Python Side

```python
# umap/hnsw_wrapper.py
class HnswIndexWrapper:
    def __init__(data, n_neighbors, metric, metric_kwds, random_state,
                 n_trees, n_iters, max_candidates, low_memory, n_jobs,
                 verbose, compressed)

    def query(query_data, k, epsilon) -> (indices, distances)
    def neighbor_graph() -> (indices, distances) | None
    def prepare() -> None
    def update(X) -> None

    # Properties
    @property neighbor_graph
    @property _angular_trees
    @property _raw_data
```

---

## Supported Distance Metrics

| Metric | Formula | Support |
|--------|---------|---------|
| **Euclidean (L2)** | `√(Σ(a_i - b_i)²)` | ✅ Full |
| **Manhattan (L1)** | `Σ\|a_i - b_i\|` | ✅ Full |
| **Cosine** | `1 - (a·b / \|a\|\|b\|)` | ✅ Full |
| **Chebyshev (L∞)** | `max(\|a_i - b_i\|)` | ✅ Full |
| **Minkowski** | `(Σ\|a_i - b_i\|^p)^(1/p)` | ✅ Full |
| **Hamming** | `count(a_i ≠ b_i)` | ✅ Full |
| **All others** | (PyNNDescent metrics) | ⚠️ Fallback to PyNNDescent |

---

## Architecture Layers

### Layer 1: Python User API
```python
umap.UMAP(metric='euclidean')
  └─> fit_transform(X)
      └─> nearest_neighbors(X, n_neighbors, metric)
```

### Layer 2: Backend Selection
```python
_get_nn_backend(metric, sparse_data, use_hnsw=None)
  Checks:
  1. Explicit user choice (use_hnsw parameter)
  2. HNSW availability (try/except import)
  3. Metric support (in hnsw_metrics set)
  4. Sparse data (scipy.sparse check)
  5. Default to HNSW if all checks pass
```

### Layer 3: Python Wrapper
```python
HnswIndexWrapper
  └─> Converts PyNNDescent params to HNSW params
      └─> Calls Rust backend via PyO3
          └─> Caches results
              └─> Returns PyNNDescent-compatible API
```

### Layer 4: Rust Implementation
```rust
HnswIndex (PyO3 struct)
  └─> Pure Rust implementation
      └─> Vec<Vec<f32>> data storage
          └─> Distance metric dispatch
              └─> Brute-force k-NN search
                  └─> Result caching
```

---

## Parameter Mappings

### From PyNNDescent to HNSW

```
PyNNDescent Parameter → HNSW Parameter → Formula
──────────────────────────────────────────────────────
n_trees               → m (M parameter)
  (None)              → min(64, 5 + round(√n / 20))
  → clamped to        → max(8, min(64, n_trees))

n_iters               → ef_construction (search parameter)
  (None)              → max(5, round(log₂(n)))
  × max_candidates    → n_iters × max_candidates ÷ 2
  → clamped to        → max(200, min(800, value))

epsilon               → ef (query search parameter)
  (0.1 default)       → max(k, int(k × (1.0 + epsilon × 30)))
  → capped at         → min(ef, 500)
```

---

## Current Performance Profile

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Index Construction | O(n × d) | Linear data copy |
| Single Query | O(n × d) | Brute-force full scan |
| K Queries (batch) | O(k_q × n × d) | Per-query O(n×d) |
| Neighbor Graph | O(n² × d) | All-pairs computation, cached |
| Update (add m) | O(m × d) | Append only, no rebuild |

### Space Complexity

| Data | Size | Notes |
|------|------|-------|
| Indexed Data | 4 × n × d bytes | float32 |
| Neighbor Cache | 8 × n × k + 4 × n × k bytes | indices + distances |
| Total | ~4nd + 12nk bytes | Minimal overhead |

### Known Limitations

- **Search Complexity**: O(n) instead of O(log n) for true HNSW
- **No Hierarchical Graph**: Single-layer search only
- **Sparse Data**: Not supported (automatic fallback to PyNNDescent)
- **Determinism**: Fully deterministic (no randomization unlike PyNNDescent)

---

## Optimization Opportunities (Roadmap)

### Phase 2: Immediate (2-3 weeks)
- [ ] Parallel queries with rayon → 4-8x speedup
- [ ] SIMD vectorization for distance metrics → 2-4x
- [ ] Cosine preprocessing (normalize once) → 2x
- [ ] Partial sort instead of full sort → 1.5-2x

### Phase 3: Medium-term (2-3 months)
- [ ] True HNSW hierarchical graph → 10-100x for large n
- [ ] Sparse data support
- [ ] Index serialization (serde infrastructure ready)
- [ ] GPU acceleration (CUDA/Metal)

### Phase 4: Long-term
- [ ] Pluggable ANN backends (FAISS, Annoy, HGG)
- [ ] Runtime algorithm selection
- [ ] Distributed computing support

---

## Testing Status

| Test Category | Status | Notes |
|---------------|--------|-------|
| Unit Tests | ✅ All passing | Distance metrics fully tested |
| Integration Tests | ✅ All passing | UMAP core functionality |
| Trustworthiness Score | 0.978 | Excellent quality |
| Performance Tests | ⚠️ Limited | Benchmarking infrastructure planned |

---

## Known Issues & Design Notes

### Current Implementation Reality

The current implementation is **NOT true HNSW** despite the name:
- Uses brute-force k-NN search (O(n) per query)
- No hierarchical multi-layer graph
- No logarithmic search complexity
- **Intentionally simplified** for correctness verification

This is **Phase 1** of a multi-phase optimization roadmap:
1. **Foundation**: Brute-force with full API (current) ← You are here
2. **Optimization**: Parallel + SIMD improvements
3. **True HNSW**: Hierarchical graph structure
4. **Pluggable Backends**: Multiple algorithm support

### Why This Approach?

1. **De-risks** project by validating infrastructure first
2. **Enables incremental** optimization without breaking changes
3. **Provides fallback** if HNSW proves problematic
4. **Facilitates comparison** with other algorithms

### Unused Dependencies (for Phase 2+)

```toml
rayon = "1.8"           # Will be used for parallelization
ndarray = "0.15"        # Will be used for SIMD vectorization
parking_lot = "0.12"    # Will be used for sync primitives
serde = "1.0"          # Will be used for serialization
serde_json = "1.0"     # Will be used for serialization
```

These are included proactively to simplify Phase 2 implementation.

---

## Backend Selection Logic (Decision Tree)

```
User Query: UMAP(metric='cosine', use_pynndescent=None)
    │
    ├─ Is use_pynndescent explicitly True?
    │  YES → Use PyNNDescent
    │
    ├─ Is use_hnsw explicitly False?
    │  YES → Use PyNNDescent
    │
    ├─ Is HNSW available (import successful)?
    │  NO  → Use PyNNDescent (warn if use_hnsw=True)
    │
    ├─ Is metric supported by HNSW?
    │  (euclidean, l2, manhattan, l1, taxicab, cosine, chebyshev, linfinity, hamming)
    │  NO  → Use PyNNDescent (warn if use_hnsw=True)
    │
    ├─ Is data sparse (scipy.sparse)?
    │  YES → Use PyNNDescent (warn if use_hnsw=True)
    │
    └─ Default to HnswIndexWrapper ✅
```

---

## Build Configuration

### Rust Build (Cargo.toml)
```toml
[package]
name = "_hnsw_backend"
version = "0.1.0"
edition = "2021"
rust-version = "1.74"

[lib]
crate-type = ["cdylib"]  # Compiled Python extension

[profile.release]
opt-level = 3            # Maximum optimizations
lto = "fat"              # Full Link-Time Optimization
codegen-units = 1        # Single-unit for better optimization
strip = true             # Remove symbols for smaller binary
```

### Python Build (pyproject.toml)
```toml
[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "umap"
module-name = "umap._hnsw_backend"
```

### Build Output
- **Binary**: `target/release/lib_hnsw_backend.so`
- **Size**: ~1-2 MB (after stripping)
- **Python Support**: 3.9+ (via ABI3)

---

## Comparison: HNSW vs PyNNDescent

| Feature | HNSW | PyNNDescent | Notes |
|---------|------|------------|-------|
| **Backend** | Rust + PyO3 | Python + Numba | HNSW is compiled |
| **Search Complexity** | O(n) current / O(log n) planned | O(n log n) to O(n) | HNSW will improve |
| **Metrics** | 6 (euclidean, manhattan, cosine, chebyshev, minkowski, hamming) | All | HNSW more limited |
| **Sparse Support** | ❌ No | ✅ Yes | HNSW planned for Phase 3 |
| **Determinism** | ✅ Fully deterministic | ❌ Randomized | HNSW better for reproducibility |
| **Dependencies** | Rust compiler | Python only | HNSW adds build requirement |
| **Memory** | Low | Higher | HNSW more efficient |
| **Dynamic Updates** | ✅ Yes | ✅ Yes | Both support updates |
| **Serialization** | 🚧 Planned | ✅ Yes | HNSW infrastructure ready |

---

## Quick Debugging Checklist

- [ ] Is HNSW imported successfully? Check `umap/hnsw_wrapper.py` line 18-24
- [ ] Is metric in `hnsw_metrics` set? Check `umap/umap_.py` line 290-300
- [ ] Is data sparse (scipy.sparse)? Automatic PyNNDescent fallback
- [ ] Is data float32? Check `HnswIndexWrapper.__init__()` line 91
- [ ] Is index built before querying? Call `prepare()` (no-op but expected)
- [ ] Check cache invalidation after `update()`? Should happen automatically
- [ ] Are parameter conversions correct? Check `_compute_m()` and `_compute_ef_construction()`

---

## Important Git History

| Commit | Message | Change |
|--------|---------|--------|
| `79ebada` | Add Rust-based HNSW nearest neighbor backend | Initial implementation (Phase 1) |
| `31c18f6` | Add comprehensive type annotations | Type safety improvements |
| `38f1d59` | Fix most ruff errors in utils.py | Code quality |
| `281d5bc` | Fix all ruff errors in __init__.py and validation.py | Code quality |

---

## Related Documentation

- **Full Architecture Analysis**: `/HNSW_ARCHITECTURE_SUMMARY.md` (20KB detailed breakdown)
- **Visual Diagram**: `/HNSW_ARCHITECTURE_DIAGRAM.txt` (ASCII diagrams of all layers)
- **Development Roadmap**: `/doc/development_roadmap.md` (multi-phase optimization plan)
- **Performance**: `/doc/benchmarking.md` (UMAP vs other algorithms)

---

## Contact & Contribution

For questions about HNSW-RS architecture:
1. Check `/HNSW_ARCHITECTURE_SUMMARY.md` for detailed analysis
2. Review `/doc/development_roadmap.md` for planned improvements
3. Check recent commits (especially `79ebada`) for context
4. File issue with "HNSW" label on GitHub
