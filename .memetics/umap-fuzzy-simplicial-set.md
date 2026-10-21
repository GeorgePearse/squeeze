---
upstream:
  - repo: lmcinnes/umap
    paths:
      - umap/umap_.py
      - umap/layouts.py
      - umap/spectral.py
      - umap/utils.py
      - umap/distances.py
      - umap/sparse.py
    ref: master
    last_seen: 33d7e6f2f7b10d41c615ebba34ad66c7138b141f
local:
  paths:
    - squeeze/umap_.py
    - squeeze/layouts.py
    - squeeze/spectral.py
    - squeeze/utils.py
    - squeeze/distances.py
    - squeeze/sparse.py
---

# UMAP core (fuzzy simplicial set, SGD layout, spectral init, numba metrics)

Squeeze started life as a fork of umap-learn and its Python UMAP estimator is still the upstream code:
`smooth_knn_dist` / `fuzzy_simplicial_set` / `simplicial_set_embedding` and the `UMAP` estimator in
`umap_.py`, the numba SGD epoch kernels in `layouts.py` (euclidean, generic, inverse, aligned,
densMAP), the spectral / multi-component initialisation in `spectral.py`, the tau-RNG and CSR helpers
in `utils.py`, and the numba dense and sparse metric tables in `distances.py` / `sparse.py`. It was kept
in-tree rather than imported so the neighbour-search stage could be swapped for the Rust HNSW index
and so the rest of squeeze (composition, streaming, benchmarks) can rely on one pinned behaviour.
Upstream bug fixes and new metrics are the drift worth porting.

Adaptation notes — keep these stable when syncing:

- Package is `squeeze`, not `umap`; all intra-package imports are `from squeeze.X import ...`.
- `nearest_neighbors` goes through `_get_nn_backend(metric, sparse_data, use_hnsw)`, which prefers
  the Rust `HnswIndexWrapper` for the supported metric set and falls back to `pynndescent.NNDescent`.
  The `UMAP(use_hnsw=...)` parameter and the warning behaviour must survive any rewrite.
- Type annotations were added throughout; preserve them rather than dropping to upstream's untyped
  signatures.
- Public API (function names, `UMAP` constructor arguments, `fit` / `transform` / `inverse_transform`
  semantics, densMAP support, disconnection handling) tracks upstream and should stay a superset.
- Tests live in `squeeze/tests/test_umap*.py`; upstream test changes are a hint, not a target.
