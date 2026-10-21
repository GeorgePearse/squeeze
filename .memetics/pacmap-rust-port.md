---
upstream:
  - repo: YingfanWang/PaCMAP
    paths:
      - source/pacmap/pacmap.py
    ref: master
    last_seen: 2f45defb3cfbb340b550994de30bdfa1535d628a
local:
  paths:
    - src/pacmap.rs
---

# PaCMAP in Rust

A compact Rust/PyO3 reimplementation of PaCMAP (Pairwise Controlled Manifold Approximation) following
the reference `pacmap.py`: three pair types (near neighbours, mid-near pairs, uniform far pairs), the
three loss kernels `10 / (10 + d^2)`, `10000 / (10000 + d^2)` and `1 / (1 + d^2)`, and the three-phase
weight schedule switching at iterations 100 and 200. It was rewritten rather than wrapped so it runs
without numba and shares squeeze's SIMD distance kernels and exact distance matrix.

Adaptation notes — deliberate differences from upstream that should not be "fixed" by a sync:

- Pair generation works from a full exact distance matrix (`mds::compute_distance_matrix`), not an
  approximate kNN graph; sizes are therefore quadratic. Keep the heap-based top-k selection.
- Weight schedule is simplified: `w_MN` starts at 3 rather than annealing from 1000; the phase
  boundaries (100 / 200 / `n_iter`) and final weights `(1, 0, 1)` match upstream.
- Optimiser is plain gradient descent with `lr / (1 + iter/100)` decay and per-iteration centering,
  not Adam. Initial embedding is Gaussian scaled by 100.
- Constructor signature `PaCMAP::new(n_components, n_neighbors, mn_ratio, fp_ratio, n_iter,
  learning_rate, random_state)` and `fit_transform` are the only public surface; no `transform`.
