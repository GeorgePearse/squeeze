---
upstream:
  - repo: eamid/trimap
    paths:
      - trimap/trimap_.py
    ref: master
    last_seen: 6a31d53ad7f328ad547ef3fe6d29dadb91f28242
local:
  paths:
    - src/trimap.rs
---

# TriMap in Rust

A Rust/PyO3 reimplementation of TriMap (Large-scale Dimensionality Reduction Using Triplets) following
the reference `trimap_.py`: per-anchor triplets built from `n_inliers` nearest and `n_outliers`
farthest points plus `n_random` random triplets, distance-based triplet weights, and gradient descent on
the triplet loss. Rewritten rather than imported to avoid the numba / annoy dependency chain and to reuse
squeeze's SIMD distances and rayon parallelism.

Adaptation notes — intentional simplifications relative to upstream:

- Neighbour/outlier selection is exact over a dense distance matrix (heap top-k), not annoy-based;
  "outliers" are the farthest points instead of upstream's random far samples.
- Weights are `1 + weight_adj / (1 + d_ij)` when the margin is positive, otherwise 1 — a simplified
  stand-in for upstream's log-scaled weights; `weight_adj` keeps its upstream name.
- Initialisation is scaled Gaussian noise (no PCA init); optimiser is plain gradient descent rather
  than delta-bar-delta.
- Public surface is `TriMap::new(n_components, n_inliers, n_outliers, n_random, weight_adj, n_iter,
  learning_rate, random_state)` plus `fit_transform`.
