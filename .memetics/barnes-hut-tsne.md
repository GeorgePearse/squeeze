---
upstream:
  - repo: lvdmaaten/bhtsne
    paths:
      - tsne.cpp
      - sptree.cpp
      - sptree.h
    ref: master
    last_seen: 89bef2c0c7313c23617e8ef67a258ed73187397c
local:
  paths:
    - src/tsne.rs
    - src/barnes_hut.rs
---

# Barnes-Hut t-SNE

Rust t-SNE whose large-`n` path follows van der Maaten's reference Barnes-Hut implementation: a 2D
quadtree (`QuadTreeNode` ~ `SPTree`) storing centre of mass and cumulative size, the
`cell_size^2 / dist^2 < theta^2` acceptance test, `(1 + d^2)^-1` Student-t kernels for the repulsive
"non-edge" forces, and the attractive/repulsive split of the gradient with the normalisation `Z`
accumulated from the tree. Written in Rust so it shares squeeze's SIMD distance kernels, rayon
parallelism and PyO3 bindings instead of shelling out to the `bh_tsne` binary.

Adaptation notes:

- Exact O(n^2) gradient remains the default; Barnes-Hut is auto-selected only for `n > 1000` and
  `n_components == 2` (`should_use_barnes_hut`), or forced via `use_barnes_hut`. Keep both paths.
- The tree is 2D only (quadtree); no generalised `SPTree` for 3D.
- Perplexity binary search and early exaggeration follow the standard algorithm; input similarities
  are computed from a dense distance matrix, not a kNN sparse `P` as in upstream's BH mode.
- Constructor `TSNE::new(n_components, perplexity, learning_rate, n_iter, early_exaggeration,
  use_barnes_hut, theta, random_state)` and `fit_transform` are the stable surface.
