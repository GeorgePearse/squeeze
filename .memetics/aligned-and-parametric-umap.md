---
upstream:
  - repo: lmcinnes/umap
    paths:
      - umap/aligned_umap.py
      - umap/parametric_umap.py
    ref: master
    last_seen: 2050e64a460cccb9af2a3df3915b29627f6d826f
local:
  paths:
    - squeeze/aligned_umap.py
    - squeeze/parametric_umap.py
---

# AlignedUMAP and ParametricUMAP

The two UMAP variants carried over from the umap-learn fork. `AlignedUMAP` embeds a sequence of related
datasets with Procrustes-style regularisation between consecutive layouts (`expand_relations`,
`build_neighborhood_similarities`, `init_from_existing`). `ParametricUMAP` learns an encoder (and
optional decoder) network against the UMAP cross-entropy on sampled graph edges (`construct_edge_dataset`,
`compute_cross_entropy`, `prepare_networks`, save/load helpers). They were kept in-tree so they sit on
squeeze's own `UMAP`, `layouts.optimize_layout_aligned_euclidean`, `spectral_layout` and sparse helpers
rather than on an external umap install.

Adaptation notes:

- Imports are `from squeeze...`; `ParametricUMAP` subclasses `squeeze.UMAP`.
- Upstream's Keras 3 backend handling (TensorFlow / torch optional imports) is kept as-is; the roadmap
  notes a future PyTorch-first rewrite, so do not deepen the TensorFlow dependency when porting changes.
- Keep the fitted-transform contract (`fit`, `transform`, `update` for aligned; `fit`, `transform`,
  `inverse_transform`, `save` / `load_ParametricUMAP` for parametric) identical to upstream.
- Type annotations on module-level helpers are local additions; preserve them.
