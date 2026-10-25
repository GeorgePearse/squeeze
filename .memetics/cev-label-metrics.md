---
upstream:
  - repo: manzt/cev-metrics
    paths:
      - src/lib.rs
      - src/step_function.rs
    ref: main
    last_seen: 3fda725d5db268fd0fa626ec5bf28013addf961c
local:
  paths:
    - src/label_metrics.rs
    - squeeze/evaluation.py
---

# Label confusion and neighbourhood on the Delaunay graph (cev-metrics)

`cev-metrics` is the Rust core behind the Comparative Embedding Visualization (cev) tool: per-class
label matrices computed on a 2D embedding via its Delaunay graph, so two embeddings of the same
points can be compared class by class instead of by one global scalar. squeeze's evaluation suite
was all unlabelled scalars (trustworthiness, continuity, co-ranking, Spearman, density) plus
downstream accuracy, so this view was missing. `src/label_metrics.rs` reimplements the idea and
`squeeze.evaluation` exposes it as `label_confusion`, `label_neighborhood` and
`compare_label_structure`:

- Graph: Delaunay triangulation of the 2D points, undirected, edge weight = Euclidean length.
- Confusion (`n_labels x n_labels` counts): per label `L`, threshold = mean + 3 std of the
  intra-label edge lengths; from every `L` node follow edges no longer than the threshold (one hop);
  row `L` is the label histogram of the reached set. Off-diagonals = points of other classes that
  `L`'s region swallows.
- Neighbourhood (`n_labels x n_labels` scores in [0, 1]): walk up to `max_depth` hops outward from
  the boundary of each confusion set, record `(label, path distance)` for every node reached, then
  score each cell `ECDF(count) * (1 - ECDF(mean distance))` with the ECDFs over all non-empty cells.
- Graph stats: triangle count and ambiguous-circumcircle (non-unique Delaunay) count.

Method: Manz, Trevor et al., "A General Framework for Comparing Embedding Visualizations Across
Class-Label Hierarchies", IEEE VIS 2024 (parent project OzetteTech/comparative-embedding-visualization,
Apache-2.0).

Licence: manzt/cev-metrics carries no licence file, so its code is all-rights-reserved and was NOT
copied. `src/label_metrics.rs` is a clean-room reimplementation from the description above and the
paper, in squeeze's own style. A sync against upstream should compare behaviour, never paste code.

Adaptation notes — deliberate differences from upstream that should not be "fixed" by a sync:

- Upstream's Python `neighborhood(df, max_depth)` silently drops `max_depth` (never passed to Rust).
  squeeze honours it; depth 1 reproduces upstream's effective behaviour.
- Upstream's confusion threshold uses the variance where it says "std" (no square root). squeeze uses
  the real population standard deviation, so thresholds (and off-diagonal counts) differ.
- A label with no intra-label edge gives upstream a NaN threshold (0/0); empty input panics on
  `max()`. squeeze follows no edges for such a label (it counts only itself) and returns empty
  matrices for empty input.
- Input is numpy arrays plus any label dtype (`np.unique` codes), not a pandas categorical; rows and
  columns follow `np.unique(labels)`. No pandas in the new code.
- Delaunay comes from the crates.io `delaunator` crate (ISC), not manzt's git fork; adjacency is a
  CSR array rather than petgraph. Collinear input falls back to the ordered hull as a chain graph.
- Only 2D embeddings are accepted; other widths raise a clear `ValueError`.
- `compare_label_structure(emb_a, emb_b, labels)` returns both matrices for each embedding and their
  difference, which is the comparison framing cev is built around; upstream exposes only the
  per-embedding matrices.
