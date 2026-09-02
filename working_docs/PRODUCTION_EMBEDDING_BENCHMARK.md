# Production embedding benchmark (2026-09-02)

First run of squeeze on real data instead of Digits: Visia's dinov3 annotation embeddings
(1280-d, cosine), with the production UMAP configuration `n_neighbors=200, min_dist=0,
metric=cosine, init=pca, n_epochs=500`, supervised on class labels. Two datasets:
`deka` (4,621 rows) and `autoclear_ewaste` (149,534 rows). Machine: 16-core VM, no GPU used
for squeeze.

Metrics on fixed random samples: trustworthiness@15 (5k rows), kNN-15 class accuracy in the
2-D layout on a 40k/10k split, silhouette by class (10k rows).

## What the first run showed

| deka, 4,621 rows | seconds | kNN acc 2-D | silhouette | trustworthiness |
| --- | --- | --- | --- | --- |
| umap-learn 0.5.9 | 47.7 | 0.999 | 0.217 | 0.953 |
| squeeze (HNSW backend, defaults) | 33.7 | 0.914 | 0.014 | 0.939 |
| squeeze with PyNNDescent backend | 59.7 | 0.999 | 0.281 | 0.955 |

Faster, but the layout is wrong. Isolating the k-NN stage:

| deka k-NN only, k=200 | seconds | recall vs exact |
| --- | --- | --- |
| squeeze HNSW, defaults (M=8, ef_construction=360) | 22.2 | 0.654 |
| squeeze HNSW, M=32, ef_c=400 | 71.3 | 0.889 |
| squeeze HNSW, M=64, ef_c=800 | 207.5 | 0.961 |
| PyNNDescent | 36.7 | 0.9999 |
| exact blocked matmul (numpy prototype) | 1.7 | 1.000 |

So the HNSW index loses a third of the true 200-NN in 1280 dimensions, and buying the
recall back costs more than brute force. Three things in the index contribute:

- `M` is derived from umap's `n_trees` heuristic (`5 + sqrt(n)/20`), which gives M=8 at
  5k rows and 24 at 150k. That is far too sparse a graph for k=200.
- Construction is a serial `for i in 0..n { insert }` loop; only the self-query is parallel.
- Cosine recomputes both norms on every distance call.

The layout code is fine: given PyNNDescent's graph, squeeze matches umap-learn exactly.

## The exact backend

`ExactKnnIndex` (`src/exact_knn.rs`, wrapper `squeeze/exact_knn.py`): rayon over row blocks,
each block a dense product against the whole dataset via `ndarray::dot`, then a per-row
partial sort (`select_nth_unstable`) for the k smallest distances. Cosine on pre-normalised
rows, euclidean via precomputed norms. Recall 1.0, deterministic. The numpy prototype was
dominated by single-threaded `argpartition` (9.5 s per 2048-row block vs 2.8 s for the
sgemm), which is exactly the part the Rust version parallelises.

`squeeze.UMAP(use_hnsw=None)` now picks it automatically for dense data up to
`EXACT_AUTO_MAX_SAMPLES` (250k) rows with a cosine/euclidean metric; `use_hnsw="exact"`
forces it, `use_hnsw=True/False` keep the old behaviour.

At 150k rows, k-NN stage on 16 cores: PyNNDescent 202 s (recall 0.999), numpy prototype
360 s (recall 1.0). The Rust backend's number goes here once measured.

## Not in scope of this note

GPU numbers from the same benchmark, for context only (the repo is CPU-only by policy):
cuML supervised UMAP on one L40S took 14 s at 150k rows and 35 s at 508k rows with the same
layout quality as umap-learn CPU (118 s and ~2,100 s).

## Build note

On a host with mixed nix / conda toolchains the `openblas-static` link fails. The branch
switches `ndarray-linalg` to `openblas-system` and builds the wheel inside
`rust:1-bookworm` with `libopenblas-dev` and `patchelf` so OpenBLAS is bundled into an
abi3 manylinux wheel.
