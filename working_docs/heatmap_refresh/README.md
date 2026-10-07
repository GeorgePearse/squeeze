# Algorithm vs Metrics heatmap refresh

Both `NeighborMap` (PCA initialization, 160 epochs) and `SpectralMap` (128 block
iterations) now appear alongside the nine established Squeeze algorithms.
All algorithms are refitted with seed 42 and one full-data warmup excluded from
timing. Rayon, OpenMP, Numba and BLAS use one thread. Timing covers the full
`fit_transform`; metrics are calculated afterward. This is a single-seed
illustrative comparison, not a replacement for the earlier three-seed Digits
validation or evidence of generalization.

Digits remains the default (all 1,797 rows, 64 raw pixel features). At George's
request, **Fashion-MNIST is an additional benchmark dataset**. Its standard run
uses 2,000 examples from the official test split: 200 per class sampled without
replacement with NumPy seed 42. Images are flattened to 784 raw float64 pixels;
there is no feature scaling or PCA preprocessing. Labels stratify sampling and
support evaluation only; no labels are passed to the embedding algorithms.

The [official Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist)
is pinned at `b2617bb6d3ffa2e429640350f613e3291e10b141`. Downloads use HTTPS,
validate the authors' published MD5 checksums and IDX payloads, and cache outside
the checkout at `~/.cache/squeeze/fashion-mnist`. Raw dataset files are not
committed. Protocol JSON stores the selected indices, dataset hash and versions.
The measured Fashion generator is retained at commit `99f4935`; the later
output-directory cleanup does not change default paths, fitting or metrics.
`--samples` permits multiples of 10 from 100 through the full 10,000 test images;
larger counts are expensive for the quadratic implementations.

## Outputs

- Digits: root `metrics_heatmap.png` and `metrics_results.csv`.
- Fashion-MNIST: [heatmap](fashion-mnist/metrics_heatmap.png) and
  [CSV](fashion-mnist/metrics_results.csv).
- Per-dataset `embeddings.npz` and `protocol.json` preserve raw coordinates and
  run settings. The Fashion sample is independent of the algorithm seed.

The radar renderer reuses these saved CSV measurements without refitting. New
benchmark runs also emit `metrics_radar.png`, `.svg` and `.html`. The documentation
stores both dataset renderings under `docs/assets/benchmarks/*-radar/`; see the
[radar guide](../../docs/benchmarking.md#circular-performance-profiles) for the
relative normalization and regeneration commands.

## Metric definitions

Trustworthiness at k=5/15/30 now uses sklearn's rank-based implementation.
Continuity is the same rank penalty with original and embedded spaces reversed.
Neighbor recall is the original repository's overlap metric, now named correctly
instead of displaying it three times as trustworthiness, continuity and co-ranking.
The shared evaluation API remains unchanged.

Spearman uses **all non-self pairs** for these sample sizes. Global structure
compares class-centroid distances. Density preservation and linear reconstruction
R² retain the existing evaluator. Silhouette uses KMeans-assigned clusters;
ARI/NMI compare those clusters to the supplied class labels. Classification is
5-fold RandomForest accuracy on embeddings fitted on **all** samples: it is
transductive, not a held-out embedding or supervised Fashion-MNIST benchmark.

Every metric color is normalized within its own column. Time is lower-is-better
and uses reversed log normalization. Cell annotations are raw values, not colors'
normalized values. New algorithm names are bold. Failed algorithms/metrics stop
the run instead of silently dropping rows.

## Reproduce

Build the current Rust extension first, as described in the root README. Install
the benchmark extra (`uv sync --extra dev --extra benchmark --no-install-project`).

```bash
uv run --no-sync python benchmark_metrics_heatmap.py
uv run --no-sync python benchmark_metrics_heatmap.py --dataset fashion-mnist --samples 2000
```

The repeated-seed comparison runner and HTML explorer also support Fashion-MNIST:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
uv run --no-sync python -m scripts.benchmark_neighbors \
  --dataset fashion-mnist --samples 2000 --output working_docs/fashion_comparison
uv run --no-sync python -m scripts.report_neighbors working_docs/fashion_comparison
```

GitHub's Performance Benchmarks workflow offers a dataset selector for manual
runs. Pull requests continue to use Digits; dataset loader tests are offline.
