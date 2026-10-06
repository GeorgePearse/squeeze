# Benchmark results

The current heatmaps compare **11 algorithms** on Digits and a balanced
2,000-image Fashion-MNIST sample. Both include NeighborMap and SpectralMap.
Each method uses seed 42, one full-data warmup excluded from timing, and a
single thread per numerical library. Time measures `fit_transform`, not scoring.

These are single-seed snapshots. Use the repeated-seed runner below to inspect
variation; no general significance or large-data scalability claim follows from
the heatmaps alone.

## Digits · 1,797 × 64

![Algorithm versus metrics for Digits, including NeighborMap and SpectralMap](assets/benchmarks/digits.png)

[Download the full Digits CSV](assets/benchmarks/digits.csv)

## Fashion-MNIST · 2,000 × 784

![Algorithm versus metrics for Fashion-MNIST, including NeighborMap and SpectralMap](assets/benchmarks/fashion-mnist.png)

[Download the full Fashion-MNIST CSV](assets/benchmarks/fashion-mnist.csv)

The new method names are bold. Green indicates better performance within a column.
Rank trustworthiness, neighbor recall, and transductive classification accuracy
have different meanings: read the [metric definitions](evaluation_metrics.md)
before comparing them. Sample selection is documented in [Datasets](datasets.md).

## Reproduce the heatmaps

[Build the current extension](installation.md), then run:

```bash
uv run --no-sync python benchmark_metrics_heatmap.py
uv run --no-sync python benchmark_metrics_heatmap.py --dataset fashion-mnist --samples 2000
```

Digits writes the root PNG/CSV and saves embeddings/protocol under
`working_docs/heatmap_refresh/digits`. Fashion-MNIST writes all four artifacts
under `working_docs/heatmap_refresh/fashion-mnist`. `--output-dir PATH` keeps
all four output files together at a custom location. Use `--help` for options.

## Repeated-seed comparisons

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
uv run --no-sync python -m scripts.benchmark_neighbors \
  --dataset fashion-mnist --samples 2000 --output working_docs/fashion_comparison
uv run --no-sync python -m scripts.report_neighbors working_docs/fashion_comparison
```

Omit the dataset/sample arguments to use Digits. The runner records per-seed
measurements and coordinates; the report provides interactive scatter plots and
runtime/quality comparisons. The Performance Benchmarks GitHub workflow also
offers a dataset selector for manual runs. Pull-request CI uses Digits.

## Evidence and provenance

- [Heatmap protocol and source revision](https://github.com/GeorgePearse/squeeze/blob/main/working_docs/heatmap_refresh/README.md)
- [Saved Digits embeddings and protocol](https://github.com/GeorgePearse/squeeze/blob/main/working_docs/heatmap_refresh/digits)
- [Saved Fashion-MNIST embeddings and protocol](https://github.com/GeorgePearse/squeeze/blob/main/working_docs/heatmap_refresh/fashion-mnist)
- [Earlier three-seed graph comparison](https://github.com/GeorgePearse/squeeze/blob/main/working_docs/neighbor_benchmarks/README.md)

The measured Fashion implementation is retained at commit `99f4935`. Later
output-path and documentation changes do not change its fits or metrics.
Older working reports may label overlap as trustworthiness; their original
measurements are retained as historical evidence, not rewritten as rank scores.
