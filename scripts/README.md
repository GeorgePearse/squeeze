# Development scripts

Run these modules from the repository root using the project environment:

| Command | Purpose | Default output |
|---|---|---|
| `uv run --no-sync python -m scripts.benchmark_metrics_heatmap` | Full-algorithm fit-speed and quality comparison | `assets/benchmarks/digits/` |
| `uv run --no-sync python -m scripts.benchmark_metrics_heatmap --dataset fashion-mnist` | Fashion-MNIST comparison | `assets/benchmarks/fashion-mnist/` |
| `uv run --no-sync python -m scripts.benchmark_optimizations` | Older strategy comparison | `assets/benchmarks/strategies/` |
| `uv run --no-sync python -m scripts.benchmark_hybrid_techniques` | Exploratory hybrid comparison | `assets/benchmarks/hybrid/` |
| `uv run --no-sync python -m scripts.benchmark_neighbors` | Repeated-seed graph comparison | See `--help` |
| `uv run --no-sync python -m scripts.benchmark_radar CSV --output-dir PATH --title TITLE` | Redraw radar and fit-time figures from saved results | Explicit output directory |

The first four runners accept `--output-dir PATH`. The full-algorithm runner
stores embeddings and protocol records under `working_docs/heatmap_refresh/`
unless a custom output directory is given, in which case it keeps all files
together. It does not write results to the repository root by default.

`just benchmark-metrics` runs the primary fit-speed and quality suite.
`just benchmark` runs the older strategy suite.

`maintenance/` preserves one-off RST conversion and Ruff repair scripts from
earlier development. They contain historical machine paths and are not part of
the normal build, test, or documentation workflow.
