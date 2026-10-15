# Squeeze

## Explore structure. Measure the tradeoffs.

Squeeze is a Python library for dimensionality reduction, combining a
Python/Numba UMAP implementation with ten Rust algorithms. It runs on the CPU
everywhere and moves the heavy kernels to a GPU when one is present. Fit a linear baseline,
compare neighborhood layouts, and inspect the quality you keep when reducing data
to two dimensions.

[Install Squeeze](installation.md){ .md-button .md-button--primary }
[Compare the algorithms](algorithms/index.md){ .md-button }

### One project, several ways to reduce dimensions

| Start here | What you get |
| --- | --- |
| [Quick start](basic_usage.md) | A runnable Digits embedding with a quality measurement |
| [Algorithm guide](algorithms/index.md) | Eleven methods, their actual APIs, and their constraints |
| [Benchmark results](benchmarking.md) | Digits and Fashion-MNIST heatmaps, timing and raw data |
| [New graph methods](algorithms/neighbor-map.md) | NeighborMap and SpectralMap, experimental Rust alternatives |
| [API reference](api.md) | Constructors, supported methods, and composition utilities |

### What is implemented

UMAP offers the richest estimator interface, including out-of-sample transforms.
Rust PCA provides fit and transform; the other Rust methods primarily expose
`fit_transform`. They do **not** all implement the full scikit-learn estimator
contract. [Check the capability table](algorithms/index.md) before building a pipeline.

This is an alpha research library. The benchmarks document particular datasets,
parameters, and machines; they are not a blanket speed or quality guarantee.
Squeeze runs on the CPU with no GPU software installed. When a GPU driver is
present (Vulkan, Metal or DX12; MLX on Apple Silicon) the pairwise, k-NN and
gradient kernels use it automatically; see [GPU compute](gpu.md). No CUDA
installation is needed.

### Built on previous work

Squeeze derives its UMAP implementation from umap-learn. The project keeps that
lineage explicit while developing its own Rust methods and comparison tooling.
See [credits and licensing](credits.md) for attribution.
