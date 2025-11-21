# Squeeze

**High-performance dimensionality reduction for Python**

Squeeze is a fast, CPU-optimized library for dimensionality reduction techniques including UMAP, t-SNE, PCA, and more. Built with a Rust backend and SIMD vectorization for maximum performance.

## Why Squeeze?

Dimensionality reduction "squeezes" high-dimensional data into lower dimensions while preserving structure. Squeeze provides:

- **Multiple Algorithms**: UMAP today, t-SNE, PCA, Isomap, and more coming soon
- **Fast**: 27x faster k-NN construction than PyNNDescent via HNSW with SIMD
- **CPU-Optimized**: No GPU required - runs anywhere
- **Production Ready**: Scikit-learn compatible API

## Installation

```bash
pip install squeeze
```

Or with uv (recommended):

```bash
uv pip install squeeze
```

## Quick Start

```python
import squeeze
from sklearn.datasets import load_digits

digits = load_digits()

# UMAP embedding
umap_embedding = squeeze.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(digits.data)

# t-SNE embedding
tsne_embedding = squeeze.TSNE(perplexity=30, n_iter=1000).fit_transform(digits.data)

# PCA embedding
pca_embedding = squeeze.PCA(n_components=2).fit_transform(digits.data)

# All algorithms share a consistent API: fit_transform(X)
```

## Supported Algorithms

All algorithms are implemented in **Rust** for maximum performance.

| Algorithm | Status | Description |
|-----------|--------|-------------|
| **UMAP** | ✅ Implemented | Uniform Manifold Approximation and Projection |
| **t-SNE** | ✅ Implemented | t-Distributed Stochastic Neighbor Embedding |
| **PCA** | ✅ Implemented | Principal Component Analysis (eigendecomposition) |
| **Isomap** | ✅ Implemented | Isometric Mapping (geodesic distances + MDS) |
| **LLE** | ✅ Implemented | Locally Linear Embedding |
| **MDS** | ✅ Implemented | Multidimensional Scaling (classical + metric SMACOF) |
| **PHATE** | ✅ Implemented | Potential of Heat-diffusion for Affinity-based Trajectory Embedding |
| **TriMap** | ✅ Implemented | Large-scale Dimensionality Reduction Using Triplets |
| **PaCMAP** | ✅ Implemented | Pairwise Controlled Manifold Approximation |

## Benchmark Results

All algorithms benchmarked on the sklearn Digits dataset (1,797 samples, 64 features):

![Benchmark Results](benchmark_results.png)

![Embeddings Comparison](embeddings_comparison.png)

**Key findings:**
- **Best quality**: t-SNE (0.99 trustworthiness) and UMAP (0.98)
- **Best speed/quality tradeoff**: PaCMAP (0.23s, 0.98 trustworthiness)
- **Fastest**: PCA (0.01s)

Run the benchmark yourself:

```bash
just benchmark
```

### k-NN Backend Performance

Squeeze includes a Rust-based HNSW (Hierarchical Navigable Small World) backend with SIMD-accelerated distance computations:

```
k-NN Backend Comparison (sklearn digits dataset)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Backend              Build Time   Recall    Speedup
─────────────────────────────────────────────────
PyNNDescent          6.512s       93.3%     1.00x
HNSW Simple          0.242s       95.1%     26.96x
HNSW Robust α=1.2    0.256s       97.6%     25.46x
```

## Features

### All Algorithms

```python
import squeeze
import numpy as np

# Load your data
X = np.random.randn(1000, 50)  # 1000 samples, 50 features

# PCA - fast linear projection
pca = squeeze.PCA(n_components=2)
X_pca = pca.fit_transform(X)

# t-SNE - preserves local structure
tsne = squeeze.TSNE(n_components=2, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X)

# MDS - preserves pairwise distances
mds = squeeze.MDS(n_components=2, metric=True, n_iter=300)
X_mds = mds.fit_transform(X)

# Isomap - geodesic distances on manifold
isomap = squeeze.Isomap(n_components=2, n_neighbors=10)
X_isomap = isomap.fit_transform(X)

# LLE - local linear relationships
lle = squeeze.LLE(n_components=2, n_neighbors=10)
X_lle = lle.fit_transform(X)

# PHATE - diffusion-based embedding
phate = squeeze.PHATE(n_components=2, k=15, t=10)
X_phate = phate.fit_transform(X)

# TriMap - triplet-based embedding
trimap = squeeze.TriMap(n_components=2, n_inliers=10, n_outliers=5)
X_trimap = trimap.fit_transform(X)

# PaCMAP - pair-based embedding
pacmap = squeeze.PaCMAP(n_components=2, n_neighbors=10)
X_pacmap = pacmap.fit_transform(X)
```

### UMAP with HNSW Backend

```python
import squeeze

# Use the fast HNSW backend (default)
reducer = squeeze.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    use_hnsw=True,  # Default
    hnsw_prune_strategy="robust",  # Better graph quality
    hnsw_alpha=1.2
)
embedding = reducer.fit_transform(data)
```

### Composition Pipeline

Chain multiple reduction techniques:

```python
from squeeze.composition import DRPipeline
from sklearn.decomposition import PCA
import squeeze

# 2048D → 100D → 2D
pipeline = DRPipeline([
    ('pca', PCA(n_components=100)),
    ('umap', squeeze.UMAP(n_components=2))
])
embedding = pipeline.fit_transform(high_dim_data)
```

### Ensemble Methods

Blend multiple algorithms:

```python
from squeeze.composition import EnsembleDR
from sklearn.decomposition import PCA
import squeeze

ensemble = EnsembleDR([
    ('pca', PCA(n_components=2), 0.3),
    ('umap', squeeze.UMAP(n_components=2), 0.7)
], alignment='procrustes')

blended = ensemble.fit_transform(data)
```

### Sparse Data Support

Efficient handling of sparse matrices:

```python
from squeeze.sparse_ops import SparseUMAP
import scipy.sparse as sp

sparse_data = sp.random(10000, 5000, density=0.05, format='csr')
embedding = SparseUMAP(n_components=2).fit_transform(sparse_data)
```

### Evaluation Metrics

```python
from squeeze.metrics import trustworthiness, continuity, DREvaluator

# Single metric
trust = trustworthiness(X_original, X_embedded, k=15)

# All metrics
evaluator = DREvaluator(k=15)
metrics = evaluator.evaluate(X_original, X_embedded)
print(evaluator.summary())
```

## Development

```bash
# Clone the repo
git clone https://github.com/georgepearse/squeeze
cd squeeze

# Install with uv
uv sync --extra dev

# Build Rust extension
uv run maturin develop --release

# Run tests
uv run pytest squeeze/tests/ -v

# Run benchmarks
uv run python benchmark_optimizations.py
```

Or use the justfile:

```bash
just install    # Install deps + build
just test       # Run tests
just benchmark  # Run benchmarks
just lint       # Check code style
```

## Project Philosophy

1. **Algorithm Agnostic**: One library for all DR techniques
2. **Performance First**: SIMD, Rust backend, optimized algorithms
3. **CPU-Focused**: No GPU dependencies - runs everywhere
4. **Research Platform**: Easy experimentation with techniques and parameters
5. **Production Ready**: Reliable, tested, well-documented

## Citation

If you use Squeeze in your research, please cite the original UMAP paper:

```bibtex
@article{mcinnes2018umap,
  title={UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction},
  author={McInnes, Leland and Healy, John and Melville, James},
  journal={arXiv preprint arXiv:1802.03426},
  year={2018}
}
```

## License

BSD 3-Clause License

## Acknowledgments

Squeeze builds on the excellent work of:
- [UMAP](https://github.com/lmcinnes/umap) by Leland McInnes
- [PyNNDescent](https://github.com/lmcinnes/pynndescent) for approximate nearest neighbors
- The scientific Python ecosystem (NumPy, SciPy, scikit-learn, Numba)
