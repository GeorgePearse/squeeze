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

All algorithms are implemented in **Rust** for maximum performance. See the [Algorithm Guide](docs/algorithms/index.md) for detailed documentation on each algorithm.

| Algorithm | Status | Description | Docs |
|-----------|--------|-------------|------|
| **UMAP** | ✅ Implemented | Uniform Manifold Approximation and Projection | [Guide](docs/how_umap_works.md) |
| **t-SNE** | ✅ Implemented | t-Distributed Stochastic Neighbor Embedding | [Guide](docs/algorithms/tsne.md) |
| **PCA** | ✅ Implemented | Principal Component Analysis (eigendecomposition) | [Guide](docs/algorithms/pca.md) |
| **Isomap** | ✅ Implemented | Isometric Mapping (geodesic distances + MDS) | [Guide](docs/algorithms/isomap.md) |
| **LLE** | ✅ Implemented | Locally Linear Embedding | [Guide](docs/algorithms/lle.md) |
| **MDS** | ✅ Implemented | Multidimensional Scaling (classical + metric SMACOF) | [Guide](docs/algorithms/mds.md) |
| **PHATE** | ✅ Implemented | Potential of Heat-diffusion for Affinity-based Trajectory Embedding | [Guide](docs/algorithms/phate.md) |
| **TriMap** | ✅ Implemented | Large-scale Dimensionality Reduction Using Triplets | [Guide](docs/algorithms/trimap.md) |
| **PaCMAP** | ✅ Implemented | Pairwise Controlled Manifold Approximation | [Guide](docs/algorithms/pacmap.md) |

### Choosing an Algorithm

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Quick exploration | PCA | Fast, interpretable |
| Cluster visualization | t-SNE, UMAP | Best local structure |
| Large datasets (>100k) | PaCMAP, TriMap | Fast, scalable |
| Biological trajectories | PHATE | Designed for this |
| Best speed/quality | PaCMAP | 0.13s, 0.978 trustworthiness |

## Benchmark Results

All algorithms benchmarked on the sklearn Digits dataset (1,797 samples, 64 features):

### Algorithm vs Metrics Heatmap

![Metrics Heatmap](metrics_heatmap.png)

*Green = best performance, Red = worst performance (relative within each metric)*

### Metrics Comparison Table

| Algorithm | Trust. k=15 | Spearman | Global | Silhouette | Classif. Acc | Time |
|-----------|:-----------:|:--------:|:------:|:----------:|:------------:|:----:|
| **t-SNE** | 0.59 | 0.41 | 0.63 | 0.65 | 0.97 | 24.2s |
| **UMAP** | 0.51 | 0.36 | 0.62 | **0.78** | **0.98** | 9.2s |
| **PaCMAP** | 0.48 | 0.24 | 0.30 | 0.71 | 0.97 | **0.2s** |
| **MDS** | 0.20 | **0.73** | **0.83** | 0.39 | 0.72 | 10.6s |
| **PHATE** | 0.16 | 0.55 | 0.79 | 0.40 | 0.61 | 7.4s |
| **PCA** | 0.15 | 0.58 | 0.82 | 0.39 | 0.61 | **<0.01s** |
| **Isomap** | 0.09 | 0.29 | 0.55 | 0.39 | 0.41 | 6.3s |
| **LLE** | 0.02 | -0.03 | 0.05 | 0.32 | 0.13 | 13.7s |
| **TriMap** | 0.01 | -0.06 | 0.05 | 0.33 | 0.09 | 0.6s |

**Legend:**
- **Trust. k=15**: Trustworthiness - local neighborhood preservation (higher = better)
- **Spearman**: Distance correlation - global structure preservation (higher = better)
- **Global**: Inter-cluster distance preservation (higher = better)
- **Silhouette**: Cluster separation quality (higher = better)
- **Classif. Acc**: 5-fold CV classification accuracy (higher = better)

### Overall Rankings

| Rank | Algorithm | Best For |
|:----:|-----------|----------|
| 1 | **t-SNE** | Local structure, cluster visualization |
| 2 | **UMAP** | Balanced local/global, fast |
| 3 | **MDS** | Global structure preservation |
| 4 | **PaCMAP** | Speed + quality tradeoff |
| 5 | **PHATE** | Biological trajectories |

### Hybrid Techniques

Intelligent combinations of algorithms can outperform individual methods:

![Hybrid Comparison](hybrid_comparison.png)

| Hybrid Technique | Trust. k=15 | Silhouette | Classif. Acc | Time | Key Benefit |
|------------------|:-----------:|:----------:|:------------:|:----:|-------------|
| **PCA(50)→t-SNE** | **0.59** | 0.66 | 0.97 | 32.3s | Best local structure |
| **PCA(30)→UMAP** | 0.52 | **0.80** | 0.98 | 6.5s | Best silhouette, 35% faster |
| **Multi-scale UMAP** | 0.51 | 0.77 | 0.98 | 16.4s | Captures multiple scales |
| **MDS+UMAP Ensemble** | 0.34 | 0.63 | 0.91 | 18.8s | Best global structure |
| **Progressive PaCMAP→UMAP** | 0.51 | 0.76 | 0.98 | 8.7s | Fast + refined |

**Key Findings:**
- **PCA(50)→t-SNE** achieves the best trustworthiness (0.59), beating pure t-SNE
- **PCA(30)→UMAP** is Pareto optimal: 35% faster than UMAP with better silhouette
- Hybrid pipelines inherit PCA's noise reduction + manifold method's structure preservation

```python
from squeeze.composition import DRPipeline, EnsembleDR, ProgressiveDR
from sklearn.decomposition import PCA
import squeeze

# Best overall: PCA → t-SNE
pipeline = DRPipeline([
    ('pca', PCA(n_components=50)),
    ('tsne', squeeze.TSNE(n_components=2))
])

# Fastest high-quality: PCA → UMAP  
pipeline = DRPipeline([
    ('pca', PCA(n_components=30)),
    ('umap', squeeze.UMAP(n_components=2))
])

# Multi-scale structure
ensemble = EnsembleDR([
    ('local', squeeze.UMAP(n_neighbors=5), 0.5),
    ('global', squeeze.UMAP(n_neighbors=30), 0.5)
], blend_mode='procrustes')
```

Run the hybrid benchmark:

```bash
python benchmark_hybrid_techniques.py
```

![Benchmark Results](benchmark_results.png)

![Embeddings Comparison](embeddings_comparison.png)

Run the base benchmark:

```bash
just benchmark              # Quick benchmark
python benchmark_metrics_heatmap.py  # Full metrics heatmap
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

Comprehensive metrics for evaluating DR quality:

```python
from squeeze.evaluation import trustworthiness, continuity, DREvaluator, quick_evaluate

# Quick evaluation (3 core metrics)
metrics = quick_evaluate(X_original, X_embedded, k=15)
print(f"Trustworthiness: {metrics['trustworthiness']:.3f}")
print(f"Continuity: {metrics['continuity']:.3f}")
print(f"Spearman: {metrics['spearman_correlation']:.3f}")

# Comprehensive evaluation
evaluator = DREvaluator(X_original, X_embedded, labels=y, method_name='UMAP')
report = evaluator.evaluate_all()
print(report)

# Individual metrics
from squeeze.evaluation import (
    spearman_distance_correlation,
    global_structure_preservation,
    local_density_preservation,
    clustering_quality,
    classification_accuracy,
)
```

See [Evaluation Metrics Guide](docs/evaluation_metrics.md) for full documentation.

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

Apache License 2.0

## Acknowledgments

Squeeze builds on the excellent work of:
- [UMAP](https://github.com/lmcinnes/umap) by Leland McInnes
- [PyNNDescent](https://github.com/lmcinnes/pynndescent) for approximate nearest neighbors
- The scientific Python ecosystem (NumPy, SciPy, scikit-learn, Numba)
