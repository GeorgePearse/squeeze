# PCA (Principal Component Analysis)

## Overview

PCA is the simplest and fastest dimensionality reduction technique. It finds the directions (principal components) along which your data varies the most and projects the data onto these directions. PCA is a linear method, meaning it can only capture linear relationships in your data.

## When to Use PCA

**Best for:**
- Quick data exploration and visualization
- Pre-processing before other algorithms (reduce 1000D → 50D → 2D)
- Data with linear structure
- When interpretability matters (components have meaning)
- Very large datasets where speed is critical

**Avoid when:**
- Data lies on a curved manifold
- Local cluster structure is important
- Nonlinear relationships dominate

## How It Works

### Intuitive Explanation

Imagine your data as a cloud of points in high-dimensional space. PCA finds the "longest axis" through this cloud—the direction along which points are most spread out. This becomes the first principal component. The second component is the longest axis perpendicular to the first, and so on.

### Mathematical Foundation

1. **Center the data**: Subtract the mean from each feature
2. **Compute covariance matrix**: C = (1/n) X^T X
3. **Eigendecomposition**: Find eigenvectors and eigenvalues of C
4. **Select components**: Keep top k eigenvectors (largest eigenvalues)
5. **Project**: Multiply data by selected eigenvectors

**Complexity**: O(min(n²d, nd²)) where n = samples, d = features

## Parameters

### `n_components`
- **Type**: int
- **Default**: 2
- **Description**: Number of principal components to compute
- **Recommendations**:
  - Use 2-3 for visualization
  - For pre-processing, keep enough to explain 90-95% of variance
  - Check `explained_variance_ratio_` to decide

## Quick Start

```python
import squeeze
import numpy as np

# Generate sample data
X = np.random.randn(1000, 50)

# Basic usage
pca = squeeze.PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Check explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
```

## Attributes After Fitting

| Attribute | Description |
|-----------|-------------|
| `components_` | Principal component vectors (shape: n_components × n_features) |
| `explained_variance_` | Variance explained by each component |
| `explained_variance_ratio_` | Percentage of total variance per component |

## Examples

### Choosing Number of Components

```python
import squeeze
import numpy as np

X = np.random.randn(1000, 100)

# Fit with many components to see variance explained
pca = squeeze.PCA(n_components=20)
pca.fit(X)

# Cumulative variance
cumsum = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")
```

### Pre-processing Pipeline

```python
from squeeze.composition import DRPipeline
import squeeze

# High-dim → Medium-dim → 2D
pipeline = DRPipeline([
    ('pca', squeeze.PCA(n_components=50)),
    ('umap', squeeze.UMAP(n_components=2))
])
embedding = pipeline.fit_transform(high_dim_data)
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Time Complexity | O(min(n²d, nd²)) |
| Memory | O(nd) |
| Scalability | Excellent (millions of samples) |
| Benchmark (Digits) | <0.01s |
| Trustworthiness | 0.829 |

## Strengths and Limitations

### Strengths
- Extremely fast
- Deterministic (no random initialization)
- Interpretable components
- No hyperparameters to tune (except n_components)
- Preserves global variance structure
- Works well as pre-processing step

### Limitations
- Only captures linear relationships
- May miss complex cluster structure
- Components may not align with meaningful features
- Sensitive to feature scaling (standardize first!)

## Tips

1. **Always standardize your data** before PCA if features have different scales
2. **Use PCA first** to reduce dimensionality before slower algorithms like t-SNE
3. **Check explained variance** to ensure you're not losing important information
4. **Visualize components** to understand what each component represents

## Citation

```bibtex
@article{pearson1901pca,
  title={On lines and planes of closest fit to systems of points in space},
  author={Pearson, Karl},
  journal={The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science},
  volume={2},
  number={11},
  pages={559--572},
  year={1901}
}
```
