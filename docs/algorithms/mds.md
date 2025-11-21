# MDS (Multidimensional Scaling)

## Overview

MDS finds a low-dimensional representation that preserves the pairwise distances between points as faithfully as possible. Unlike PCA (which preserves variance) or t-SNE (which preserves neighborhoods), MDS explicitly tries to maintain the actual distances between all pairs of points.

## When to Use MDS

**Best for:**
- When preserving pairwise distances is important
- Data where you have a distance/dissimilarity matrix
- Moderate-sized datasets
- When global structure matters

**Avoid when:**
- Dataset is very large (O(n²) distance computation)
- Local cluster structure is more important than distances
- You need very fast results

## How It Works

### Intuitive Explanation

Imagine you have a table of distances between cities. MDS finds coordinates on a map such that when you measure distances between the plotted cities, they match the original distance table as closely as possible.

### Two Variants

**Classical MDS** (metric=False):
- Uses eigendecomposition of doubly-centered distance matrix
- Fast, closed-form solution
- Equivalent to PCA on centered distance matrix

**Metric MDS / SMACOF** (metric=True):
- Iteratively minimizes "stress" (mismatch between distances)
- More flexible, handles non-Euclidean distances better
- Uses the SMACOF algorithm (Scaling by MAjorizing a Complicated Function)

### Mathematical Foundation

**Classical MDS:**
1. Compute squared distance matrix D²
2. Double-center: B = -½ J D² J, where J = I - 11ᵀ/n
3. Eigendecompose B = VΛVᵀ
4. Coordinates: X = V_k Λ_k^(1/2)

**Metric SMACOF:**
1. Initialize coordinates randomly
2. Compute stress: σ = Σᵢⱼ (dᵢⱼ - d̂ᵢⱼ)²
3. Update coordinates to reduce stress
4. Repeat until convergence

**Complexity**: O(n²d) for classical, O(n² × iterations) for SMACOF

## Parameters

### `n_components`
- **Type**: int
- **Default**: 2
- **Description**: Number of output dimensions
- **Recommendations**: 2-3 for visualization

### `metric`
- **Type**: bool
- **Default**: True
- **Description**: If True, use metric SMACOF; if False, use classical MDS
- **Effect**:
  - True: Iterative optimization, better for non-Euclidean
  - False: Faster eigendecomposition-based solution
- **Recommendations**: Start with True (default)

### `n_iter`
- **Type**: int
- **Default**: 300
- **Description**: Maximum iterations for SMACOF (only used when metric=True)
- **Recommendations**: 300-500 usually sufficient

### `random_state`
- **Type**: int or None
- **Default**: None
- **Description**: Random seed for SMACOF initialization
- **Recommendations**: Set for reproducibility

## Quick Start

```python
import squeeze
import numpy as np

# Generate sample data
X = np.random.randn(500, 20)

# Basic MDS
mds = squeeze.MDS(n_components=2, metric=True)
X_embedded = mds.fit_transform(X)

# Check stress (lower is better)
print(f"Stress: {mds.stress_:.4f}")
```

## Attributes After Fitting

| Attribute | Description |
|-----------|-------------|
| `stress_` | Final stress value (goodness of fit, lower is better) |

## Examples

### Classical vs Metric MDS

```python
import squeeze
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Classical MDS
mds_classical = squeeze.MDS(n_components=2, metric=False)
X_classical = mds_classical.fit_transform(X)
axes[0].scatter(X_classical[:, 0], X_classical[:, 1], c=labels, s=10)
axes[0].set_title('Classical MDS')

# Metric SMACOF MDS
mds_metric = squeeze.MDS(n_components=2, metric=True, n_iter=500)
X_metric = mds_metric.fit_transform(X)
axes[1].scatter(X_metric[:, 0], X_metric[:, 1], c=labels, s=10)
axes[1].set_title(f'Metric MDS (stress={mds_metric.stress_:.4f})')

plt.show()
```

### From Distance Matrix

```python
import squeeze
import numpy as np
from sklearn.metrics import pairwise_distances

# If you have a precomputed distance matrix
X = np.random.randn(200, 50)
D = pairwise_distances(X)

# MDS can work with distances directly
# (internally computes distances, but structure is preserved)
mds = squeeze.MDS(n_components=2)
X_embedded = mds.fit_transform(X)
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Time Complexity | O(n²d) classical, O(n² × iter) SMACOF |
| Memory | O(n²) for distance matrix |
| Scalability | Medium (practical limit ~5-10k points) |
| Benchmark (Digits) | 5.50s |
| Trustworthiness | 0.889 |

## Strengths and Limitations

### Strengths
- Explicitly preserves pairwise distances
- Meaningful coordinates (distances in embedding reflect original)
- Classical MDS is deterministic and fast
- Good for data where distances are the primary structure

### Limitations
- O(n²) memory for distance matrix
- Doesn't focus on local neighborhood structure
- May not reveal clusters as well as t-SNE/UMAP
- SMACOF can get stuck in local minima

## When to Choose MDS Over Other Methods

| Scenario | MDS vs Alternative |
|----------|-------------------|
| Distance preservation critical | MDS > UMAP/t-SNE |
| Cluster visualization | t-SNE/UMAP > MDS |
| Very large data | PCA > MDS |
| Geodesic distances needed | Isomap > MDS |

## Tips

1. **Check stress value**: Lower stress indicates better fit. Stress > 0.2 suggests poor representation.

2. **Try both variants**: Classical MDS is faster; metric MDS often gives better results.

3. **Increase n_iter** if SMACOF hasn't converged (stress still decreasing).

4. **Use for distance data**: MDS is ideal when you care about preserving distances, not just neighborhoods.

## Citation

```bibtex
@book{borg2005modern,
  title={Modern Multidimensional Scaling: Theory and Applications},
  author={Borg, Ingwer and Groenen, Patrick JF},
  year={2005},
  publisher={Springer Science & Business Media}
}

@article{de1977smacof,
  title={Applications of convex analysis to multidimensional scaling},
  author={De Leeuw, Jan},
  journal={Recent Developments in Statistics},
  pages={133--145},
  year={1977}
}
```
