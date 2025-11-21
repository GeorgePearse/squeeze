# Isomap (Isometric Mapping)

## Overview

Isomap is a manifold learning algorithm that preserves geodesic distances—the shortest path distances along the manifold surface—rather than straight-line Euclidean distances. This makes it excellent for "unfolding" curved manifolds like the famous Swiss roll.

## When to Use Isomap

**Best for:**
- Data lying on a curved manifold (Swiss roll, S-curve)
- When you want to "unfold" nonlinear structure
- Recovering intrinsic geometry of data
- Data with clear manifold structure

**Avoid when:**
- Data has disconnected components (breaks shortest path)
- Very large datasets (O(n²) complexity)
- Data with holes in the manifold
- Noisy data with unclear manifold structure

## How It Works

### Intuitive Explanation

Imagine an ant walking on a curved surface (like a rolled-up piece of paper). Isomap measures distances the way the ant would—by walking along the surface—rather than drilling through it. Then it finds a flat representation where these "walking distances" are preserved.

### Algorithm Steps

1. **Build k-NN graph**: Connect each point to its k nearest neighbors
2. **Compute geodesic distances**: Use Dijkstra's algorithm to find shortest path between all pairs
3. **Apply classical MDS**: Embed using the geodesic distance matrix

### Mathematical Foundation

1. **Neighborhood graph**: G where edge (i,j) exists if j ∈ kNN(i)
2. **Geodesic distances**: dG(i,j) = shortest path in G from i to j
3. **Classical MDS**: Eigendecomposition of doubly-centered squared geodesic distance matrix

**Complexity**: O(n² log n) for Dijkstra + O(n²d) for MDS

## Parameters

### `n_components`
- **Type**: int
- **Default**: 2
- **Description**: Number of output dimensions
- **Recommendations**: 2-3 for visualization, higher for preprocessing

### `n_neighbors`
- **Type**: int
- **Default**: 10
- **Description**: Number of neighbors for k-NN graph construction
- **Effect**:
  - Too low: Graph may become disconnected
  - Too high: "Short-circuits" the manifold
- **Recommendations**:
  - Start with 10-15
  - Increase if you get disconnected graph errors
  - Decrease if manifold structure is lost

## Quick Start

```python
import squeeze
import numpy as np

# Generate Swiss roll data
from sklearn.datasets import make_swiss_roll
X, color = make_swiss_roll(n_samples=1000, noise=0.1)

# Apply Isomap
isomap = squeeze.Isomap(n_components=2, n_neighbors=12)
X_embedded = isomap.fit_transform(X)

# The Swiss roll should be "unrolled"
import matplotlib.pyplot as plt
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=color, cmap='viridis', s=5)
plt.title('Isomap Unrolls the Swiss Roll')
plt.show()
```

## Examples

### Tuning n_neighbors

```python
import squeeze
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

X, color = make_swiss_roll(n_samples=1500, noise=0.1)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
n_neighbors_list = [5, 10, 20, 50]

for ax, n_neighbors in zip(axes, n_neighbors_list):
    isomap = squeeze.Isomap(n_components=2, n_neighbors=n_neighbors)
    try:
        X_emb = isomap.fit_transform(X)
        ax.scatter(X_emb[:, 0], X_emb[:, 1], c=color, s=5, cmap='viridis')
        ax.set_title(f'n_neighbors = {n_neighbors}')
    except Exception as e:
        ax.set_title(f'n_neighbors = {n_neighbors}\n(Error: disconnected)')
```

### Classic Manifold Examples

```python
import squeeze
from sklearn.datasets import make_s_curve, make_swiss_roll

# S-curve
X_s, color_s = make_s_curve(n_samples=1000, noise=0.1)
X_s_embedded = squeeze.Isomap(n_components=2, n_neighbors=10).fit_transform(X_s)

# Swiss roll
X_swiss, color_swiss = make_swiss_roll(n_samples=1000, noise=0.1)
X_swiss_embedded = squeeze.Isomap(n_components=2, n_neighbors=10).fit_transform(X_swiss)
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Time Complexity | O(n² log n) + O(n²d) |
| Memory | O(n²) for distance matrix |
| Scalability | Poor (practical limit ~5k points) |
| Benchmark (Digits) | 6.07s |
| Trustworthiness | 0.658 |

## Strengths and Limitations

### Strengths
- Excellent for manifold "unfolding"
- Preserves geodesic (intrinsic) distances
- No iterative optimization (deterministic)
- Theoretically grounded

### Limitations
- Fails if graph is disconnected
- Sensitive to noise and outliers
- "Short-circuits" with too many neighbors
- O(n²) scaling limits to small datasets
- Assumes single connected manifold
- Lower trustworthiness on non-manifold data

## Common Issues

### Disconnected Graph Error

If you get an error about disconnected components:

```python
# Solution 1: Increase n_neighbors
isomap = squeeze.Isomap(n_neighbors=20)  # Try larger value

# Solution 2: Remove outliers first
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20)
mask = lof.fit_predict(X) == 1
X_clean = X[mask]
```

### Short-Circuiting

If the manifold structure is lost:
- The embedding looks like PCA
- Decrease n_neighbors to prevent "cutting across" the manifold

## When to Choose Isomap Over Other Methods

| Scenario | Recommendation |
|----------|---------------|
| Clear manifold structure | Isomap |
| Cluster visualization | t-SNE/UMAP |
| Large dataset | UMAP/PaCMAP |
| Unknown structure | UMAP |
| Preserving local neighborhoods | LLE |

## Tips

1. **Visualize your data first** in 3D to see if it has manifold structure
2. **Start with default n_neighbors** (10) and adjust based on results
3. **Check for disconnected components** before fitting
4. **Compare with LLE** which also does manifold learning but differently
5. **Use for preprocessing** when you know data lies on a manifold

## Citation

```bibtex
@article{tenenbaum2000isomap,
  title={A global geometric framework for nonlinear dimensionality reduction},
  author={Tenenbaum, Joshua B and De Silva, Vin and Langford, John C},
  journal={Science},
  volume={290},
  number={5500},
  pages={2319--2323},
  year={2000}
}
```
