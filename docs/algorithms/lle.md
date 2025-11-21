# LLE (Locally Linear Embedding)

## Overview

LLE is a manifold learning algorithm that preserves local linear relationships. It assumes each point can be reconstructed as a weighted linear combination of its neighbors, and finds an embedding where these same reconstruction weights still apply.

## When to Use LLE

**Best for:**
- Smooth manifolds with locally linear structure
- When local geometry matters more than global
- Data where points can be approximated by their neighbors
- Manifold learning on moderate-sized datasets

**Avoid when:**
- Data has many disconnected clusters
- Global structure preservation is important
- Very large datasets
- Data with high noise

## How It Works

### Intuitive Explanation

LLE asks: "How can I describe each point using only its neighbors?" It finds weights that reconstruct each point from its neighbors, then finds low-dimensional coordinates where the same weights still work. This preserves the local "shape" of neighborhoods.

### Algorithm Steps

1. **Find neighbors**: Identify k-nearest neighbors for each point
2. **Compute reconstruction weights**: Find weights W that minimize reconstruction error
3. **Embed**: Find low-dimensional coordinates that minimize reconstruction error using the same weights

### Mathematical Foundation

**Step 1 - Reconstruction weights:**
Minimize: Σᵢ ||xᵢ - Σⱼ Wᵢⱼ xⱼ||² subject to Σⱼ Wᵢⱼ = 1

**Step 2 - Embedding:**
Minimize: Σᵢ ||yᵢ - Σⱼ Wᵢⱼ yⱼ||²

This is equivalent to finding the smallest non-zero eigenvectors of (I - W)ᵀ(I - W).

**Complexity**: O(dn²) for weight computation + O(n³) for eigendecomposition

## Parameters

### `n_components`
- **Type**: int
- **Default**: 2
- **Description**: Number of output dimensions
- **Recommendations**: 2-3 for visualization

### `n_neighbors`
- **Type**: int
- **Default**: 12
- **Description**: Number of neighbors for reconstruction
- **Effect**:
  - Too low: Poor reconstruction, unstable
  - Too high: Loses local linearity assumption
- **Recommendations**:
  - Start with 10-15
  - Should be > n_components
  - Increase for noisy data

### `reg`
- **Type**: float
- **Default**: 1e-3
- **Description**: Regularization parameter for numerical stability
- **Effect**: Prevents singular matrices in weight computation
- **Recommendations**: Usually default is fine; increase if you get numerical errors

## Quick Start

```python
import squeeze
from sklearn.datasets import make_swiss_roll

# Generate manifold data
X, color = make_swiss_roll(n_samples=1000, noise=0.1)

# Apply LLE
lle = squeeze.LLE(n_components=2, n_neighbors=12)
X_embedded = lle.fit_transform(X)

# Visualize
import matplotlib.pyplot as plt
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=color, cmap='viridis', s=5)
plt.title('LLE Embedding')
plt.show()
```

## Examples

### Comparing LLE and Isomap

```python
import squeeze
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

X, color = make_swiss_roll(n_samples=1500, noise=0.1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# LLE
lle = squeeze.LLE(n_components=2, n_neighbors=12)
X_lle = lle.fit_transform(X)
axes[0].scatter(X_lle[:, 0], X_lle[:, 1], c=color, s=5, cmap='viridis')
axes[0].set_title('LLE')

# Isomap
isomap = squeeze.Isomap(n_components=2, n_neighbors=12)
X_isomap = isomap.fit_transform(X)
axes[1].scatter(X_isomap[:, 0], X_isomap[:, 1], c=color, s=5, cmap='viridis')
axes[1].set_title('Isomap')

plt.tight_layout()
plt.show()
```

### Tuning n_neighbors

```python
import squeeze
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
n_neighbors_list = [5, 10, 20, 50]

for ax, n_neighbors in zip(axes, n_neighbors_list):
    lle = squeeze.LLE(n_components=2, n_neighbors=n_neighbors)
    X_emb = lle.fit_transform(X)
    ax.scatter(X_emb[:, 0], X_emb[:, 1], c=color, s=5, cmap='viridis')
    ax.set_title(f'n_neighbors = {n_neighbors}')
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Time Complexity | O(dn²) + O(n³) |
| Memory | O(n²) for weight matrix |
| Scalability | Medium (practical limit ~10k points) |
| Benchmark (Digits) | 11.46s |
| Trustworthiness | 0.512 |

## Strengths and Limitations

### Strengths
- Preserves local geometry well
- No iterative optimization (single eigendecomposition)
- Good for smooth manifolds
- Theoretically elegant

### Limitations
- Assumes locally linear structure
- Sensitive to n_neighbors choice
- Lower trustworthiness on complex data
- Can produce degenerate embeddings
- Doesn't handle non-uniform sampling well
- O(n³) eigendecomposition limits scalability

## Common Issues

### Collapsed Embedding

If points collapse to similar values:
- Increase n_neighbors
- Increase regularization (`reg` parameter)
- Check if data has enough local variation

### Numerical Instability

If you get numerical errors:
```python
# Increase regularization
lle = squeeze.LLE(n_components=2, n_neighbors=15, reg=1e-2)
```

## LLE vs Other Manifold Methods

| Aspect | LLE | Isomap |
|--------|-----|--------|
| Preserves | Local linearity | Geodesic distances |
| Assumption | Points reconstructible from neighbors | Single connected manifold |
| Robustness | More sensitive to parameters | More robust |
| Speed | Similar | Similar |

## Tips

1. **n_neighbors must be > n_components** for the algorithm to work
2. **Start with n_neighbors ≈ 10-15** and adjust based on results
3. **Standardize your data** before applying LLE
4. **Compare with Isomap** to see which captures your manifold better
5. **Use for preprocessing** if you believe local structure is most important

## Citation

```bibtex
@article{roweis2000lle,
  title={Nonlinear dimensionality reduction by locally linear embedding},
  author={Roweis, Sam T and Saul, Lawrence K},
  journal={Science},
  volume={290},
  number={5500},
  pages={2323--2326},
  year={2000}
}
```
