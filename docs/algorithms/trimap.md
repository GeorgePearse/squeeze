# TriMap (Large-scale Dimensionality Reduction Using Triplets)

## Overview

TriMap is a fast dimensionality reduction method that uses triplet constraints to preserve both local and global structure. For each point, it creates triplets of (anchor, similar point, dissimilar point) and optimizes the embedding so that similar points stay closer than dissimilar ones.

## When to Use TriMap

**Best for:**
- Large datasets where speed matters
- When you want both local and global structure
- As a faster alternative to t-SNE
- General-purpose visualization

**Avoid when:**
- You need the best possible quality (use t-SNE/UMAP)
- Very small datasets (overhead not worth it)
- You need deterministic results without setting random_state

## How It Works

### Intuitive Explanation

TriMap asks: "For each point A, should point B be closer than point C?" It creates many such triplet comparisons and optimizes coordinates so these relationships hold. By mixing nearby neighbors (local structure) with random far points (global structure), it preserves both scales.

### Algorithm Steps

1. **Find nearest neighbors**: k-NN for each point
2. **Generate triplets**:
   - Inlier triplets: (anchor, near neighbor, far neighbor)
   - Outlier triplets: (anchor, near neighbor, random far point)
   - Random triplets: (anchor, random point 1, random point 2)
3. **Compute weights**: Based on distance margins
4. **Optimize**: Gradient descent to satisfy triplet constraints

### Mathematical Foundation

**Triplet loss**: For triplet (i, j, k) where j should be closer to i than k:

L = Σ wᵢⱼₖ · loss(||yᵢ - yⱼ||, ||yᵢ - yₖ||)

where loss penalizes when ||yᵢ - yⱼ|| > ||yᵢ - yₖ||

**Complexity**: O(n × (n_inliers + n_outliers + n_random) × n_iter)

## Parameters

### `n_components`
- **Type**: int
- **Default**: 2
- **Description**: Output dimensions
- **Recommendations**: 2-3 for visualization

### `n_inliers`
- **Type**: int
- **Default**: 12
- **Description**: Number of nearest neighbors per point (similar points)
- **Effect**: Higher preserves more local structure
- **Recommendations**: 10-20 range

### `n_outliers`
- **Type**: int
- **Default**: 4
- **Description**: Number of far neighbors per point (dissimilar points)
- **Effect**: Higher preserves more global structure
- **Recommendations**: 3-10 range

### `n_random`
- **Type**: int
- **Default**: 3
- **Description**: Number of random triplets per point
- **Effect**: Adds global structure preservation
- **Recommendations**: 1-5 range

### `n_iter`
- **Type**: int
- **Default**: 800
- **Description**: Number of optimization iterations
- **Recommendations**: 400-1000

### `learning_rate`
- **Type**: float
- **Default**: 0.1
- **Description**: Gradient descent step size
- **Recommendations**: 0.05-0.5 range

### `weight_adj`
- **Type**: float
- **Default**: 50.0
- **Description**: Weight adjustment for triplet importance
- **Effect**: Higher emphasizes hard triplets
- **Recommendations**: Default usually works

### `random_state`
- **Type**: int or None
- **Default**: None
- **Description**: Random seed for reproducibility

## Quick Start

```python
import squeeze
from sklearn.datasets import load_digits

# Load data
digits = load_digits()
X = digits.data

# Apply TriMap
trimap = squeeze.TriMap(n_components=2, n_inliers=12, n_outliers=4, random_state=42)
X_embedded = trimap.fit_transform(X)

# Visualize
import matplotlib.pyplot as plt
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=digits.target, cmap='tab10', s=5)
plt.title('TriMap Embedding')
plt.show()
```

## Examples

### Tuning Local vs Global Balance

```python
import squeeze
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

configs = [
    {'n_inliers': 20, 'n_outliers': 2, 'title': 'More Local (20 inliers, 2 outliers)'},
    {'n_inliers': 12, 'n_outliers': 4, 'title': 'Balanced (12 inliers, 4 outliers)'},
    {'n_inliers': 5, 'n_outliers': 10, 'title': 'More Global (5 inliers, 10 outliers)'},
]

for ax, config in zip(axes, configs):
    trimap = squeeze.TriMap(
        n_inliers=config['n_inliers'],
        n_outliers=config['n_outliers'],
        random_state=42
    )
    X_emb = trimap.fit_transform(X)
    ax.scatter(X_emb[:, 0], X_emb[:, 1], c=y, s=5, cmap='tab10')
    ax.set_title(config['title'])
```

### Large Dataset

```python
import squeeze
import numpy as np

# Large dataset
X_large = np.random.randn(50000, 100)

# TriMap handles this efficiently
trimap = squeeze.TriMap(n_components=2, n_iter=500)
X_embedded = trimap.fit_transform(X_large)  # Much faster than t-SNE
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Time Complexity | O(n × triplets × iterations) |
| Memory | O(n × triplets) |
| Scalability | Good (handles 100k+ points) |
| Benchmark (Digits) | 0.30s |
| Trustworthiness | 0.500 |

## Strengths and Limitations

### Strengths
- Fast, scales to large datasets
- Preserves both local and global structure
- Simple, interpretable triplet constraints
- Good for quick exploration

### Limitations
- Lower quality than t-SNE/UMAP on small datasets
- Triplet sampling introduces variance
- Results depend on random triplet selection
- May not separate clusters as cleanly

## TriMap vs Other Fast Methods

| Method | Speed | Quality | Global Structure |
|--------|-------|---------|------------------|
| TriMap | Fast | Good | Good |
| PaCMAP | Fast | Better | Good |
| UMAP | Medium | Best | Good |
| t-SNE | Slow | Best | Poor |

## Tips

1. **Start with defaults**: The default parameters work well for most data
2. **Increase n_inliers** if local structure is most important
3. **Increase n_outliers** if global structure is most important
4. **Set random_state** for reproducible results
5. **Consider PaCMAP** which often gives better results at similar speed

## Citation

```bibtex
@article{amid2019trimap,
  title={TriMap: Large-scale Dimensionality Reduction Using Triplets},
  author={Amid, Ehsan and Warmuth, Manfred K},
  journal={arXiv preprint arXiv:1910.00204},
  year={2019}
}
```
