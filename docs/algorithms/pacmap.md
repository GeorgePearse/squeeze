# PaCMAP (Pairwise Controlled Manifold Approximation)

## Overview

PaCMAP is a modern dimensionality reduction method that achieves an excellent balance between speed and quality. It uses three types of point pairs (near, mid-near, and far) and a three-phase optimization schedule to preserve both local and global structure effectively.

## When to Use PaCMAP

**Best for:**
- When you want the best speed/quality tradeoff
- Large datasets (10k-100k+ points)
- General-purpose visualization
- When UMAP is too slow but you need good quality

**Avoid when:**
- You need the absolute best cluster separation (use t-SNE)
- Very small datasets where any method works
- You need deterministic results without random_state

## How It Works

### Intuitive Explanation

PaCMAP identifies three types of relationships for each point:
1. **Near pairs**: Your closest neighbors (local structure)
2. **Mid-near pairs**: Moderately close points (medium-range structure)
3. **Far pairs**: Random distant points (global structure)

It then optimizes in three phases, gradually shifting focus from global to local structure.

### Algorithm Steps

1. **Identify pairs**:
   - Near pairs: k-nearest neighbors
   - Mid-near pairs: Next closest points (scaled by mn_ratio)
   - Far pairs: Random distant points (scaled by fp_ratio)

2. **Three-phase optimization**:
   - **Phase 1 (0-100 iter)**: Strong far pair repulsion, establish global layout
   - **Phase 2 (100-200 iter)**: Transition, balanced forces
   - **Phase 3 (200-450 iter)**: Focus on near pairs, refine local structure

### Mathematical Foundation

**Loss function with three pair types**:
L = w_near × Σ(near pairs) + w_mid × Σ(mid-near pairs) + w_far × Σ(far pairs)

**Weight schedule** changes across phases:
- Phase 1: High w_far, moderate w_mid, low w_near
- Phase 2: Decreasing w_far, stable w_mid
- Phase 3: Low w_far, low w_mid, high w_near

**Complexity**: O(n × pairs × iterations)

## Parameters

### `n_components`
- **Type**: int
- **Default**: 2
- **Description**: Output dimensions
- **Recommendations**: 2-3 for visualization

### `n_neighbors`
- **Type**: int
- **Default**: 10
- **Description**: Number of near pairs per point
- **Effect**: Higher preserves more local detail
- **Recommendations**: 10-30 range

### `mn_ratio`
- **Type**: float
- **Default**: 0.5
- **Description**: Ratio for mid-near pairs (multiplier on n_neighbors)
- **Effect**: Higher includes more medium-range structure
- **Recommendations**: 0.3-1.0 range

### `fp_ratio`
- **Type**: float
- **Default**: 2.0
- **Description**: Ratio for far pairs (multiplier on n_neighbors)
- **Effect**: Higher emphasizes global structure
- **Recommendations**: 1.0-3.0 range

### `n_iter`
- **Type**: int
- **Default**: 450
- **Description**: Total optimization iterations
- **Recommendations**: 400-500 usually sufficient

### `learning_rate`
- **Type**: float
- **Default**: 1.0
- **Description**: Base learning rate
- **Recommendations**: 0.5-2.0 range

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

# Apply PaCMAP - fast with great quality
pacmap = squeeze.PaCMAP(n_components=2, random_state=42)
X_embedded = pacmap.fit_transform(X)

# Visualize
import matplotlib.pyplot as plt
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=digits.target, cmap='tab10', s=5)
plt.title('PaCMAP Embedding (0.13s, 0.978 trustworthiness)')
plt.show()
```

## Examples

### Comparing with Other Methods

```python
import squeeze
import matplotlib.pyplot as plt
import time

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
methods = [
    ('PaCMAP', squeeze.PaCMAP(random_state=42)),
    ('UMAP', squeeze.UMAP(random_state=42)),
    ('t-SNE', squeeze.TSNE(random_state=42)),
    ('TriMap', squeeze.TriMap(random_state=42)),
]

for ax, (name, reducer) in zip(axes, methods):
    start = time.time()
    X_emb = reducer.fit_transform(X)
    elapsed = time.time() - start
    ax.scatter(X_emb[:, 0], X_emb[:, 1], c=y, s=5, cmap='tab10')
    ax.set_title(f'{name} ({elapsed:.2f}s)')
```

### Tuning the Three-Phase Balance

```python
import squeeze
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

configs = [
    {'mn_ratio': 0.2, 'fp_ratio': 3.0, 'title': 'More Global'},
    {'mn_ratio': 0.5, 'fp_ratio': 2.0, 'title': 'Balanced (default)'},
    {'mn_ratio': 1.0, 'fp_ratio': 1.0, 'title': 'More Local'},
]

for ax, config in zip(axes, configs):
    pacmap = squeeze.PaCMAP(
        mn_ratio=config['mn_ratio'],
        fp_ratio=config['fp_ratio'],
        random_state=42
    )
    X_emb = pacmap.fit_transform(X)
    ax.scatter(X_emb[:, 0], X_emb[:, 1], c=y, s=5, cmap='tab10')
    ax.set_title(config['title'])
```

### Large Dataset

```python
import squeeze
import numpy as np

# Large dataset - PaCMAP handles this well
X_large = np.random.randn(100000, 50)

# Fast and high quality
pacmap = squeeze.PaCMAP(n_components=2)
X_embedded = pacmap.fit_transform(X_large)
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Time Complexity | O(n × pairs × iterations) |
| Memory | O(n × pairs) |
| Scalability | Excellent (handles 100k+ points) |
| Benchmark (Digits) | 0.13s (fastest nonlinear) |
| Trustworthiness | 0.978 (near best) |

**PaCMAP achieves the best speed/quality tradeoff in our benchmarks.**

## Strengths and Limitations

### Strengths
- Excellent speed/quality balance
- Preserves both local and global structure
- Three-phase schedule is well-designed
- Scales to large datasets
- Simple, interpretable pair-based approach

### Limitations
- Slightly lower quality than t-SNE on small data
- Results vary with random initialization
- Parameter tuning can improve results further

## PaCMAP vs Alternatives

| Scenario | Recommendation |
|----------|---------------|
| Best quality, small data | t-SNE |
| Best quality, any size | UMAP |
| Best speed/quality tradeoff | **PaCMAP** |
| Fastest with reasonable quality | PaCMAP or TriMap |
| Trajectory data | PHATE |

## Why PaCMAP Works Well

The three-phase optimization is key:

1. **Phase 1**: Establishes global layout by strongly repelling far pairs
2. **Phase 2**: Transitions smoothly, avoiding sudden changes
3. **Phase 3**: Refines local structure without disrupting global layout

This prevents the "crowding" issues of t-SNE and produces stable, high-quality results.

## Tips

1. **Use defaults first**: PaCMAP's defaults are well-tuned
2. **Increase n_neighbors** for more local detail
3. **Increase fp_ratio** for more global structure preservation
4. **Set random_state** for reproducibility
5. **PaCMAP is often the best choice** for general visualization tasks

## Citation

```bibtex
@article{wang2021pacmap,
  title={Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization},
  author={Wang, Yingfan and Huang, Haiyang and Rudin, Cynthia and Shaposhnik, Yaron},
  journal={Journal of Machine Learning Research},
  volume={22},
  number={201},
  pages={1--73},
  year={2021}
}
```
