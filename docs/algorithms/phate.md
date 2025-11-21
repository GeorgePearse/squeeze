# PHATE (Potential of Heat-diffusion for Affinity-based Trajectory Embedding)

## Overview

PHATE is a diffusion-based dimensionality reduction method designed to preserve both local and global structure, with a particular strength in visualizing trajectories and continuous progressions in data. It was developed for biological data like single-cell RNA sequencing where cells form continuous developmental trajectories.

## When to Use PHATE

**Best for:**
- Biological data (single-cell, gene expression)
- Data with continuous trajectories or progressions
- When you want to preserve branching structures
- Developmental or temporal data

**Avoid when:**
- Data has distinct, separated clusters
- Speed is critical
- Very large datasets (>50k points)

## How It Works

### Intuitive Explanation

PHATE simulates heat diffusion on your data. Imagine placing a heat source at each data point and letting heat spread to neighbors. Points that share heat (are connected by diffusion paths) end up close in the embedding. The "potential distance" captures how similarly heat spreads from different points.

### Algorithm Steps

1. **Build adaptive kernel**: Compute affinities with local bandwidth adaptation
2. **Create diffusion operator**: Row-normalize affinity matrix (Markov chain)
3. **Diffuse**: Raise operator to power t (diffusion time)
4. **Compute potential distances**: Log-transform diffused affinities
5. **Embed with MDS**: Apply classical MDS to potential distance matrix

### Mathematical Foundation

1. **Adaptive kernel**: K(i,j) = exp(-||xᵢ - xⱼ||² / (σᵢ σⱼ))
2. **Diffusion operator**: P = D⁻¹K (row-normalized)
3. **Diffused operator**: Pᵗ (matrix power)
4. **Potential distance**: U(i,j) = ||log(Pᵗ[i,:]) - log(Pᵗ[j,:])||
5. **Embedding**: Classical MDS on U

**Complexity**: O(n² × t) for diffusion + O(n²d) for MDS

## Parameters

### `n_components`
- **Type**: int
- **Default**: 2
- **Description**: Output dimensions
- **Recommendations**: 2-3 for visualization

### `k`
- **Type**: int
- **Default**: 15
- **Description**: Number of neighbors for adaptive kernel bandwidth
- **Effect**:
  - Low: More local, may disconnect
  - High: More global, smoother
- **Recommendations**: 5-30 range, start with 15

### `t`
- **Type**: int
- **Default**: 5
- **Description**: Diffusion time (how many steps heat spreads)
- **Effect**:
  - Low (1-3): Emphasizes local structure
  - Medium (5-10): Balanced
  - High (>10): Emphasizes global structure, trajectories
- **Recommendations**:
  - Start with 5
  - Increase for trajectory data
  - Use "auto" selection in some implementations

### `decay`
- **Type**: float
- **Default**: 2.0
- **Description**: Alpha decay for kernel (controls locality)
- **Effect**: Higher values make kernel more local
- **Recommendations**: Usually default works well

### `random_state`
- **Type**: int or None
- **Default**: None
- **Description**: Random seed (used in MDS initialization)

## Quick Start

```python
import squeeze
import numpy as np

# Simulate trajectory data (e.g., differentiation)
t = np.linspace(0, 1, 500)
X = np.column_stack([
    t + 0.1 * np.random.randn(500),
    np.sin(2 * np.pi * t) + 0.1 * np.random.randn(500),
    np.cos(2 * np.pi * t) + 0.1 * np.random.randn(500),
])

# Apply PHATE
phate = squeeze.PHATE(n_components=2, k=15, t=5)
X_embedded = phate.fit_transform(X)

# Visualize trajectory
import matplotlib.pyplot as plt
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=t, cmap='viridis', s=10)
plt.colorbar(label='Pseudotime')
plt.title('PHATE Embedding')
plt.show()
```

## Examples

### Tuning Diffusion Time

```python
import squeeze
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
t_values = [1, 5, 10, 20]

for ax, t in zip(axes, t_values):
    phate = squeeze.PHATE(n_components=2, k=15, t=t)
    X_emb = phate.fit_transform(X)
    ax.scatter(X_emb[:, 0], X_emb[:, 1], c=pseudotime, s=5, cmap='viridis')
    ax.set_title(f't = {t}')
```

### Branching Structure

```python
import squeeze
import numpy as np

# Create branching data (Y-shape)
n = 300
branch1 = np.column_stack([np.linspace(0, 1, n), np.zeros(n)])
branch2 = np.column_stack([1 + np.linspace(0, 0.5, n//2), np.linspace(0, 0.5, n//2)])
branch3 = np.column_stack([1 + np.linspace(0, 0.5, n//2), -np.linspace(0, 0.5, n//2)])
X_branch = np.vstack([branch1, branch2, branch3]) + 0.05 * np.random.randn(n + n, 2)

# Add noise dimensions
X_high = np.column_stack([X_branch, np.random.randn(len(X_branch), 10)])

# PHATE preserves branching structure
phate = squeeze.PHATE(n_components=2, t=10)
X_emb = phate.fit_transform(X_high)
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Time Complexity | O(n² × t) + O(n²d) |
| Memory | O(n²) for diffusion matrix |
| Scalability | Medium (practical limit ~20-50k) |
| Benchmark (Digits) | 6.17s |
| Trustworthiness | 0.828 |

## Strengths and Limitations

### Strengths
- Excellent for trajectory/progression data
- Preserves branching structures
- Handles noise well through diffusion
- Good balance of local and global structure
- Specifically designed for biological data

### Limitations
- Slower than UMAP/PaCMAP
- O(n²) memory limits scalability
- May over-smooth discrete clusters
- Parameter tuning (especially t) can be tricky

## PHATE vs Other Methods

| Data Type | Recommended Method |
|-----------|-------------------|
| Discrete clusters | t-SNE, UMAP |
| Trajectories | PHATE |
| Branching structures | PHATE |
| Mixed | UMAP or PHATE |
| Large data | UMAP, PaCMAP |

## Tips for Biological Data

1. **Gene expression data**: Normalize and log-transform first
2. **Single-cell RNA-seq**: Consider using highly variable genes
3. **Diffusion time**: Increase t if trajectory structure unclear
4. **Multiple runs**: Try different t values to see trajectory at different scales
5. **Combine with pseudotime**: PHATE embeddings often align well with pseudotime

## Automatic Parameter Selection

For optimal t selection, you can look for the "knee" in the Von Neumann entropy of the diffusion operator:

```python
# Conceptual - check entropy at different t values
# Choose t where entropy stabilizes
```

## Citation

```bibtex
@article{moon2019phate,
  title={Visualizing structure and transitions in high-dimensional biological data},
  author={Moon, Kevin R and van Dijk, David and Wang, Zheng and others},
  journal={Nature Biotechnology},
  volume={37},
  number={12},
  pages={1482--1492},
  year={2019}
}
```
