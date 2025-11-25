# t-SNE (t-Distributed Stochastic Neighbor Embedding)

## Overview

t-SNE is a powerful nonlinear dimensionality reduction technique that excels at preserving local structure and revealing clusters in data. It converts similarities between data points into probability distributions and minimizes the divergence between high-dimensional and low-dimensional representations.

## When to Use t-SNE

**Best for:**
- Visualizing clusters in data
- Exploring high-dimensional datasets
- When local structure matters more than global
- Publication-quality cluster visualizations

**Avoid when:**
- Dataset is very large (>10,000 points)
- You need to preserve global distances
- You need deterministic results (without setting random_state)
- Speed is critical

## How It Works

### Intuitive Explanation

t-SNE asks: "If I had to describe the neighborhood of each point using probabilities, what would that look like?" It creates a probability distribution where nearby points have high probability and far points have low probability. Then it tries to create a 2D embedding with the same probability structure.

The "t" in t-SNE refers to the Student's t-distribution used in the low-dimensional space, which has heavier tails than a Gaussian. This prevents the "crowding problem" where points collapse together.

### Mathematical Foundation

1. **Compute pairwise affinities** in high-D using Gaussian kernels:
   - p(j|i) = exp(-||xi - xj||² / 2σi²) / Σk exp(-||xi - xk||² / 2σi²)
   - σi chosen so perplexity matches target

2. **Symmetrize**: pij = (p(j|i) + p(i|j)) / 2n

3. **Define low-D affinities** using Student's t-distribution:
   - qij = (1 + ||yi - yj||²)^(-1) / Σk,l (1 + ||yk - yl||²)^(-1)

4. **Minimize KL divergence**: KL(P||Q) = Σij pij log(pij/qij)

5. **Optimize** using gradient descent with momentum

**Complexity**: O(n²) for exact, O(n log n) with Barnes-Hut approximation

## Parameters

### `n_components`
- **Type**: int
- **Default**: 2
- **Description**: Output dimensionality
- **Recommendations**: Usually 2 or 3 for visualization

### `perplexity`
- **Type**: float
- **Default**: 30.0
- **Description**: Related to the number of nearest neighbors. Can be thought of as a smooth measure of effective neighborhood size.
- **Effect**:
  - Low (5-10): Focuses on very local structure
  - Medium (30-50): Balanced local/global
  - High (50-100): More global structure
- **Recommendations**:
  - Start with 30
  - Try 5-50 range
  - Should be smaller than n_samples

### `learning_rate`
- **Type**: float
- **Default**: 200.0
- **Description**: Step size for gradient descent
- **Effect**: Too high causes instability, too low causes slow convergence
- **Recommendations**: 100-1000 range, often "auto" works well

### `n_iter`
- **Type**: int
- **Default**: 1000
- **Description**: Number of optimization iterations
- **Recommendations**:
  - 1000 is usually sufficient
  - Increase if embedding hasn't stabilized
  - Watch for convergence

### `early_exaggeration`
- **Type**: float
- **Default**: 12.0
- **Description**: Multiplier for P in early iterations to create space between clusters
- **Effect**: Higher values create more separated clusters initially
- **Recommendations**: Default works well for most cases

### `random_state`
- **Type**: int or None
- **Default**: None
- **Description**: Random seed for reproducibility
- **Recommendations**: Set for reproducible results

### `theta`
- **Type**: float
- **Default**: 0.5
- **Description**: Barnes-Hut approximation parameter. Controls the accuracy/speed tradeoff for the O(n log n) gradient computation.
- **Effect**:
  - 0.0: Exact computation (no approximation)
  - 0.5: Good balance of speed and accuracy (default)
  - 1.0: Faster but less accurate
- **Recommendations**: Use default 0.5 for most cases

### `use_barnes_hut`
- **Type**: bool or None
- **Default**: None (auto-select)
- **Description**: Whether to use Barnes-Hut approximation for gradient computation
- **Effect**:
  - `None`: Auto-selects based on dataset size (uses Barnes-Hut when n > 1000 and n_components == 2)
  - `True`: Always use Barnes-Hut
  - `False`: Always use exact computation
- **Recommendations**: Let it auto-select unless you have specific requirements

### `min_grad_norm`
- **Type**: float
- **Default**: 1e-7
- **Description**: Minimum gradient norm for early stopping. When the gradient becomes smaller than this value, optimization stops.
- **Effect**: Set to 0 to disable early stopping
- **Recommendations**: Default works well; increase for faster (but potentially less optimal) results

### `n_iter_without_progress`
- **Type**: int
- **Default**: 300
- **Description**: Number of iterations without improvement before early stopping triggers
- **Effect**: Lower values stop earlier, higher values allow more exploration
- **Recommendations**: Default 300 balances convergence and computation time

## Quick Start

```python
import squeeze
from sklearn.datasets import load_digits

# Load data
digits = load_digits()
X = digits.data

# Basic t-SNE
tsne = squeeze.TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X)

# Visualize
import matplotlib.pyplot as plt
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=digits.target, cmap='tab10', s=5)
plt.title('t-SNE of Digits Dataset')
plt.show()
```

## Examples

### Tuning Perplexity

```python
import squeeze
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
perplexities = [5, 15, 30, 50]

for ax, perp in zip(axes, perplexities):
    tsne = squeeze.TSNE(perplexity=perp, random_state=42)
    X_emb = tsne.fit_transform(X)
    ax.scatter(X_emb[:, 0], X_emb[:, 1], c=y, s=5)
    ax.set_title(f'Perplexity = {perp}')
```

### With Pre-processing

```python
import squeeze
from sklearn.preprocessing import StandardScaler

# Standardize features
X_scaled = StandardScaler().fit_transform(X)

# PCA pre-processing for speed
pca = squeeze.PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

# t-SNE on reduced data
tsne = squeeze.TSNE(perplexity=30, n_iter=1000)
X_embedded = tsne.fit_transform(X_pca)
```

### Using Barnes-Hut for Large Datasets

```python
import squeeze
import numpy as np

# Large dataset (10,000 points)
X_large = np.random.randn(10000, 100)

# Barnes-Hut is auto-selected for large datasets
# Explicitly enable for smaller datasets:
tsne = squeeze.TSNE(
    perplexity=30,
    theta=0.5,           # Barnes-Hut approximation parameter
    use_barnes_hut=True, # Force Barnes-Hut even for small datasets
    random_state=42
)
X_embedded = tsne.fit_transform(X_large)
```

### With Early Stopping

```python
import squeeze

# Enable early stopping for faster convergence
tsne = squeeze.TSNE(
    perplexity=30,
    n_iter=2000,                    # Max iterations
    min_grad_norm=1e-7,             # Stop when gradient is small
    n_iter_without_progress=300,    # Stop if no improvement
    random_state=42
)
X_embedded = tsne.fit_transform(X)
# May finish before 2000 iterations if converged
```

## Barnes-Hut Approximation

The Barnes-Hut algorithm reduces t-SNE's gradient computation from O(n²) to O(n log n) by using a quadtree to approximate forces from distant points.

### How It Works

1. **Build a quadtree** of all points in the low-dimensional embedding
2. For each point, compute forces from:
   - **Nearby points**: Exact computation
   - **Distant groups**: Approximate using center of mass
3. The `theta` parameter controls the distance threshold:
   - If `node_size / distance < theta`, use approximation
   - Lower theta = more exact, higher theta = faster

### When to Use

| Dataset Size | Recommendation |
|--------------|----------------|
| < 1,000 | Use exact (default) |
| 1,000 - 10,000 | Barnes-Hut recommended (auto-selected) |
| > 10,000 | Barnes-Hut essential |

## Early Stopping

Early stopping monitors the gradient norm and stops optimization when:
1. Gradient norm falls below `min_grad_norm`, OR
2. No improvement for `n_iter_without_progress` iterations

This prevents wasted computation when the embedding has converged.

## Performance Characteristics

| Metric | Exact | Barnes-Hut |
|--------|-------|------------|
| Time Complexity | O(n²) | O(n log n) |
| Memory | O(n²) | O(n) |
| Practical Limit | ~5,000 points | ~100,000 points |
| Accuracy | Exact | Approximate (controlled by theta) |

| Benchmark | Value |
|-----------|-------|
| Digits Dataset (1797 points) | 12.97s |
| Trustworthiness | 0.990 (best) |

## Strengths and Limitations

### Strengths
- Excellent at revealing cluster structure
- Produces visually appealing embeddings
- Handles nonlinear relationships well
- Best-in-class local structure preservation

### Limitations
- Slow for large datasets (O(n²))
- Non-deterministic without setting random_state
- Distances in embedding are not meaningful
- Global structure often distorted
- Sensitive to hyperparameters
- Can create artificial clusters in random data

## Common Pitfalls

1. **Don't interpret distances**: t-SNE distorts global distances. Cluster sizes and between-cluster distances are not meaningful.

2. **Perplexity matters**: Results vary significantly with perplexity. Always try multiple values.

3. **Run multiple times**: Without fixed random_state, results vary between runs.

4. **Don't use for preprocessing**: t-SNE is for visualization only, not for feeding into downstream ML.

5. **Scale your data**: Standardize features before t-SNE.

## Tips

1. **Use PCA first** to reduce to 50-100 dimensions for speed
2. **Set random_state** for reproducible results
3. **Try multiple perplexities** to understand your data
4. **Increase n_iter** if the embedding looks unstable
5. **Consider UMAP** for larger datasets with similar quality

## Citation

```bibtex
@article{vandermaaten2008tsne,
  title={Visualizing data using t-SNE},
  author={Van der Maaten, Laurens and Hinton, Geoffrey},
  journal={Journal of Machine Learning Research},
  volume={9},
  number={Nov},
  pages={2579--2605},
  year={2008}
}
```
