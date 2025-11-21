# Dimensionality Reduction Evaluation Metrics

Squeeze provides a comprehensive suite of metrics for evaluating the quality of dimensionality reduction embeddings. This guide explains what each metric measures, how to interpret the results, and when to use each one.

## Quick Start

```python
from squeeze import UMAP
from squeeze.evaluation import DREvaluator, quick_evaluate, trustworthiness
from sklearn.datasets import load_digits

# Load data
X, y = load_digits(return_X_y=True)

# Create embedding
reducer = UMAP(n_components=2, random_state=42)
X_reduced = reducer.fit_transform(X)

# Quick evaluation (3 core metrics)
metrics = quick_evaluate(X, X_reduced, k=15)
print(f"Trustworthiness: {metrics['trustworthiness']:.3f}")
print(f"Continuity: {metrics['continuity']:.3f}")
print(f"Spearman correlation: {metrics['spearman_correlation']:.3f}")

# Comprehensive evaluation
evaluator = DREvaluator(X, X_reduced, labels=y, reducer=reducer, method_name='UMAP')
report = evaluator.evaluate_all()
print(report)
```

## Metric Categories

| Category | Metrics | What It Measures |
|----------|---------|------------------|
| **Local Structure** | Trustworthiness, Continuity, Co-ranking | Neighborhood preservation |
| **Global Structure** | Spearman correlation, Global preservation, Density | Large-scale relationships |
| **Reconstruction** | RMSE, R-squared | Information loss |
| **Stability** | Bootstrap stability, Noise robustness | Reproducibility |
| **Downstream** | Clustering quality, Classification accuracy | Practical usefulness |

---

## Local Structure Metrics

### Trustworthiness

**Question:** "Are my k nearest neighbors in the embedding actually neighbors in the original space?"

**Intuition:** High trustworthiness means the embedding doesn't create "fake" neighbors - points that appear close in the embedding but were actually far apart originally.

```python
from squeeze.evaluation import trustworthiness

T = trustworthiness(X_original, X_reduced, k=15)
print(f"Trustworthiness: {T:.3f}")
```

**Interpretation:**

| Score | Quality | Meaning |
|-------|---------|---------|
| > 0.95 | Excellent | Almost all neighbors preserved |
| 0.90 - 0.95 | Good | Minor neighbor changes |
| 0.80 - 0.90 | Fair | Some artificial clustering |
| < 0.80 | Poor | Significant structure distortion |

**Sensitivity to k:**

- `k=5`: Very local structure (immediate neighbors)
- `k=15`: Standard choice (balanced view)
- `k=30`: Semi-local structure (larger neighborhoods)

---

### Continuity

**Question:** "Are my original neighbors still neighbors in the embedding?"

**Intuition:** High continuity means the embedding doesn't "tear apart" real neighborhoods - points that were close originally remain close in the embedding.

```python
from squeeze.evaluation import continuity

C = continuity(X_original, X_reduced, k=15)
print(f"Continuity: {C:.3f}")
```

**Trustworthiness vs Continuity:**

| Pattern | Meaning | Example |
|---------|---------|---------|
| T >> C | Artificial clusters created | Points grouped together that shouldn't be |
| C >> T | Original structure torn apart | Natural clusters split up |
| T ≈ C (both high) | Balanced, good embedding | Ideal outcome |

---

### Co-ranking Quality

**Question:** "How well do distance rankings agree between original and embedded space?"

```python
from squeeze.evaluation import co_ranking_quality

Q = co_ranking_quality(X_original, X_reduced, k=15)
print(f"Co-ranking quality: {Q:.3f}")
```

**Advantages:**
- Captures both local and global structure in a single metric
- More robust than trustworthiness alone
- Based on ranking, not absolute distances

---

## Global Structure Metrics

### Spearman Distance Correlation

**Question:** "If two points were far apart originally, are they still far apart?"

**Intuition:** Measures whether the overall geometry is preserved. Important when you care about relationships between distant clusters.

```python
from squeeze.evaluation import spearman_distance_correlation

rho = spearman_distance_correlation(X_original, X_reduced)
print(f"Spearman rho: {rho:.3f}")
```

**Interpretation:**

| Score | Quality |
|-------|---------|
| > 0.90 | Excellent global preservation |
| 0.80 - 0.90 | Good |
| 0.60 - 0.80 | Fair (local methods like t-SNE) |
| < 0.60 | Poor global structure |

**Note:** This metric is O(n²), so it subsamples for large datasets (default max 5000 samples).

---

### Global Structure Preservation

**Question:** "Are inter-cluster distances preserved?"

**Requires:** Cluster/class labels

```python
from squeeze.evaluation import global_structure_preservation

G = global_structure_preservation(X_original, X_reduced, labels=y)
print(f"Global structure: {G:.3f}")
```

This computes correlation between cluster centroid distances in original vs embedded space.

---

### Local Density Preservation

**Question:** "Are dense regions still dense and sparse regions still sparse?"

**Intuition:** Important for methods like densMAP that explicitly preserve density. Also critical when downstream clustering depends on density.

```python
from squeeze.evaluation import local_density_preservation

D = local_density_preservation(X_original, X_reduced, k=15)
print(f"Density preservation: {D:.3f}")
```

**Typical values:**

| Method | Expected Score |
|--------|---------------|
| densMAP | 0.85 - 0.95 |
| UMAP | 0.70 - 0.85 |
| t-SNE | 0.65 - 0.80 |
| PCA | 0.75 - 0.90 |

---

## Reconstruction Metrics

### Reconstruction Error

**Question:** "Can we reconstruct the original data from the embedding?"

```python
from squeeze.evaluation import reconstruction_error

error = reconstruction_error(X_original, X_reduced)
print(f"Normalized RMSE: {error['normalized_rmse']:.3f}")
print(f"R-squared: {error['r2']:.3f}")
```

**Returns:**
- `mse`: Mean squared error
- `rmse`: Root mean squared error
- `normalized_rmse`: RMSE / std(original) - lower is better
- `r2`: Coefficient of determination - higher is better

**Interpretation:**

| Normalized RMSE | Quality |
|-----------------|---------|
| < 0.1 | Excellent (almost no information loss) |
| 0.1 - 0.3 | Good |
| 0.3 - 0.5 | Fair |
| > 0.5 | Poor (significant information loss) |

---

## Stability Metrics

### Bootstrap Stability

**Question:** "Do I get the same result with different random samples?"

```python
from squeeze.evaluation import bootstrap_stability
from squeeze import UMAP

stability = bootstrap_stability(X, UMAP(random_state=42), n_bootstrap=10)
print(f"Stability score: {stability['stability_score']:.3f}")
```

**Interpretation:**

| Score | Stability |
|-------|-----------|
| > 0.95 | Very stable (deterministic-like) |
| 0.85 - 0.95 | Stable (minor variations) |
| 0.70 - 0.85 | Moderately stable |
| < 0.70 | Unstable (results vary significantly) |

---

### Noise Robustness

**Question:** "How much does the embedding change when I add noise?"

```python
from squeeze.evaluation import noise_robustness

robustness = noise_robustness(X, UMAP(), noise_levels=[0.01, 0.05, 0.1])
print(robustness)
# {0.01: 0.98, 0.05: 0.95, 0.1: 0.92}
```

Higher scores mean more robust to noise at that level.

---

### Parameter Sensitivity

**Question:** "How sensitive is the method to parameter choices?"

```python
from squeeze.evaluation import parameter_sensitivity
from squeeze import UMAP

sensitivity = parameter_sensitivity(X, UMAP, {
    'n_neighbors': [5, 15, 30],
    'min_dist': [0.0, 0.1, 0.5]
})

for param, results in sensitivity.items():
    print(f"{param}: mean sensitivity = {results['mean_sensitivity']:.3f}")
```

Lower mean sensitivity = more robust to parameter choice.

---

## Downstream Task Metrics

### Clustering Quality

**Question:** "How good are clusters in the embedded space?"

```python
from squeeze.evaluation import clustering_quality

metrics = clustering_quality(X_reduced, labels_true=y)
print(f"Silhouette: {metrics['silhouette_score']:.3f}")
print(f"Adjusted Rand Index: {metrics['adjusted_rand_index']:.3f}")
```

**Returns:**
- `silhouette_score`: [-1, 1], higher is better
- `calinski_harabasz`: Higher is better
- `davies_bouldin`: Lower is better
- `adjusted_rand_index`: [0, 1] if labels provided
- `normalized_mutual_info`: [0, 1] if labels provided

---

### Classification Accuracy

**Question:** "Can a classifier achieve good accuracy using the embedded features?"

```python
from squeeze.evaluation import classification_accuracy

results = classification_accuracy(X_reduced, y, cv=5)
print(f"Accuracy: {results['mean_accuracy']:.3f} +/- {results['std_accuracy']:.3f}")
```

Higher accuracy means the embedding preserves class-relevant information.

---

## Comprehensive Evaluation

### Using DREvaluator

The `DREvaluator` class provides a unified interface for all metrics:

```python
from squeeze.evaluation import DREvaluator
from squeeze import UMAP

# Create embedding
reducer = UMAP(n_components=2, random_state=42)
X_reduced = reducer.fit_transform(X)

# Comprehensive evaluation
evaluator = DREvaluator(
    X_original=X,
    X_reduced=X_reduced,
    labels=y,           # Optional: for supervised metrics
    reducer=reducer,    # Optional: for stability metrics
    method_name='UMAP'
)

# Run all evaluations
report = evaluator.evaluate_all(
    k_values=[5, 15, 30],    # k values for local metrics
    include_stability=True,  # Include bootstrap stability
    include_noise=False,     # Skip noise robustness (slower)
    n_bootstrap=10
)

# Print summary
print(report)

# Get as dictionary
results_dict = report.to_dict()
```

### Selective Evaluation

You can also run specific evaluations:

```python
# Local structure only
local = evaluator.evaluate_local_structure(k_values=[5, 15, 30])

# Global structure only
global_struct = evaluator.evaluate_global_structure()

# Clustering only
clustering = evaluator.evaluate_clustering()

# Classification only
classification = evaluator.evaluate_classification(cv=5)
```

---

## Method Comparison

### Typical Metric Values by Method

| Method | Trustworthiness | Spearman | Global | Stability |
|--------|-----------------|----------|--------|-----------|
| **t-SNE** | 0.95 | 0.55 | 0.50 | 0.85 |
| **UMAP** | 0.92 | 0.80 | 0.85 | 0.95 |
| **PCA** | 0.80 | 0.95 | 0.95 | 1.00 |
| **PaCMAP** | 0.90 | 0.85 | 0.90 | 0.90 |
| **Isomap** | 0.85 | 0.90 | 0.90 | 0.98 |

### Choosing the Right Metrics

| Use Case | Primary Metrics |
|----------|-----------------|
| **Visualization** | Trustworthiness, Continuity |
| **Clustering downstream** | Silhouette, Density preservation |
| **Scientific publication** | All local + global + stability |
| **Production pipeline** | Trustworthiness, Stability, Classification |
| **Exploratory analysis** | Quick evaluate (T, C, Spearman) |

---

## Quick Reference

### One-liner Evaluation

```python
from squeeze.evaluation import quick_evaluate

# Core metrics in one call
metrics = quick_evaluate(X_original, X_reduced, k=15)
```

### Full Report

```python
from squeeze.evaluation import DREvaluator

report = DREvaluator(X, X_reduced, labels=y, reducer=reducer).evaluate_all()
print(report)
```

### Individual Metrics

```python
from squeeze.evaluation import (
    trustworthiness,
    continuity,
    spearman_distance_correlation,
    local_density_preservation,
    clustering_quality,
)

T = trustworthiness(X, X_reduced, k=15)
C = continuity(X, X_reduced, k=15)
rho = spearman_distance_correlation(X, X_reduced)
D = local_density_preservation(X, X_reduced)
clust = clustering_quality(X_reduced, labels_true=y)
```

---

## API Reference

### Functions

| Function | Parameters | Returns |
|----------|------------|---------|
| `trustworthiness(X_orig, X_red, k=15)` | Original data, reduced data, neighbors | float [0, 1] |
| `continuity(X_orig, X_red, k=15)` | Original data, reduced data, neighbors | float [0, 1] |
| `co_ranking_quality(X_orig, X_red, k=15)` | Original data, reduced data, neighbors | float [0, 1] |
| `spearman_distance_correlation(X_orig, X_red)` | Original data, reduced data | float [-1, 1] |
| `global_structure_preservation(X_orig, X_red, labels)` | Original, reduced, labels | float [0, 1] |
| `local_density_preservation(X_orig, X_red, k=15)` | Original data, reduced data, neighbors | float [0, 1] |
| `reconstruction_error(X_orig, X_red)` | Original data, reduced data | dict |
| `clustering_quality(X_red, labels_true=None, n_clusters=None)` | Reduced data, optional labels | dict |
| `classification_accuracy(X_red, labels, cv=5)` | Reduced data, labels, CV folds | dict |
| `bootstrap_stability(X, reducer, n_bootstrap=10)` | Data, reducer, iterations | dict |
| `noise_robustness(X, reducer, noise_levels=None)` | Data, reducer, noise levels | dict |
| `quick_evaluate(X_orig, X_red, k=15)` | Original data, reduced data, neighbors | dict |

### Classes

| Class | Description |
|-------|-------------|
| `DREvaluator` | Comprehensive evaluation interface |
| `EvaluationReport` | Container for evaluation results |

---

## References

1. Venna, J., & Kaski, S. (2006). Local multidimensional scaling. Neural Networks, 19(6-7), 889-899.

2. Lee, J. A., & Verleysen, M. (2009). Quality assessment of dimensionality reduction: Rank-based criteria. Neurocomputing, 72(7-9), 1431-1443.

3. Becht, E., et al. (2019). Dimensionality reduction for visualizing single-cell data using UMAP. Nature biotechnology, 37(1), 38-44.
