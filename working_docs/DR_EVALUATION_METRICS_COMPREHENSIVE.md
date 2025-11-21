# Dimensionality Reduction Evaluation Metrics: Comprehensive Guide

**Status:** Technical Reference
**Date:** November 4, 2025

---

## Executive Summary

Dimensionality reduction (DR) is fundamentally about **preservation**:
- Preserve what matters while discarding what doesn't
- Different applications value different aspects
- No single metric tells the full story
- Always use multiple complementary metrics

This guide covers:
1. **Core metrics** (every DR evaluation should include)
2. **Structure preservation metrics** (local, global, density)
3. **Reconstruction metrics** (how well can we invert?)
4. **Stability metrics** (reproducible? robust?)
5. **Downstream task metrics** (does it actually help?)
6. **Domain-specific metrics** (biology, NLP, etc.)

---

## Part 1: Core Preservation Metrics

### 1.1 Trustworthiness (Local Structure Preservation)

**What it measures:** Of the k nearest neighbors in the reduced space, how many were neighbors in the original space?

**Formula:**
```
Trustworthiness(k) = 1 - (2 / (n(2n-3k-1))) * Σ r(i,j)
where r(i,j) = max(0, rank_original(i,j) - k)
```

**Interpretation:**
- 1.0 = Perfect (all k-NN preserved)
- 0.5 = Random (half preserved)
- 0.0 = Worst case (none preserved)

**Code:**
```python
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
import numpy as np

def trustworthiness(X_original, X_reduced, k=15):
    """Trustworthiness: do k-NN in reduced match original?"""
    n = X_original.shape[0]

    # Get k-NN in original space
    nbrs_orig = NearestNeighbors(n_neighbors=k+1).fit(X_original)
    _, indices_orig = nbrs_orig.kneighbors(X_original)
    indices_orig = indices_orig[:, 1:]  # Skip self

    # Get k-NN in reduced space
    nbrs_red = NearestNeighbors(n_neighbors=k+1).fit(X_reduced)
    _, indices_red = nbrs_red.kneighbors(X_reduced)
    indices_red = indices_red[:, 1:]  # Skip self

    # Count matches
    n_matches = sum(
        len(np.intersect1d(indices_orig[i], indices_red[i]))
        for i in range(n)
    )

    return n_matches / (n * k)

# Usage
T = trustworthiness(X_original, X_reduced, k=15)
print(f"Trustworthiness: {T:.3f}")  # 0.978 is excellent
```

**Sensitivity:**
- `k=5`: Measures very local structure (0-3 neighbors)
- `k=15`: Balanced (standard)
- `k=30`: Includes semi-local neighborhoods

**Typical Values:**
- t-SNE: 0.90-0.98
- UMAP: 0.92-0.98
- PCA: 0.85-0.92
- Bad method: <0.80

**Pros:**
- Easy to understand
- Fast to compute
- Well-established

**Cons:**
- Only measures local structure
- Doesn't say anything about global structure
- Sensitive to k choice

---

### 1.2 Continuity (Inverse Preservation)

**What it measures:** Of the k nearest neighbors in the original space, how many are neighbors in the reduced space?

**Formula:**
```
Continuity(k) = 1 - (2 / (n(2n-3k-1))) * Σ s(i,j)
where s(i,j) = max(0, rank_reduced(i,j) - k)
```

**Key Difference from Trustworthiness:**
- Trustworthiness: Original neighbors → stay neighbors in reduced?
- Continuity: Reduced neighbors → came from original neighbors?

**Code:**
```python
def continuity(X_original, X_reduced, k=15):
    """Continuity: are neighbors in reduced space from original neighbors?"""
    # Same as trustworthiness but with roles reversed
    n = X_original.shape[0]

    nbrs_orig = NearestNeighbors(n_neighbors=k+1).fit(X_original)
    _, indices_orig = nbrs_orig.kneighbors(X_original)
    indices_orig = indices_orig[:, 1:]

    nbrs_red = NearestNeighbors(n_neighbors=k+1).fit(X_reduced)
    _, indices_red = nbrs_red.kneighbors(X_reduced)
    indices_red = indices_red[:, 1:]

    # Switch perspective: check reduced neighbors in original
    n_matches = sum(
        len(np.intersect1d(indices_orig[i], indices_red[i]))
        for i in range(n)
    )

    return n_matches / (n * k)
```

**Interpretation:**
- Trustworthiness + Continuity together tell complete story
- If Trustworthiness >> Continuity: New artificial clusters created
- If Continuity >> Trustworthiness: Original structure torn apart

**Typical Pattern:**
- UMAP: T ≈ C ≈ 0.95 (balanced)
- t-SNE: T ≈ 0.95, C ≈ 0.92 (slight local bias)

---

### 1.3 Co-Ranking Matrix (AUC-based Quality)

**What it measures:** Combined local and global structure preservation via ranking agreement.

**Concept:**
- In original space, rank all points by distance to point i
- In reduced space, rank all points by distance to point i
- How well do these rankings agree?

**Formula:**
```
Q(k) = (1/k) * Σ_{i=1}^{min(k_orig, k_reduced)} σ(i)
where σ(i) = number of points in top-k of both rankings
```

**Code:**
```python
def co_ranking_matrix(X_original, X_reduced, k=15):
    """Co-ranking matrix quality metric."""
    from sklearn.metrics import pairwise_distances

    D_orig = pairwise_distances(X_original)
    D_red = pairwise_distances(X_reduced)

    n = X_original.shape[0]
    Q = 0

    for i in range(n):
        # Get k nearest in both spaces
        nn_orig = np.argsort(D_orig[i])[1:k+1]
        nn_red = np.argsort(D_red[i])[1:k+1]

        # Count overlap
        Q += len(np.intersect1d(nn_orig, nn_red))

    return Q / (n * k)

# Usage: Q ranges from 0 to 1, higher is better
Q = co_ranking_matrix(X_original, X_reduced, k=15)
print(f"Co-ranking quality: {Q:.3f}")  # 0.90+ is good
```

**Advantages:**
- Captures both local and global structure
- More robust than trustworthiness alone
- Single number, easy to compare

**Disadvantages:**
- Less interpretable
- Computationally expensive for large n

---

## Part 2: Structure-Specific Metrics

### 2.1 Local Density Preservation

**What it measures:** Is the local density structure preserved? Clusters should remain compact.

**Formula:**
```
For each point i:
  density_original[i] = 1 / mean(distance to k neighbors)
  density_reduced[i] = 1 / mean(distance to k neighbors)

Correlation(density_original, density_reduced) → 0.0 to 1.0
```

**Code:**
```python
def local_density_preservation(X_original, X_reduced, k=15):
    """How well is local density preserved?"""
    from sklearn.neighbors import NearestNeighbors

    # Get k-NN distances
    nbrs_orig = NearestNeighbors(n_neighbors=k).fit(X_original)
    _, distances_orig = nbrs_orig.kneighbors(X_original)

    nbrs_red = NearestNeighbors(n_neighbors=k).fit(X_reduced)
    _, distances_red = nbrs_red.kneighbors(X_reduced)

    # Density = 1 / mean distance
    density_orig = 1.0 / distances_orig.mean(axis=1)
    density_red = 1.0 / distances_red.mean(axis=1)

    # Correlation of density maps
    correlation = np.corrcoef(density_orig, density_red)[0, 1]
    return max(0, correlation)  # Clamp to [0, 1]

# Usage
D = local_density_preservation(X_original, X_reduced)
print(f"Local density preservation: {D:.3f}")
# 0.8+ is good, means clusters remain tight/loose as in original
```

**Why It Matters:**
- densMAP specifically optimizes for this
- Critical for clustering downstream
- Visible in the embedding (dense → compact, sparse → spread out)

**Typical Values:**
- densMAP: 0.85-0.95
- UMAP: 0.70-0.85
- t-SNE: 0.65-0.80 (not density-preserving)
- PCA: 0.75-0.90 (linear, so preserves well)

---

### 2.2 Global Structure Preservation

**What it measures:** Are large-scale relationships preserved? If two clusters were far apart, are they still far apart?

**Code:**
```python
def global_structure_preservation(X_original, X_reduced, labels=None):
    """Preserve inter-cluster relationships."""
    if labels is None:
        raise ValueError("Need cluster labels for global structure")

    unique_labels = np.unique(labels)

    # Distance between cluster centers
    distances_orig = []
    distances_red = []

    for i, label_i in enumerate(unique_labels):
        for label_j in unique_labels[i+1:]:
            # Centers
            center_orig_i = X_original[labels == label_i].mean(axis=0)
            center_orig_j = X_original[labels == label_j].mean(axis=0)

            center_red_i = X_reduced[labels == label_i].mean(axis=0)
            center_red_j = X_reduced[labels == label_j].mean(axis=0)

            # Distances
            distances_orig.append(np.linalg.norm(center_orig_i - center_orig_j))
            distances_red.append(np.linalg.norm(center_red_i - center_red_j))

    # Correlation of inter-cluster distances
    correlation = np.corrcoef(distances_orig, distances_red)[0, 1]
    return max(0, correlation)

# Usage
G = global_structure_preservation(X_original, X_reduced, labels=y)
print(f"Global structure preservation: {G:.3f}")
# 0.8+ means inter-cluster relationships preserved
```

**Interpretation:**
- If G ≈ 0: Method ignores cluster separation
- If G ≈ 1: Cluster organization fully preserved
- Good methods: G > 0.80

**Method Comparison:**
- PCA: 0.90-0.98 (linear, preserves global)
- UMAP: 0.80-0.92
- t-SNE: 0.40-0.70 (local focus, ignores global)
- PaCMAP: 0.85-0.95 (designed for this)

---

### 2.3 Spearman Distance Correlation

**What it measures:** How well do pairwise distances correlate?

**Formula:**
```
Rank all pairwise distances in original space
Rank all pairwise distances in reduced space
Compute Spearman correlation (rank correlation)

Range: -1 to 1 (higher is better)
```

**Code:**
```python
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances

def spearman_distance_correlation(X_original, X_reduced):
    """Correlation of pairwise distances."""
    D_orig = pairwise_distances(X_original)
    D_red = pairwise_distances(X_reduced)

    # Flatten distances (excluding diagonal)
    mask = np.triu_indices_from(D_orig, k=1)
    d_orig = D_orig[mask]
    d_red = D_red[mask]

    # Spearman correlation
    rho, p_value = spearmanr(d_orig, d_red)
    return rho

# Usage
rho = spearman_distance_correlation(X_original, X_reduced)
print(f"Spearman distance correlation: {rho:.3f}")
# 0.9+ is excellent, 0.7-0.9 is good, <0.7 is poor
```

**Advantages:**
- Captures global structure well
- Rank-based (robust to outliers)
- Single number, easy interpretation

**Disadvantages:**
- O(n²) memory for large datasets
- Slow for n > 10,000

---

## Part 3: Reconstruction & Invertibility

### 3.1 Reconstruction Error

**What it measures:** Can we reconstruct the original data from the embedding?

**Formula:**
```
Train a regression: X_reduced → X_original
MSE = mean((X_original - X_reconstructed)^2)
Normalized: RMSE / std(X_original)
```

**Code:**
```python
from sklearn.linear_model import LinearRegression

def reconstruction_error(X_original, X_reduced):
    """How well can we reconstruct original from reduced?"""
    # Train linear model: reduced → original
    model = LinearRegression()
    model.fit(X_reduced, X_original)
    X_reconstructed = model.predict(X_reduced)

    # Compute error
    mse = np.mean((X_original - X_reconstructed) ** 2)
    rmse = np.sqrt(mse)

    # Normalize by original variance
    normalized_rmse = rmse / np.std(X_original)

    return {
        "mse": mse,
        "rmse": rmse,
        "normalized_rmse": normalized_rmse,
    }

# Usage
error = reconstruction_error(X_original, X_reduced)
print(f"Normalized RMSE: {error['normalized_rmse']:.3f}")
# Lower is better, <0.1 is excellent
```

**Interpretation:**
- 0.01: Excellent (almost no information lost)
- 0.1: Good
- 0.5: Moderate (significant information loss)
- 1.0+: Bad (almost random reconstruction)

**Why It Matters:**
- Linear reconstruction shows how much structure is preserved
- Non-parametric methods can't reconstruct well (by design)
- Parametric methods should reconstruct better

---

### 3.2 Inverse Rank Correlation

**What it measures:** How well does the reverse transformation work?

**For Parametric Methods:**
```python
def inverse_rank_correlation(X_original, X_reduced, reducer):
    """For parametric methods, how good is inverse?"""
    # Apply inverse (if available)
    X_reconstructed = reducer.inverse_transform(X_reduced)

    # Compare distances
    D_orig = pairwise_distances(X_original)
    D_recon = pairwise_distances(X_reconstructed)

    # Rank correlation
    rho = spearmanr(D_orig[np.triu_indices_from(D_orig, k=1)],
                    D_recon[np.triu_indices_from(D_recon, k=1)])[0]

    return rho
```

**Typical Values:**
- Parametric UMAP: 0.85-0.95
- Parametric t-SNE: 0.80-0.92
- Non-parametric methods: Not applicable (no inverse)

---

## Part 4: Stability & Robustness

### 4.1 Bootstrap Stability (Procrustes)

**What it measures:** If we rerun DR with different random samples, do we get the same result?

**Code:**
```python
from scipy.spatial.distance import procrustes

def bootstrap_stability(X, n_bootstrap=100, sample_fraction=0.8):
    """Are results consistent across bootstrap samples?"""
    n_samples = int(X.shape[0] * sample_fraction)

    embeddings = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(X.shape[0], size=n_samples, replace=False)
        X_sample = X[indices]

        # Embed
        reducer = UMAP()
        X_emb = reducer.fit_transform(X_sample)
        embeddings.append(X_emb)

    # Compare via Procrustes alignment
    reference = embeddings[0]
    procrustes_errors = []

    for emb in embeddings[1:]:
        _, error = procrustes(reference, emb)
        procrustes_errors.append(error)

    return {
        "mean_procrustes_error": np.mean(procrustes_errors),
        "std_procrustes_error": np.std(procrustes_errors),
        "stability_score": 1.0 - np.mean(procrustes_errors),
    }

# Usage
stability = bootstrap_stability(X)
print(f"Stability score: {stability['stability_score']:.3f}")
# 0.95+ is excellent (very stable)
# 0.80-0.95 is acceptable
# <0.80 is unstable (different runs give different results)
```

**Interpretation:**
- UMAP: 0.92-0.98 (very stable)
- t-SNE: 0.85-0.95 (somewhat unstable, seed-dependent)
- PCA: 0.99+ (deterministic, maximally stable)

---

### 4.2 Noise Robustness

**What it measures:** If we add noise to the data, does the embedding change much?

**Code:**
```python
def noise_robustness(X, reducer_class=UMAP, noise_levels=[0.01, 0.05, 0.1]):
    """How robust is embedding to input noise?"""

    # Original embedding
    reducer = reducer_class()
    X_emb_clean = reducer.fit_transform(X)

    robustness_scores = []

    for noise_level in noise_levels:
        # Add Gaussian noise
        X_noisy = X + np.random.randn(*X.shape) * noise_level * X.std(axis=0)

        # Re-embed
        reducer_noisy = reducer_class()
        X_emb_noisy = reducer_noisy.fit_transform(X_noisy)

        # Compare via Procrustes
        _, error = procrustes(X_emb_clean, X_emb_noisy)
        robustness_scores.append(1.0 - error)

    return dict(zip(noise_levels, robustness_scores))

# Usage
robustness = noise_robustness(X)
print(robustness)
# {0.01: 0.98, 0.05: 0.95, 0.1: 0.92}
# Higher means more robust to noise
```

---

### 4.3 Parameter Sensitivity

**What it measures:** How much do results change with parameter variations?

**Code:**
```python
def parameter_sensitivity(X, parameters_to_vary=None):
    """Sensitivity to parameter choices."""
    if parameters_to_vary is None:
        parameters_to_vary = {
            "n_neighbors": [5, 15, 30],
            "min_dist": [0.01, 0.1, 0.5],
        }

    results = {}

    # Baseline
    reducer_baseline = UMAP()
    X_baseline = reducer_baseline.fit_transform(X)

    # Vary each parameter
    for param_name, values in parameters_to_vary.items():
        sensitivities = []

        for value in values:
            kwargs = {param_name: value}
            reducer = UMAP(**kwargs)
            X_varied = reducer.fit_transform(X)

            # Procrustes distance
            _, error = procrustes(X_baseline, X_varied)
            sensitivities.append(error)

        results[param_name] = {
            "parameter_values": values,
            "procrustes_errors": sensitivities,
            "mean_sensitivity": np.mean(sensitivities),
        }

    return results

# Usage
sensitivity = parameter_sensitivity(X)
# Low mean sensitivity = robust to parameter choice
# High mean sensitivity = need to tune carefully
```

**Typical Patterns:**
- UMAP: Low sensitivity (n_neighbors affects result, but not drastically)
- t-SNE: High sensitivity (perplexity critical, seed matters)
- PCA: Very low sensitivity (only n_components matters)

---

## Part 5: Downstream Task Metrics

### 5.1 Clustering Quality on Embedding

**What it measures:** Can we cluster better using the embedded space?

**Code:**
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def clustering_quality_on_embedding(X_reduced, y_true=None):
    """How good are the clusters in the reduced space?"""

    # Cluster the reduced space
    kmeans = KMeans(n_clusters=len(np.unique(y_true)))
    labels_pred = kmeans.fit_predict(X_reduced)

    metrics = {
        # Higher is better
        "silhouette_score": silhouette_score(X_reduced, labels_pred),
        "calinski_harabasz": calinski_harabasz_score(X_reduced, labels_pred),
        # Lower is better
        "davies_bouldin": davies_bouldin_score(X_reduced, labels_pred),
    }

    # If we have ground truth
    if y_true is not None:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        metrics.update({
            "adjusted_rand_index": adjusted_rand_score(y_true, labels_pred),
            "mutual_information": normalized_mutual_info_score(y_true, labels_pred),
        })

    return metrics

# Usage
metrics = clustering_quality_on_embedding(X_umap, y_true=y)
# Good clustering → embedding preserves structure well
```

---

### 5.2 Classification Accuracy on Embedding

**What it measures:** Can a classifier achieve good accuracy using the embedded features?

**Code:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def classification_accuracy_on_embedding(X_reduced, y):
    """How useful is the embedding for downstream classification?"""

    clf = RandomForestClassifier(n_estimators=100)

    # Cross-validated accuracy
    scores = cross_val_score(clf, X_reduced, y, cv=5, scoring='accuracy')

    return {
        "mean_accuracy": scores.mean(),
        "std_accuracy": scores.std(),
        "scores": scores,
    }

# Usage
results = classification_accuracy_on_embedding(X_umap, y)
# High accuracy → embedding is informative
```

---

## Part 6: Domain-Specific Metrics

### 6.1 Single-Cell RNA-seq

```python
def scrnaseq_embedding_quality(X_embedded, obs_metadata=None):
    """Specific metrics for scRNA-seq embeddings."""

    metrics = {}

    # 1. Cell type purity (if cell type labels available)
    if 'cell_type' in obs_metadata:
        cell_types = obs_metadata['cell_type'].values
        unique_types = np.unique(cell_types)

        # For each cell type, compute cohesion
        cohesion_scores = []
        for ct in unique_types:
            mask = cell_types == ct
            if mask.sum() > 1:
                subset = X_embedded[mask]
                # Average distance within cell type
                distances = pairwise_distances(subset).mean()
                cohesion_scores.append(distances)

        metrics['cell_type_cohesion'] = np.mean(cohesion_scores)

    # 2. Batch effect (if batch info available)
    if 'batch' in obs_metadata:
        # Compute mixed-cell index (MCI)
        # Higher = more mixing = less batch effect
        ...

    # 3. Neighborhood preservation
    # For rare cell types, check if they cluster together
    ...

    return metrics
```

### 6.2 NLP/Text

```python
def nlp_embedding_quality(X_embedded, documents, labels=None):
    """Specific metrics for text embeddings."""

    metrics = {}

    # 1. Semantic similarity (using pre-trained model)
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(X_embedded)

    metrics['mean_similarity'] = similarity.mean()
    metrics['similarity_std'] = similarity.std()

    # 2. Topic coherence (if topics available)
    if labels is not None:
        for topic_id in np.unique(labels):
            mask = labels == topic_id
            topic_embedding = X_embedded[mask].mean(axis=0)
            # Distance from members to topic center
            distances = np.linalg.norm(X_embedded[mask] - topic_embedding, axis=1)
            metrics[f'topic_{topic_id}_cohesion'] = distances.mean()

    return metrics
```

### 6.3 Imaging/Vision

```python
def vision_embedding_quality(X_embedded, images_metadata):
    """Specific metrics for image embeddings."""

    metrics = {}

    # 1. Object preservation (if object class available)
    if 'class' in images_metadata:
        class_labels = images_metadata['class'].values
        # Similar to cell type cohesion
        ...

    # 2. Visual similarity (if similar images marked)
    if 'similar_pairs' in images_metadata:
        similar_pairs = images_metadata['similar_pairs']
        distances_similar = []
        for i, j in similar_pairs:
            dist = np.linalg.norm(X_embedded[i] - X_embedded[j])
            distances_similar.append(dist)

        metrics['similar_pair_distance'] = np.mean(distances_similar)

    return metrics
```

---

## Part 7: Recommended Metric Sets

### Minimum Required (Every DR Evaluation)

```python
def comprehensive_dr_evaluation(X_original, X_reduced, y_true=None,
                                 reducer=None, name="Method"):
    """Standard metrics every DR should report."""

    results = {
        "method": name,
        "n_samples": X_original.shape[0],
        "n_features_original": X_original.shape[1],
        "n_features_reduced": X_reduced.shape[1],
        "reduction_ratio": X_original.shape[1] / X_reduced.shape[1],
    }

    # Local structure
    results['trustworthiness_15'] = trustworthiness(X_original, X_reduced, k=15)
    results['continuity_15'] = continuity(X_original, X_reduced, k=15)
    results['co_ranking_15'] = co_ranking_matrix(X_original, X_reduced, k=15)

    # Global structure
    results['spearman_distance_corr'] = spearman_distance_correlation(X_original, X_reduced)
    if y_true is not None:
        results['global_structure'] = global_structure_preservation(X_original, X_reduced, y_true)

    # Stability (if reducer available)
    if reducer is not None:
        stability = bootstrap_stability(X_original)
        results['stability_score'] = stability['stability_score']

    # Clustering quality (if labels available)
    if y_true is not None:
        clustering = clustering_quality_on_embedding(X_reduced, y_true)
        results.update({f"clustering_{k}": v for k, v in clustering.items()})

    return results

# Usage
metrics = comprehensive_dr_evaluation(X_original, X_reduced, y_true=y,
                                      reducer=reducer, name="UMAP")
```

### Publication-Quality (Comprehensive)

```python
# Include everything above PLUS:
- Density preservation
- Parameter sensitivity
- Noise robustness
- Reconstruction error
- Bootstrap stability
- Cross-validation stability
- Computational cost (time, memory)
- Comparison to 3+ baseline methods
- Statistical significance tests
```

---

## Part 8: Metric Trade-offs

### The Local vs Global Trade-off

**t-SNE:**
- Trustworthiness: 0.95 (excellent local)
- Spearman distance correlation: 0.55 (poor global)
- Use when: Local clusters matter most

**UMAP:**
- Trustworthiness: 0.92 (good local)
- Spearman distance correlation: 0.80 (decent global)
- Use when: Need balance

**PCA:**
- Trustworthiness: 0.80 (fair local)
- Spearman distance correlation: 0.95 (excellent global)
- Use when: Global structure matters

**PaCMAP:**
- Trustworthiness: 0.90 (good local)
- Spearman distance correlation: 0.85 (good global)
- Use when: Want balance, more time to compute

---

## Part 9: Implementation in Code

### One-Line Evaluation Report

```python
from umap.evaluate import DREvaluator

evaluator = DREvaluator(X_original, X_reduced, y_true=y, reducer=reducer)
report = evaluator.evaluate_and_report()
report.save_html("evaluation_report.html")
report.display()  # In Jupyter
```

### Full Metrics Dict

```python
metrics = evaluator.evaluate_all()

print(metrics)
# {
#     'local_structure': {
#         'trustworthiness_5': 0.98,
#         'trustworthiness_15': 0.95,
#         'trustworthiness_30': 0.92,
#         'continuity_5': 0.97,
#         'continuity_15': 0.94,
#         'continuity_30': 0.91,
#     },
#     'global_structure': {
#         'spearman_distance_corr': 0.82,
#         'global_structure': 0.85,
#     },
#     'stability': {
#         'bootstrap_stability': 0.94,
#         'noise_robustness_0.01': 0.98,
#         'noise_robustness_0.05': 0.95,
#     },
#     'downstream': {
#         'silhouette_score': 0.72,
#         'classification_accuracy': 0.89,
#     },
# }
```

---

## Part 10: Quick Reference Table

| Metric | Range | What to Aim For | Compute Time | Use Case |
|--------|-------|---|---|---|
| Trustworthiness | 0-1 | >0.90 | Fast | Local structure |
| Continuity | 0-1 | >0.90 | Fast | Local structure |
| Co-ranking | 0-1 | >0.90 | Fast | Overall quality |
| Spearman Distance | -1 to 1 | >0.80 | Slow | Global structure |
| Global Structure | 0-1 | >0.80 | Fast | Inter-cluster distance |
| Local Density | 0-1 | >0.80 | Fast | Cluster density |
| Bootstrap Stability | 0-1 | >0.90 | Very Slow | Reproducibility |
| Silhouette Score | -1 to 1 | >0.60 | Fast | Clustering quality |
| Classification | 0-1 | >0.85 | Slow | Downstream task |
| Reconstruction RMSE | 0+ | <0.10 | Fast | Information loss |

---

## Summary

The right metrics depend on your use case:

**For Publishing:**
- Trustworthiness (k=5, 15, 30)
- Continuity (k=5, 15, 30)
- Spearman distance correlation
- Global structure preservation
- Bootstrap stability

**For Production Quality Control:**
- Trustworthiness + Continuity (quick check)
- Silhouette score (clustering quality)
- Bootstrap stability (reproducibility)
- Noise robustness (if dealing with noisy data)

**For Research/Diagnosis:**
- Everything above
- Parameter sensitivity
- Feature importance
- Local density preservation
- Reconstruction error

**For Domain-Specific Use:**
- Add domain-specific metrics (cell type cohesion, semantic similarity, etc.)
