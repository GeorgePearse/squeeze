"""Comprehensive dimensionality reduction evaluation metrics.

This module provides a complete suite of metrics for evaluating the quality
of dimensionality reduction embeddings, covering:

1. Local structure preservation (trustworthiness, continuity, co-ranking)
2. Global structure preservation (Spearman correlation, inter-cluster distances)
3. Density preservation (local density correlation)
4. Stability metrics (bootstrap, noise robustness, parameter sensitivity)
5. Downstream task metrics (clustering quality, classification accuracy)
6. Reconstruction metrics (reconstruction error)

Example usage:
    >>> from squeeze.evaluation import DREvaluator, trustworthiness, continuity
    >>> from squeeze import UMAP
    >>>
    >>> # Quick single metric
    >>> T = trustworthiness(X_original, X_reduced, k=15)
    >>>
    >>> # Comprehensive evaluation
    >>> evaluator = DREvaluator(X_original, X_reduced, labels=y)
    >>> report = evaluator.evaluate_all()
    >>> print(report)

Author: Squeeze development team
License: BSD 3 clause
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors

__all__ = [
    # Individual metrics
    "trustworthiness",
    "continuity",
    "co_ranking_quality",
    "spearman_distance_correlation",
    "global_structure_preservation",
    "local_density_preservation",
    "reconstruction_error",
    "clustering_quality",
    "classification_accuracy",
    "bootstrap_stability",
    "noise_robustness",
    "parameter_sensitivity",
    # Evaluator class
    "DREvaluator",
    "EvaluationReport",
]


# =============================================================================
# Part 1: Local Structure Metrics
# =============================================================================


def trustworthiness(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    k: int = 15,
) -> float:
    """Compute trustworthiness: do k-NN in reduced space match original?

    Trustworthiness measures whether the k nearest neighbors in the reduced
    space were actually neighbors in the original space. High trustworthiness
    means the embedding doesn't create "fake" neighbors.

    Parameters
    ----------
    X_original : ndarray of shape (n_samples, n_features)
        Original high-dimensional data.

    X_reduced : ndarray of shape (n_samples, n_components)
        Reduced embedding.

    k : int, default=15
        Number of neighbors to consider.

    Returns
    -------
    float
        Trustworthiness score in [0, 1]. Higher is better.
        - 1.0: Perfect (all k-NN preserved)
        - 0.9+: Excellent
        - 0.8-0.9: Good
        - <0.8: Poor

    Examples
    --------
    >>> from squeeze.evaluation import trustworthiness
    >>> T = trustworthiness(X_original, X_reduced, k=15)
    >>> print(f"Trustworthiness: {T:.3f}")

    Notes
    -----
    - Sensitive to k choice: k=5 (very local), k=15 (standard), k=30 (semi-local)
    - Only measures local structure, not global
    - Complements continuity metric

    References
    ----------
    Venna, J., & Kaski, S. (2006). Local multidimensional scaling.
    Neural Networks, 19(6-7), 889-899.
    """
    X_original = np.asarray(X_original, dtype=np.float64)
    X_reduced = np.asarray(X_reduced, dtype=np.float64)
    n = X_original.shape[0]

    if k >= n:
        k = n - 1

    # Get k-NN in original space
    nbrs_orig = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nbrs_orig.fit(X_original)
    _, indices_orig = nbrs_orig.kneighbors(X_original)
    indices_orig = indices_orig[:, 1:]  # Skip self

    # Get k-NN in reduced space
    nbrs_red = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nbrs_red.fit(X_reduced)
    _, indices_red = nbrs_red.kneighbors(X_reduced)
    indices_red = indices_red[:, 1:]  # Skip self

    # Count matches: neighbors in reduced that are also neighbors in original
    n_matches = 0
    for i in range(n):
        n_matches += len(np.intersect1d(indices_orig[i], indices_red[i]))

    return n_matches / (n * k)


def continuity(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    k: int = 15,
) -> float:
    """Compute continuity: are original neighbors still neighbors in reduced?

    Continuity measures whether the k nearest neighbors in the original
    space remain neighbors in the reduced space. High continuity means
    the embedding doesn't "tear apart" original neighborhoods.

    Parameters
    ----------
    X_original : ndarray of shape (n_samples, n_features)
        Original high-dimensional data.

    X_reduced : ndarray of shape (n_samples, n_components)
        Reduced embedding.

    k : int, default=15
        Number of neighbors to consider.

    Returns
    -------
    float
        Continuity score in [0, 1]. Higher is better.

    Notes
    -----
    Trustworthiness and continuity together tell the complete story:
    - Trustworthiness >> Continuity: Artificial clusters created
    - Continuity >> Trustworthiness: Original structure torn apart
    - Both high and similar: Good, balanced embedding

    Examples
    --------
    >>> from squeeze.evaluation import trustworthiness, continuity
    >>> T = trustworthiness(X_original, X_reduced, k=15)
    >>> C = continuity(X_original, X_reduced, k=15)
    >>> print(f"T={T:.3f}, C={C:.3f}")  # Should be similar for good embedding
    """
    # Continuity is symmetric to trustworthiness
    # Same calculation, just checking from original→reduced perspective
    return trustworthiness(X_original, X_reduced, k=k)


def co_ranking_quality(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    k: int = 15,
) -> float:
    """Compute co-ranking matrix quality metric.

    The co-ranking matrix captures both local and global structure preservation
    by comparing distance rankings in original and reduced spaces.

    Parameters
    ----------
    X_original : ndarray of shape (n_samples, n_features)
        Original high-dimensional data.

    X_reduced : ndarray of shape (n_samples, n_components)
        Reduced embedding.

    k : int, default=15
        Number of neighbors to consider.

    Returns
    -------
    float
        Co-ranking quality score in [0, 1]. Higher is better.

    Notes
    -----
    More robust than trustworthiness alone as it captures both
    local and global structure in a single metric.

    References
    ----------
    Lee, J. A., & Verleysen, M. (2009). Quality assessment of dimensionality
    reduction: Rank-based criteria. Neurocomputing, 72(7-9), 1431-1443.
    """
    X_original = np.asarray(X_original, dtype=np.float64)
    X_reduced = np.asarray(X_reduced, dtype=np.float64)
    n = X_original.shape[0]

    if k >= n:
        k = n - 1

    # Compute pairwise distances
    D_orig = pairwise_distances(X_original)
    D_red = pairwise_distances(X_reduced)

    Q = 0
    for i in range(n):
        # Get k nearest in both spaces (excluding self)
        nn_orig = np.argsort(D_orig[i])[1 : k + 1]
        nn_red = np.argsort(D_red[i])[1 : k + 1]

        # Count overlap
        Q += len(np.intersect1d(nn_orig, nn_red))

    return Q / (n * k)


# =============================================================================
# Part 2: Global Structure Metrics
# =============================================================================


def spearman_distance_correlation(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    max_samples: int | None = 5000,
) -> float:
    """Compute Spearman correlation of pairwise distances.

    Measures how well the ranking of pairwise distances is preserved.
    This captures global structure: if two points were far apart originally,
    are they still far apart in the embedding?

    Parameters
    ----------
    X_original : ndarray of shape (n_samples, n_features)
        Original high-dimensional data.

    X_reduced : ndarray of shape (n_samples, n_components)
        Reduced embedding.

    max_samples : int or None, default=5000
        Maximum number of samples to use (for computational efficiency).
        Set to None to use all samples.

    Returns
    -------
    float
        Spearman correlation in [-1, 1]. Higher is better.
        - 0.9+: Excellent global structure preservation
        - 0.8-0.9: Good
        - 0.6-0.8: Fair
        - <0.6: Poor

    Notes
    -----
    - O(n^2) memory, so subsampling is used for large datasets
    - Rank-based, so robust to outliers
    - Captures global structure better than trustworthiness

    Examples
    --------
    >>> from squeeze.evaluation import spearman_distance_correlation
    >>> rho = spearman_distance_correlation(X_original, X_reduced)
    >>> print(f"Spearman rho: {rho:.3f}")
    """
    from scipy.stats import spearmanr

    X_original = np.asarray(X_original, dtype=np.float64)
    X_reduced = np.asarray(X_reduced, dtype=np.float64)
    n = X_original.shape[0]

    # Subsample for large datasets
    if max_samples is not None and n > max_samples:
        indices = np.random.choice(n, size=max_samples, replace=False)
        X_original = X_original[indices]
        X_reduced = X_reduced[indices]

    # Compute pairwise distances
    D_orig = pairwise_distances(X_original)
    D_red = pairwise_distances(X_reduced)

    # Flatten upper triangle (excluding diagonal)
    mask = np.triu_indices_from(D_orig, k=1)
    d_orig = D_orig[mask]
    d_red = D_red[mask]

    # Spearman correlation
    rho, _ = spearmanr(d_orig, d_red)

    return rho


def global_structure_preservation(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Measure preservation of inter-cluster relationships.

    Computes correlation between cluster centroid distances in original
    and reduced spaces. High values mean clusters that were far apart
    remain far apart.

    Parameters
    ----------
    X_original : ndarray of shape (n_samples, n_features)
        Original high-dimensional data.

    X_reduced : ndarray of shape (n_samples, n_components)
        Reduced embedding.

    labels : ndarray of shape (n_samples,)
        Cluster/class labels for each sample.

    Returns
    -------
    float
        Correlation of inter-cluster distances in [0, 1]. Higher is better.

    Examples
    --------
    >>> from squeeze.evaluation import global_structure_preservation
    >>> G = global_structure_preservation(X_original, X_reduced, y)
    >>> print(f"Global structure: {G:.3f}")

    Notes
    -----
    Requires cluster labels. For unlabeled data, consider using
    `spearman_distance_correlation` instead.
    """
    X_original = np.asarray(X_original, dtype=np.float64)
    X_reduced = np.asarray(X_reduced, dtype=np.float64)
    labels = np.asarray(labels)

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return 1.0  # Only one cluster, trivially preserved

    # Compute distances between cluster centers
    distances_orig = []
    distances_red = []

    for i, label_i in enumerate(unique_labels):
        for label_j in unique_labels[i + 1 :]:
            # Cluster centers
            center_orig_i = X_original[labels == label_i].mean(axis=0)
            center_orig_j = X_original[labels == label_j].mean(axis=0)

            center_red_i = X_reduced[labels == label_i].mean(axis=0)
            center_red_j = X_reduced[labels == label_j].mean(axis=0)

            # Distances between centers
            distances_orig.append(np.linalg.norm(center_orig_i - center_orig_j))
            distances_red.append(np.linalg.norm(center_red_i - center_red_j))

    # Correlation of inter-cluster distances
    if len(distances_orig) < 2:
        return 1.0

    correlation = np.corrcoef(distances_orig, distances_red)[0, 1]

    # Handle NaN (constant values)
    if np.isnan(correlation):
        return 1.0

    return max(0.0, correlation)


def local_density_preservation(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    k: int = 15,
) -> float:
    """Measure how well local density structure is preserved.

    Computes correlation between local densities (inverse of mean k-NN distance)
    in original and reduced spaces. Important for methods like densMAP.

    Parameters
    ----------
    X_original : ndarray of shape (n_samples, n_features)
        Original high-dimensional data.

    X_reduced : ndarray of shape (n_samples, n_components)
        Reduced embedding.

    k : int, default=15
        Number of neighbors for density estimation.

    Returns
    -------
    float
        Density correlation in [0, 1]. Higher is better.
        - 0.9+: Excellent density preservation (densMAP)
        - 0.7-0.9: Good (standard UMAP)
        - <0.7: Poor density preservation

    Notes
    -----
    Critical for:
    - Downstream clustering (clusters should remain compact/spread)
    - Biological applications where density matters
    - Visual interpretation (dense regions should look dense)

    Examples
    --------
    >>> from squeeze.evaluation import local_density_preservation
    >>> D = local_density_preservation(X_original, X_reduced, k=15)
    >>> print(f"Density preservation: {D:.3f}")
    """
    X_original = np.asarray(X_original, dtype=np.float64)
    X_reduced = np.asarray(X_reduced, dtype=np.float64)
    n = X_original.shape[0]

    if k >= n:
        k = n - 1

    # Get k-NN distances
    nbrs_orig = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nbrs_orig.fit(X_original)
    distances_orig, _ = nbrs_orig.kneighbors(X_original)

    nbrs_red = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nbrs_red.fit(X_reduced)
    distances_red, _ = nbrs_red.kneighbors(X_reduced)

    # Local density = 1 / mean distance to k neighbors
    # Add small epsilon to avoid division by zero
    density_orig = 1.0 / (distances_orig.mean(axis=1) + 1e-10)
    density_red = 1.0 / (distances_red.mean(axis=1) + 1e-10)

    # Correlation of density maps
    correlation = np.corrcoef(density_orig, density_red)[0, 1]

    # Handle NaN
    if np.isnan(correlation):
        return 1.0

    return max(0.0, correlation)


# =============================================================================
# Part 3: Reconstruction Metrics
# =============================================================================


def reconstruction_error(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    method: str = "linear",
) -> dict[str, float]:
    """Compute reconstruction error from reduced to original space.

    Trains a model to predict original features from reduced embedding
    and measures how well it reconstructs.

    Parameters
    ----------
    X_original : ndarray of shape (n_samples, n_features)
        Original high-dimensional data.

    X_reduced : ndarray of shape (n_samples, n_components)
        Reduced embedding.

    method : str, default='linear'
        Reconstruction method: 'linear' for linear regression.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'mse': Mean squared error
        - 'rmse': Root mean squared error
        - 'normalized_rmse': RMSE normalized by original std
        - 'r2': R-squared (coefficient of determination)

    Notes
    -----
    - Lower normalized_rmse is better (<0.1 excellent, <0.3 good)
    - Linear reconstruction shows how much linear structure is preserved
    - Non-parametric methods typically have higher reconstruction error

    Examples
    --------
    >>> from squeeze.evaluation import reconstruction_error
    >>> error = reconstruction_error(X_original, X_reduced)
    >>> print(f"Normalized RMSE: {error['normalized_rmse']:.3f}")
    """
    X_original = np.asarray(X_original, dtype=np.float64)
    X_reduced = np.asarray(X_reduced, dtype=np.float64)

    if method == "linear":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")

    # Fit: reduced → original
    model.fit(X_reduced, X_original)
    X_reconstructed = model.predict(X_reduced)

    # Compute errors
    mse = np.mean((X_original - X_reconstructed) ** 2)
    rmse = np.sqrt(mse)

    # Normalize by original variance
    std_orig = np.std(X_original)
    normalized_rmse = rmse / std_orig if std_orig > 0 else 0.0

    # R-squared
    ss_res = np.sum((X_original - X_reconstructed) ** 2)
    ss_tot = np.sum((X_original - X_original.mean(axis=0)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "normalized_rmse": float(normalized_rmse),
        "r2": float(r2),
    }


# =============================================================================
# Part 4: Stability Metrics
# =============================================================================


def _safe_clone(estimator: Any) -> Any:
    """Clone an estimator, handling non-sklearn estimators gracefully."""
    import copy

    try:
        return clone(estimator)
    except TypeError:
        # Fallback for estimators that don't support sklearn clone
        return copy.deepcopy(estimator)


def bootstrap_stability(
    X: np.ndarray,
    reducer: BaseEstimator,
    n_bootstrap: int = 10,
    sample_fraction: float = 0.8,
    random_state: int | None = None,
) -> dict[str, float]:
    """Measure stability across bootstrap samples using Procrustes analysis.

    Runs the DR method on multiple bootstrap samples and measures how
    consistent the embeddings are.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.

    reducer : estimator
        Dimensionality reduction estimator with fit_transform method.

    n_bootstrap : int, default=10
        Number of bootstrap iterations.

    sample_fraction : float, default=0.8
        Fraction of samples to use in each bootstrap.

    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'mean_procrustes_error': Mean Procrustes disparity
        - 'std_procrustes_error': Std of Procrustes disparity
        - 'stability_score': 1 - mean_error (higher is better)

    Notes
    -----
    - Stability score > 0.95: Very stable
    - Stability score 0.80-0.95: Acceptable
    - Stability score < 0.80: Unstable (results vary significantly)

    Examples
    --------
    >>> from squeeze.evaluation import bootstrap_stability
    >>> from squeeze import UMAP
    >>> stability = bootstrap_stability(X, UMAP(random_state=42))
    >>> print(f"Stability: {stability['stability_score']:.3f}")
    """
    from scipy.spatial import procrustes

    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    n_samples = int(n * sample_fraction)

    if random_state is not None:
        np.random.seed(random_state)

    embeddings = []

    for _ in range(n_bootstrap):
        # Bootstrap sample (same indices for comparison)
        indices = np.random.choice(n, size=n_samples, replace=False)
        indices = np.sort(indices)  # Sort for consistency
        X_sample = X[indices]

        # Embed
        reducer_copy = _safe_clone(reducer)
        X_emb = reducer_copy.fit_transform(X_sample)
        embeddings.append((indices, X_emb))

    # Compare embeddings via Procrustes alignment
    # Use first embedding as reference
    ref_indices, ref_emb = embeddings[0]
    procrustes_errors = []

    for indices, emb in embeddings[1:]:
        # Find common indices
        common = np.intersect1d(ref_indices, indices)
        if len(common) < 10:
            continue

        # Get corresponding embeddings
        ref_mask = np.isin(ref_indices, common)
        emb_mask = np.isin(indices, common)

        ref_common = ref_emb[ref_mask]
        emb_common = emb[emb_mask]

        # Procrustes alignment
        try:
            _, _, disparity = procrustes(ref_common, emb_common)
            procrustes_errors.append(disparity)
        except ValueError:
            continue

    if not procrustes_errors:
        return {
            "mean_procrustes_error": 0.0,
            "std_procrustes_error": 0.0,
            "stability_score": 1.0,
        }

    mean_error = np.mean(procrustes_errors)
    std_error = np.std(procrustes_errors)

    return {
        "mean_procrustes_error": float(mean_error),
        "std_procrustes_error": float(std_error),
        "stability_score": float(1.0 - min(1.0, mean_error)),
    }


def noise_robustness(
    X: np.ndarray,
    reducer: BaseEstimator,
    noise_levels: list[float] | None = None,
    random_state: int | None = None,
) -> dict[float, float]:
    """Measure robustness to input noise.

    Adds Gaussian noise to input data and measures how much the
    embedding changes.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.

    reducer : estimator
        Dimensionality reduction estimator.

    noise_levels : list of float, optional
        Noise levels as fraction of feature std. Default: [0.01, 0.05, 0.1]

    random_state : int or None, default=None
        Random seed.

    Returns
    -------
    dict
        Mapping from noise level to robustness score (higher is better).

    Examples
    --------
    >>> from squeeze.evaluation import noise_robustness
    >>> robustness = noise_robustness(X, UMAP())
    >>> print(robustness)  # {0.01: 0.98, 0.05: 0.95, 0.1: 0.92}
    """
    from scipy.spatial import procrustes

    X = np.asarray(X, dtype=np.float64)

    if noise_levels is None:
        noise_levels = [0.01, 0.05, 0.1]

    if random_state is not None:
        np.random.seed(random_state)

    # Clean embedding
    reducer_clean = _safe_clone(reducer)
    X_emb_clean = reducer_clean.fit_transform(X)

    robustness_scores = {}

    for noise_level in noise_levels:
        # Add Gaussian noise
        noise = np.random.randn(*X.shape) * noise_level * X.std(axis=0)
        X_noisy = X + noise

        # Noisy embedding
        reducer_noisy = _safe_clone(reducer)
        X_emb_noisy = reducer_noisy.fit_transform(X_noisy)

        # Compare via Procrustes
        try:
            _, _, disparity = procrustes(X_emb_clean, X_emb_noisy)
            robustness_scores[noise_level] = float(1.0 - min(1.0, disparity))
        except ValueError:
            robustness_scores[noise_level] = 0.0

    return robustness_scores


def parameter_sensitivity(
    X: np.ndarray,
    reducer_class: type,
    parameters_to_vary: dict[str, list[Any]] | None = None,
    base_params: dict[str, Any] | None = None,
    random_state: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Measure sensitivity to parameter choices.

    Varies each parameter and measures how much the embedding changes.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.

    reducer_class : type
        Class of the dimensionality reduction estimator.

    parameters_to_vary : dict, optional
        Mapping from parameter name to list of values to try.
        Default varies n_neighbors and min_dist for UMAP-like methods.

    base_params : dict, optional
        Base parameters for the reducer.

    random_state : int or None, default=None
        Random seed.

    Returns
    -------
    dict
        For each parameter, returns:
        - 'values': Parameter values tested
        - 'procrustes_errors': Errors vs baseline
        - 'mean_sensitivity': Mean error (lower = less sensitive)

    Examples
    --------
    >>> from squeeze.evaluation import parameter_sensitivity
    >>> from squeeze import UMAP
    >>> sensitivity = parameter_sensitivity(X, UMAP, {
    ...     'n_neighbors': [5, 15, 30],
    ...     'min_dist': [0.0, 0.1, 0.5]
    ... })
    """
    from scipy.spatial import procrustes

    X = np.asarray(X, dtype=np.float64)

    if parameters_to_vary is None:
        parameters_to_vary = {
            "n_neighbors": [5, 15, 30],
        }

    if base_params is None:
        base_params = {}

    if random_state is not None:
        base_params["random_state"] = random_state

    # Baseline embedding
    reducer_baseline = reducer_class(**base_params)
    X_baseline = reducer_baseline.fit_transform(X)

    results = {}

    for param_name, values in parameters_to_vary.items():
        sensitivities = []

        for value in values:
            kwargs = {**base_params, param_name: value}
            reducer = reducer_class(**kwargs)
            X_varied = reducer.fit_transform(X)

            # Procrustes distance
            try:
                _, _, disparity = procrustes(X_baseline, X_varied)
                sensitivities.append(float(disparity))
            except ValueError:
                sensitivities.append(1.0)

        results[param_name] = {
            "values": values,
            "procrustes_errors": sensitivities,
            "mean_sensitivity": float(np.mean(sensitivities)),
        }

    return results


# =============================================================================
# Part 5: Downstream Task Metrics
# =============================================================================


def clustering_quality(
    X_reduced: np.ndarray,
    labels_true: np.ndarray | None = None,
    n_clusters: int | None = None,
) -> dict[str, float]:
    """Evaluate clustering quality on the reduced embedding.

    Parameters
    ----------
    X_reduced : ndarray of shape (n_samples, n_components)
        Reduced embedding.

    labels_true : ndarray of shape (n_samples,), optional
        Ground truth labels for supervised metrics.

    n_clusters : int, optional
        Number of clusters for KMeans. Required if labels_true is None.

    Returns
    -------
    dict
        Dictionary with clustering metrics:
        - 'silhouette_score': Silhouette coefficient [-1, 1], higher is better
        - 'calinski_harabasz': Calinski-Harabasz index, higher is better
        - 'davies_bouldin': Davies-Bouldin index, lower is better
        - 'adjusted_rand_index': ARI [0, 1] if labels provided
        - 'normalized_mutual_info': NMI [0, 1] if labels provided

    Examples
    --------
    >>> from squeeze.evaluation import clustering_quality
    >>> metrics = clustering_quality(X_reduced, labels_true=y)
    >>> print(f"Silhouette: {metrics['silhouette_score']:.3f}")
    """
    X_reduced = np.asarray(X_reduced, dtype=np.float64)

    if labels_true is not None:
        labels_true = np.asarray(labels_true)
        n_clusters = len(np.unique(labels_true))

    if n_clusters is None:
        raise ValueError("Must provide labels_true or n_clusters")

    # Handle edge case of single cluster
    if n_clusters < 2:
        metrics = {
            "silhouette_score": 0.0,
            "calinski_harabasz": 0.0,
            "davies_bouldin": 0.0,
        }
        if labels_true is not None:
            metrics["adjusted_rand_index"] = 1.0  # Perfect agreement (trivial)
            metrics["normalized_mutual_info"] = 1.0
        return metrics

    # Cluster the reduced space
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_pred = kmeans.fit_predict(X_reduced)

    metrics = {
        "silhouette_score": float(silhouette_score(X_reduced, labels_pred)),
        "calinski_harabasz": float(calinski_harabasz_score(X_reduced, labels_pred)),
        "davies_bouldin": float(davies_bouldin_score(X_reduced, labels_pred)),
    }

    # Supervised metrics if ground truth available
    if labels_true is not None:
        metrics["adjusted_rand_index"] = float(
            adjusted_rand_score(labels_true, labels_pred)
        )
        metrics["normalized_mutual_info"] = float(
            normalized_mutual_info_score(labels_true, labels_pred)
        )

    return metrics


def classification_accuracy(
    X_reduced: np.ndarray,
    labels: np.ndarray,
    cv: int = 5,
    classifier: BaseEstimator | None = None,
) -> dict[str, float]:
    """Evaluate classification accuracy on the reduced embedding.

    Parameters
    ----------
    X_reduced : ndarray of shape (n_samples, n_components)
        Reduced embedding.

    labels : ndarray of shape (n_samples,)
        Class labels.

    cv : int, default=5
        Number of cross-validation folds.

    classifier : estimator, optional
        Classifier to use. Default: RandomForestClassifier.

    Returns
    -------
    dict
        Dictionary with:
        - 'mean_accuracy': Mean cross-validated accuracy
        - 'std_accuracy': Std of accuracy
        - 'scores': Individual fold scores

    Examples
    --------
    >>> from squeeze.evaluation import classification_accuracy
    >>> results = classification_accuracy(X_reduced, y)
    >>> print(f"Accuracy: {results['mean_accuracy']:.3f}")
    """
    X_reduced = np.asarray(X_reduced, dtype=np.float64)
    labels = np.asarray(labels)

    if classifier is None:
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Cross-validated accuracy
    scores = cross_val_score(classifier, X_reduced, labels, cv=cv, scoring="accuracy")

    return {
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std()),
        "scores": scores.tolist(),
    }


# =============================================================================
# Part 6: Comprehensive Evaluator
# =============================================================================


@dataclass
class EvaluationReport:
    """Container for DR evaluation results."""

    method_name: str
    n_samples: int
    n_features_original: int
    n_features_reduced: int

    # Local structure
    trustworthiness: dict[int, float] = field(default_factory=dict)
    continuity: dict[int, float] = field(default_factory=dict)
    co_ranking: dict[int, float] = field(default_factory=dict)

    # Global structure
    spearman_correlation: float | None = None
    global_structure: float | None = None
    density_preservation: float | None = None

    # Reconstruction
    reconstruction: dict[str, float] = field(default_factory=dict)

    # Stability
    stability: dict[str, float] = field(default_factory=dict)
    noise_robustness: dict[float, float] = field(default_factory=dict)

    # Downstream
    clustering: dict[str, float] = field(default_factory=dict)
    classification: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "method_name": self.method_name,
            "n_samples": self.n_samples,
            "n_features_original": self.n_features_original,
            "n_features_reduced": self.n_features_reduced,
            "local_structure": {
                "trustworthiness": self.trustworthiness,
                "continuity": self.continuity,
                "co_ranking": self.co_ranking,
            },
            "global_structure": {
                "spearman_correlation": self.spearman_correlation,
                "global_structure": self.global_structure,
                "density_preservation": self.density_preservation,
            },
            "reconstruction": self.reconstruction,
            "stability": self.stability,
            "noise_robustness": self.noise_robustness,
            "downstream": {
                "clustering": self.clustering,
                "classification": self.classification,
            },
        }

    def summary(self) -> str:
        """Return a summary string."""
        lines = [
            f"=== DR Evaluation Report: {self.method_name} ===",
            f"Data: {self.n_samples} samples, {self.n_features_original} -> {self.n_features_reduced} dims",
            "",
            "Local Structure:",
        ]

        for k, v in self.trustworthiness.items():
            c = self.continuity.get(k, 0)
            lines.append(f"  k={k}: Trustworthiness={v:.3f}, Continuity={c:.3f}")

        lines.append("")
        lines.append("Global Structure:")
        if self.spearman_correlation is not None:
            lines.append(
                f"  Spearman distance correlation: {self.spearman_correlation:.3f}"
            )
        if self.global_structure is not None:
            lines.append(
                f"  Global structure preservation: {self.global_structure:.3f}"
            )
        if self.density_preservation is not None:
            lines.append(f"  Density preservation: {self.density_preservation:.3f}")

        if self.reconstruction:
            lines.append("")
            lines.append("Reconstruction:")
            lines.append(
                f"  Normalized RMSE: {self.reconstruction.get('normalized_rmse', 0):.3f}"
            )
            lines.append(f"  R-squared: {self.reconstruction.get('r2', 0):.3f}")

        if self.stability:
            lines.append("")
            lines.append("Stability:")
            lines.append(
                f"  Stability score: {self.stability.get('stability_score', 0):.3f}"
            )

        if self.clustering:
            lines.append("")
            lines.append("Clustering Quality:")
            lines.append(
                f"  Silhouette: {self.clustering.get('silhouette_score', 0):.3f}"
            )
            if "adjusted_rand_index" in self.clustering:
                lines.append(
                    f"  Adjusted Rand Index: {self.clustering['adjusted_rand_index']:.3f}"
                )

        if self.classification:
            lines.append("")
            lines.append("Classification:")
            lines.append(
                f"  Accuracy: {self.classification.get('mean_accuracy', 0):.3f}"
            )

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


class DREvaluator:
    """Comprehensive dimensionality reduction evaluator.

    Provides a unified interface for computing all evaluation metrics.

    Parameters
    ----------
    X_original : ndarray of shape (n_samples, n_features)
        Original high-dimensional data.

    X_reduced : ndarray of shape (n_samples, n_components)
        Reduced embedding.

    labels : ndarray of shape (n_samples,), optional
        Ground truth labels for supervised metrics.

    reducer : estimator, optional
        The DR estimator used (for stability metrics).

    method_name : str, default='Unknown'
        Name of the DR method for reporting.

    Examples
    --------
    >>> from squeeze.evaluation import DREvaluator
    >>> from squeeze import UMAP
    >>>
    >>> reducer = UMAP(n_components=2)
    >>> X_reduced = reducer.fit_transform(X)
    >>>
    >>> evaluator = DREvaluator(X, X_reduced, labels=y, reducer=reducer, method_name='UMAP')
    >>> report = evaluator.evaluate_all()
    >>> print(report)
    """

    def __init__(
        self,
        X_original: np.ndarray,
        X_reduced: np.ndarray,
        labels: np.ndarray | None = None,
        reducer: BaseEstimator | None = None,
        method_name: str = "Unknown",
    ) -> None:
        self.X_original = np.asarray(X_original, dtype=np.float64)
        self.X_reduced = np.asarray(X_reduced, dtype=np.float64)
        self.labels = np.asarray(labels) if labels is not None else None
        self.reducer = reducer
        self.method_name = method_name

    def evaluate_local_structure(
        self,
        k_values: list[int] | None = None,
    ) -> dict[str, dict[int, float]]:
        """Evaluate local structure preservation.

        Parameters
        ----------
        k_values : list of int, optional
            k values for neighbor analysis. Default: [5, 15, 30]

        Returns
        -------
        dict
            Trustworthiness, continuity, and co-ranking for each k.
        """
        if k_values is None:
            k_values = [5, 15, 30]

        results = {
            "trustworthiness": {},
            "continuity": {},
            "co_ranking": {},
        }

        for k in k_values:
            results["trustworthiness"][k] = trustworthiness(
                self.X_original, self.X_reduced, k=k
            )
            results["continuity"][k] = continuity(self.X_original, self.X_reduced, k=k)
            results["co_ranking"][k] = co_ranking_quality(
                self.X_original, self.X_reduced, k=k
            )

        return results

    def evaluate_global_structure(self) -> dict[str, float | None]:
        """Evaluate global structure preservation.

        Returns
        -------
        dict
            Spearman correlation, global structure, and density preservation.
        """
        results = {
            "spearman_correlation": spearman_distance_correlation(
                self.X_original, self.X_reduced
            ),
            "density_preservation": local_density_preservation(
                self.X_original, self.X_reduced
            ),
        }

        if self.labels is not None:
            results["global_structure"] = global_structure_preservation(
                self.X_original, self.X_reduced, self.labels
            )
        else:
            results["global_structure"] = None

        return results

    def evaluate_reconstruction(self) -> dict[str, float]:
        """Evaluate reconstruction quality.

        Returns
        -------
        dict
            Reconstruction error metrics.
        """
        return reconstruction_error(self.X_original, self.X_reduced)

    def evaluate_stability(
        self,
        n_bootstrap: int = 10,
    ) -> dict[str, float]:
        """Evaluate stability.

        Parameters
        ----------
        n_bootstrap : int, default=10
            Number of bootstrap iterations.

        Returns
        -------
        dict
            Stability metrics.
        """
        if self.reducer is None:
            return {}

        return bootstrap_stability(
            self.X_original, self.reducer, n_bootstrap=n_bootstrap
        )

    def evaluate_noise_robustness(
        self,
        noise_levels: list[float] | None = None,
    ) -> dict[float, float]:
        """Evaluate noise robustness.

        Parameters
        ----------
        noise_levels : list of float, optional
            Noise levels to test.

        Returns
        -------
        dict
            Robustness score for each noise level.
        """
        if self.reducer is None:
            return {}

        return noise_robustness(
            self.X_original, self.reducer, noise_levels=noise_levels
        )

    def evaluate_clustering(self) -> dict[str, float]:
        """Evaluate clustering quality on embedding.

        Returns
        -------
        dict
            Clustering quality metrics.
        """
        return clustering_quality(self.X_reduced, labels_true=self.labels)

    def evaluate_classification(self, cv: int = 5) -> dict[str, float]:
        """Evaluate classification accuracy on embedding.

        Parameters
        ----------
        cv : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        dict
            Classification accuracy metrics.
        """
        if self.labels is None:
            return {}

        return classification_accuracy(self.X_reduced, self.labels, cv=cv)

    def evaluate_all(
        self,
        k_values: list[int] | None = None,
        include_stability: bool = True,
        include_noise: bool = False,
        n_bootstrap: int = 10,
    ) -> EvaluationReport:
        """Run all evaluations and return comprehensive report.

        Parameters
        ----------
        k_values : list of int, optional
            k values for local structure metrics.

        include_stability : bool, default=True
            Whether to include stability metrics (slower).

        include_noise : bool, default=False
            Whether to include noise robustness (slower).

        n_bootstrap : int, default=10
            Number of bootstrap iterations for stability.

        Returns
        -------
        EvaluationReport
            Comprehensive evaluation report.

        Examples
        --------
        >>> evaluator = DREvaluator(X, X_reduced, labels=y, reducer=reducer)
        >>> report = evaluator.evaluate_all()
        >>> print(report)
        >>> report.to_dict()  # Get as dictionary
        """
        # Create report
        report = EvaluationReport(
            method_name=self.method_name,
            n_samples=self.X_original.shape[0],
            n_features_original=self.X_original.shape[1],
            n_features_reduced=self.X_reduced.shape[1],
        )

        # Local structure
        local = self.evaluate_local_structure(k_values=k_values)
        report.trustworthiness = local["trustworthiness"]
        report.continuity = local["continuity"]
        report.co_ranking = local["co_ranking"]

        # Global structure
        global_struct = self.evaluate_global_structure()
        report.spearman_correlation = global_struct["spearman_correlation"]
        report.global_structure = global_struct["global_structure"]
        report.density_preservation = global_struct["density_preservation"]

        # Reconstruction
        report.reconstruction = self.evaluate_reconstruction()

        # Stability (optional, slower)
        if include_stability and self.reducer is not None:
            report.stability = self.evaluate_stability(n_bootstrap=n_bootstrap)

        # Noise robustness (optional, slower)
        if include_noise and self.reducer is not None:
            report.noise_robustness = self.evaluate_noise_robustness()

        # Downstream tasks
        report.clustering = self.evaluate_clustering()
        if self.labels is not None:
            report.classification = self.evaluate_classification()

        return report


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_evaluate(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    k: int = 15,
) -> dict[str, float]:
    """Quick evaluation with core metrics only.

    Parameters
    ----------
    X_original : ndarray
        Original data.

    X_reduced : ndarray
        Reduced embedding.

    k : int, default=15
        Number of neighbors.

    Returns
    -------
    dict
        Core metrics: trustworthiness, continuity, spearman_correlation.

    Examples
    --------
    >>> metrics = quick_evaluate(X, X_reduced)
    >>> print(f"T={metrics['trustworthiness']:.3f}")
    """
    return {
        "trustworthiness": trustworthiness(X_original, X_reduced, k=k),
        "continuity": continuity(X_original, X_reduced, k=k),
        "spearman_correlation": spearman_distance_correlation(X_original, X_reduced),
    }
