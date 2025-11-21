"""Evaluation metrics for dimensionality reduction quality assessment.

This module provides comprehensive metrics for evaluating the quality of
dimensionality reduction embeddings, measuring different aspects like
local structure preservation, global structure, reconstruction error,
stability, and downstream task performance.

Basic usage:
    >>> from squeeze.metrics import trustworthiness, continuity
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.decomposition import PCA
    >>> from squeeze import UMAP
    >>>
    >>> iris = load_iris()
    >>> X = iris.data
    >>>
    >>> # Compute 2D embedding
    >>> reducer = UMAP(n_components=2)
    >>> X_embedded = reducer.fit_transform(X)
    >>>
    >>> # Evaluate embedding quality
    >>> trust = trustworthiness(X, X_embedded)
    >>> cont = continuity(X, X_embedded)
    >>> print(f"Trustworthiness: {trust:.3f}, Continuity: {cont:.3f}")

Author: UMAP development team
License: BSD 3 clause
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def trustworthiness(
    X: np.ndarray,
    X_embedded: np.ndarray,
    k: int = 15,
    metric: str = "euclidean",
) -> float:
    r"""Measure trustworthiness of an embedding.

    Trustworthiness measures the degree to which the k-nearest neighbors
    in the embedding space are the same as in the original space. A value
    of 1.0 indicates perfect trustworthiness (all neighbors match), while
    0.0 indicates no trustworthiness.

    .. math::
        T(k) = 1 - \frac{2}{nk(2n-3k-1)} \sum_{i=1}^{n} \sum_{j \in U_i(k)} (r_i(x_j) - k)

    where U_i(k) is the set of k-nearest neighbors in embedding space
    that are not in the original k-neighbors, and r_i(x_j) is the rank
    of point x_j in the original space.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The original data.

    X_embedded : ndarray of shape (n_samples, n_components)
        The embedded data.

    k : int, default=15
        The number of nearest neighbors to consider.

    metric : str, default='euclidean'
        The distance metric to use for computing neighbors in the original space.

    Returns
    -------
    trustworthiness : float
        A score between 0 and 1, where 1 indicates perfect trustworthiness.

    References
    ----------
    Venna, J., & Kaski, S. (2006). Local multidimensional scaling.
    """
    n_samples = X.shape[0]

    # Compute k-nearest neighbors in original space
    nbrs_original = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    _, indices_original = nbrs_original.kneighbors(X)
    indices_original = indices_original[:, 1:]  # Remove self

    # Compute k-nearest neighbors in embedding space
    nbrs_embedded = NearestNeighbors(n_neighbors=k + 1).fit(X_embedded)
    _, indices_embedded = nbrs_embedded.kneighbors(X_embedded)
    indices_embedded = indices_embedded[:, 1:]  # Remove self

    # For each point, find neighbors in embedding that are not in original
    trust_sum = 0
    for i in range(n_samples):
        original_neighbors = set(indices_original[i])
        embedding_neighbors = indices_embedded[i]

        for rank, j in enumerate(embedding_neighbors, 1):
            if j not in original_neighbors:
                # j is a neighbor in embedding but not in original
                # Find its rank in the original space
                original_ranks = np.where(indices_original[i] == j)[0]
                if len(original_ranks) == 0:
                    # Not in top k of original, so rank is > k
                    original_rank = k + 1
                else:
                    original_rank = original_ranks[0] + 1

                trust_sum += max(0, original_rank - k)

    # Normalize
    trustworthiness_score = 1 - (2 / (n_samples * k * (2 * n_samples - 3 * k - 1))) * trust_sum

    return max(0, min(1, trustworthiness_score))


def continuity(
    X: np.ndarray,
    X_embedded: np.ndarray,
    k: int = 15,
    metric: str = "euclidean",
) -> float:
    r"""Measure continuity of an embedding.

    Continuity is the inverse of trustworthiness - it measures whether
    neighbors in the original space are also neighbors in the embedding.
    A value of 1.0 indicates perfect continuity.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The original data.

    X_embedded : ndarray of shape (n_samples, n_components)
        The embedded data.

    k : int, default=15
        The number of nearest neighbors to consider.

    metric : str, default='euclidean'
        The distance metric for the original space.

    Returns
    -------
    continuity : float
        A score between 0 and 1, where 1 indicates perfect continuity.

    References
    ----------
    Venna, J., & Kaski, S. (2006). Local multidimensional scaling.
    """
    n_samples = X.shape[0]

    # Compute k-nearest neighbors in original space
    nbrs_original = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    _, indices_original = nbrs_original.kneighbors(X)
    indices_original = indices_original[:, 1:]  # Remove self

    # Compute k-nearest neighbors in embedding space
    nbrs_embedded = NearestNeighbors(n_neighbors=k + 1).fit(X_embedded)
    _, indices_embedded = nbrs_embedded.kneighbors(X_embedded)
    indices_embedded = indices_embedded[:, 1:]  # Remove self

    # For each point, find neighbors in original that are not in embedding
    cont_sum = 0
    for i in range(n_samples):
        embedding_neighbors = set(indices_embedded[i])
        original_neighbors = indices_original[i]

        for rank, j in enumerate(original_neighbors, 1):
            if j not in embedding_neighbors:
                # j is a neighbor in original but not in embedding
                # Find its rank in the embedding space
                embedding_ranks = np.where(indices_embedded[i] == j)[0]
                if len(embedding_ranks) == 0:
                    # Not in top k of embedding, so rank is > k
                    embedding_rank = k + 1
                else:
                    embedding_rank = embedding_ranks[0] + 1

                cont_sum += max(0, embedding_rank - k)

    # Normalize
    continuity_score = 1 - (2 / (n_samples * k * (2 * n_samples - 3 * k - 1))) * cont_sum

    return max(0, min(1, continuity_score))


def local_continuity_meta_estimate(
    X: np.ndarray,
    X_embedded: np.ndarray,
    k: int = 15,
    metric: str = "euclidean",
) -> float:
    """Compute local continuity meta-estimate (LCMC).

    LCMC measures the degree to which the local structure of the data
    is preserved in the embedding. It's based on the correlation between
    original and embedded distances for neighbors.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The original data.

    X_embedded : ndarray of shape (n_samples, n_components)
        The embedded data.

    k : int, default=15
        The number of nearest neighbors to consider.

    metric : str, default='euclidean'
        The distance metric for the original space.

    Returns
    -------
    lcmc : float
        Local continuity meta-estimate score.
    """
    n_samples = X.shape[0]

    # Compute pairwise distances
    D_original = pairwise_distances(X, metric=metric)
    D_embedded = pairwise_distances(X_embedded, metric="euclidean")

    # For each point, look at distances to k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    _, indices = nbrs.kneighbors(X)
    indices = indices[:, 1:]  # Remove self

    correlations = []
    for i in range(n_samples):
        neighbors = indices[i]
        original_distances = D_original[i, neighbors]
        embedded_distances = D_embedded[i, neighbors]

        # Compute Spearman correlation between distances
        original_ranks = np.argsort(np.argsort(original_distances))
        embedded_ranks = np.argsort(np.argsort(embedded_distances))

        correlation = np.corrcoef(original_ranks, embedded_ranks)[0, 1]
        if not np.isnan(correlation):
            correlations.append(correlation)

    return np.mean(correlations) if correlations else 0.0


def reconstruction_error(
    X: np.ndarray,
    X_embedded: np.ndarray,
    fit_intercept: bool = True,
) -> float:
    r"""Compute reconstruction error using linear regression.

    Measures how well the original high-dimensional data can be
    reconstructed from the embedding using linear regression.

    .. math::
        \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} ||X_i - \hat{X}_i||^2}

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The original data.

    X_embedded : ndarray of shape (n_samples, n_components)
        The embedded data.

    fit_intercept : bool, default=True
        Whether to fit an intercept term.

    Returns
    -------
    reconstruction_error : float
        The normalized RMSE of reconstruction.
    """
    from sklearn.linear_model import LinearRegression

    # Fit linear regression from embedding to original
    regressor = LinearRegression(fit_intercept=fit_intercept)
    regressor.fit(X_embedded, X)
    X_reconstructed = regressor.predict(X_embedded)

    # Compute RMSE
    rmse = np.sqrt(np.mean((X - X_reconstructed) ** 2))

    # Normalize by the average norm of X
    avg_norm = np.mean(np.linalg.norm(X, axis=1))
    if avg_norm > 0:
        normalized_error = rmse / avg_norm
    else:
        normalized_error = rmse

    return normalized_error


def spearman_distance_correlation(
    X: np.ndarray,
    X_embedded: np.ndarray,
    sample_size: Optional[int] = None,
) -> float:
    r"""Compute Spearman correlation of pairwise distances.

    Measures the rank correlation between pairwise distances in the
    original and embedded spaces. Values closer to 1 indicate better
    preservation of distance relationships.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The original data.

    X_embedded : ndarray of shape (n_samples, n_components)
        The embedded data.

    sample_size : int, optional
        If provided, sample this many point pairs to compute correlation.
        Useful for large datasets to speed up computation.

    Returns
    -------
    correlation : float
        Spearman correlation between original and embedded distances.
    """
    from scipy.stats import spearmanr

    # Compute pairwise distances
    D_original = pairwise_distances(X, metric="euclidean")
    D_embedded = pairwise_distances(X_embedded, metric="euclidean")

    # Get upper triangular indices (avoid redundant pairs and self-distances)
    n = X.shape[0]
    indices = np.triu_indices(n, k=1)

    original_distances = D_original[indices]
    embedded_distances = D_embedded[indices]

    # If dataset is large, sample pairs
    if sample_size is not None and len(original_distances) > sample_size:
        sample_indices = np.random.choice(
            len(original_distances), size=sample_size, replace=False
        )
        original_distances = original_distances[sample_indices]
        embedded_distances = embedded_distances[sample_indices]

    # Compute Spearman correlation
    correlation, _ = spearmanr(original_distances, embedded_distances)

    return float(np.clip(correlation, -1, 1))


class DREvaluator:
    """Comprehensive evaluator for dimensionality reduction embeddings.

    Computes multiple quality metrics to assess different aspects of
    an embedding.

    Parameters
    ----------
    k : int, default=15
        Number of neighbors for local structure metrics.

    sample_size : int, optional
        For large datasets, sample this many distances for correlation.

    Attributes
    ----------
    metrics_ : dict
        Dictionary of computed metrics after evaluation.

    Examples
    --------
    >>> from squeeze.metrics import DREvaluator
    >>> from sklearn.datasets import load_iris
    >>> from squeeze import UMAP
    >>>
    >>> iris = load_iris()
    >>> X = iris.data
    >>> X_embedded = UMAP(n_components=2).fit_transform(X)
    >>>
    >>> evaluator = DREvaluator()
    >>> evaluator.evaluate(X, X_embedded)
    >>> print(evaluator.metrics_)
    """

    def __init__(self, k: int = 15, sample_size: Optional[int] = None):
        """Initialize the evaluator."""
        self.k = k
        self.sample_size = sample_size
        self.metrics_ = {}

    def evaluate(
        self,
        X: np.ndarray,
        X_embedded: np.ndarray,
        metric: str = "euclidean",
    ) -> dict[str, float]:
        """Compute all metrics for an embedding.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The original data.

        X_embedded : ndarray of shape (n_samples, n_components)
            The embedded data.

        metric : str, default='euclidean'
            Distance metric for original space.

        Returns
        -------
        metrics : dict
            Dictionary with metric names and values.
        """
        self.metrics_ = {}

        # Local structure metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.metrics_["trustworthiness"] = trustworthiness(
                X, X_embedded, k=self.k, metric=metric
            )
            self.metrics_["continuity"] = continuity(
                X, X_embedded, k=self.k, metric=metric
            )
            self.metrics_["lcmc"] = local_continuity_meta_estimate(
                X, X_embedded, k=self.k, metric=metric
            )

            # Global structure metrics
            self.metrics_["reconstruction_error"] = reconstruction_error(X, X_embedded)
            self.metrics_["spearman_distance_correlation"] = (
                spearman_distance_correlation(X, X_embedded, self.sample_size)
            )

        return self.metrics_

    def summary(self) -> str:
        """Return a text summary of the evaluation metrics.

        Returns
        -------
        summary : str
            Formatted summary of metrics.
        """
        if not self.metrics_:
            return "No metrics computed yet. Call evaluate() first."

        summary_lines = ["Dimensionality Reduction Quality Metrics", "=" * 40]

        for metric_name, value in self.metrics_.items():
            summary_lines.append(f"{metric_name:.<35} {value:.4f}")

        return "\n".join(summary_lines)
