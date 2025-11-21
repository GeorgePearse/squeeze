"""Out-of-sample extension and streaming support for dimensionality reduction.

This module provides wrappers that add transform() capabilities to DR algorithms
that only support fit_transform(), and streaming/incremental DR support.

Example usage:
    >>> from squeeze.extensions import OutOfSampleDR, StreamingDR
    >>> from squeeze import TSNE
    >>>
    >>> # Add transform capability to t-SNE
    >>> tsne = OutOfSampleDR(TSNE(n_components=2))
    >>> tsne.fit(X_train)
    >>> X_test_embedded = tsne.transform(X_test)
    >>>
    >>> # Streaming DR for incremental updates
    >>> streaming = StreamingDR(base_reducer=UMAP())
    >>> embedding = streaming.fit_transform(X_initial)
    >>> embedding = streaming.partial_fit_transform(X_new)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

__all__ = ["OutOfSampleDR", "StreamingDR"]


class OutOfSampleDR(BaseEstimator):
    """Wrapper that adds out-of-sample extension (transform) to any DR algorithm.

    This wrapper stores the training data and embeddings, then uses k-NN
    interpolation to embed new points. For each new point, it finds the
    k nearest neighbors in the original space and computes a weighted
    average of their embeddings.

    Parameters
    ----------
    base_reducer : estimator
        A dimensionality reduction estimator with fit_transform() method.
        Examples: TSNE, MDS, Isomap, LLE, PHATE, TriMap, PaCMAP

    n_neighbors : int, default=5
        Number of neighbors to use for interpolation.

    weights : str, default='distance'
        Weight function for interpolation:
        - 'uniform': All neighbors weighted equally
        - 'distance': Weight by inverse distance

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        The embedding of the training data.

    X_train_ : ndarray of shape (n_samples, n_features)
        The training data (stored for neighbor lookup).

    Examples
    --------
    >>> from squeeze import TSNE
    >>> from squeeze.extensions import OutOfSampleDR
    >>>
    >>> # Wrap t-SNE to add transform capability
    >>> tsne = OutOfSampleDR(TSNE(n_components=2, n_iter=500))
    >>> tsne.fit(X_train)
    >>> X_test_2d = tsne.transform(X_test)
    >>>
    >>> # Or fit and transform in one step
    >>> X_train_2d = tsne.fit_transform(X_train)

    Notes
    -----
    This approach works well when:
    - New points are similar to training data (interpolation, not extrapolation)
    - The embedding preserves local structure (most DR methods do)

    For methods with native transform() like PCA or UMAP, use them directly
    instead of this wrapper for better results.
    """

    def __init__(
        self,
        base_reducer: Any,
        n_neighbors: int = 5,
        weights: str = "distance",
    ) -> None:
        self.base_reducer = base_reducer
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> OutOfSampleDR:
        """Fit the base reducer and store training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,), optional
            Target values (ignored).

        Returns
        -------
        self : OutOfSampleDR
        """
        X = np.asarray(X, dtype=np.float64)
        self.X_train_ = X

        # Fit the base reducer directly (no cloning needed)
        self.embedding_ = self.base_reducer.fit_transform(X)

        # Build k-NN index for fast neighbor lookup
        self.nn_ = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(X)),
            algorithm="auto",
        )
        self.nn_.fit(X)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using k-NN interpolation.

        For each new point, finds k nearest neighbors in the training data
        and computes a weighted average of their embeddings.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_embedded : ndarray of shape (n_samples, n_components)
            Embedded coordinates.
        """
        check_is_fitted(self, ["embedding_", "nn_"])

        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_components = self.embedding_.shape[1]

        # Find k nearest neighbors
        distances, indices = self.nn_.kneighbors(X)

        # Compute weighted average of neighbor embeddings
        X_embedded = np.zeros((n_samples, n_components))

        for i in range(n_samples):
            neighbor_embeddings = self.embedding_[indices[i]]

            if self.weights == "uniform":
                X_embedded[i] = neighbor_embeddings.mean(axis=0)
            elif self.weights == "distance":
                # Inverse distance weighting (add small epsilon to avoid div by zero)
                dist = distances[i] + 1e-10
                weights = 1.0 / dist
                weights /= weights.sum()
                X_embedded[i] = np.average(neighbor_embeddings, axis=0, weights=weights)
            else:
                raise ValueError(f"Unknown weights: {self.weights}")

        return X_embedded

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit and return the embedding of training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,), optional
            Target values (ignored).

        Returns
        -------
        X_embedded : ndarray of shape (n_samples, n_components)
            Embedded coordinates.
        """
        self.fit(X, y)
        return self.embedding_


class StreamingDR(BaseEstimator):
    """Streaming/incremental dimensionality reduction.

    This wrapper enables incremental updates to an embedding as new data
    arrives. It uses k-NN interpolation to embed new points based on
    existing embeddings.

    Parameters
    ----------
    base_reducer : estimator
        A dimensionality reduction estimator.

    n_neighbors : int, default=5
        Number of neighbors for interpolation.

    Attributes
    ----------
    X_all_ : ndarray
        All data seen so far.

    embedding_ : ndarray
        Current embedding of all data.

    Examples
    --------
    >>> from squeeze import UMAP
    >>> from squeeze.extensions import StreamingDR
    >>>
    >>> # Create streaming reducer
    >>> streaming = StreamingDR(UMAP())
    >>>
    >>> # Initial fit
    >>> embedding = streaming.fit_transform(X_initial)
    >>>
    >>> # Add new data incrementally
    >>> for X_batch in data_stream:
    ...     embedding = streaming.partial_fit_transform(X_batch)
    """

    def __init__(
        self,
        base_reducer: Any,
        n_neighbors: int = 5,
    ) -> None:
        self.base_reducer = base_reducer
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> StreamingDR:
        """Initial fit on data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Initial training data.

        y : ignored

        Returns
        -------
        self : StreamingDR
        """
        X = np.asarray(X, dtype=np.float64)
        self.X_all_ = X.copy()

        # Fit base reducer
        self.embedding_ = self.base_reducer.fit_transform(X)

        # Build k-NN index
        self._rebuild_nn_index()

        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray | None = None) -> StreamingDR:
        """Add new data to the model using k-NN interpolation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to add.

        y : ignored

        Returns
        -------
        self : StreamingDR
        """
        check_is_fitted(self, ["X_all_", "embedding_"])

        X = np.asarray(X, dtype=np.float64)

        # Use k-NN interpolation for new points
        new_embeddings = self._interpolate_new_points(X)
        self.X_all_ = np.vstack([self.X_all_, X])
        self.embedding_ = np.vstack([self.embedding_, new_embeddings])
        self._rebuild_nn_index()

        return self

    def partial_fit_transform(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        """Add new data and return updated full embedding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to add.

        y : ignored

        Returns
        -------
        embedding : ndarray of shape (n_total_samples, n_components)
            The embedding of all data seen so far.
        """
        self.partial_fit(X, y)
        return self.embedding_

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit and return embedding."""
        self.fit(X, y)
        return self.embedding_

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data without adding to model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_embedded : ndarray of shape (n_samples, n_components)
            Embedded coordinates.
        """
        check_is_fitted(self, ["embedding_", "nn_"])
        return self._interpolate_new_points(X)

    def _interpolate_new_points(self, X: np.ndarray) -> np.ndarray:
        """Embed new points using k-NN interpolation."""
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_components = self.embedding_.shape[1]

        # Find k nearest neighbors in existing data
        k = min(self.n_neighbors, len(self.X_all_))
        distances, indices = self.nn_.kneighbors(X, n_neighbors=k)

        # Weighted average of neighbor embeddings
        new_embeddings = np.zeros((n_samples, n_components))

        for i in range(n_samples):
            neighbor_embeddings = self.embedding_[indices[i]]
            dist = distances[i] + 1e-10
            weights = 1.0 / dist
            weights /= weights.sum()
            new_embeddings[i] = np.average(neighbor_embeddings, axis=0, weights=weights)

        return new_embeddings

    def _rebuild_nn_index(self) -> None:
        """Rebuild the k-NN index."""
        self.nn_ = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(self.X_all_)),
            algorithm="auto",
        )
        self.nn_.fit(self.X_all_)
