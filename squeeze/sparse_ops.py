"""Sparse matrix operations and utilities for dimensionality reduction.

This module provides efficient sparse matrix operations optimized for
dimensionality reduction, including sparse distance computations,
sparse k-NN graph construction, and format detection/conversion.

Use Cases:
- Single-cell RNA-seq (95-98% sparse)
- NLP/text analysis (TF-IDF vectors)
- Network/graph analysis
- Sensor data with many zeros

Example:
    >>> import scipy.sparse as sp
    >>> from squeeze.sparse_ops import SparseFormatDetector, sparse_euclidean
    >>>
    >>> # Create sparse data
    >>> X = sp.random(1000, 5000, density=0.01, format='csr')
    >>>
    >>> # Auto-detect and convert to canonical format
    >>> X_canonical = SparseFormatDetector.to_canonical(X, target_format='csr')
    >>>
    >>> # Compute sparse distances efficiently
    >>> distances = sparse_euclidean(X[:100], X[100:110])

Author: UMAP development team
License: BSD 3 clause
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp


class SparseFormatDetector:
    """Detect and convert between sparse matrix formats.

    Provides utilities for working with various sparse matrix formats
    (CSR, CSC, COO, DOK, etc.) and converting between them.
    """

    SPARSE_FORMATS = {"csr", "csc", "coo", "dok", "lil", "bsr"}

    @staticmethod
    def is_sparse(X: object) -> bool:
        """Check if X is a sparse matrix.

        Parameters
        ----------
        X : object
            Object to check.

        Returns
        -------
        is_sparse : bool
            True if X is a scipy.sparse matrix.
        """
        return sp.issparse(X)

    @staticmethod
    def get_format(X: object) -> str | None:
        """Get the format of a sparse matrix.

        Parameters
        ----------
        X : object
            Sparse matrix.

        Returns
        -------
        format : str or None
            Format name ('csr', 'csc', etc.) or None if not sparse.
        """
        if not sp.issparse(X):
            return None
        return X.format

    @staticmethod
    def to_canonical(
        X: np.ndarray | sp.spmatrix,
        target_format: str = "csr",
    ) -> sp.spmatrix:
        """Convert to canonical sparse format.

        Handles both dense arrays and sparse matrices, converting to
        the specified sparse format.

        Parameters
        ----------
        X : ndarray or sparse matrix
            Input data (dense or sparse).

        target_format : str, default='csr'
            Target format ('csr', 'csc', 'coo', etc.).

        Returns
        -------
        X_sparse : sparse matrix
            Data in target sparse format.
        """
        if isinstance(X, np.ndarray):
            # Convert dense to sparse
            X_sparse = sp.csr_matrix(X)
        elif sp.issparse(X):
            X_sparse = X
        else:
            msg = f"Expected ndarray or sparse matrix, got {type(X)}"
            raise TypeError(msg)

        # Convert to target format
        if target_format == "csr":
            return X_sparse.tocsr()
        elif target_format == "csc":
            return X_sparse.tocsc()
        elif target_format == "coo":
            return X_sparse.tocoo()
        elif target_format == "dok":
            return X_sparse.todok()
        elif target_format == "lil":
            return X_sparse.tolil()
        else:
            msg = f"Unknown format: {target_format}"
            raise ValueError(msg)

    @staticmethod
    def get_sparsity(X: np.ndarray | sp.spmatrix) -> float:
        """Compute sparsity of a matrix.

        Parameters
        ----------
        X : ndarray or sparse matrix
            Input data.

        Returns
        -------
        sparsity : float
            Fraction of zero elements (0 to 1).
        """
        if isinstance(X, np.ndarray):
            n_zeros = np.sum(X == 0)
            n_total = X.size
        elif sp.issparse(X):
            n_zeros = X.shape[0] * X.shape[1] - X.nnz
            n_total = X.shape[0] * X.shape[1]
        else:
            msg = f"Expected ndarray or sparse matrix, got {type(X)}"
            raise TypeError(msg)

        return n_zeros / n_total if n_total > 0 else 0.0

    @staticmethod
    def suggest_format(X: np.ndarray | sp.spmatrix) -> str:
        """Suggest optimal sparse format based on sparsity and usage pattern.

        Parameters
        ----------
        X : ndarray or sparse matrix
            Input data.

        Returns
        -------
        format : str
            Suggested format ('csr', 'csc', 'coo', etc.).
        """
        sparsity = SparseFormatDetector.get_sparsity(X)

        if sparsity > 0.9:
            # Very sparse: COO good for construction, CSR good for ops
            return "csr"
        elif sparsity > 0.5:
            # Moderately sparse: CSR for row access
            return "csr"
        else:
            # Less sparse: maybe just use dense
            return "csr"


def sparse_euclidean(
    X: np.ndarray | sp.spmatrix,
    Y: np.ndarray | sp.spmatrix,
    squared: bool = False,
) -> np.ndarray:
    """Compute Euclidean distances between sparse vectors.

    Efficiently computes pairwise Euclidean distances between rows of X
    and rows of Y, handling sparse matrices properly to avoid densification.

    Parameters
    ----------
    X : ndarray or sparse matrix, shape (n_samples, n_features)
        First set of vectors.

    Y : ndarray or sparse matrix, shape (m_samples, n_features)
        Second set of vectors.

    squared : bool, default=False
        If True, return squared distances.

    Returns
    -------
    distances : ndarray, shape (n_samples, m_samples)
        Pairwise distances.
    """
    # Convert to sparse if needed
    X = SparseFormatDetector.to_canonical(X, target_format="csr")
    Y = SparseFormatDetector.to_canonical(Y, target_format="csr")

    # Compute ||x||^2 and ||y||^2
    X_norm_sq = np.asarray(X.multiply(X).sum(axis=1)).ravel()
    Y_norm_sq = np.asarray(Y.multiply(Y).sum(axis=1)).ravel()

    # Compute <x, y> via matrix multiplication
    XY = X.dot(Y.T)
    if sp.issparse(XY):
        XY = XY.toarray()

    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
    distances_sq = X_norm_sq[:, np.newaxis] + Y_norm_sq[np.newaxis, :] - 2 * XY
    distances_sq = np.maximum(distances_sq, 0)  # Avoid numerical errors

    if squared:
        return distances_sq
    else:
        return np.sqrt(distances_sq)


def sparse_cosine(
    X: np.ndarray | sp.spmatrix,
    Y: np.ndarray | sp.spmatrix,
) -> np.ndarray:
    """Compute cosine distances between sparse vectors.

    Parameters
    ----------
    X : ndarray or sparse matrix, shape (n_samples, n_features)
        First set of vectors.

    Y : ndarray or sparse matrix, shape (m_samples, n_features)
        Second set of vectors.

    Returns
    -------
    distances : ndarray, shape (n_samples, m_samples)
        Pairwise cosine distances (1 - cosine similarity).
    """
    from sklearn.metrics.pairwise import cosine_distances

    X = SparseFormatDetector.to_canonical(X, target_format="csr")
    Y = SparseFormatDetector.to_canonical(Y, target_format="csr")

    return cosine_distances(X, Y)


def sparse_manhattan(
    X: np.ndarray | sp.spmatrix,
    Y: np.ndarray | sp.spmatrix,
) -> np.ndarray:
    """Compute Manhattan (L1) distances between sparse vectors.

    Parameters
    ----------
    X : ndarray or sparse matrix, shape (n_samples, n_features)
        First set of vectors.

    Y : ndarray or sparse matrix, shape (m_samples, n_features)
        Second set of vectors.

    Returns
    -------
    distances : ndarray, shape (n_samples, m_samples)
        Pairwise Manhattan distances.
    """
    X = SparseFormatDetector.to_canonical(X, target_format="csr")
    Y = SparseFormatDetector.to_canonical(Y, target_format="csr")

    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    distances = np.zeros((n_samples_X, n_samples_Y))

    for i in range(n_samples_X):
        x_i = X[i].toarray().ravel()
        for j in range(n_samples_Y):
            y_j = Y[j].toarray().ravel()
            distances[i, j] = np.sum(np.abs(x_i - y_j))

    return distances


def sparse_jaccard(
    X: np.ndarray | sp.spmatrix,
    Y: np.ndarray | sp.spmatrix,
) -> np.ndarray:
    """Compute Jaccard distances between sparse binary vectors.

    Jaccard distance = 1 - (intersection / union).

    Parameters
    ----------
    X : ndarray or sparse matrix, shape (n_samples, n_features)
        First set of binary vectors.

    Y : ndarray or sparse matrix, shape (m_samples, n_features)
        Second set of binary vectors.

    Returns
    -------
    distances : ndarray, shape (n_samples, m_samples)
        Pairwise Jaccard distances.
    """
    X = SparseFormatDetector.to_canonical(X, target_format="csr")
    Y = SparseFormatDetector.to_canonical(Y, target_format="csr")

    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    distances = np.zeros((n_samples_X, n_samples_Y))

    for i in range(n_samples_X):
        x_i = X[i].toarray().ravel()
        for j in range(n_samples_Y):
            y_j = Y[j].toarray().ravel()
            # Convert to binary
            x_bin = (x_i > 0).astype(int)
            y_bin = (y_j > 0).astype(int)
            # Jaccard: 1 - intersection / union
            intersection = np.sum(x_bin & y_bin)
            union = np.sum(x_bin | y_bin)
            if union > 0:
                distances[i, j] = 1 - (intersection / union)
            else:
                distances[i, j] = 0

    return distances


class SparseKNNGraph:
    """Build k-NN graphs efficiently for sparse data.

    Constructs approximate or exact k-nearest neighbor graphs using
    efficient algorithms suitable for sparse high-dimensional data.

    Parameters
    ----------
    n_neighbors : int, default=15
        Number of nearest neighbors.

    metric : str, default='euclidean'
        Distance metric ('euclidean', 'cosine', 'manhattan', 'jaccard').

    method : str, default='auto'
        Construction method:
        - 'exact': Brute force (slow but accurate)
        - 'auto': Choose based on data characteristics

    Attributes
    ----------
    knn_indices_ : ndarray
        Indices of k nearest neighbors for each point.

    knn_distances_ : ndarray
        Distances to k nearest neighbors.

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> from squeeze.sparse_ops import SparseKNNGraph
    >>>
    >>> X = sp.random(100, 1000, density=0.01, format='csr')
    >>> knn = SparseKNNGraph(n_neighbors=15)
    >>> knn.fit(X)
    >>> indices = knn.knn_indices_
    >>> distances = knn.knn_distances_
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        method: str = "auto",
    ):
        """Initialize sparse k-NN graph builder."""
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.method = method
        self.knn_indices_ = None
        self.knn_distances_ = None

    def fit(self, X: np.ndarray | sp.spmatrix) -> SparseKNNGraph:
        """Build k-NN graph for data X.

        Parameters
        ----------
        X : ndarray or sparse matrix, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        self : SparseKNNGraph
        """
        X = SparseFormatDetector.to_canonical(X, target_format="csr")
        n_samples = X.shape[0]

        # For now, use brute force
        # In future: add HNSW, LSH approximations
        if self.metric == "euclidean":
            distances = sparse_euclidean(X, X)
        elif self.metric == "cosine":
            distances = sparse_cosine(X, X)
        elif self.metric == "manhattan":
            distances = sparse_manhattan(X, X)
        elif self.metric == "jaccard":
            distances = sparse_jaccard(X, X)
        else:
            msg = f"Unknown metric: {self.metric}"
            raise ValueError(msg)

        # Find k nearest neighbors (excluding self at distance 0)
        self.knn_indices_ = np.zeros((n_samples, self.n_neighbors), dtype=np.int32)
        self.knn_distances_ = np.zeros((n_samples, self.n_neighbors))

        for i in range(n_samples):
            # Get sorted indices of distances
            sorted_indices = np.argsort(distances[i])
            # Exclude self (index i which has distance 0)
            sorted_indices = sorted_indices[sorted_indices != i]
            # Take first k
            self.knn_indices_[i] = sorted_indices[: self.n_neighbors]
            self.knn_distances_[i] = distances[i, self.knn_indices_[i]]

        return self

    def fit_predict(self, X: np.ndarray | sp.spmatrix):
        """Fit and return k-NN graph.

        Parameters
        ----------
        X : ndarray or sparse matrix
            Input data.

        Returns
        -------
        knn_indices : ndarray, shape (n_samples, n_neighbors)
            Indices of k nearest neighbors.
        """
        self.fit(X)
        return self.knn_indices_


class SparseUMAP:
    """Wrapper for UMAP that handles sparse input efficiently.

    This is a placeholder class showing how UMAP would be integrated
    with sparse data support. Full implementation would optimize the
    k-NN computation step.

    Parameters
    ----------
    n_components : int, default=2
        Dimensionality of embedding.

    n_neighbors : int, default=15
        Size of local neighborhood.

    metric : str, default='euclidean'
        Distance metric.

    **kwargs
        Additional arguments passed to UMAP.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        metric: str = "euclidean",
        **kwargs,
    ):
        """Initialize SparseUMAP."""
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.kwargs = kwargs
        self.umap_ = None

    def fit_transform(self, X: np.ndarray | sp.spmatrix) -> np.ndarray:
        """Fit UMAP on sparse data and return embedding.

        Parameters
        ----------
        X : ndarray or sparse matrix, shape (n_samples, n_features)
            Input data.

        Returns
        -------
        embedding : ndarray, shape (n_samples, n_components)
            Low-dimensional embedding.
        """
        from squeeze import UMAP

        # For sparse data, we could precompute k-NN using efficient sparse methods
        # For now, convert and use standard UMAP
        if sp.issparse(X):
            # Check sparsity level
            sparsity = SparseFormatDetector.get_sparsity(X)
            if sparsity > 0.9:
                # Very sparse - in future, use optimized sparse k-NN
                pass

        self.umap_ = UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            **self.kwargs,
        )

        return self.umap_.fit_transform(X)

    def fit(self, X: np.ndarray | sp.spmatrix) -> SparseUMAP:
        """Fit UMAP on sparse data.

        Parameters
        ----------
        X : ndarray or sparse matrix
            Input data.

        Returns
        -------
        self : SparseUMAP
        """
        from squeeze import UMAP

        self.umap_ = UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            **self.kwargs,
        )
        self.umap_.fit(X)
        return self

    def transform(self, X: np.ndarray | sp.spmatrix) -> np.ndarray:
        """Transform sparse data using fitted UMAP.

        Parameters
        ----------
        X : ndarray or sparse matrix
            Input data.

        Returns
        -------
        embedding : ndarray
            Low-dimensional embedding.
        """
        if self.umap_ is None:
            msg = "Model must be fit before transform"
            raise ValueError(msg)
        return self.umap_.transform(X)
