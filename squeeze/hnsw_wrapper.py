"""Python wrapper for the Rust-based HNSW nearest neighbor index.

This module provides a PyNNDescent-compatible API to the Rust-based
HNSW (or brute-force) nearest neighbor search backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    from _hnsw_backend import HnswIndex as _HnswIndex  # type: ignore[import-not-found]
    from _hnsw_backend import (
        SparseHnswIndex as _SparseHnswIndex,  # type: ignore[import-not-found]
    )
except ImportError:
    try:
        from squeeze._hnsw_backend import (
            HnswIndex as _HnswIndex,  # type: ignore[import-not-found]
        )
        from squeeze._hnsw_backend import (
            SparseHnswIndex as _SparseHnswIndex,  # type: ignore[import-not-found]
        )
    except ImportError:
        _HnswIndex = None  # type: ignore[assignment]
        _SparseHnswIndex = None  # type: ignore[assignment]


class HnswIndexWrapper:
    """Drop-in replacement for PyNNDescent.NNDescent using Rust backend.

    This class wraps the Rust-based nearest neighbor index and exposes
    it through a PyNNDescent-compatible interface, including parameter
    mapping and additional properties for backward compatibility.

    Parameters
    ----------
    data : ndarray
        The training data to index, shape (n_samples, n_features)
    n_neighbors : int, default=30
        Number of neighbors to find
    metric : str, default='euclidean'
        Distance metric to use
    metric_kwds : dict, optional
        Additional keyword arguments for the metric
    random_state : int or RandomState, optional
        Random state for reproducibility
    n_trees : int, optional
        Number of trees to use in random projection forests.
        Maps to HNSW M parameter.
    n_iters : int, optional
        Number of iterations for tree descent.
        Maps to HNSW ef_construction parameter.
    max_candidates : int, default=60
        Max candidates per iteration
    low_memory : bool, default=True
        Whether to use low memory mode
    n_jobs : int, default=-1
        Number of parallel jobs
    verbose : bool, default=False
        Whether to log progress
    compressed : bool, default=False
        Whether to return compressed neighbor graph

    """

    def __init__(
        self,
        data: NDArray,
        n_neighbors: int = 30,
        metric: str = "euclidean",
        metric_kwds: dict | None = None,
        random_state: int | None = None,
        n_trees: int | None = None,
        n_iters: int | None = None,
        max_candidates: int = 60,
        low_memory: bool = True,
        n_jobs: int = -1,
        verbose: bool = False,
        compressed: bool = False,
        prune_strategy: str = "simple",
        prune_alpha: float = 1.2,
    ) -> None:
        """Initialize the HNSW index wrapper."""
        if _HnswIndex is None:
            msg = (
                "HNSW backend not available. "
                "Please install umap with: pip install --upgrade umap"
            )
            raise ImportError(
                msg,
            )

        self._n_neighbors = n_neighbors
        self._metric = metric
        self._metric_kwds = metric_kwds or {}
        self._random_state = random_state
        self._low_memory = low_memory
        self._n_jobs = n_jobs
        self._verbose = verbose
        self._compressed = compressed
        self._prune_strategy = prune_strategy
        self._prune_alpha = prune_alpha

        # Compute HNSW parameters from PyNNDescent-style parameters
        self._m = self._compute_m(n_trees, data.shape[0])
        self._ef_construction = self._compute_ef_construction(
            n_iters,
            max_candidates,
            data.shape[0],
        )

        # Extract 'p' parameter for Minkowski distance
        p_val = metric_kwds.get("p", 2.0) if metric_kwds else 2.0

        # Convert random_state to u64 seed
        seed = None
        if random_state is not None:
            if isinstance(random_state, int):
                seed = abs(random_state) % (2**64)  # Ensure it fits in u64

        import scipy.sparse

        if scipy.sparse.isspmatrix_csr(data):
            self._is_sparse = True
            self._data = data
            data.sort_indices()

            if _SparseHnswIndex is None:
                msg = "Sparse HNSW backend not available."
                raise ImportError(msg)

            self._index = _SparseHnswIndex(
                data.data.astype(np.float32),
                data.indices.astype(np.int32),
                data.indptr.astype(np.int32),
                data.shape[0],
                data.shape[1],
                n_neighbors,
                metric,
                self._m,
                self._ef_construction,
                p_val,
                seed,
                prune_strategy,
                prune_alpha,
            )
        else:
            self._is_sparse = False
            # Ensure data is float32 (required by Rust backend)
            data = np.asarray(data, dtype=np.float32)
            self._data = data

            # Create the Rust-based index
            # Note: PyO3 exposes positional arguments, not keyword arguments
            self._index = _HnswIndex(
                data,
                n_neighbors,
                metric,
                self._m,
                self._ef_construction,
                p_val,
                seed,
                prune_strategy,
                prune_alpha,
            )

        # Store state for API compatibility
        self._neighbor_graph_cache: tuple[NDArray, NDArray] | None = None

    @staticmethod
    def _compute_m(n_trees: int | None, n_samples: int) -> int:
        """Compute HNSW M parameter from PyNNDescent n_trees.

        The M parameter controls the maximum number of bidirectional
        connections per node in the HNSW graph. Higher M values produce
        denser graphs with potentially better recall but more memory usage.
        """
        if n_trees is None:
            # UMAP default: min(64, 5 + round((n_samples ** 0.5) / 20.0))
            n_trees = min(64, 5 + round((n_samples**0.5) / 20.0))

        # Map PyNNDescent parameter to HNSW M
        # Typically 8-64, default 16 for balanced performance
        return max(8, min(64, n_trees))

    @staticmethod
    def _compute_ef_construction(
        n_iters: int | None,
        max_candidates: int,
        n_samples: int,
    ) -> int:
        """Compute HNSW ef_construction parameter.

        ef_construction controls the size of the dynamic candidate list
        during construction. Larger values produce more accurate indices
        but require longer construction time.
        """
        if n_iters is None:
            # UMAP default: max(5, round(log2(n_samples)))
            n_iters = max(5, round(np.log2(n_samples)))

        # Map to HNSW ef_construction
        # Typically 100-800, higher = better quality but slower
        return max(200, min(800, n_iters * max_candidates // 2))

    @property
    def neighbor_graph(
        self,
    ) -> tuple[NDArray, NDArray] | None:
        """Get the k-nearest neighbor graph.

        Returns
        -------
        indices : ndarray, shape (n_samples, n_neighbors)
            The indices of the k nearest neighbors for each sample
        distances : ndarray, shape (n_samples, n_neighbors)
            The distances to the k nearest neighbors for each sample

        Returns None if the index is compressed.

        """
        if self._compressed:
            return None

        if self._neighbor_graph_cache is None:
            # neighbor_graph is a method in the Rust backend
            indices, distances = self._index.neighbor_graph()
            self._neighbor_graph_cache = (indices, distances)

        return self._neighbor_graph_cache

    def query(
        self,
        query_data: NDArray,
        k: int,
        epsilon: float = 0.1,
        filter_mask: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Query the index for k nearest neighbors.

        Parameters
        ----------
        query_data : ndarray, shape (n_queries, n_features)
            The query data
        k : int
            Number of neighbors to return
        epsilon : float, default=0.1
            Accuracy/speed tradeoff parameter. Higher values search more
            candidates and return more accurate results.
        filter_mask : ndarray of bool, optional
            Boolean mask with length equal to the indexed dataset. When provided,
            candidates with False values are ignored during search.

        Returns
        -------
        indices : ndarray, shape (n_queries, k)
            The indices of the k nearest neighbors
        distances : ndarray, shape (n_queries, k)
            The distances to the k nearest neighbors

        """
        mask_arg = None
        if filter_mask is not None:
            mask = np.asarray(filter_mask)
            if mask.dtype != np.bool_:
                msg = "filter_mask must be a boolean array"
                raise ValueError(msg)
            if mask.ndim != 1:
                msg = "filter_mask must be 1-dimensional"
                raise ValueError(msg)
            if mask.shape[0] != self._data.shape[0]:
                msg = "filter_mask length must match the number of indexed samples"
                raise ValueError(
                    msg,
                )
            mask_arg = np.ascontiguousarray(mask, dtype=bool)

        # Map epsilon to ef parameter
        # epsilon controls search relaxation in PyNNDescent
        # ef controls candidate list size in HNSW
        ef = self._epsilon_to_ef(epsilon, k)

        if self._is_sparse:
            import scipy.sparse

            if not scipy.sparse.isspmatrix_csr(query_data):
                query_data = scipy.sparse.csr_matrix(query_data)

            query_data.sort_indices()

            # Call Rust sparse query method
            # Note: filter_mask is not yet supported in sparse backend query signature
            # We should update Rust side to accept it or ignore it for now.
            # The Rust SparseHnswIndex.query signature is (query_data, query_indices, query_indptr, k, ef)
            return self._index.query(
                query_data.data.astype(np.float32),
                query_data.indices.astype(np.int32),
                query_data.indptr.astype(np.int32),
                k,
                ef,
            )
        else:
            # Ensure data is float32
            query_data = np.asarray(query_data, dtype=np.float32)

            # Call Rust query method with positional arguments
            return self._index.query(query_data, k, ef, mask_arg)

    @staticmethod
    def _epsilon_to_ef(epsilon: float, k: int) -> int:
        """Map PyNNDescent epsilon to HNSW ef parameter.

        epsilon semantics: search (1 + epsilon) x optimal distance
        ef semantics: size of candidate list (must be >= k)
        """
        # Empirical mapping: higher epsilon → larger ef
        # ef should be at least k
        ef = max(k, int(k * (1.0 + epsilon * 30)))
        # Cap ef to avoid excessive computation
        return min(ef, 500)

    def prepare(self) -> None:
        """Prepare the index for querying.

        This is a no-op for the Rust backend but required for
        API compatibility with PyNNDescent.
        """
        self._index.prepare()

    def update(self, X: NDArray) -> None:
        """Update the index with new data.

        Parameters
        ----------
        X : ndarray, shape (n_new, n_features)
            New data to add to the index

        """
        X = np.asarray(X, dtype=np.float32)
        self._index.update(X)
        self._data = np.vstack([self._data, X])
        self._neighbor_graph_cache = None  # Invalidate cache

    @property
    def _angular_trees(self) -> bool:
        """Get whether angular (cosine/correlation) trees are used.

        This property is used by UMAP to determine epsilon values
        for queries.
        """
        return self._index._angular_trees  # noqa: SLF001

    @property
    def _raw_data(self) -> NDArray:
        """Get the raw data used to build the index."""
        return self._data

    def __repr__(self) -> str:
        """Return string representation of the index."""
        type_str = "sparse" if getattr(self, "_is_sparse", False) else "dense"
        return (
            f"HnswIndexWrapper(n_samples={self._index.n_samples}, "
            f"n_features={self._index.n_features}, "
            f"metric='{self._metric}', "
            f"type='{type_str}')"
        )
