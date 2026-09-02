"""Exact brute-force k-NN backend, same call shape as :class:`squeeze.hnsw_wrapper.HnswIndexWrapper`.

Use it when the neighbour count is large relative to what a graph index can recover, which
is the normal case for UMAP over deep-learning embeddings (k of 100-200 in 1000+ dims). On a
149k x 1280 production embedding set the HNSW backend returned 65% of the true 200-NN; this
backend returns all of them and, because the work is a blocked matrix product, is faster on
CPU than either HNSW or PyNNDescent up to a few hundred thousand rows.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from ._hnsw_backend import ExactKnnIndex as _ExactKnnIndex
except ImportError:  # pragma: no cover - Rust extension missing
    _ExactKnnIndex = None

SUPPORTED_METRICS = {"cosine", "correlation", "euclidean", "l2"}

# Above this many rows the O(n^2 d) product stops paying for its exactness on a CPU and the
# approximate backends take over in "auto" mode. Measured crossover on 16 cores at d=1280,
# k=200: exact ~100 s at 150k rows vs 200 s for PyNNDescent; quadratic growth overtakes at
# roughly 300k.
EXACT_AUTO_MAX_SAMPLES = 250_000


class ExactKnnIndexWrapper:
    """Exact k-NN with the wrapper interface UMAP expects (``neighbor_graph``, ``query``)."""

    def __init__(
        self,
        data: NDArray,
        n_neighbors: int = 30,
        metric: str = "euclidean",
        metric_kwds: dict | None = None,
        block: int = 128,
        **_ignored,
    ) -> None:
        if _ExactKnnIndex is None:
            msg = "Exact k-NN backend not available: the squeeze Rust extension is not installed."
            raise ImportError(msg)
        if metric not in SUPPORTED_METRICS:
            msg = f"ExactKnnIndexWrapper supports {sorted(SUPPORTED_METRICS)}, got {metric!r}"
            raise ValueError(msg)
        if metric_kwds:
            msg = "ExactKnnIndexWrapper does not take metric_kwds"
            raise ValueError(msg)
        data = np.ascontiguousarray(data, dtype=np.float32)
        self._data = data
        self._metric = metric
        self._n_neighbors = n_neighbors
        self._index = _ExactKnnIndex(data, n_neighbors, metric, block)
        self._neighbor_graph_cache: tuple[NDArray, NDArray] | None = None

    @property
    def neighbor_graph(self) -> tuple[NDArray, NDArray]:
        if self._neighbor_graph_cache is None:
            self._neighbor_graph_cache = self._index.neighbor_graph()
        return self._neighbor_graph_cache

    def query(
        self,
        query_data: NDArray,
        k: int,
        epsilon: float = 0.1,
        filter_mask: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        del epsilon  # exact search has no accuracy knob
        if filter_mask is not None:
            msg = "filter_mask is not supported by the exact backend"
            raise ValueError(msg)
        query_data = np.ascontiguousarray(query_data, dtype=np.float32)
        return self._index.query(query_data, k)

    def prepare(self) -> None:
        """No-op; kept for interface parity."""

    @property
    def _angular_trees(self) -> bool:
        return self._metric in ("cosine", "correlation")

    @property
    def _raw_data(self) -> NDArray:
        return self._data

    def __repr__(self) -> str:
        return (
            f"ExactKnnIndexWrapper(n_samples={self._data.shape[0]}, n_features={self._data.shape[1]}, "
            f"n_neighbors={self._n_neighbors}, metric='{self._metric}')"
        )
