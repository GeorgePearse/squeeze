"""Validation and trustworthiness calculations for UMAP embeddings."""

from typing import Any, Callable

import numba
import numpy as np
from sklearn.neighbors import KDTree

from umap.distances import named_distances


@numba.njit()
def trustworthiness_vector_bulk(
    indices_source: np.ndarray,
    indices_embedded: np.ndarray,
    max_k: int,
) -> np.ndarray:  # pragma: no cover
    """Compute trustworthiness vector in bulk for precomputed indices.

    Parameters
    ----------
    indices_source : np.ndarray
        Indices of neighbors in the source space.
    indices_embedded : np.ndarray
        Indices of neighbors in the embedded space.
    max_k : int
        Maximum number of neighbors to consider.

    Returns
    -------
    np.ndarray
        Trustworthiness values for each k from 0 to max_k.

    """
    n_samples = indices_embedded.shape[0]
    trustworthiness = np.zeros(max_k + 1, dtype=np.float64)

    for i in range(n_samples):
        for j in range(max_k):
            rank = 0
            while indices_source[i, rank] != indices_embedded[i, j]:
                rank += 1

            for k in range(j + 1, max_k + 1):
                if rank > k:
                    trustworthiness[k] += rank - k

    for k in range(1, max_k + 1):
        trustworthiness[k] = 1.0 - trustworthiness[k] * (
            2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0))
        )

    return trustworthiness


def make_trustworthiness_calculator(
    metric: Callable[[Any, Any], float],
) -> Callable[[np.ndarray, np.ndarray, int], np.ndarray]:  # pragma: no cover
    """Create a trustworthiness calculator function for a given metric.

    Parameters
    ----------
    metric : callable
        Distance metric function.

    Returns
    -------
    callable
        Function to compute trustworthiness vector with low memory usage.

    """

    @numba.njit(parallel=True)
    def trustworthiness_vector_lowmem(
        source: np.ndarray,
        indices_embedded: np.ndarray,
        max_k: int,
    ) -> np.ndarray:
        n_samples = indices_embedded.shape[0]
        trustworthiness = np.zeros(max_k + 1, dtype=np.float64)
        dist_vector = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            for j in numba.prange(n_samples):
                dist_vector[j] = metric(source[i], source[j])

            indices_source = np.argsort(dist_vector)

            for j in range(max_k):
                rank = 0
                while indices_source[rank] != indices_embedded[i, j]:
                    rank += 1

                for k in range(j + 1, max_k + 1):
                    if rank > k:
                        trustworthiness[k] += rank - k

        for k in range(1, max_k + 1):
            trustworthiness[k] = 1.0 - trustworthiness[k] * (
                2.0 / (n_samples * k * (2.0 * n_samples - 3.0 * k - 1.0))
            )

        trustworthiness[0] = 1.0

        return trustworthiness

    return trustworthiness_vector_lowmem


def trustworthiness_vector(
    source: np.ndarray,
    embedding: np.ndarray,
    max_k: int,
    metric: str = "euclidean",
) -> np.ndarray:  # pragma: no cover
    """Compute trustworthiness vector for an embedding.

    Parameters
    ----------
    source : np.ndarray
        Original high-dimensional data.
    embedding : np.ndarray
        Low-dimensional embedding.
    max_k : int
        Maximum number of neighbors to consider.
    metric : str, optional
        Distance metric to use, by default "euclidean".

    Returns
    -------
    np.ndarray
        Trustworthiness values for each k from 0 to max_k.

    """
    tree = KDTree(embedding, metric=metric)
    indices_embedded = tree.query(embedding, k=max_k, return_distance=False)
    # Drop the actual point itself
    indices_embedded = indices_embedded[:, 1:]

    dist = named_distances[metric]

    vec_calculator = make_trustworthiness_calculator(dist)

    return vec_calculator(source, indices_embedded, max_k)
