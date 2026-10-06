"""Sanity checks for benchmark metrics and their shared pair alignment."""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import ConstantInputWarning
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances

from scripts.benchmark_neighbors import score


def reference_neighbors(data: NDArray[np.float64]) -> NDArray[np.int64]:
    """Build the original-space neighbor reference with no self edges."""
    distances = pairwise_distances(data)
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1, kind="stable")[:, :15]


def test_identical_geometry_and_shuffled_rows() -> None:
    """Metrics are perfect for identity and penalize wrong sample alignment."""
    data = load_digits().data[:80].astype(np.float64)
    rng = np.random.default_rng(42)
    # Break integer-distance ties: sklearn may choose different tied neighbors.
    data += rng.normal(scale=1e-4, size=data.shape)
    reference = reference_neighbors(data)
    pairs = rng.integers(0, len(data), size=(2000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    identical = score(data, data, reference, pairs)
    assert all(np.isclose(value, 1.0) for value in identical.values())
    shuffled = score(data, data[rng.permutation(len(data))], reference, pairs)
    assert all(shuffled[key] < identical[key] for key in identical)


def test_constant_embedding_has_explicit_undefined_correlation() -> None:
    """Undefined correlations can be serialized as null instead of JSON NaN."""
    data = load_digits().data[:80].astype(np.float64)
    pairs = np.column_stack((np.arange(79), np.arange(1, 80)))
    with pytest.warns(ConstantInputWarning):
        result = score(data, np.zeros((80, 2)), reference_neighbors(data), pairs)
    assert result["distance_spearman"] is None
