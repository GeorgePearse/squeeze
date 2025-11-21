"""Tests for HNSW filtered queries (mask support and validation)."""

from __future__ import annotations

import numpy as np
import pytest

from squeeze import hnsw_wrapper
from squeeze.hnsw_wrapper import HnswIndexWrapper


@pytest.fixture
def sample_index() -> HnswIndexWrapper:
    """Create a sample HnswIndexWrapper for testing."""
    if hnsw_wrapper._HnswIndex is None:  # type: ignore[attr-defined]  # noqa: SLF001
        pytest.skip("Rust HNSW backend is not available in this environment")

    data = np.random.RandomState(42).rand(50, 4).astype(np.float32)
    return HnswIndexWrapper(data, n_neighbors=10)


def test_filtered_query_masks_out_indices(sample_index: HnswIndexWrapper) -> None:
    """Test that filtered queries correctly mask out indices."""
    filter_mask = np.ones(sample_index._data.shape[0], dtype=bool)  # noqa: SLF001
    filter_mask[: sample_index._data.shape[0] // 2] = False  # noqa: SLF001

    indices, _ = sample_index.query(
        sample_index._data[:3],  # noqa: SLF001
        sample_index._n_neighbors,  # noqa: SLF001
        filter_mask=filter_mask,
    )

    valid_indices = indices[indices != -1]
    assert valid_indices.size > 0
    assert np.all(valid_indices >= sample_index._data.shape[0] // 2)  # noqa: SLF001

def test_filtered_query_invalid_mask_type(sample_index: HnswIndexWrapper) -> None:
    """Test that invalid mask type raises an error."""
    with pytest.raises(ValueError, match="filter_mask must be a boolean array"):
        sample_index.query(
            sample_index._data[:2],  # noqa: SLF001
            3,
            filter_mask=np.arange(sample_index._data.shape[0]),  # noqa: SLF001
        )


def test_filtered_query_invalid_mask_shape(sample_index: HnswIndexWrapper) -> None:
    """Test that invalid mask shape raises an error."""
    bad_mask = np.ones((sample_index._data.shape[0], 1), dtype=bool)  # noqa: SLF001
    with pytest.raises(ValueError, match="filter_mask must be 1-dimensional"):
        sample_index.query(sample_index._data[:2], 3, filter_mask=bad_mask)  # noqa: SLF001


def test_filtered_query_invalid_mask_length(sample_index: HnswIndexWrapper) -> None:
    """Test that invalid mask length raises an error."""
    bad_mask = np.ones(sample_index._data.shape[0] + 1, dtype=bool)  # noqa: SLF001
    with pytest.raises(
        ValueError,
        match="filter_mask length must match the number of indexed samples",
    ):
        sample_index.query(sample_index._data[:2], 3, filter_mask=bad_mask)  # noqa: SLF001


def test_regular_query_unchanged(sample_index: HnswIndexWrapper) -> None:
    """Test that regular queries work unchanged."""
    queries = sample_index._data[:3]  # noqa: SLF001
    indices, distances = sample_index.query(
        queries,
        sample_index._n_neighbors,  # noqa: SLF001
    )

    assert indices.shape == (queries.shape[0], sample_index._n_neighbors)  # noqa: SLF001
    assert distances.shape == (queries.shape[0], sample_index._n_neighbors)  # noqa: SLF001
