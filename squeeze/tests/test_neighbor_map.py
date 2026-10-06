"""Contracts for the additive Rust embedding options."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray
from sklearn.datasets import load_digits
from sklearn.manifold import trustworthiness

from squeeze import NeighborMap, SpectralMap


@pytest.fixture
def digits() -> NDArray[np.float64]:
    """Return a small real-data slice for API tests."""
    return load_digits().data[:80].astype(np.float64)


@pytest.mark.parametrize("method", [NeighborMap, SpectralMap])
def test_reproducibility_layout_and_input_unchanged(
    method: type,
    digits: NDArray[np.float64],
) -> None:
    """Accept strided/Fortran arrays and reproduce fixed-seed results."""
    original = digits.copy()
    expected = method(n_neighbors=8, random_state=7).fit_transform(digits)
    assert expected.shape == (80, 2)
    assert np.isfinite(expected).all()
    assert_array_equal(
        expected,
        method(n_neighbors=8, random_state=7).fit_transform(digits),
    )
    assert_allclose(
        expected,
        method(n_neighbors=8, random_state=7).fit_transform(np.asfortranarray(digits)),
        atol=1e-10,
    )
    strided = np.repeat(digits, 2, axis=1)[:, ::2]
    assert_allclose(
        expected,
        method(n_neighbors=8, random_state=7).fit_transform(strided),
        atol=1e-10,
    )
    assert_array_equal(digits, original)


@pytest.mark.parametrize("method", [NeighborMap, SpectralMap])
@pytest.mark.parametrize(
    "data",
    [
        np.zeros((2, 4)),
        np.zeros((20, 0)),
        np.full((20, 4), np.nan),
        np.full((20, 4), np.inf),
    ],
)
def test_invalid_input(method: type, data: NDArray[np.float64]) -> None:
    """Reject malformed/nonfinite inputs without panicking."""
    with pytest.raises(ValueError, match=r".+"):
        method().fit_transform(data)


@pytest.mark.parametrize("method", [NeighborMap, SpectralMap])
def test_neighbor_limits(method: type, digits: NDArray[np.float64]) -> None:
    """Validate graph sizes at construction and fit time."""
    with pytest.raises(ValueError, match=r".+"):
        method(n_neighbors=0)
    with pytest.raises(ValueError, match=r".+"):
        method(n_neighbors=len(digits)).fit_transform(digits)
    assert np.isfinite(method(n_neighbors=len(digits) - 1).fit_transform(digits)).all()


@pytest.mark.parametrize("method", [NeighborMap, SpectralMap])
@pytest.mark.parametrize(
    "data",
    [np.zeros((20, 1)), np.ones((20, 4)), np.eye(20) * 1e300, np.eye(20) * 1e-300],
)
def test_degenerate_and_extreme_inputs(method: type, data: NDArray[np.float64]) -> None:
    """Duplicate, constant, single-feature and extreme inputs stay finite."""
    assert np.isfinite(method(n_neighbors=3).fit_transform(data)).all()


@pytest.mark.parametrize(
    "parameters",
    [
        {"n_epochs": 0},
        {"negative_samples": 0},
        {"learning_rate": 0},
        {"learning_rate": np.inf},
        {"learning_rate": np.nan},
        {"init": "unknown"},
    ],
)
def test_invalid_optimizer_parameters(parameters: dict) -> None:
    """Invalid optimization settings fail before computation."""
    with pytest.raises(ValueError, match=r".+"):
        NeighborMap(**parameters)


def test_spectral_iterations() -> None:
    """Zero block iterations are not a spectral embedding."""
    with pytest.raises(ValueError, match=r".+"):
        SpectralMap(n_iter=0)


@pytest.mark.parametrize("init", ["pca", "spectral"])
def test_digits_quality_smoke(init: str) -> None:
    """Both optimization paths retain meaningful Digits neighborhoods."""
    data = load_digits().data.astype(np.float64)
    embedding = NeighborMap(init=init, n_epochs=80, random_state=42).fit_transform(data)
    assert trustworthiness(data, embedding, n_neighbors=15) > 0.9  # noqa: PLR2004 - loose quality regression floor, not a speed assertion
