"""UMAP Python wrapper.

Squeeze aims to provide Rust implementations for all algorithms.

`UMAP` is therefore implemented as a thin Python wrapper around the Rust
extension type `UMAPRust` exposed from `squeeze._hnsw_backend`.

The legacy pure-Python implementation previously living in `squeeze.umap_`
has been removed from the public API.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin
except Exception:  # pragma: no cover
    # Avoid import-time hard failures if sklearn isn't available in some envs.
    BaseEstimator = object  # type: ignore[assignment]
    ClassNamePrefixFeaturesOutMixin = object  # type: ignore[assignment]


try:
    from ._hnsw_backend import UMAPRust as _UMAPRust
except Exception:  # pragma: no cover
    _UMAPRust = None


class UMAP(BaseEstimator, ClassNamePrefixFeaturesOutMixin):
    """Uniform Manifold Approximation and Projection (Rust backend)."""

    _BACKEND = "rust"

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        n_epochs: int = 0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        metric: str = "euclidean",
        dist_p: float = 2.0,
        initial_alpha: float = 1.0,
        negative_sample_rate: float = 5.0,
        gamma: float = 1.0,
        random_state: Optional[int] = None,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        **_kwargs: Any,
    ) -> None:
        if _UMAPRust is None:
            raise ImportError(
                "Rust backend not available; build the extension to use UMAP"
            )

        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.n_epochs = n_epochs
        self.min_dist = min_dist
        self.spread = spread
        self.metric = metric
        self.dist_p = dist_p
        self.initial_alpha = initial_alpha
        self.negative_sample_rate = negative_sample_rate
        self.gamma = gamma
        self.random_state = random_state
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self._model = _UMAPRust(
            n_components=n_components,
            n_neighbors=n_neighbors,
            n_epochs=n_epochs,
            min_dist=min_dist,
            spread=spread,
            metric=metric,
            dist_p=dist_p,
            initial_alpha=initial_alpha,
            negative_sample_rate=negative_sample_rate,
            gamma=gamma,
            random_state=random_state,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )

    def fit(self, X: np.ndarray, y: Any = None, **_kwargs: Any) -> "UMAP":
        _ = y
        self.fit_transform(X)
        return self

    def fit_transform(self, X: np.ndarray, y: Any = None, **_kwargs: Any) -> np.ndarray:
        _ = y
        X_arr = np.asarray(X, dtype=np.float64, order="C")
        embedding = self._model.fit_transform(X_arr)
        # Store as attribute expected by sklearn-style callers.
        self.embedding_ = np.asarray(embedding)
        return self.embedding_

    def transform(self, X: np.ndarray, **_kwargs: Any) -> np.ndarray:
        raise NotImplementedError(
            "UMAP.transform is not yet implemented in the Rust backend"
        )

    def inverse_transform(self, X: np.ndarray, **_kwargs: Any) -> np.ndarray:
        raise NotImplementedError(
            "UMAP.inverse_transform is not yet implemented in the Rust backend"
        )

    def update(self, X: np.ndarray, **_kwargs: Any) -> None:
        raise NotImplementedError(
            "UMAP.update is not yet implemented in the Rust backend"
        )

    def get_feature_names_out(self, input_features: Optional[list[str]] = None):
        # Match scikit-learn transformer conventions: names are independent of input names.
        _ = input_features
        return np.asarray([f"umap{i}" for i in range(self.n_components)], dtype=object)
