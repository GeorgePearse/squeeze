"""Strategy registry for dimensionality reduction algorithms.

This module provides a unified interface to access all DR algorithms
by name, iterate over them, and get their default configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

# Import all algorithms
from .umap_ import UMAP

try:
    from ._hnsw_backend import (
        PCA,
        TSNE,
        MDS,
        Isomap,
        LLE,
        PHATE,
        TriMap,
        PaCMAP,
    )

    RUST_BACKEND_AVAILABLE = True
except ImportError:
    RUST_BACKEND_AVAILABLE = False
    PCA = None
    TSNE = None
    MDS = None
    Isomap = None
    LLE = None
    PHATE = None
    TriMap = None
    PaCMAP = None


@dataclass
class Strategy:
    """A dimensionality reduction strategy with its configuration."""

    name: str
    algorithm_class: type
    default_params: dict = field(default_factory=dict)
    description: str = ""
    category: str = "other"  # linear, nonlinear, graph-based, etc.

    def create(self, **kwargs) -> Any:
        """Create an instance of the algorithm with given parameters.

        Parameters override defaults.
        """
        params = {**self.default_params, **kwargs}
        return self.algorithm_class(**params)

    def __repr__(self) -> str:
        return f"Strategy(name='{self.name}', category='{self.category}')"


class StrategyRegistry:
    """Registry of all available dimensionality reduction strategies."""

    def __init__(self):
        self._strategies: dict[str, Strategy] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register all default strategies."""

        # UMAP (always available)
        self.register(
            Strategy(
                name="umap",
                algorithm_class=UMAP,
                default_params={"n_components": 2, "n_neighbors": 15, "min_dist": 0.1},
                description="Uniform Manifold Approximation and Projection",
                category="nonlinear",
            )
        )

        if not RUST_BACKEND_AVAILABLE:
            return

        # PCA
        self.register(
            Strategy(
                name="pca",
                algorithm_class=PCA,
                default_params={"n_components": 2},
                description="Principal Component Analysis",
                category="linear",
            )
        )

        # t-SNE
        self.register(
            Strategy(
                name="tsne",
                algorithm_class=TSNE,
                default_params={
                    "n_components": 2,
                    "perplexity": 30.0,
                    "learning_rate": 200.0,
                    "n_iter": 1000,
                },
                description="t-Distributed Stochastic Neighbor Embedding",
                category="nonlinear",
            )
        )

        # MDS
        self.register(
            Strategy(
                name="mds",
                algorithm_class=MDS,
                default_params={"n_components": 2, "metric": True, "n_iter": 300},
                description="Multidimensional Scaling",
                category="linear",
            )
        )

        # Isomap
        self.register(
            Strategy(
                name="isomap",
                algorithm_class=Isomap,
                default_params={"n_components": 2, "n_neighbors": 10},
                description="Isometric Mapping (geodesic distances)",
                category="graph-based",
            )
        )

        # LLE
        self.register(
            Strategy(
                name="lle",
                algorithm_class=LLE,
                default_params={"n_components": 2, "n_neighbors": 12, "reg": 1e-3},
                description="Locally Linear Embedding",
                category="graph-based",
            )
        )

        # PHATE
        self.register(
            Strategy(
                name="phate",
                algorithm_class=PHATE,
                default_params={"n_components": 2, "k": 15, "t": 5, "decay": 2.0},
                description="Potential of Heat-diffusion for Affinity-based Trajectory Embedding",
                category="diffusion",
            )
        )

        # TriMap
        self.register(
            Strategy(
                name="trimap",
                algorithm_class=TriMap,
                default_params={
                    "n_components": 2,
                    "n_inliers": 12,
                    "n_outliers": 4,
                    "n_random": 3,
                    "n_iter": 800,
                    "learning_rate": 0.1,
                },
                description="Large-scale Dimensionality Reduction Using Triplets",
                category="nonlinear",
            )
        )

        # PaCMAP
        self.register(
            Strategy(
                name="pacmap",
                algorithm_class=PaCMAP,
                default_params={
                    "n_components": 2,
                    "n_neighbors": 10,
                    "mn_ratio": 0.5,
                    "fp_ratio": 2.0,
                    "n_iter": 450,
                },
                description="Pairwise Controlled Manifold Approximation",
                category="nonlinear",
            )
        )

    def register(self, strategy: Strategy) -> None:
        """Register a new strategy."""
        self._strategies[strategy.name.lower()] = strategy

    def get(self, name: str) -> Strategy:
        """Get a strategy by name."""
        name_lower = name.lower()
        if name_lower not in self._strategies:
            available = ", ".join(self._strategies.keys())
            raise KeyError(f"Unknown strategy '{name}'. Available: {available}")
        return self._strategies[name_lower]

    def __getitem__(self, name: str) -> Strategy:
        """Get a strategy by name using bracket notation."""
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        """Check if a strategy exists."""
        return name.lower() in self._strategies

    def __iter__(self) -> Iterator[Strategy]:
        """Iterate over all strategies."""
        return iter(self._strategies.values())

    def __len__(self) -> int:
        """Return number of registered strategies."""
        return len(self._strategies)

    def names(self) -> list[str]:
        """Return list of all strategy names."""
        return list(self._strategies.keys())

    def by_category(self, category: str) -> list[Strategy]:
        """Get all strategies in a category."""
        return [s for s in self._strategies.values() if s.category == category]

    def categories(self) -> list[str]:
        """Return list of all categories."""
        return list(set(s.category for s in self._strategies.values()))

    def create(self, name: str, **kwargs) -> Any:
        """Create an algorithm instance by name."""
        return self.get(name).create(**kwargs)

    def summary(self) -> str:
        """Return a formatted summary of all strategies."""
        lines = ["Squeeze Dimensionality Reduction Strategies", "=" * 50, ""]

        for category in sorted(self.categories()):
            lines.append(f"{category.upper()}")
            lines.append("-" * len(category))
            for strategy in self.by_category(category):
                lines.append(f"  {strategy.name:<12} - {strategy.description}")
            lines.append("")

        return "\n".join(lines)


# Global registry instance
STRATEGIES = StrategyRegistry()


def get_strategy(name: str) -> Strategy:
    """Get a strategy by name from the global registry."""
    return STRATEGIES.get(name)


def list_strategies() -> list[str]:
    """List all available strategy names."""
    return STRATEGIES.names()


def create_reducer(name: str, **kwargs):
    """Create a dimensionality reducer by strategy name.

    Parameters
    ----------
    name : str
        Name of the strategy (e.g., 'umap', 'tsne', 'pca')
    **kwargs
        Parameters to override defaults

    Returns
    -------
    reducer
        An instance of the dimensionality reduction algorithm

    Examples
    --------
    >>> reducer = create_reducer('tsne', perplexity=50, n_iter=500)
    >>> embedding = reducer.fit_transform(X)
    """
    return STRATEGIES.create(name, **kwargs)
