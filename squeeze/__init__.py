"""Squeeze: High-performance dimensionality reduction library.

This package provides Python implementations of various dimension reduction
techniques including UMAP, t-SNE, PCA, and more. All implementations are
optimized for CPU performance with SIMD vectorization and Rust backends.

Implemented algorithms:
- UMAP: Uniform Manifold Approximation and Projection
- PCA: Principal Component Analysis
- TSNE: t-Distributed Stochastic Neighbor Embedding
- MDS: Multidimensional Scaling
- Isomap: Isometric Mapping
- LLE: Locally Linear Embedding
- PHATE: Potential of Heat-diffusion for Affinity-based Trajectory Embedding
- TriMap: Large-scale Dimensionality Reduction Using Triplets
- PaCMAP: Pairwise Controlled Manifold Approximation
"""

from warnings import catch_warnings, simplefilter, warn

from .umap_ import UMAP

# Import Rust-based algorithms
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
except ImportError as e:
    warn(
        f"Rust backend not available: {e}. Some algorithms may not be available.",
        stacklevel=2,
        category=ImportWarning,
    )
    # Create dummy classes
    PCA = None
    TSNE = None
    MDS = None
    Isomap = None
    LLE = None
    PHATE = None
    TriMap = None
    PaCMAP = None

try:
    with catch_warnings():
        simplefilter("ignore")
        from .parametric_umap import ParametricUMAP
except ImportError:
    warn(
        "Tensorflow not installed; ParametricUMAP will be unavailable",
        stacklevel=2,
        category=ImportWarning,
    )

    class ParametricUMAP:
        """Dummy ParametricUMAP class for when Tensorflow is not installed."""

        def __init__(self, **_kwds: object) -> None:
            warn(
                "The squeeze.parametric_umap package requires Tensorflow > 2.0 "
                "to be installed.",
                stacklevel=2,
            )
            msg = "squeeze.parametric_umap requires Tensorflow >= 2.0"
            raise ImportError(msg) from None


from importlib.metadata import PackageNotFoundError, version

from .aligned_umap import AlignedUMAP
from .composition import AdaptiveDR, DRPipeline, EnsembleDR, ProgressiveDR
from .extensions import OutOfSampleDR, StreamingDR
from .strategies import (
    STRATEGIES,
    Strategy,
    StrategyRegistry,
    get_strategy,
    list_strategies,
    create_reducer,
)

try:
    __version__ = version("squeeze")
except PackageNotFoundError:
    __version__ = "0.1-dev"

__all__ = [
    # Core UMAP
    "UMAP",
    "AlignedUMAP",
    "ParametricUMAP",
    # Rust-based DR algorithms
    "PCA",
    "TSNE",
    "MDS",
    "Isomap",
    "LLE",
    "PHATE",
    "TriMap",
    "PaCMAP",
    # Composition utilities
    "AdaptiveDR",
    "DRPipeline",
    "EnsembleDR",
    "ProgressiveDR",
    # Extension utilities
    "OutOfSampleDR",
    "StreamingDR",
    # Strategy registry
    "STRATEGIES",
    "Strategy",
    "StrategyRegistry",
    "get_strategy",
    "list_strategies",
    "create_reducer",
]
