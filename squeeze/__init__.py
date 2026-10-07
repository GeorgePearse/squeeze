"""Squeeze: High-performance dimensionality reduction library.

This package provides Python implementations of various dimension reduction
techniques including UMAP, t-SNE, PCA, and more. The implementations are
SIMD-optimised Rust with a CPU reference path; when a GPU is present the heavy
kernels (pairwise distances, exact k-NN, embedding gradients) run on it through
wgpu (Vulkan/Metal/DX12) or MLX on Apple Silicon. See ``squeeze.devices()``.

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
        LLE,
        MDS,
        PCA,
        PHATE,
        TSNE,
        Isomap,
        PaCMAP,
        TriMap,
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
    from ._hnsw_backend import NeighborMap, SpectralMap
except ImportError:
    # Older compiled extensions can still provide the established algorithms.
    NeighborMap = None
    SpectralMap = None

try:
    from ._hnsw_backend import default_device, devices, resolve_device
except ImportError:

    def devices() -> str:
        """Report the compute devices (CPU only: the Rust backend is not built)."""
        return "squeeze compute devices\n  chosen: cpu (Rust backend not available)\n"

    def default_device() -> str:
        """Return the device ``device="auto"`` resolves to."""
        return "cpu"

    def resolve_device(device: "str | None" = None) -> str:  # noqa: ARG001
        """Return what a ``device=`` argument resolves to."""
        return "cpu"


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
            """Explain the missing optional dependency."""
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
from .evaluation import (
    DREvaluator,
    EvaluationReport,
    bootstrap_stability,
    classification_accuracy,
    clustering_quality,
    co_ranking_quality,
    continuity,
    global_structure_preservation,
    local_density_preservation,
    noise_robustness,
    parameter_sensitivity,
    quick_evaluate,
    reconstruction_error,
    spearman_distance_correlation,
    trustworthiness,
)
from .extensions import OutOfSampleDR, StreamingDR
from .strategies import (
    STRATEGIES,
    Strategy,
    StrategyRegistry,
    create_reducer,
    get_strategy,
    list_strategies,
)

try:
    __version__ = version("squeeze")
except PackageNotFoundError:
    __version__ = "0.1-dev"

__all__ = [  # noqa: RUF022 - grouped by API category
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
    "NeighborMap",
    "SpectralMap",
    # Compute devices
    "devices",
    "default_device",
    "resolve_device",
    # Composition utilities
    "AdaptiveDR",
    "DRPipeline",
    "EnsembleDR",
    "ProgressiveDR",
    # Extension utilities
    "OutOfSampleDR",
    "StreamingDR",
    # Evaluation metrics
    "DREvaluator",
    "EvaluationReport",
    "trustworthiness",
    "continuity",
    "co_ranking_quality",
    "spearman_distance_correlation",
    "global_structure_preservation",
    "local_density_preservation",
    "reconstruction_error",
    "clustering_quality",
    "classification_accuracy",
    "bootstrap_stability",
    "noise_robustness",
    "parameter_sensitivity",
    "quick_evaluate",
    # Strategy registry
    "STRATEGIES",
    "Strategy",
    "StrategyRegistry",
    "get_strategy",
    "list_strategies",
    "create_reducer",
]
