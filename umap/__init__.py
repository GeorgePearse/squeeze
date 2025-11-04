"""UMAP: Uniform Manifold Approximation and Projection.

This package provides a Python implementation of UMAP, a dimension reduction
technique that can be used for visualization similarly to t-SNE, but also for
general non-linear dimension reduction.
"""

from warnings import catch_warnings, simplefilter, warn

from .umap_ import UMAP

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

    # Add a dummy class to raise an error
    class ParametricUMAP:
        """Dummy ParametricUMAP class for when Tensorflow is not installed.

        This class is created when Tensorflow is not available to provide
        a helpful error message to users attempting to use ParametricUMAP
        functionality.
        """

        def __init__(self, **_kwds: object) -> None:
            """Initialize the dummy ParametricUMAP class.

            Parameters
            ----------
            **_kwds : object
                Keyword arguments (ignored).

            Raises
            ------
            ImportError
                Always raised to indicate Tensorflow is not installed.

            """
            warn(
                "The umap.parametric_umap package requires Tensorflow > 2.0 "
                "to be installed. You can install Tensorflow at "
                "https://www.tensorflow.org/install or you can install "
                "the CPU version of Tensorflow using "
                "pip install umap[parametric_umap]",
                stacklevel=2,
            )
            msg = "umap.parametric_umap requires Tensorflow >= 2.0"
            raise ImportError(
                msg,
            ) from None


from importlib.metadata import PackageNotFoundError, version

from .aligned_umap import AlignedUMAP

try:
    __version__ = version("umap")
except PackageNotFoundError:
    __version__ = "0.5-dev"

__all__ = ["UMAP", "AlignedUMAP", "ParametricUMAP"]
