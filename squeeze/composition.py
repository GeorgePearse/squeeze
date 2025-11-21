"""Hybrid dimensionality reduction composition patterns.

This module provides APIs for composing multiple dimensionality reduction
techniques in various patterns (sequential pipelines, ensembles, progressive
refinement, adaptive routing).

Basic usage - Sequential Pipeline (2048 -> 100 -> 2):
    >>> from squeeze import UMAP
    >>> from squeeze.composition import DRPipeline
    >>> from sklearn.decomposition import PCA
    >>>
    >>> # Create a pipeline: PCA for coarse reduction, then UMAP for fine embedding
    >>> pipeline = DRPipeline(steps=[
    ...     ('pca', PCA(n_components=100)),
    ...     ('umap', UMAP(n_components=2))
    ... ])
    >>>
    >>> # Fit and transform data
    >>> X_embedded = pipeline.fit_transform(X)

Author: UMAP development team
License: BSD 3 clause
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_is_fitted

__all__ = ["AdaptiveDR", "DRPipeline", "EnsembleDR", "ProgressiveDR"]


class DRPipeline(BaseEstimator):
    """Chain multiple dimensionality reduction techniques in sequence.

    This is the simplest and most practical composition pattern. Data flows
    through each step sequentially: X -> step1 -> intermediate1 -> step2 -> ... -> final embedding.

    For example, to reduce from 2048 dimensions to 2 via intermediate 100-d space:
        pipeline = DRPipeline([
            ('pca', PCA(n_components=100)),
            ('umap', UMAP(n_components=2))
        ])
        X_2d = pipeline.fit_transform(X_2048d)

    Parameters
    ----------
    steps : list of (str, estimator) tuples
        List of (name, estimator) tuples where each estimator must implement
        fit() and transform() methods (following scikit-learn conventions).
        The output dimensionality of step i should match or be compatible
        with the input dimensionality expected by step i+1.

    Attributes
    ----------
    steps_ : list
        The fitted steps (set after calling fit).

    named_steps_ : dict
        Dict-like object with named access to steps.
        Access with: pipeline.named_steps_['step_name']

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.manifold import TSNE
    >>> from squeeze import UMAP
    >>> from squeeze.composition import DRPipeline
    >>>
    >>> # Generate synthetic data
    >>> X = np.random.randn(100, 2048)
    >>>
    >>> # Sequential reduction: 2048 -> 100 -> 2
    >>> pipeline = DRPipeline(steps=[
    ...     ('pca_1', PCA(n_components=100)),
    ...     ('umap', UMAP(n_components=2))
    ... ])
    >>>
    >>> X_embedded = pipeline.fit_transform(X)
    >>> print(X_embedded.shape)  # (100, 2)
    >>>
    >>> # Or use intermediate embeddings
    >>> X_100d = pipeline.named_steps_['pca_1'].transform(X)
    >>> print(X_100d.shape)  # (100, 100)

    Notes
    -----
    - Each step's output becomes the next step's input
    - All steps except the last should have transform() method
    - Useful for coarse-to-fine reduction: start with fast method (PCA),
      refine with accurate method (UMAP/t-SNE)
    - Memory-efficient for very high-dimensional data
    - Preserves scikit-learn pipeline semantics

    """

    def __init__(self, steps: list[tuple[str, Any]]) -> None:
        """Initialize the pipeline.

        Parameters
        ----------
        steps : list of (str, estimator) tuples
            The pipeline steps.

        """
        self.steps = steps

    def _validate_steps(self) -> None:
        """Validate the steps list."""
        if not isinstance(self.steps, list):
            msg = "steps must be a list"
            raise TypeError(msg)

        if not self.steps:
            msg = "steps list cannot be empty"
            raise ValueError(msg)

        for name, estimator in self.steps:
            if not isinstance(name, str):
                msg = f"step name must be string, got {type(name)}"
                raise TypeError(msg)

            if not hasattr(estimator, "fit"):
                msg = f"step '{name}' must implement fit() method"
                raise TypeError(msg)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **fit_params,
    ) -> DRPipeline:
        """Fit all steps of the pipeline.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,), optional
            Target values. Used for supervised steps if provided.

        **fit_params : dict
            Parameters passed to the fit method of each step.
            Parameters for step i should be prefixed with step_name + '__'.

        Returns
        -------
        self : DRPipeline
            Returns self.

        Examples
        --------
        >>> pipeline.fit(X)
        >>> # Or with target values for supervised reduction:
        >>> pipeline.fit(X, y=labels)

        """
        self._validate_steps()

        X = np.asarray(X)
        self.steps_ = list(self.steps)
        self.named_steps_ = dict(self.steps_)

        # Fit and transform through the pipeline
        for _i, (name, estimator) in enumerate(self.steps_[:-1]):
            # Extract fit params for this step
            step_fit_params = {}
            for key, value in fit_params.items():
                if key.startswith(f"{name}__"):
                    step_fit_params[key.split("__", 1)[1]] = value

            # Fit and transform
            X = estimator.fit_transform(X, y=y, **step_fit_params)

        # Fit the last step (don't transform)
        name, estimator = self.steps_[-1]
        step_fit_params = {}
        for key, value in fit_params.items():
            if key.startswith(f"{name}__"):
                step_fit_params[key.split("__", 1)[1]] = value

        estimator.fit(X, y=y, **step_fit_params)

        return self

    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **fit_params,
    ) -> np.ndarray:
        """Fit all steps and transform with the last step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples,), optional
            Target values.

        **fit_params : dict
            Parameters passed to fit methods of steps.

        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_components)
            The transformed data.

        Examples
        --------
        >>> X_2d = pipeline.fit_transform(X)

        """
        self._validate_steps()

        X = np.asarray(X)
        self.steps_ = list(self.steps)
        self.named_steps_ = dict(self.steps_)

        # Fit and transform through all steps
        for _i, (name, estimator) in enumerate(self.steps_[:-1]):
            # Extract fit params for this step
            step_fit_params = {}
            for key, value in fit_params.items():
                if key.startswith(f"{name}__"):
                    step_fit_params[key.split("__", 1)[1]] = value

            # Fit and transform
            X = estimator.fit_transform(X, y=y, **step_fit_params)

        # Fit and transform the last step
        name, estimator = self.steps_[-1]
        step_fit_params = {}
        for key, value in fit_params.items():
            if key.startswith(f"{name}__"):
                step_fit_params[key.split("__", 1)[1]] = value

        if hasattr(estimator, "fit_transform"):
            X = estimator.fit_transform(X, y=y, **step_fit_params)
        else:
            X = estimator.fit(X, y=y, **step_fit_params).transform(X)

        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data through all steps of the pipeline.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data.

        Examples
        --------
        >>> pipeline.fit(X_train)
        >>> X_test_transformed = pipeline.transform(X_test)

        """
        check_is_fitted(self, "steps_")

        X = np.asarray(X)

        # Transform through all steps
        for _name, estimator in self.steps_:
            X = estimator.transform(X)

        return X

    def fit_predict(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit the model and predict with the last step if it has predict().

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, optional
            Target values.

        Returns
        -------
        y_pred : array-like
            Predictions from the last step (if it has predict()).

        """
        name, last_step = self.steps[-1]

        if not hasattr(last_step, "predict"):
            msg = f"Final step '{name}' does not have a predict() method"
            raise AttributeError(msg)

        # Fit pipeline
        self.fit(X, y)

        # Get intermediate data
        X_transformed = self.transform(X)

        return last_step.predict(X_transformed)

    @property
    def named_steps(self) -> dict[str, Any]:
        """Read-only view of step names and estimators."""
        if hasattr(self, "named_steps_"):
            return self.named_steps_
        return {}

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this pipeline.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.

        """
        out = {"steps": self.steps}

        if deep:
            for name, step in self.steps:
                if hasattr(step, "get_params"):
                    for param_name, param_value in step.get_params(deep=True).items():
                        out[f"{name}__{param_name}"] = param_value

        return out

    def set_params(self, **params) -> DRPipeline:
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : DRPipeline

        """
        if not params:
            return self

        valid_params = self.get_params(deep=True)

        for key, value in params.items():
            if key not in valid_params:
                msg = f"Invalid parameter {key} for estimator {self}."
                raise ValueError(msg)

            if "__" in key:
                name, param = key.split("__", 1)
                # Find the step by name
                for step_name, step_est in self.steps:
                    if step_name == name:
                        step_est.set_params(**{param: value})
                        break
            else:
                setattr(self, key, value)

        return self


class EnsembleDR(BaseEstimator):
    """Blend outputs from multiple dimensionality reduction methods.

    This composition pattern runs multiple DR methods on the same input data
    and combines their embeddings using various blend modes (weighted average,
    voting, consensus).

    Parameters
    ----------
    methods : list of (str, estimator, weight) tuples
        List of (name, estimator, weight) tuples. Each estimator should implement
        fit() and transform(). Weights should sum to 1.0 for weighted_average mode.

    blend_mode : str, default='weighted_average'
        How to combine embeddings:
        - 'weighted_average': Linear combination of embeddings (requires weights to sum to 1)
        - 'procrustes': Align each embedding to first using Procrustes analysis, then average
        - 'stacking': Use weighted average then apply Procrustes alignment to consensus

    alignment : str, default='procrustes'
        How to align embeddings before blending:
        - 'procrustes': Procrustes alignment
        - 'none': No alignment, direct averaging

    Attributes
    ----------
    steps_ : list
        The fitted steps.

    Examples
    --------
    >>> from squeeze import UMAP
    >>> from sklearn.manifold import TSNE
    >>> from sklearn.decomposition import PCA
    >>> from squeeze.composition import EnsembleDR
    >>>
    >>> ensemble = EnsembleDR(methods=[
    ...     ('umap', UMAP(n_neighbors=15), 0.5),
    ...     ('tsne', TSNE(), 0.3),
    ...     ('pca', PCA(n_components=2), 0.2)
    ... ], blend_mode='weighted_average')
    >>>
    >>> X_blended = ensemble.fit_transform(X)

    """

    def __init__(
        self,
        methods: list[tuple[str, Any, float]],
        blend_mode: str = "weighted_average",
        alignment: str = "procrustes",
    ) -> None:
        """Initialize the ensemble.

        Parameters
        ----------
        methods : list of (str, estimator, weight) tuples
            The DR methods and their weights.

        blend_mode : str
            How to blend embeddings.

        alignment : str
            How to align embeddings.

        """
        self.methods = methods
        self.blend_mode = blend_mode
        self.alignment = alignment

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> EnsembleDR:
        """Fit all DR methods.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, optional
            Target values.

        Returns
        -------
        self : EnsembleDR

        """
        self.steps_ = [(name, clone(est)) for name, est, _ in self.methods]
        self.weights_ = np.array([w for _, _, w in self.methods])

        # Normalize weights if needed
        if self.blend_mode == "weighted_average":
            if not np.isclose(self.weights_.sum(), 1.0):
                msg = f"Weights should sum to 1.0 for weighted_average, got {self.weights_.sum()}"
                raise ValueError(msg)

        for _name, estimator in self.steps_:
            estimator.fit(X, y)

        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit all methods and return blended embedding.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, optional
            Target values.

        Returns
        -------
        X_blended : array-like, shape (n_samples, n_components)
            Blended embedding.

        """
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data and blend embeddings.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_blended : array-like, shape (n_samples, n_components)
            Blended embedding.

        """
        check_is_fitted(self, "steps_")

        embeddings = []
        for _name, estimator in self.steps_:
            X_transformed = estimator.transform(X)
            embeddings.append(X_transformed)

        if self.blend_mode == "weighted_average":
            # Weighted average of embeddings
            result = np.average(embeddings, axis=0, weights=self.weights_)
        elif self.blend_mode == "procrustes":
            # Align each to first, then average
            result = self._procrustes_blend(embeddings)
        elif self.blend_mode == "stacking":
            # Weighted average then Procrustes alignment
            result = np.average(embeddings, axis=0, weights=self.weights_)
            # Apply light Procrustes regularization
            result = self._procrustes_blend([result, embeddings[0]])[0]
        else:
            msg = f"Unknown blend_mode: {self.blend_mode}"
            raise ValueError(msg)

        return result

    def _procrustes_blend(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """Blend embeddings using Procrustes alignment.

        Aligns all embeddings to the first one, then computes weighted average.
        """
        try:
            from scipy.spatial import procrustes
        except ImportError:
            msg = "scipy required for Procrustes alignment"
            raise ImportError(msg) from None

        reference = embeddings[0]
        aligned = [reference]

        for emb in embeddings[1:]:
            # Procrustes alignment returns (ref_aligned, emb_aligned, disparity)
            _, aligned_emb, _ = procrustes(reference, emb)
            aligned.append(aligned_emb)

        # Weighted average of aligned embeddings
        return np.average(aligned, axis=0, weights=self.weights_)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "methods": self.methods,
            "blend_mode": self.blend_mode,
            "alignment": self.alignment,
        }

    def set_params(self, **params) -> EnsembleDR:
        """Set parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ProgressiveDR(BaseEstimator):
    """Start with coarse approximation, progressively refine to accurate solution.

    This pattern uses two DR methods: a fast/coarse one for initialization
    and a slow/accurate one for refinement. The blending parameter controls
    the trade-off between speed and accuracy.

    Parameters
    ----------
    coarse : estimator
        Fast but coarse dimensionality reducer (e.g., PCA, coarse UMAP).

    fine : estimator
        Slow but accurate dimensionality reducer (e.g., fine UMAP, t-SNE).

    blend_steps : int, default=10
        Number of interpolation steps from coarse to fine.

    blend_function : str, default='linear'
        How to interpolate between coarse and fine:
        - 'linear': Linear interpolation
        - 'exponential': Exponential approach to fine
        - 'sigmoid': Sigmoid blending

    Examples
    --------
    >>> from squeeze import UMAP
    >>> from sklearn.decomposition import PCA
    >>> from squeeze.composition import ProgressiveDR
    >>>
    >>> progressive = ProgressiveDR(
    ...     coarse=UMAP(n_neighbors=30, n_epochs=100),  # Fast, coarse
    ...     fine=UMAP(n_neighbors=15, n_epochs=500),    # Slow, accurate
    ...     blend_steps=10
    ... )
    >>>
    >>> X_refined = progressive.fit_transform(X)

    """

    def __init__(
        self,
        coarse: Any,
        fine: Any,
        blend_steps: int = 10,
        blend_function: str = "linear",
    ) -> None:
        """Initialize progressive DR."""
        self.coarse = coarse
        self.fine = fine
        self.blend_steps = blend_steps
        self.blend_function = blend_function

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> ProgressiveDR:
        """Fit both coarse and fine methods."""
        self.coarse_ = clone(self.coarse)
        self.fine_ = clone(self.fine)

        self.coarse_.fit(X, y)
        self.fine_.fit(X, y)

        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit and return progressively refined embedding."""
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform and progressively blend coarse and fine embeddings."""
        check_is_fitted(self, "coarse_")

        X_coarse = self.coarse_.transform(X)
        X_fine = self.fine_.transform(X)

        # Compute blend weights based on blend_function
        alpha = self._get_blend_weights()

        # Final blend uses the last alpha value
        return (1 - alpha[-1]) * X_coarse + alpha[-1] * X_fine

    def _get_blend_weights(self) -> np.ndarray:
        """Get blending weights for progressive refinement."""
        steps = np.arange(self.blend_steps) / max(1, self.blend_steps - 1)

        if self.blend_function == "linear":
            return steps
        if self.blend_function == "exponential":
            return 1 - np.exp(-3 * steps)
        if self.blend_function == "sigmoid":
            return 1 / (1 + np.exp(-5 * (steps - 0.5)))
        msg = f"Unknown blend_function: {self.blend_function}"
        raise ValueError(msg)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters."""
        return {
            "coarse": self.coarse,
            "fine": self.fine,
            "blend_steps": self.blend_steps,
            "blend_function": self.blend_function,
        }

    def set_params(self, **params) -> ProgressiveDR:
        """Set parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class AdaptiveDR(BaseEstimator):
    """Automatically choose DR algorithm based on data characteristics.

    This pattern provides intelligent algorithm selection based on data
    properties (size, intrinsic dimensionality, sparsity, etc.).

    Parameters
    ----------
    method_map : dict
        Dict mapping condition keys to DR estimators.
        Keys are in form 'size:small', 'size:large', 'sparse:true', etc.

    strategy : str, default='size'
        Selection strategy:
        - 'size': Select based on number of samples
        - 'dimensionality': Select based on feature dimensionality
        - 'sparsity': Select based on data sparsity
        - 'composite': Use multiple criteria

    Examples
    --------
    >>> from squeeze import UMAP
    >>> from sklearn.decomposition import PCA
    >>> from squeeze.composition import AdaptiveDR
    >>>
    >>> adaptive = AdaptiveDR(
    ...     method_map={
    ...         'size:small': UMAP(n_neighbors=5),
    ...         'size:large': UMAP(n_neighbors=15),
    ...     },
    ...     strategy='size'
    ... )
    >>>
    >>> X_embedded = adaptive.fit_transform(X)

    """

    def __init__(
        self,
        method_map: dict[str, Any],
        strategy: str = "size",
    ) -> None:
        """Initialize adaptive DR."""
        self.method_map = method_map
        self.strategy = strategy

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> AdaptiveDR:
        """Select and fit the appropriate method."""
        X = np.asarray(X)

        # Select method based on strategy
        method_key = self._select_method(X)
        self.selected_method_ = clone(self.method_map[method_key])
        self.selected_method_.fit(X, y)

        return self

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Fit and transform with selected method."""
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using selected method."""
        check_is_fitted(self, "selected_method_")
        return self.selected_method_.transform(X)

    def _select_method(self, X: np.ndarray) -> str:
        """Select which method to use based on data."""
        n_samples, n_features = X.shape

        if self.strategy == "size":
            if n_samples < 1000:
                return "size:small"
            if n_samples < 100000:
                return "size:medium"
            return "size:large"

        if self.strategy == "dimensionality":
            if n_features < 50:
                return "dim:low"
            if n_features < 1000:
                return "dim:medium"
            return "dim:high"

        if self.strategy == "sparsity":
            # Estimate sparsity
            import scipy.sparse

            if scipy.sparse.issparse(X):
                sparsity = 1 - (X.nnz / (X.shape[0] * X.shape[1]))
            else:
                sparsity = np.mean(X == 0)

            if sparsity > 0.9:
                return "sparse:high"
            if sparsity > 0.5:
                return "sparse:medium"
            return "sparse:low"

        msg = f"Unknown strategy: {self.strategy}"
        raise ValueError(msg)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters."""
        return {
            "method_map": self.method_map,
            "strategy": self.strategy,
        }

    def set_params(self, **params) -> AdaptiveDR:
        """Set parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
