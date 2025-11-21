"""Tests for dimensionality reduction composition patterns."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from squeeze import UMAP
from squeeze.composition import AdaptiveDR, DRPipeline, EnsembleDR, ProgressiveDR


class TestDRPipeline:
    """Tests for DRPipeline sequential composition."""

    @pytest.fixture
    def data(self):
        """Load iris dataset."""
        iris = load_iris()
        X = iris.data
        y = iris.target
        return X, y

    def test_pipeline_initialization(self) -> None:
        """Test basic pipeline creation."""
        pipeline = DRPipeline(
            steps=[
                ("pca", PCA(n_components=2)),
                ("umap", UMAP(n_components=2)),
            ],
        )
        assert len(pipeline.steps) == 2

    def test_pipeline_fit_transform(self, data) -> None:
        """Test fit_transform on sequential pipeline."""
        X, _ = data

        pipeline = DRPipeline(
            steps=[
                ("pca", PCA(n_components=2)),
                ("umap", UMAP(n_components=2, random_state=42)),
            ],
        )

        X_transformed = pipeline.fit_transform(X)

        assert X_transformed.shape == (len(X), 2)

    def test_pipeline_transform(self, data) -> None:
        """Test transform after fit."""
        X, _ = data

        pipeline = DRPipeline(
            steps=[
                ("pca", PCA(n_components=2)),
                ("umap", UMAP(n_components=2, random_state=42)),
            ],
        )

        pipeline.fit(X)
        X_test = X[:10]
        X_transformed = pipeline.transform(X_test)

        assert X_transformed.shape == (10, 2)

    def test_pipeline_fit_with_y(self, data) -> None:
        """Test pipeline fit with target values."""
        X, y = data

        pipeline = DRPipeline(
            steps=[
                ("pca", PCA(n_components=2)),
                ("umap", UMAP(n_components=2, random_state=42)),
            ],
        )

        pipeline.fit(X, y)
        assert hasattr(pipeline, "steps_")

    def test_pipeline_intermediate_access(self, data) -> None:
        """Test access to intermediate embeddings."""
        X, _ = data

        pipeline = DRPipeline(
            steps=[
                ("pca", PCA(n_components=2)),
                ("umap", UMAP(n_components=2, random_state=42)),
            ],
        )

        pipeline.fit(X)

        # Access intermediate PCA embedding
        X_pca = pipeline.named_steps_["pca"].transform(X)
        assert X_pca.shape == (len(X), 2)

    def test_pipeline_get_params(self) -> None:
        """Test get_params method."""
        pipeline = DRPipeline(
            steps=[
                ("pca", PCA(n_components=10)),
                ("umap", UMAP(n_components=2)),
            ],
        )

        params = pipeline.get_params()
        assert "steps" in params
        assert "pca__n_components" in params
        assert params["pca__n_components"] == 10

    def test_pipeline_set_params(self) -> None:
        """Test set_params method."""
        pipeline = DRPipeline(
            steps=[
                ("pca", PCA(n_components=3)),
                ("umap", UMAP(n_components=2)),
            ],
        )

        pipeline.set_params(pca__n_components=2)
        assert pipeline.steps[0][1].n_components == 2

    def test_pipeline_empty_steps_error(self) -> None:
        """Test error on empty steps."""
        with pytest.raises(ValueError):
            pipeline = DRPipeline(steps=[])
            pipeline.fit(np.random.randn(10, 5))

    def test_pipeline_invalid_step_type_error(self) -> None:
        """Test error on invalid step type."""
        with pytest.raises(TypeError):
            pipeline = DRPipeline(steps=[("pca", "not an estimator")])
            pipeline.fit(np.random.randn(10, 5))

    def test_pipeline_dimensionality_reduction_chain(self, data) -> None:
        """Test 2048 -> 100 -> 2 style reduction."""
        X, _ = data
        # Create high-dimensional data
        X_high = np.random.randn(len(X), 100)

        pipeline = DRPipeline(
            steps=[
                ("pca_coarse", PCA(n_components=30)),
                ("pca_medium", PCA(n_components=10)),
                ("umap_fine", UMAP(n_components=2, random_state=42)),
            ],
        )

        X_2d = pipeline.fit_transform(X_high)
        assert X_2d.shape == (len(X), 2)

    def test_pipeline_reproducibility(self, data) -> None:
        """Test that same random state gives same results."""
        X, _ = data

        pipeline1 = DRPipeline(
            steps=[
                ("pca", PCA(n_components=2, random_state=42)),
                ("umap", UMAP(n_components=2, random_state=42)),
            ],
        )

        pipeline2 = DRPipeline(
            steps=[
                ("pca", PCA(n_components=2, random_state=42)),
                ("umap", UMAP(n_components=2, random_state=42)),
            ],
        )

        X1 = pipeline1.fit_transform(X)
        X2 = pipeline2.fit_transform(X)

        np.testing.assert_array_almost_equal(X1, X2, decimal=5)


class TestEnsembleDR:
    """Tests for EnsembleDR ensemble composition."""

    @pytest.fixture
    def data(self):
        """Load iris dataset."""
        iris = load_iris()
        return iris.data, iris.target

    def test_ensemble_initialization(self) -> None:
        """Test ensemble creation."""
        ensemble = EnsembleDR(
            methods=[
                ("pca", PCA(n_components=2), 0.5),
                ("umap", UMAP(n_components=2), 0.5),
            ],
        )
        assert len(ensemble.methods) == 2

    def test_ensemble_fit_transform(self, data) -> None:
        """Test ensemble fit_transform."""
        X, _ = data

        ensemble = EnsembleDR(
            methods=[
                ("pca", PCA(n_components=2), 0.5),
                ("umap", UMAP(n_components=2, random_state=42), 0.5),
            ],
            blend_mode="weighted_average",
        )

        X_blended = ensemble.fit_transform(X)
        assert X_blended.shape == (len(X), 2)

    def test_ensemble_weights_validation(self) -> None:
        """Test that weights must sum to 1.0."""
        with pytest.raises(ValueError):
            ensemble = EnsembleDR(
                methods=[
                    ("pca", PCA(n_components=2), 0.6),
                    ("umap", UMAP(n_components=2), 0.6),
                ],
            )
            ensemble.fit(np.random.randn(10, 4))

    def test_ensemble_procrustes_blend(self, data) -> None:
        """Test Procrustes alignment blending."""
        X, _ = data

        ensemble = EnsembleDR(
            methods=[
                ("pca", PCA(n_components=2), 0.5),
                ("umap", UMAP(n_components=2, random_state=42), 0.5),
            ],
            blend_mode="procrustes",
        )

        X_blended = ensemble.fit_transform(X)
        assert X_blended.shape == (len(X), 2)

    def test_ensemble_transform(self, data) -> None:
        """Test transform after fit."""
        X, _ = data

        ensemble = EnsembleDR(
            methods=[
                ("pca", PCA(n_components=2), 0.5),
                ("umap", UMAP(n_components=2, random_state=42), 0.5),
            ],
        )

        ensemble.fit(X)
        X_test = X[:10]
        X_blended = ensemble.transform(X_test)

        assert X_blended.shape == (10, 2)


class TestProgressiveDR:
    """Tests for ProgressiveDR progressive refinement."""

    @pytest.fixture
    def data(self):
        """Load iris dataset."""
        iris = load_iris()
        return iris.data, iris.target

    def test_progressive_initialization(self) -> None:
        """Test progressive DR creation."""
        progressive = ProgressiveDR(
            coarse=PCA(n_components=10),
            fine=UMAP(n_components=2),
            blend_steps=5,
        )
        assert progressive.blend_steps == 5

    def test_progressive_fit_transform(self, data) -> None:
        """Test progressive fit_transform."""
        X, _ = data

        progressive = ProgressiveDR(
            coarse=PCA(n_components=2),
            fine=UMAP(n_components=2, random_state=42),
            blend_steps=10,
        )

        X_refined = progressive.fit_transform(X)
        assert X_refined.shape == (len(X), 2)

    def test_progressive_blend_functions(self, data) -> None:
        """Test different blend functions."""
        X, _ = data

        for blend_func in ["linear", "exponential", "sigmoid"]:
            progressive = ProgressiveDR(
                coarse=PCA(n_components=2),
                fine=UMAP(n_components=2, random_state=42),
                blend_function=blend_func,
            )

            X_refined = progressive.fit_transform(X)
            assert X_refined.shape == (len(X), 2)

    def test_progressive_invalid_blend_function(self, data) -> None:
        """Test error on invalid blend function."""
        X, _ = data

        progressive = ProgressiveDR(
            coarse=PCA(n_components=10),
            fine=UMAP(n_components=2),
            blend_function="invalid",
        )

        with pytest.raises(ValueError):
            progressive.fit_transform(X)


class TestAdaptiveDR:
    """Tests for AdaptiveDR adaptive algorithm selection."""

    def test_adaptive_size_strategy(self) -> None:
        """Test size-based adaptive selection."""
        adaptive = AdaptiveDR(
            method_map={
                "size:small": PCA(n_components=2),
                "size:medium": PCA(n_components=2),
                "size:large": UMAP(n_components=2, random_state=42),
            },
            strategy="size",
        )

        # Small dataset
        X_small = np.random.randn(100, 10)
        X_small_embedded = adaptive.fit_transform(X_small)
        assert X_small_embedded.shape == (100, 2)

        # Large dataset
        X_large = np.random.randn(10000, 10)
        X_large_embedded = adaptive.fit_transform(X_large)
        assert X_large_embedded.shape == (10000, 2)

    def test_adaptive_dimensionality_strategy(self) -> None:
        """Test dimensionality-based adaptive selection."""
        adaptive = AdaptiveDR(
            method_map={
                "dim:low": PCA(n_components=2),
                "dim:high": UMAP(n_components=2, random_state=42),
            },
            strategy="dimensionality",
        )

        # Low-dimensional data
        X_low = np.random.randn(100, 10)
        X_low_embedded = adaptive.fit_transform(X_low)
        assert X_low_embedded.shape == (100, 2)

        # High-dimensional data
        X_high = np.random.randn(100, 1000)
        X_high_embedded = adaptive.fit_transform(X_high)
        assert X_high_embedded.shape == (100, 2)

    def test_adaptive_sparsity_strategy(self) -> None:
        """Test sparsity-based adaptive selection."""
        import scipy.sparse

        adaptive = AdaptiveDR(
            method_map={
                "sparse:low": PCA(n_components=2),
                "sparse:medium": PCA(n_components=2),
                "sparse:high": PCA(n_components=2),
            },
            strategy="sparsity",
        )

        # Dense data
        X_dense = np.random.randn(100, 20)
        X_dense_embedded = adaptive.fit_transform(X_dense)
        assert X_dense_embedded.shape == (100, 2)

        # Sparse data
        X_sparse = scipy.sparse.random(100, 20, density=0.05, format="csr")
        X_sparse_embedded = adaptive.fit_transform(X_sparse.toarray())
        assert X_sparse_embedded.shape == (100, 2)


class TestCompositionIntegration:
    """Integration tests combining multiple composition patterns."""

    def test_pipeline_of_pipelines(self) -> None:
        """Test nested pipelines."""
        X = np.random.randn(50, 100)

        inner_pipeline = DRPipeline(
            steps=[
                ("pca1", PCA(n_components=30)),
                ("pca2", PCA(n_components=10)),
            ],
        )

        outer_pipeline = DRPipeline(
            steps=[
                ("inner", inner_pipeline),
                ("umap", UMAP(n_components=2, random_state=42)),
            ],
        )

        X_embedded = outer_pipeline.fit_transform(X)
        assert X_embedded.shape == (50, 2)

    def test_pipeline_then_ensemble(self) -> None:
        """Test pipeline followed by ensemble."""
        X = np.random.randn(50, 100)

        # First reduce with pipeline
        pipeline = DRPipeline(
            steps=[
                ("pca", PCA(n_components=20)),
            ],
        )
        X_reduced = pipeline.fit_transform(X)

        # Then apply ensemble on reduced data
        ensemble = EnsembleDR(
            methods=[
                ("umap1", UMAP(n_components=2, n_neighbors=5, random_state=42), 0.5),
                ("umap2", UMAP(n_components=2, n_neighbors=15, random_state=42), 0.5),
            ],
        )

        X_blended = ensemble.fit_transform(X_reduced)
        assert X_blended.shape == (50, 2)

    def test_realistic_workflow_high_dim_to_2d(self) -> None:
        """Test realistic workflow: high-dim -> low-dim -> 2D."""
        # Simulate high-dimensional data (e.g., image embeddings)
        X = np.random.randn(200, 500)

        # Multi-stage reduction
        pipeline = DRPipeline(
            steps=[
                ("pca_1", PCA(n_components=100)),
                ("pca_2", PCA(n_components=30)),
                ("umap", UMAP(n_components=2, random_state=42)),
            ],
        )

        X_2d = pipeline.fit_transform(X)
        assert X_2d.shape == (200, 2)

        # Test transform on new data
        X_new = np.random.randn(10, 500)
        X_new_2d = pipeline.transform(X_new)
        assert X_new_2d.shape == (10, 2)
