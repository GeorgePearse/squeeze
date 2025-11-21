"""Tests for dimensionality reduction evaluation metrics."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from squeeze import UMAP
from squeeze.metrics import (
    DREvaluator,
    continuity,
    local_continuity_meta_estimate,
    reconstruction_error,
    spearman_distance_correlation,
    trustworthiness,
)


class TestTrustworthiness:
    """Tests for trustworthiness metric."""

    @pytest.fixture
    def data(self):
        """Load iris dataset."""
        iris = load_iris()
        return iris.data, iris.target

    def test_trustworthiness_perfect_embedding(self):
        """Test trustworthiness with perfect embedding (identity)."""
        X = np.random.randn(50, 5)
        # Perfect embedding: same space
        trust = trustworthiness(X, X)
        assert trust > 0.99

    def test_trustworthiness_random_embedding(self):
        """Test trustworthiness with random embedding."""
        X = np.random.randn(50, 5)
        X_random = np.random.randn(50, 2)
        trust = trustworthiness(X, X_random)
        assert 0 <= trust <= 1

    def test_trustworthiness_umap_embedding(self, data):
        """Test trustworthiness with UMAP embedding."""
        X, _ = data
        X_embedded = UMAP(n_components=2, random_state=42).fit_transform(X)
        trust = trustworthiness(X, X_embedded, k=5)
        # UMAP should have good trustworthiness
        assert 0.7 < trust <= 1.0

    def test_trustworthiness_with_different_k(self, data):
        """Test trustworthiness with different k values."""
        X, _ = data
        X_embedded = UMAP(n_components=2, random_state=42).fit_transform(X)

        for k in [5, 10, 15]:
            trust = trustworthiness(X, X_embedded, k=k)
            assert 0 <= trust <= 1

    def test_trustworthiness_range(self, data):
        """Test that trustworthiness is always in [0, 1]."""
        X, _ = data

        X_bad = np.random.randn(*X.shape)
        for _ in range(5):
            X_bad = np.random.randn(*X.shape)
            trust = trustworthiness(X, X_bad)
            assert 0 <= trust <= 1


class TestContinuity:
    """Tests for continuity metric."""

    @pytest.fixture
    def data(self):
        """Load iris dataset."""
        iris = load_iris()
        return iris.data, iris.target

    def test_continuity_perfect_embedding(self):
        """Test continuity with perfect embedding."""
        X = np.random.randn(50, 5)
        cont = continuity(X, X)
        assert cont > 0.99

    def test_continuity_range(self, data):
        """Test that continuity is always in [0, 1]."""
        X, _ = data
        X_embedded = UMAP(n_components=2, random_state=42).fit_transform(X)
        cont = continuity(X, X_embedded)
        assert 0 <= cont <= 1

    def test_continuity_with_pca(self, data):
        """Test continuity with PCA embedding."""
        X, _ = data
        X_pca = PCA(n_components=2).fit_transform(X)
        cont = continuity(X, X_pca, k=5)
        assert 0 <= cont <= 1


class TestLocalContinuityMetaEstimate:
    """Tests for local continuity meta-estimate."""

    def test_lcmc_perfect_embedding(self):
        """Test LCMC with perfect embedding."""
        X = np.random.randn(50, 5)
        lcmc = local_continuity_meta_estimate(X, X)
        assert lcmc > 0.99

    def test_lcmc_range(self):
        """Test that LCMC is in reasonable range."""
        X = np.random.randn(50, 5)
        X_embedded = PCA(n_components=2).fit_transform(X)
        lcmc = local_continuity_meta_estimate(X, X_embedded)
        assert -1 <= lcmc <= 1


class TestReconstructionError:
    """Tests for reconstruction error metric."""

    def test_reconstruction_error_perfect(self):
        """Test reconstruction error with perfect embedding (identity)."""
        X = np.random.randn(50, 5)
        error = reconstruction_error(X, X)
        assert error < 0.01  # Should be nearly zero

    def test_reconstruction_error_linear_embedding(self):
        """Test reconstruction error with linear embedding."""
        X = np.random.randn(50, 5)
        # Simple linear projection
        A = np.random.randn(5, 2)
        X_embedded = X @ A
        error = reconstruction_error(X, X_embedded)
        # Should be relatively low for linear embedding
        assert 0 <= error <= 10

    def test_reconstruction_error_random_embedding(self):
        """Test reconstruction error with random embedding."""
        X = np.random.randn(50, 5)
        X_random = np.random.randn(50, 2)
        error = reconstruction_error(X, X_random)
        assert error >= 0


class TestSpearmanDistanceCorrelation:
    """Tests for Spearman distance correlation."""

    def test_spearman_perfect_embedding(self):
        """Test Spearman correlation with perfect embedding."""
        X = np.random.randn(50, 5)
        corr = spearman_distance_correlation(X, X)
        assert corr > 0.99

    def test_spearman_range(self):
        """Test that Spearman correlation is in [-1, 1]."""
        X = np.random.randn(50, 5)
        X_embedded = PCA(n_components=2).fit_transform(X)
        corr = spearman_distance_correlation(X, X_embedded)
        assert -1 <= corr <= 1

    def test_spearman_with_sampling(self):
        """Test Spearman correlation with distance sampling."""
        X = np.random.randn(100, 10)
        X_embedded = PCA(n_components=2).fit_transform(X)

        # Full computation
        corr_full = spearman_distance_correlation(X, X_embedded)

        # With sampling
        corr_sampled = spearman_distance_correlation(X, X_embedded, sample_size=100)

        # Should be similar
        assert abs(corr_full - corr_sampled) < 0.1


class TestDREvaluator:
    """Tests for DREvaluator class."""

    @pytest.fixture
    def data(self):
        """Load iris dataset."""
        iris = load_iris()
        return iris.data, iris.target

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = DREvaluator(k=10)
        assert evaluator.k == 10
        assert not evaluator.metrics_

    def test_evaluator_evaluate(self, data):
        """Test evaluator.evaluate() method."""
        X, _ = data
        X_embedded = UMAP(n_components=2, random_state=42).fit_transform(X)

        evaluator = DREvaluator(k=5)
        metrics = evaluator.evaluate(X, X_embedded)

        # Check that all expected metrics are present
        expected_metrics = [
            "trustworthiness",
            "continuity",
            "lcmc",
            "reconstruction_error",
            "spearman_distance_correlation",
        ]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])

    def test_evaluator_metrics_stored(self, data):
        """Test that metrics are stored in evaluator."""
        X, _ = data
        X_embedded = PCA(n_components=2).fit_transform(X)

        evaluator = DREvaluator()
        evaluator.evaluate(X, X_embedded)

        assert len(evaluator.metrics_) > 0
        assert "trustworthiness" in evaluator.metrics_

    def test_evaluator_summary(self, data):
        """Test evaluator summary output."""
        X, _ = data
        X_embedded = PCA(n_components=2).fit_transform(X)

        evaluator = DREvaluator()
        evaluator.evaluate(X, X_embedded)

        summary = evaluator.summary()
        assert isinstance(summary, str)
        assert "trustworthiness" in summary
        assert "Dimensionality Reduction Quality Metrics" in summary

    def test_evaluator_summary_before_evaluation(self):
        """Test evaluator summary before evaluation."""
        evaluator = DREvaluator()
        summary = evaluator.summary()
        assert "No metrics computed" in summary


class TestMetricsComparison:
    """Comparison tests between different embeddings."""

    def test_umap_vs_pca_trustworthiness(self):
        """Test that UMAP typically has better trustworthiness than PCA."""
        X = np.random.randn(100, 50)

        X_pca = PCA(n_components=2).fit_transform(X)
        X_umap = UMAP(n_components=2, random_state=42).fit_transform(X)

        trust_pca = trustworthiness(X, X_pca)
        trust_umap = trustworthiness(X, X_umap)

        # UMAP should typically preserve local structure better
        assert trust_umap >= trust_pca * 0.8  # Allow some variation

    def test_metrics_are_independent(self):
        """Test that different metrics measure different things."""
        X = np.random.randn(50, 10)
        X_embedded = PCA(n_components=2).fit_transform(X)

        trust = trustworthiness(X, X_embedded)
        cont = continuity(X, X_embedded)
        recon = reconstruction_error(X, X_embedded)
        spear = spearman_distance_correlation(X, X_embedded)

        # These metrics should be somewhat different
        metrics = [trust, cont, recon, spear]
        # Check they're not all the same
        assert len(set(np.round(metrics, 2))) > 1


class TestMetricsEdgeCases:
    """Test edge cases and error handling."""

    def test_single_dimension_embedding(self):
        """Test metrics with 1D embedding."""
        X = np.random.randn(20, 5)
        X_1d = np.random.randn(20, 1)

        trust = trustworthiness(X, X_1d, k=3)
        assert 0 <= trust <= 1

    def test_identical_points(self):
        """Test metrics with identical points."""
        X = np.ones((10, 5))
        X_embedded = np.ones((10, 2))

        # Should handle gracefully
        trust = trustworthiness(X, X_embedded, k=3)
        assert isinstance(trust, (float, int, np.floating))

    def test_large_k_relative_to_samples(self):
        """Test with k close to n_samples."""
        X = np.random.randn(15, 5)
        X_embedded = PCA(n_components=2).fit_transform(X)

        trust = trustworthiness(X, X_embedded, k=10)
        assert 0 <= trust <= 1
