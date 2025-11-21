"""Tests for dimensionality reduction evaluation metrics."""

import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA as SklearnPCA

from squeeze import PCA
from squeeze.evaluation import (
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


@pytest.fixture
def digits_data():
    """Load a subset of digits for testing."""
    X, y = load_digits(return_X_y=True)
    # Use smaller subset for faster tests
    return X[:300].astype(np.float64), y[:300]


@pytest.fixture
def reduced_data(digits_data):
    """Create PCA reduced data."""
    X, y = digits_data
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    return X, X_reduced, y


class TestLocalStructureMetrics:
    """Tests for local structure metrics."""

    def test_trustworthiness_range(self, reduced_data):
        """Test trustworthiness returns value in [0, 1]."""
        X, X_reduced, _ = reduced_data
        T = trustworthiness(X, X_reduced, k=15)
        assert 0 <= T <= 1

    def test_trustworthiness_k_values(self, reduced_data):
        """Test trustworthiness with different k values."""
        X, X_reduced, _ = reduced_data
        for k in [5, 10, 15, 30]:
            T = trustworthiness(X, X_reduced, k=k)
            assert 0 <= T <= 1

    def test_trustworthiness_perfect(self):
        """Test trustworthiness is 1.0 for identity mapping."""
        X = np.random.randn(100, 10)
        T = trustworthiness(X, X, k=15)
        assert T == pytest.approx(1.0)

    def test_continuity_range(self, reduced_data):
        """Test continuity returns value in [0, 1]."""
        X, X_reduced, _ = reduced_data
        C = continuity(X, X_reduced, k=15)
        assert 0 <= C <= 1

    def test_continuity_perfect(self):
        """Test continuity is 1.0 for identity mapping."""
        X = np.random.randn(100, 10)
        C = continuity(X, X, k=15)
        assert C == pytest.approx(1.0)

    def test_co_ranking_quality_range(self, reduced_data):
        """Test co-ranking quality returns value in [0, 1]."""
        X, X_reduced, _ = reduced_data
        Q = co_ranking_quality(X, X_reduced, k=15)
        assert 0 <= Q <= 1

    def test_co_ranking_quality_perfect(self):
        """Test co-ranking quality is 1.0 for identity mapping."""
        X = np.random.randn(100, 10)
        Q = co_ranking_quality(X, X, k=15)
        assert Q == pytest.approx(1.0)


class TestGlobalStructureMetrics:
    """Tests for global structure metrics."""

    def test_spearman_correlation_range(self, reduced_data):
        """Test Spearman correlation returns value in [-1, 1]."""
        X, X_reduced, _ = reduced_data
        rho = spearman_distance_correlation(X, X_reduced)
        assert -1 <= rho <= 1

    def test_spearman_correlation_perfect(self):
        """Test Spearman correlation is 1.0 for identity mapping."""
        X = np.random.randn(100, 10)
        rho = spearman_distance_correlation(X, X)
        assert rho == pytest.approx(1.0, abs=0.001)

    def test_spearman_correlation_subsampling(self, reduced_data):
        """Test Spearman correlation with subsampling."""
        X, X_reduced, _ = reduced_data
        rho = spearman_distance_correlation(X, X_reduced, max_samples=100)
        assert -1 <= rho <= 1

    def test_global_structure_preservation_range(self, reduced_data):
        """Test global structure preservation returns value in [0, 1]."""
        X, X_reduced, y = reduced_data
        G = global_structure_preservation(X, X_reduced, y)
        assert 0 <= G <= 1

    def test_global_structure_preservation_single_label(self):
        """Test global structure preservation with single label."""
        X = np.random.randn(50, 10)
        X_reduced = np.random.randn(50, 2)
        y = np.zeros(50)  # Single label
        G = global_structure_preservation(X, X_reduced, y)
        assert G == 1.0  # Trivially preserved

    def test_local_density_preservation_range(self, reduced_data):
        """Test local density preservation returns value in [0, 1]."""
        X, X_reduced, _ = reduced_data
        D = local_density_preservation(X, X_reduced, k=15)
        assert 0 <= D <= 1

    def test_local_density_preservation_perfect(self):
        """Test local density preservation is high for identity mapping."""
        X = np.random.randn(100, 10)
        D = local_density_preservation(X, X, k=15)
        assert D == pytest.approx(1.0, abs=0.001)


class TestReconstructionMetrics:
    """Tests for reconstruction metrics."""

    def test_reconstruction_error_keys(self, reduced_data):
        """Test reconstruction error returns expected keys."""
        X, X_reduced, _ = reduced_data
        error = reconstruction_error(X, X_reduced)
        assert "mse" in error
        assert "rmse" in error
        assert "normalized_rmse" in error
        assert "r2" in error

    def test_reconstruction_error_values(self, reduced_data):
        """Test reconstruction error values are reasonable."""
        X, X_reduced, _ = reduced_data
        error = reconstruction_error(X, X_reduced)
        assert error["mse"] >= 0
        assert error["rmse"] >= 0
        assert error["normalized_rmse"] >= 0

    def test_reconstruction_error_perfect(self):
        """Test reconstruction error is 0 for identity mapping."""
        X = np.random.randn(100, 10)
        error = reconstruction_error(X, X)
        assert error["mse"] == pytest.approx(0.0, abs=1e-10)
        assert error["r2"] == pytest.approx(1.0, abs=1e-10)


class TestStabilityMetrics:
    """Tests for stability metrics."""

    def test_bootstrap_stability_keys(self, digits_data):
        """Test bootstrap stability returns expected keys."""
        X, _ = digits_data
        X = X[:100]  # Smaller for speed
        # Use sklearn PCA which supports cloning
        stability = bootstrap_stability(X, SklearnPCA(n_components=2), n_bootstrap=3)
        assert "mean_procrustes_error" in stability
        assert "std_procrustes_error" in stability
        assert "stability_score" in stability

    def test_bootstrap_stability_range(self, digits_data):
        """Test bootstrap stability score is in [0, 1]."""
        X, _ = digits_data
        X = X[:100]
        stability = bootstrap_stability(X, SklearnPCA(n_components=2), n_bootstrap=3)
        assert 0 <= stability["stability_score"] <= 1

    def test_noise_robustness_keys(self, digits_data):
        """Test noise robustness returns expected keys."""
        X, _ = digits_data
        X = X[:100]
        robustness = noise_robustness(
            X, SklearnPCA(n_components=2), noise_levels=[0.01, 0.05]
        )
        assert 0.01 in robustness
        assert 0.05 in robustness

    def test_noise_robustness_range(self, digits_data):
        """Test noise robustness scores are in [0, 1]."""
        X, _ = digits_data
        X = X[:100]
        robustness = noise_robustness(
            X, SklearnPCA(n_components=2), noise_levels=[0.01]
        )
        for score in robustness.values():
            assert 0 <= score <= 1

    def test_parameter_sensitivity_structure(self, digits_data):
        """Test parameter sensitivity returns expected structure."""
        X, _ = digits_data
        X = X[:100]
        # Use sklearn PCA which supports cloning and get_params
        sensitivity = parameter_sensitivity(X, SklearnPCA, {"n_components": [2, 5, 10]})
        assert "n_components" in sensitivity
        assert "values" in sensitivity["n_components"]
        assert "procrustes_errors" in sensitivity["n_components"]
        assert "mean_sensitivity" in sensitivity["n_components"]


class TestDownstreamMetrics:
    """Tests for downstream task metrics."""

    def test_clustering_quality_keys(self, reduced_data):
        """Test clustering quality returns expected keys."""
        _, X_reduced, y = reduced_data
        metrics = clustering_quality(X_reduced, labels_true=y)
        assert "silhouette_score" in metrics
        assert "calinski_harabasz" in metrics
        assert "davies_bouldin" in metrics
        assert "adjusted_rand_index" in metrics
        assert "normalized_mutual_info" in metrics

    def test_clustering_quality_ranges(self, reduced_data):
        """Test clustering quality values are in expected ranges."""
        _, X_reduced, y = reduced_data
        metrics = clustering_quality(X_reduced, labels_true=y)
        assert -1 <= metrics["silhouette_score"] <= 1
        assert metrics["calinski_harabasz"] >= 0
        assert metrics["davies_bouldin"] >= 0
        assert 0 <= metrics["adjusted_rand_index"] <= 1
        assert 0 <= metrics["normalized_mutual_info"] <= 1

    def test_clustering_quality_no_labels(self, reduced_data):
        """Test clustering quality without ground truth labels."""
        _, X_reduced, y = reduced_data
        n_clusters = len(np.unique(y))
        metrics = clustering_quality(X_reduced, n_clusters=n_clusters)
        assert "silhouette_score" in metrics
        assert "adjusted_rand_index" not in metrics

    def test_classification_accuracy_keys(self, reduced_data):
        """Test classification accuracy returns expected keys."""
        _, X_reduced, y = reduced_data
        results = classification_accuracy(X_reduced, y, cv=3)
        assert "mean_accuracy" in results
        assert "std_accuracy" in results
        assert "scores" in results

    def test_classification_accuracy_range(self, reduced_data):
        """Test classification accuracy is in [0, 1]."""
        _, X_reduced, y = reduced_data
        results = classification_accuracy(X_reduced, y, cv=3)
        assert 0 <= results["mean_accuracy"] <= 1
        assert results["std_accuracy"] >= 0


class TestQuickEvaluate:
    """Tests for quick_evaluate function."""

    def test_quick_evaluate_keys(self, reduced_data):
        """Test quick_evaluate returns expected keys."""
        X, X_reduced, _ = reduced_data
        metrics = quick_evaluate(X, X_reduced, k=15)
        assert "trustworthiness" in metrics
        assert "continuity" in metrics
        assert "spearman_correlation" in metrics

    def test_quick_evaluate_ranges(self, reduced_data):
        """Test quick_evaluate values are in expected ranges."""
        X, X_reduced, _ = reduced_data
        metrics = quick_evaluate(X, X_reduced, k=15)
        assert 0 <= metrics["trustworthiness"] <= 1
        assert 0 <= metrics["continuity"] <= 1
        assert -1 <= metrics["spearman_correlation"] <= 1


class TestDREvaluator:
    """Tests for DREvaluator class."""

    def test_evaluator_initialization(self, reduced_data):
        """Test DREvaluator initialization."""
        X, X_reduced, y = reduced_data
        evaluator = DREvaluator(X, X_reduced, labels=y, method_name="PCA")
        assert evaluator.method_name == "PCA"

    def test_evaluate_local_structure(self, reduced_data):
        """Test evaluate_local_structure method."""
        X, X_reduced, y = reduced_data
        evaluator = DREvaluator(X, X_reduced, labels=y)
        local = evaluator.evaluate_local_structure(k_values=[5, 15])
        assert 5 in local["trustworthiness"]
        assert 15 in local["trustworthiness"]
        assert 5 in local["continuity"]
        assert 15 in local["continuity"]

    def test_evaluate_global_structure(self, reduced_data):
        """Test evaluate_global_structure method."""
        X, X_reduced, y = reduced_data
        evaluator = DREvaluator(X, X_reduced, labels=y)
        global_struct = evaluator.evaluate_global_structure()
        assert "spearman_correlation" in global_struct
        assert "global_structure" in global_struct
        assert "density_preservation" in global_struct

    def test_evaluate_reconstruction(self, reduced_data):
        """Test evaluate_reconstruction method."""
        X, X_reduced, y = reduced_data
        evaluator = DREvaluator(X, X_reduced, labels=y)
        recon = evaluator.evaluate_reconstruction()
        assert "mse" in recon
        assert "r2" in recon

    def test_evaluate_clustering(self, reduced_data):
        """Test evaluate_clustering method."""
        X, X_reduced, y = reduced_data
        evaluator = DREvaluator(X, X_reduced, labels=y)
        clustering = evaluator.evaluate_clustering()
        assert "silhouette_score" in clustering

    def test_evaluate_classification(self, reduced_data):
        """Test evaluate_classification method."""
        X, X_reduced, y = reduced_data
        evaluator = DREvaluator(X, X_reduced, labels=y)
        classification = evaluator.evaluate_classification(cv=3)
        assert "mean_accuracy" in classification

    def test_evaluate_all(self, reduced_data):
        """Test evaluate_all method."""
        X, X_reduced, y = reduced_data
        evaluator = DREvaluator(X, X_reduced, labels=y, method_name="PCA")
        report = evaluator.evaluate_all(
            k_values=[5, 15], include_stability=False, include_noise=False
        )
        assert isinstance(report, EvaluationReport)
        assert report.method_name == "PCA"
        assert 5 in report.trustworthiness
        assert 15 in report.trustworthiness

    def test_evaluate_all_with_reducer(self, digits_data):
        """Test evaluate_all with reducer for stability metrics."""
        X, y = digits_data
        X = X[:100]  # Smaller for speed
        y = y[:100]
        # Use sklearn PCA which supports cloning
        reducer = SklearnPCA(n_components=2)
        X_reduced = reducer.fit_transform(X)
        evaluator = DREvaluator(X, X_reduced, labels=y, reducer=reducer)
        report = evaluator.evaluate_all(
            k_values=[15], include_stability=True, include_noise=False, n_bootstrap=3
        )
        assert report.stability.get("stability_score") is not None


class TestEvaluationReport:
    """Tests for EvaluationReport class."""

    def test_report_to_dict(self, reduced_data):
        """Test EvaluationReport.to_dict method."""
        X, X_reduced, y = reduced_data
        evaluator = DREvaluator(X, X_reduced, labels=y, method_name="TestMethod")
        report = evaluator.evaluate_all(include_stability=False, include_noise=False)
        result = report.to_dict()
        assert isinstance(result, dict)
        assert result["method_name"] == "TestMethod"
        assert "local_structure" in result
        assert "global_structure" in result

    def test_report_summary(self, reduced_data):
        """Test EvaluationReport.summary method."""
        X, X_reduced, y = reduced_data
        evaluator = DREvaluator(X, X_reduced, labels=y, method_name="TestMethod")
        report = evaluator.evaluate_all(include_stability=False, include_noise=False)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "TestMethod" in summary
        assert "Trustworthiness" in summary

    def test_report_str(self, reduced_data):
        """Test EvaluationReport __str__ method."""
        X, X_reduced, y = reduced_data
        evaluator = DREvaluator(X, X_reduced, labels=y)
        report = evaluator.evaluate_all(include_stability=False, include_noise=False)
        assert str(report) == report.summary()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_k(self):
        """Test with small k value."""
        X = np.random.randn(50, 10)
        X_reduced = np.random.randn(50, 2)
        T = trustworthiness(X, X_reduced, k=2)
        assert 0 <= T <= 1

    def test_large_k(self):
        """Test with k larger than n_samples."""
        X = np.random.randn(20, 10)
        X_reduced = np.random.randn(20, 2)
        T = trustworthiness(X, X_reduced, k=50)  # k > n_samples
        assert 0 <= T <= 1

    def test_single_class(self):
        """Test clustering metrics with single class."""
        X_reduced = np.random.randn(50, 2)
        y = np.zeros(50)  # Single class
        # Should handle gracefully
        metrics = clustering_quality(X_reduced, labels_true=y)
        assert "silhouette_score" in metrics

    def test_two_samples(self):
        """Test with minimal number of samples."""
        X = np.random.randn(10, 5)
        X_reduced = np.random.randn(10, 2)
        T = trustworthiness(X, X_reduced, k=3)
        assert 0 <= T <= 1
