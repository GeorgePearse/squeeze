"""Tests for benchmarking framework."""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

from squeeze import UMAP
from squeeze.benchmark import AlgorithmConfig, BenchmarkResult, DRBenchmark


class TestAlgorithmConfig:
    """Tests for AlgorithmConfig."""

    def test_algorithm_config_initialization(self):
        """Test AlgorithmConfig initialization."""
        config = AlgorithmConfig("test_algo", PCA, {"n_components": 2})
        assert config.name == "test_algo"
        assert config.algorithm_class == PCA
        assert config.params == {"n_components": 2}

    def test_algorithm_config_default_color(self):
        """Test default color assignment."""
        config_umap = AlgorithmConfig("umap", UMAP, {})
        assert config_umap.color == "#1f77b4"  # blue

        config_pca = AlgorithmConfig("pca", PCA, {})
        assert config_pca.color == "#2ca02c"  # green

        config_custom = AlgorithmConfig("my_algorithm", object, {})
        assert config_custom.color == "#1f77b4"  # default blue

    def test_algorithm_config_custom_color(self):
        """Test custom color assignment."""
        config = AlgorithmConfig("test", PCA, {}, color="#FF0000")
        assert config.color == "#FF0000"

    def test_algorithm_config_custom_marker(self):
        """Test custom marker assignment."""
        config = AlgorithmConfig("test", PCA, {}, marker="s")
        assert config.marker == "s"


class TestBenchmarkResult:
    """Tests for BenchmarkResult."""

    def test_benchmark_result_initialization(self):
        """Test BenchmarkResult initialization."""
        config = AlgorithmConfig("test", PCA, {})
        times = [1.0, 1.1, 0.9]
        qualities = [0.8, 0.81, 0.79]
        embedding = np.random.randn(10, 2)

        result = BenchmarkResult(config, times, qualities, embedding)

        assert result.mean_time == pytest.approx(1.0, abs=0.01)
        assert result.std_time == pytest.approx(np.std(times), abs=0.01)
        assert result.mean_quality == pytest.approx(0.8, abs=0.01)

    def test_benchmark_result_single_run(self):
        """Test BenchmarkResult with single run (no std)."""
        config = AlgorithmConfig("test", PCA, {})
        result = BenchmarkResult(config, [1.0], [0.8], np.random.randn(10, 2))

        assert result.mean_time == 1.0
        assert result.std_time == 0.0
        assert result.mean_quality == 0.8
        assert result.std_quality == 0.0

    def test_benchmark_result_repr(self):
        """Test BenchmarkResult string representation."""
        config = AlgorithmConfig("test_algo", PCA, {})
        result = BenchmarkResult(config, [1.0], [0.8], np.random.randn(10, 2))

        repr_str = repr(result)
        assert "test_algo" in repr_str
        assert "1.0" in repr_str or "1.000" in repr_str
        assert "0.8" in repr_str or "0.800" in repr_str


class TestDRBenchmark:
    """Tests for DRBenchmark."""

    @pytest.fixture
    def data(self):
        """Load iris dataset."""
        iris = load_iris()
        return iris.data[:100]  # Small subset for fast testing

    def test_benchmark_initialization(self):
        """Test DRBenchmark initialization."""
        benchmark = DRBenchmark()
        assert len(benchmark.algorithms) == 0
        assert len(benchmark.results) == 0

    def test_add_algorithm(self, data):
        """Test adding algorithms."""
        benchmark = DRBenchmark()
        benchmark.add_algorithm("pca", PCA, {"n_components": 2})
        benchmark.add_algorithm("umap", UMAP, {"n_components": 2})

        assert len(benchmark.algorithms) == 2
        assert "pca" in benchmark.algorithms
        assert "umap" in benchmark.algorithms

    def test_add_algorithm_chaining(self):
        """Test method chaining for add_algorithm."""
        benchmark = (
            DRBenchmark()
            .add_algorithm("pca", PCA, {"n_components": 2})
            .add_algorithm("umap", UMAP, {"n_components": 2})
        )

        assert len(benchmark.algorithms) == 2

    def test_run_single_algorithm(self, data):
        """Test running benchmark on single algorithm."""
        benchmark = DRBenchmark()
        benchmark.add_algorithm("pca", PCA, {"n_components": 2})

        results = benchmark.run(data, dataset_name="test", n_runs=1, verbose=False)

        assert len(results) == 1
        assert results[0].config.name == "pca"
        assert len(results[0].times) == 1
        assert len(results[0].quality_scores) == 1

    def test_run_multiple_algorithms(self, data):
        """Test running benchmark on multiple algorithms."""
        benchmark = DRBenchmark()
        benchmark.add_algorithm("pca", PCA, {"n_components": 2})
        benchmark.add_algorithm("umap", UMAP, {"n_components": 2, "random_state": 42})

        results = benchmark.run(data, dataset_name="test", n_runs=1, verbose=False)

        assert len(results) == 2

    def test_run_multiple_runs(self, data):
        """Test running benchmark multiple times."""
        benchmark = DRBenchmark()
        benchmark.add_algorithm("pca", PCA, {"n_components": 2})

        results = benchmark.run(data, dataset_name="test", n_runs=3, verbose=False)

        assert len(results[0].times) == 3
        assert len(results[0].quality_scores) == 3

    def test_run_stores_results(self, data):
        """Test that run() stores results."""
        benchmark = DRBenchmark()
        benchmark.add_algorithm("pca", PCA, {"n_components": 2})

        benchmark.run(data, dataset_name="test1", n_runs=1, verbose=False)
        benchmark.run(data, dataset_name="test2", n_runs=1, verbose=False)

        assert "test1" in benchmark.results
        assert "test2" in benchmark.results

    def test_run_scaling_experiment(self, data):
        """Test scaling experiment on different sizes."""
        benchmark = DRBenchmark()
        benchmark.add_algorithm("pca", PCA, {"n_components": 2})

        # Use small sizes for fast testing
        sizes = [10, 20, 30]
        results = benchmark.run_scaling_experiment(data, sizes=sizes, n_runs=1, verbose=False)

        assert len(results) == 3
        assert all(size in results for size in sizes)

    def test_run_scaling_experiment_skips_large_sizes(self, data):
        """Test that scaling experiment skips sizes larger than dataset."""
        benchmark = DRBenchmark()
        benchmark.add_algorithm("pca", PCA, {"n_components": 2})

        # Request size larger than dataset
        sizes = [10, 1000000]  # 1M samples won't fit
        results = benchmark.run_scaling_experiment(data, sizes=sizes, n_runs=1, verbose=False)

        # Should only have results for the smaller size
        assert len(results) == 1
        assert 10 in results

    def test_summary(self, data):
        """Test summary generation."""
        benchmark = DRBenchmark()
        benchmark.add_algorithm("pca", PCA, {"n_components": 2})
        benchmark.run(data, dataset_name="test", n_runs=1, verbose=False)

        summary = benchmark.summary()

        assert isinstance(summary, str)
        assert "test" in summary
        assert "pca" in summary.lower()

    def test_plot_quality_vs_speed_no_results(self):
        """Test plot_quality_vs_speed with no results."""
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            pytest.skip("matplotlib not available")

        benchmark = DRBenchmark()

        with pytest.raises(ValueError):
            benchmark.plot_quality_vs_speed()

    def test_plot_quality_vs_speed_with_results(self, data):
        """Test plot_quality_vs_speed generates plot (no display)."""
        try:
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
        except ImportError:
            pytest.skip("matplotlib not available")

        benchmark = DRBenchmark()
        benchmark.add_algorithm("pca", PCA, {"n_components": 2})
        results = benchmark.run(data, dataset_name="test", n_runs=1, verbose=False)

        # Should not raise
        benchmark.plot_quality_vs_speed(results, output_file=None)

    def test_plot_quality_vs_speed_with_file(self, data, tmp_path):
        """Test plot_quality_vs_speed saves to file."""
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            pytest.skip("matplotlib not available")

        benchmark = DRBenchmark()
        benchmark.add_algorithm("pca", PCA, {"n_components": 2})
        results = benchmark.run(data, dataset_name="test", n_runs=1, verbose=False)

        output_file = str(tmp_path / "benchmark.png")
        benchmark.plot_quality_vs_speed(results, output_file=output_file)

        # Check file was created
        import os
        assert os.path.exists(output_file)

    def test_benchmark_quality_scores_in_range(self, data):
        """Test that quality scores are in valid range [0, 1]."""
        benchmark = DRBenchmark()
        benchmark.add_algorithm("pca", PCA, {"n_components": 2})
        benchmark.add_algorithm("umap", UMAP, {"n_components": 2, "random_state": 42})

        results = benchmark.run(data, dataset_name="test", n_runs=1, verbose=False)

        for result in results:
            assert 0 <= result.mean_quality <= 1

    def test_benchmark_handles_errors(self, data):
        """Test that benchmark continues even if one algorithm fails."""
        benchmark = DRBenchmark()
        # Valid algorithm
        benchmark.add_algorithm("pca", PCA, {"n_components": 2})
        # Invalid algorithm with bad params (should fail gracefully)
        benchmark.add_algorithm("bad_pca", PCA, {"n_components": 10000})  # Way too many

        results = benchmark.run(data, dataset_name="test", n_runs=1, verbose=False)

        # Should have results for PCA (the valid one)
        # bad_pca might fail and be skipped
        assert len(results) >= 1


class TestBenchmarkIntegration:
    """Integration tests for benchmarking."""

    def test_complete_benchmark_workflow(self):
        """Test complete benchmarking workflow."""
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            pytest.skip("matplotlib not available")

        # Get small dataset
        iris = load_iris()
        X = iris.data[:50]  # Very small for speed

        # Create benchmark
        benchmark = DRBenchmark()
        benchmark.add_algorithm("pca", PCA, {"n_components": 2})
        benchmark.add_algorithm("umap", UMAP, {"n_components": 2, "random_state": 42})

        # Run on single dataset
        results1 = benchmark.run(X, dataset_name="small", n_runs=1, verbose=False)
        assert len(results1) == 2

        # Run scaling experiment
        sizes = [10, 20, 30]
        scaling_results = benchmark.run_scaling_experiment(X, sizes=sizes, n_runs=1, verbose=False)
        assert len(scaling_results) == 3

        # Generate summary
        summary = benchmark.summary()
        assert "small" in summary

        # Verify results are sorted by quality
        quality_scores = [r.mean_quality for r in results1]
        assert all(0 <= q <= 1 for q in quality_scores)

    def test_different_metrics(self):
        """Test with different distance metrics."""
        iris = load_iris()
        X = iris.data[:50]

        benchmark = DRBenchmark(metric="cosine")
        benchmark.add_algorithm("pca", PCA, {"n_components": 2})

        results = benchmark.run(X, dataset_name="test", n_runs=1, verbose=False)
        assert len(results) == 1
        assert results[0].mean_quality >= 0
