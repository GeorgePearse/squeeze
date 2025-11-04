"""Benchmarking framework for dimensionality reduction algorithms.

Provides tools for systematic benchmarking of DR algorithms, measuring
both embedding quality and computational efficiency to visualize
quality vs speed trade-offs.

The benchmark results are visualized as scatter plots where:
- X-axis: Computation time (seconds, log scale)
- Y-axis: Embedding quality (trustworthiness + continuity / 2)
- Each point: One algorithm/parameter configuration
- Color/marker: Different algorithms

Example:
    >>> from umap.benchmark import DRBenchmark
    >>> from sklearn.datasets import load_digits
    >>>
    >>> X, y = load_digits(return_X_y=True)
    >>> benchmark = DRBenchmark()
    >>> benchmark.add_algorithm('umap', UMAP, dict(n_neighbors=15))
    >>> benchmark.add_algorithm('pca', PCA, dict(n_components=2))
    >>> results = benchmark.run(X, n_runs=3)
    >>> benchmark.plot_quality_vs_speed(results, 'digits_benchmark.png')

Author: UMAP development team
License: BSD 3 clause
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import numpy as np
from sklearn.datasets import load_digits, load_iris

from umap.metrics import DREvaluator


class AlgorithmConfig:
    """Configuration for a dimensionality reduction algorithm.

    Parameters
    ----------
    name : str
        Display name for the algorithm.

    algorithm_class : type
        Class of the algorithm (e.g., UMAP, PCA).

    params : dict
        Parameters to pass to the algorithm.

    color : str, optional
        Color for plotting.

    marker : str, optional
        Marker style for plotting.
    """

    def __init__(
        self,
        name: str,
        algorithm_class: type,
        params: dict[str, Any],
        color: Optional[str] = None,
        marker: Optional[str] = None,
    ):
        """Initialize algorithm configuration."""
        self.name = name
        self.algorithm_class = algorithm_class
        self.params = params
        self.color = color or self._default_color(name)
        self.marker = marker or "o"

    @staticmethod
    def _default_color(name: str) -> str:
        """Get default color based on algorithm name."""
        colors = {
            "umap": "#1f77b4",  # blue
            "tsne": "#ff7f0e",  # orange
            "pca": "#2ca02c",   # green
            "pacmap": "#d62728",  # red
            "densmap": "#9467bd",  # purple
            "trimap": "#8c564b",  # brown
            "parametric_umap": "#e377c2",  # pink
            "phate": "#7f7f7f",  # gray
        }
        for key, color in colors.items():
            if key.lower() in name.lower():
                return color
        return "#1f77b4"  # default blue


class BenchmarkResult:
    """Results from benchmarking a single algorithm.

    Attributes
    ----------
    config : AlgorithmConfig
        Algorithm configuration.

    times : list of float
        Computation times for each run.

    quality_scores : list of float
        Quality scores (trustworthiness + continuity) / 2.

    embedding : ndarray
        Final embedding.
    """

    def __init__(
        self,
        config: AlgorithmConfig,
        times: list[float],
        quality_scores: list[float],
        embedding: np.ndarray,
    ):
        """Initialize benchmark result."""
        self.config = config
        self.times = times
        self.quality_scores = quality_scores
        self.embedding = embedding

        # Compute statistics
        self.mean_time = np.mean(times)
        self.std_time = np.std(times) if len(times) > 1 else 0.0
        self.mean_quality = np.mean(quality_scores)
        self.std_quality = np.std(quality_scores) if len(quality_scores) > 1 else 0.0

    def __repr__(self) -> str:
        """String representation of result."""
        return (
            f"{self.config.name}: "
            f"time={self.mean_time:.3f}±{self.std_time:.3f}s, "
            f"quality={self.mean_quality:.3f}±{self.std_quality:.3f}"
        )


class DRBenchmark:
    """Benchmarking framework for dimensionality reduction algorithms.

    Systematically measures quality vs speed trade-offs across different
    algorithms and parameter configurations.

    Attributes
    ----------
    algorithms : dict
        Registered algorithms as {name: AlgorithmConfig}.

    results : dict
        Benchmark results as {dataset_name: [BenchmarkResult]}.
    """

    def __init__(self, metric: str = "euclidean"):
        """Initialize benchmarking framework.

        Parameters
        ----------
        metric : str, default='euclidean'
            Distance metric for quality evaluation.
        """
        self.algorithms: dict[str, AlgorithmConfig] = {}
        self.results: dict[str, list[BenchmarkResult]] = {}
        self.metric = metric
        self.evaluator = DREvaluator(k=15)

    def add_algorithm(
        self,
        name: str,
        algorithm_class: type,
        params: dict[str, Any],
        color: Optional[str] = None,
        marker: Optional[str] = None,
    ) -> DRBenchmark:
        """Register an algorithm for benchmarking.

        Parameters
        ----------
        name : str
            Display name for algorithm.

        algorithm_class : type
            Algorithm class (e.g., UMAP, PCA).

        params : dict
            Parameters to pass to algorithm.

        color : str, optional
            Color for plotting.

        marker : str, optional
            Marker style for plotting.

        Returns
        -------
        self : DRBenchmark
        """
        config = AlgorithmConfig(name, algorithm_class, params, color, marker)
        self.algorithms[name] = config
        return self

    def run(
        self,
        X: np.ndarray,
        dataset_name: str = "default",
        n_runs: int = 1,
        verbose: bool = True,
    ) -> list[BenchmarkResult]:
        """Run benchmarks on dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data.

        dataset_name : str, default='default'
            Name of dataset (for results tracking).

        n_runs : int, default=1
            Number of times to run each algorithm.

        verbose : bool, default=True
            Print progress information.

        Returns
        -------
        results : list of BenchmarkResult
            Benchmark results for each algorithm.
        """
        results = []

        for name, config in self.algorithms.items():
            if verbose:
                print(f"Benchmarking {name}...", end=" ", flush=True)

            times = []
            quality_scores = []

            for run in range(n_runs):
                try:
                    # Create fresh instance
                    algorithm = config.algorithm_class(**config.params)

                    # Time the fit_transform
                    start_time = time.time()
                    X_embedded = algorithm.fit_transform(X)
                    elapsed = time.time() - start_time
                    times.append(elapsed)

                    # Evaluate quality
                    self.evaluator.evaluate(X, X_embedded, metric=self.metric)
                    quality = (
                        self.evaluator.metrics_["trustworthiness"]
                        + self.evaluator.metrics_["continuity"]
                    ) / 2
                    quality_scores.append(quality)

                except Exception as e:
                    if verbose:
                        print(f"\nError: {e}")
                    continue

            if quality_scores:  # Only add if at least one run succeeded
                result = BenchmarkResult(config, times, quality_scores, X_embedded)
                results.append(result)

                if verbose:
                    print(f"✓ ({result.mean_time:.3f}s, quality={result.mean_quality:.3f})")

        self.results[dataset_name] = results
        return results

    def run_scaling_experiment(
        self,
        X_base: np.ndarray,
        sizes: Optional[list[int]] = None,
        n_runs: int = 1,
        verbose: bool = True,
    ) -> dict[int, list[BenchmarkResult]]:
        """Run benchmarks on different dataset sizes (scaling experiment).

        Parameters
        ----------
        X_base : ndarray, shape (n_samples, n_features)
            Base dataset (will be subsampled).

        sizes : list of int, optional
            Sample sizes to test. If None, uses [100, 500, 1000, 5000].

        n_runs : int, default=1
            Number of runs per size.

        verbose : bool, default=True
            Print progress.

        Returns
        -------
        results : dict
            Results as {size: [BenchmarkResult]}.
        """
        if sizes is None:
            sizes = [100, 500, 1000, 5000]

        all_results = {}
        n_samples = X_base.shape[0]

        for size in sizes:
            if size > n_samples:
                if verbose:
                    print(f"Skipping size {size} (exceeds dataset size {n_samples})")
                continue

            if verbose:
                print(f"\nTesting size: {size}")

            # Sample subset
            indices = np.random.choice(n_samples, size=size, replace=False)
            X_subset = X_base[indices]

            # Benchmark on this size
            results = self.run(X_subset, dataset_name=f"size_{size}", n_runs=n_runs, verbose=verbose)
            all_results[size] = results

        return all_results

    def plot_quality_vs_speed(
        self,
        results: Optional[list[BenchmarkResult]] = None,
        output_file: Optional[str] = None,
        title: str = "Embedding Quality vs Computation Speed",
        log_scale: bool = True,
    ) -> None:
        """Plot quality vs speed trade-off.

        Parameters
        ----------
        results : list of BenchmarkResult, optional
            Results to plot. If None, plots all results from last run.

        output_file : str, optional
            File path to save plot. If None, displays interactively.

        title : str, default='Embedding Quality vs Computation Speed'
            Plot title.

        log_scale : bool, default=True
            Use logarithmic scale for time axis.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            msg = "matplotlib required for plotting"
            raise ImportError(msg) from None

        if results is None:
            # Use results from last run
            if not self.results:
                msg = "No results to plot. Run benchmarks first."
                raise ValueError(msg)
            results = list(self.results.values())[-1]

        fig, ax = plt.subplots(figsize=(10, 6))

        for result in results:
            ax.scatter(
                result.mean_time,
                result.mean_quality,
                s=200,
                color=result.config.color,
                marker=result.config.marker,
                alpha=0.7,
                label=result.config.name,
                edgecolors="black",
                linewidth=1.5,
            )

            # Add error bars
            if result.std_time > 0 or result.std_quality > 0:
                ax.errorbar(
                    result.mean_time,
                    result.mean_quality,
                    xerr=result.std_time,
                    yerr=result.std_quality,
                    fmt="none",
                    color=result.config.color,
                    alpha=0.3,
                    capsize=5,
                )

        # Formatting
        if log_scale:
            ax.set_xscale("log")
        ax.set_xlabel("Computation Time (seconds)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Embedding Quality (Trustworthiness + Continuity) / 2", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {output_file}")
        else:
            plt.show()

        plt.close()

    def summary(self) -> str:
        """Get summary of all benchmark results.

        Returns
        -------
        summary : str
            Formatted summary text.
        """
        lines = ["Dimensionality Reduction Benchmark Summary", "=" * 50]

        for dataset_name, results in self.results.items():
            lines.append(f"\nDataset: {dataset_name}")
            lines.append("-" * 50)

            for result in sorted(results, key=lambda r: r.mean_quality, reverse=True):
                lines.append(str(result))

        return "\n".join(lines)


def benchmark_standard_datasets() -> None:
    """Run benchmarks on standard datasets and generate plots."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    from umap import UMAP

    # Setup algorithms
    benchmark = DRBenchmark()
    benchmark.add_algorithm("UMAP (default)", UMAP, {})
    benchmark.add_algorithm("UMAP (fast)", UMAP, {"n_neighbors": 10, "n_epochs": 100})
    benchmark.add_algorithm("UMAP (accurate)", UMAP, {"n_neighbors": 15, "min_dist": 0.01})
    benchmark.add_algorithm("PCA", PCA, {"n_components": 2})
    benchmark.add_algorithm("t-SNE", TSNE, {"perplexity": 30})

    # Benchmark on Iris
    print("Benchmarking on Iris dataset...")
    iris = load_iris()
    results_iris = benchmark.run(iris.data, dataset_name="iris", n_runs=3, verbose=True)
    benchmark.plot_quality_vs_speed(results_iris, "iris_benchmark.png", "Iris Dataset: Quality vs Speed")

    # Benchmark on Digits with scaling
    print("\nBenchmarking on Digits dataset (scaling experiment)...")
    digits = load_digits()
    scaling_results = benchmark.run_scaling_experiment(digits.data, n_runs=2, verbose=True)

    # Plot scaling results
    for size, results in scaling_results.items():
        benchmark.plot_quality_vs_speed(
            results, f"digits_size_{size}_benchmark.png", f"Digits ({size} samples): Quality vs Speed"
        )

    print("\n" + benchmark.summary())
