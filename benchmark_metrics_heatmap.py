#!/usr/bin/env python
"""Generate a heatmap comparison of all DR algorithms against all evaluation metrics.

This script runs all dimensionality reduction algorithms on the sklearn Digits dataset
and computes comprehensive evaluation metrics, then visualizes the results as a heatmap
with green indicating good performance and red indicating poor performance.

Usage:
    python benchmark_metrics_heatmap.py

Output:
    - metrics_heatmap.png: Heatmap visualization of all algorithms vs metrics
    - metrics_results.csv: Raw metric values for further analysis
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import load_digits

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


@dataclass
class AlgorithmResult:
    """Container for algorithm benchmark results."""

    name: str
    embedding: np.ndarray
    fit_time: float


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the sklearn Digits dataset."""
    print("Loading sklearn Digits dataset...")
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64)
    print(f"  Shape: {X.shape} ({len(np.unique(y))} classes)")
    return X, y


def get_algorithms() -> list[tuple[str, object]]:
    """Get list of DR algorithms to benchmark."""
    import squeeze as sqz

    algorithms = [("UMAP", sqz.UMAP(n_components=2, random_state=42))]

    # Try to import Rust-based algorithms
    try:
        if sqz.PCA is not None:
            algorithms.append(("PCA", sqz.PCA(n_components=2)))
    except (ImportError, TypeError, AttributeError):
        pass

    try:
        if sqz.TSNE is not None:
            algorithms.append(("t-SNE", sqz.TSNE(n_components=2, random_state=42)))
    except (ImportError, TypeError, AttributeError):
        pass

    try:
        if sqz.MDS is not None:
            algorithms.append(("MDS", sqz.MDS(n_components=2)))
    except (ImportError, TypeError, AttributeError):
        pass

    try:
        if sqz.Isomap is not None:
            algorithms.append(("Isomap", sqz.Isomap(n_components=2, n_neighbors=15)))
    except (ImportError, TypeError, AttributeError):
        pass

    try:
        if sqz.LLE is not None:
            algorithms.append(("LLE", sqz.LLE(n_components=2, n_neighbors=15)))
    except (ImportError, TypeError, AttributeError):
        pass

    try:
        if sqz.PHATE is not None:
            algorithms.append(("PHATE", sqz.PHATE(n_components=2)))
    except (ImportError, TypeError, AttributeError):
        pass

    try:
        if sqz.TriMap is not None:
            algorithms.append(("TriMap", sqz.TriMap(n_components=2)))
    except (ImportError, TypeError, AttributeError):
        pass

    try:
        if sqz.PaCMAP is not None:
            algorithms.append(("PaCMAP", sqz.PaCMAP(n_components=2)))
    except (ImportError, TypeError, AttributeError):
        pass

    return algorithms


def run_algorithms(
    X: np.ndarray, algorithms: list[tuple[str, object]]
) -> list[AlgorithmResult]:
    """Run all algorithms and collect embeddings."""
    results = []

    for name, reducer in algorithms:
        print(f"Running {name}...", end=" ", flush=True)
        try:
            start = time.perf_counter()
            embedding = reducer.fit_transform(X)
            elapsed = time.perf_counter() - start
            print(f"done ({elapsed:.2f}s)")
            results.append(
                AlgorithmResult(name=name, embedding=embedding, fit_time=elapsed)
            )
        except Exception as e:
            print(f"FAILED: {e}")

    return results


def compute_metrics(
    X: np.ndarray, y: np.ndarray, results: list[AlgorithmResult]
) -> pd.DataFrame:
    """Compute all evaluation metrics for each algorithm."""
    from squeeze.evaluation import (
        classification_accuracy,
        clustering_quality,
        co_ranking_quality,
        continuity,
        global_structure_preservation,
        local_density_preservation,
        reconstruction_error,
        spearman_distance_correlation,
        trustworthiness,
    )

    metrics_data = []

    for result in results:
        print(f"Computing metrics for {result.name}...", end=" ", flush=True)

        X_reduced = result.embedding

        try:
            # Local structure metrics
            T_5 = trustworthiness(X, X_reduced, k=5)
            T_15 = trustworthiness(X, X_reduced, k=15)
            T_30 = trustworthiness(X, X_reduced, k=30)
            C_15 = continuity(X, X_reduced, k=15)
            Q_15 = co_ranking_quality(X, X_reduced, k=15)

            # Global structure metrics
            spearman = spearman_distance_correlation(X, X_reduced)
            global_struct = global_structure_preservation(X, X_reduced, y)
            density = local_density_preservation(X, X_reduced, k=15)

            # Reconstruction
            recon = reconstruction_error(X, X_reduced)
            r2 = recon["r2"]

            # Downstream tasks
            clust = clustering_quality(X_reduced, labels_true=y)
            silhouette = clust["silhouette_score"]
            ari = clust["adjusted_rand_index"]
            nmi = clust["normalized_mutual_info"]

            classif = classification_accuracy(X_reduced, y, cv=5)
            accuracy = classif["mean_accuracy"]

            metrics_data.append(
                {
                    "Algorithm": result.name,
                    "Trust. (k=5)": T_5,
                    "Trust. (k=15)": T_15,
                    "Trust. (k=30)": T_30,
                    "Continuity": C_15,
                    "Co-ranking": Q_15,
                    "Spearman": spearman,
                    "Global Struct.": global_struct,
                    "Density Pres.": density,
                    "Reconstr. R²": r2,
                    "Silhouette": silhouette,
                    "Adj. Rand Idx": ari,
                    "Norm. MI": nmi,
                    "Classif. Acc.": accuracy,
                    "Time (s)": result.fit_time,
                }
            )
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")

    return pd.DataFrame(metrics_data)


def create_heatmap(df: pd.DataFrame, output_path: str = "metrics_heatmap.png") -> None:
    """Create a heatmap visualization of algorithms vs metrics."""
    # Separate algorithm names and metrics
    algorithms = df["Algorithm"].tolist()

    # Select metric columns (exclude Algorithm and Time)
    metric_cols = [col for col in df.columns if col not in ["Algorithm", "Time (s)"]]
    metrics_df = df[metric_cols].copy()

    # Define which metrics are "higher is better" vs "lower is better"
    # All our metrics are "higher is better" except we need to handle ranges
    higher_is_better = {
        "Trust. (k=5)": True,
        "Trust. (k=15)": True,
        "Trust. (k=30)": True,
        "Continuity": True,
        "Co-ranking": True,
        "Spearman": True,
        "Global Struct.": True,
        "Density Pres.": True,
        "Reconstr. R²": True,
        "Silhouette": True,  # Range [-1, 1], higher is better
        "Adj. Rand Idx": True,
        "Norm. MI": True,
        "Classif. Acc.": True,
    }

    # Normalize metrics to [0, 1] for color mapping
    normalized = metrics_df.copy()
    for col in metric_cols:
        values = metrics_df[col].values
        min_val = values.min()
        max_val = values.max()

        if max_val > min_val:
            normalized[col] = (values - min_val) / (max_val - min_val)
        else:
            normalized[col] = 0.5  # All same value

        # For silhouette, shift from [-1, 1] to [0, 1] before normalizing
        if col == "Silhouette":
            shifted = (values + 1) / 2  # Now in [0, 1]
            min_val = shifted.min()
            max_val = shifted.max()
            if max_val > min_val:
                normalized[col] = (shifted - min_val) / (max_val - min_val)
            else:
                normalized[col] = 0.5

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))

    # Create custom colormap: red -> yellow -> green
    colors = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]
    cmap = LinearSegmentedColormap.from_list("RdYlGn", colors, N=256)

    # Create heatmap data
    heatmap_data = normalized.values

    # Plot heatmap
    im = ax.imshow(heatmap_data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_yticks(np.arange(len(algorithms)))

    # Set tick labels
    ax.set_xticklabels(metric_cols, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(algorithms, fontsize=11)

    # Add text annotations with actual values
    for i in range(len(algorithms)):
        for j in range(len(metric_cols)):
            value = metrics_df.iloc[i, j]
            norm_value = normalized.iloc[i, j]

            # Choose text color based on background brightness
            text_color = "white" if norm_value < 0.5 else "black"

            # Format value
            if abs(value) < 0.01:
                text = f"{value:.3f}"
            elif abs(value) < 1:
                text = f"{value:.2f}"
            else:
                text = f"{value:.1f}"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
                fontweight="bold",
            )

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel(
        "Relative Performance (within metric)", rotation=-90, va="bottom", fontsize=10
    )

    # Labels and title
    ax.set_xlabel("Evaluation Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Algorithm", fontsize=12, fontweight="bold")
    ax.set_title(
        "Dimensionality Reduction: Algorithm vs Metric Comparison\n"
        "(Green = Best, Red = Worst, relative to other algorithms)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Add grid
    ax.set_xticks(np.arange(len(metric_cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(algorithms) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nHeatmap saved to: {output_path}")
    plt.close()


def create_summary_table(df: pd.DataFrame) -> None:
    """Print a summary table of rankings."""
    print("\n" + "=" * 80)
    print("ALGORITHM RANKINGS BY METRIC")
    print("=" * 80)

    metric_cols = [col for col in df.columns if col not in ["Algorithm", "Time (s)"]]

    # For each metric, rank algorithms
    rankings = {}
    for col in metric_cols:
        sorted_df = df.sort_values(col, ascending=False)
        rankings[col] = sorted_df["Algorithm"].tolist()

    # Print rankings
    for col in metric_cols:
        print(f"\n{col}:")
        for i, alg in enumerate(rankings[col], 1):
            value = df[df["Algorithm"] == alg][col].values[0]
            print(f"  {i}. {alg}: {value:.3f}")

    # Compute overall ranking (average rank across metrics)
    print("\n" + "=" * 80)
    print("OVERALL RANKING (by average rank across all metrics)")
    print("=" * 80)

    avg_ranks = {}
    for alg in df["Algorithm"]:
        ranks = []
        for col in metric_cols:
            sorted_algs = df.sort_values(col, ascending=False)["Algorithm"].tolist()
            ranks.append(sorted_algs.index(alg) + 1)
        avg_ranks[alg] = np.mean(ranks)

    sorted_overall = sorted(avg_ranks.items(), key=lambda x: x[1])
    for i, (alg, avg_rank) in enumerate(sorted_overall, 1):
        print(f"  {i}. {alg}: avg rank = {avg_rank:.2f}")


def main():
    """Main entry point."""
    print("=" * 80)
    print("DIMENSIONALITY REDUCTION METRICS BENCHMARK")
    print("=" * 80)
    print()

    # Load data
    X, y = load_data()
    print()

    # Get algorithms
    algorithms = get_algorithms()
    print(f"Benchmarking {len(algorithms)} algorithms:")
    for name, _ in algorithms:
        print(f"  - {name}")
    print()

    # Run algorithms
    print("Running algorithms...")
    results = run_algorithms(X, algorithms)
    print()

    # Compute metrics
    print("Computing evaluation metrics...")
    df = compute_metrics(X, y, results)
    print()

    # Save raw results
    df.to_csv("metrics_results.csv", index=False)
    print("Raw results saved to: metrics_results.csv")

    # Create heatmap
    print("\nGenerating heatmap visualization...")
    create_heatmap(df)

    # Print summary
    create_summary_table(df)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
