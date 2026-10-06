"""Generate a heatmap comparison of all DR algorithms against all evaluation metrics.

Run all dimensionality reduction algorithms on Digits or a Fashion-MNIST sample.
The benchmark
computes comprehensive evaluation metrics, then visualizes the results as a heatmap
with green indicating good performance and red indicating poor performance.

Usage:
    python benchmark_metrics_heatmap.py

Output:
    - metrics_heatmap.png: Heatmap visualization of all algorithms vs metrics
    - metrics_results.csv: Raw metric values for further analysis
"""

# ruff: noqa: T201, N803, N806, PLC0415
# This CLI retains sklearn X/y notation and lazy optional evaluator imports.

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import trustworthiness as rank_trustworthiness
from threadpoolctl import threadpool_limits

from scripts.benchmark_datasets import load_benchmark_data


@dataclass
class AlgorithmResult:
    """Container for algorithm benchmark results."""

    name: str
    embedding: np.ndarray
    fit_time: float


def get_algorithms() -> list[tuple[str, object]]:
    """Get list of DR algorithms to benchmark."""
    import squeeze as sqz

    # Fail explicitly if the extension is stale; do not silently omit rows.
    return [
        ("UMAP", sqz.UMAP(n_components=2, n_epochs=200, n_jobs=1, random_state=42)),
        ("PCA", sqz.PCA(n_components=2)),
        ("t-SNE", sqz.TSNE(n_components=2, random_state=42)),
        ("MDS", sqz.MDS(n_components=2, random_state=42)),
        ("Isomap", sqz.Isomap(n_components=2, n_neighbors=15)),
        ("LLE", sqz.LLE(n_components=2, n_neighbors=15)),
        ("PHATE", sqz.PHATE(n_components=2, random_state=42)),
        ("TriMap", sqz.TriMap(n_components=2, random_state=42)),
        ("PaCMAP", sqz.PaCMAP(n_components=2, random_state=42)),
        ("NeighborMap", sqz.NeighborMap(n_neighbors=15, n_epochs=160, random_state=42)),
        ("SpectralMap", sqz.SpectralMap(n_neighbors=15, n_iter=128, random_state=42)),
    ]


def run_algorithms(
    X: np.ndarray,
    algorithms: list[tuple[str, object]],
) -> list[AlgorithmResult]:
    """Run all algorithms and collect embeddings."""
    results = []

    for name, reducer in algorithms:
        print(f"Running {name}...", end=" ", flush=True)
        try:
            reducer.fit_transform(X)  # Full-data warmup, excluded from timing.
            start = time.perf_counter()
            embedding = reducer.fit_transform(X)
            elapsed = time.perf_counter() - start
            print(f"done ({elapsed:.2f}s)")
            results.append(
                AlgorithmResult(name=name, embedding=embedding, fit_time=elapsed),
            )
        except Exception as e:
            print(f"FAILED: {e}")
            raise

    return results


def compute_metrics(
    X: np.ndarray,
    y: np.ndarray,
    results: list[AlgorithmResult],
) -> pd.DataFrame:
    """Compute all evaluation metrics for each algorithm."""
    from squeeze.evaluation import (
        classification_accuracy,
        clustering_quality,
        global_structure_preservation,
        local_density_preservation,
        reconstruction_error,
        spearman_distance_correlation,
    )
    from squeeze.evaluation import (
        trustworthiness as neighbor_recall,
    )

    metrics_data = []

    for result in results:
        print(f"Computing metrics for {result.name}...", end=" ", flush=True)

        X_reduced = result.embedding

        try:
            # Local structure metrics
            T_5 = rank_trustworthiness(X, X_reduced, n_neighbors=5)
            T_15 = rank_trustworthiness(X, X_reduced, n_neighbors=15)
            T_30 = rank_trustworthiness(X, X_reduced, n_neighbors=30)
            C_15 = rank_trustworthiness(X_reduced, X, n_neighbors=15)
            Q_15 = neighbor_recall(X, X_reduced, k=15)

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
                    "Neighbor Recall": Q_15,
                    "Spearman": spearman,
                    "Global Struct.": global_struct,
                    "Density Pres.": density,
                    "Reconstr. R²": r2,
                    "Silhouette": silhouette,
                    "Adj. Rand Idx": ari,
                    "Norm. MI": nmi,
                    "Transductive Acc.": accuracy,
                    "Time (s)": result.fit_time,
                },
            )
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")
            raise

    return pd.DataFrame(metrics_data)


def create_heatmap(
    df: pd.DataFrame,
    output_path: str = "metrics_heatmap.png",
    dataset_title: str = "Digits",
) -> None:
    """Create a heatmap visualization of algorithms vs metrics."""
    # Separate algorithm names and metrics
    algorithms = df["Algorithm"].tolist()

    # Runtime is the only lower-is-better column.
    metric_cols = [col for col in df.columns if col != "Algorithm"]
    metrics_df = df[metric_cols].copy()

    # Normalize metrics to [0, 1] for color mapping
    normalized = metrics_df.copy()
    for col in metric_cols:
        values = metrics_df[col].to_numpy()
        if col == "Time (s)":
            values = -np.log10(values)  # Faster is greener; spans orders of magnitude.
        min_val = values.min()
        max_val = values.max()

        if max_val > min_val:
            normalized[col] = (values - min_val) / (max_val - min_val)
        else:
            normalized[col] = 0.5  # All same value

    # Create figure
    _fig, ax = plt.subplots(figsize=(18, max(8, len(algorithms) * 0.55 + 3)))

    # Create custom colormap: red -> yellow -> green
    colors = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]
    cmap = LinearSegmentedColormap.from_list("RdYlGn", colors, N=256)

    # Create heatmap data
    heatmap_data = normalized.to_numpy()

    # Plot heatmap
    im = ax.imshow(heatmap_data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_yticks(np.arange(len(algorithms)))

    # Set tick labels
    ax.set_xticklabels(metric_cols, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(algorithms, fontsize=11)
    for label in ax.get_yticklabels():
        if label.get_text() in {"NeighborMap", "SpectralMap"}:
            label.set_fontweight("bold")

    # Add text annotations with actual values
    for i in range(len(algorithms)):
        for j in range(len(metric_cols)):
            value = metrics_df.iloc[i, j]
            norm_value = normalized.iloc[i, j]

            red, green, blue, _ = cmap(norm_value)
            text_color = (
                "black"
                if 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.5  # noqa: PLR2004 - luminance threshold
                else "white"
            )
            text = f"{value:.3g}" if metric_cols[j] == "Time (s)" else f"{value:.3f}"

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
        "Relative Performance (within metric)",
        rotation=-90,
        va="bottom",
        fontsize=10,
    )

    # Labels and title
    ax.set_xlabel("Evaluation Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Algorithm", fontsize=12, fontweight="bold")
    ax.set_title(
        "Dimensionality Reduction: Algorithm vs Metric Comparison\n"
        f"{dataset_title} · seed 42 · warmed fits · one thread per library\n"
        "Green = best within each metric; runtime uses a reversed log scale",
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
            value = df[df["Algorithm"] == alg][col].to_numpy()[0]
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


def main() -> None:
    """Run the chosen dataset and save its heatmap plus raw evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["digits", "fashion-mnist"],
        default="digits",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Fashion-MNIST sample count; default 2000",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or (
        Path()
        if args.dataset == "digits"
        else Path("working_docs/heatmap_refresh/fashion-mnist")
    )
    output.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("DIMENSIONALITY REDUCTION METRICS BENCHMARK")
    print("=" * 80)
    print()

    # Load data
    X, y, provenance = load_benchmark_data(
        args.dataset,
        args.samples,
        cache_dir=args.cache_dir,
    )
    print(f"Dataset: {provenance['name']} — {X.shape}")
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

    artifact_dir = Path("working_docs/heatmap_refresh") / args.dataset
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_dir / "embeddings.npz",
        **{r.name: r.embedding for r in results},
    )
    (artifact_dir / "protocol.json").write_text(
        json.dumps(
            {
                "dataset": provenance,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "numpy",
                        "scipy",
                        "scikit-learn",
                        "numba",
                        "pandas",
                        "matplotlib",
                    )
                },
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes(),
                ).hexdigest(),
                "seed": 42,
                "warmup": "one full-data fit per algorithm, excluded",
                "threads": 1,
                "umap_epochs": 200,
                "neighbor_map_epochs": 160,
                "spectral_map_iterations": 128,
                "classification": (
                    "5-fold RandomForest on full-data embeddings; "
                    "transductive, not held-out embedding evaluation"
                ),
            },
            indent=2,
        )
        + "\n",
    )

    # Save raw results
    df.to_csv(output / "metrics_results.csv", index=False)
    print("Raw results saved to: metrics_results.csv")

    # Create heatmap
    print("\nGenerating heatmap visualization...")
    create_heatmap(
        df,
        str(output / "metrics_heatmap.png"),
        f"{provenance['name']} ({len(X):,} samples)",
    )

    # Print summary
    create_summary_table(df)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    with threadpool_limits(limits=1):
        main()
