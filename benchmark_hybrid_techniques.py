#!/usr/bin/env python
"""Benchmark hybrid dimensionality reduction techniques.

This script evaluates intelligent hybrid combinations of DR algorithms,
comparing them against baseline methods to find optimal combinations.

Hybrid Techniques:
1. PCA(30) → UMAP: Speed + quality pipeline
2. MDS + UMAP Ensemble: Global + local structure blend
3. PCA(50) → t-SNE: Global init + local refinement
4. Multi-scale UMAP: Different neighborhood scales
5. Progressive PaCMAP → UMAP: Fast init + fine refinement
6. MDS + PaCMAP Ensemble: Global + fast blend

Usage:
    python benchmark_hybrid_techniques.py

Output:
    - hybrid_metrics_heatmap.png: Comparison visualization
    - hybrid_results.csv: Raw metric values
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA as SklearnPCA

warnings.filterwarnings("ignore")


@dataclass
class HybridResult:
    """Container for hybrid benchmark results."""

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


def create_hybrid_techniques() -> list[tuple[str, Any]]:
    """Create all hybrid technique configurations."""
    import squeeze as sqz
    from squeeze.composition import DRPipeline, EnsembleDR, ProgressiveDR

    UMAP, PaCMAP, MDS, TSNE = sqz.UMAP, sqz.PaCMAP, sqz.MDS, sqz.TSNE

    hybrids = []

    # ==========================================================================
    # 1. PCA(30) → UMAP Pipeline
    # Rationale: PCA removes noise fast, UMAP refines local structure
    # ==========================================================================
    hybrids.append(
        (
            "PCA(30)→UMAP",
            DRPipeline(
                [
                    ("pca", SklearnPCA(n_components=30)),
                    ("umap", UMAP(n_components=2, random_state=42)),
                ]
            ),
        )
    )

    # ==========================================================================
    # 2. MDS + UMAP Ensemble (Global + Local)
    # Rationale: MDS has best global (0.83), UMAP has best silhouette (0.78)
    # ==========================================================================
    hybrids.append(
        (
            "MDS+UMAP Ensemble",
            EnsembleDR(
                methods=[
                    ("mds", MDS(n_components=2), 0.4),
                    ("umap", UMAP(n_components=2, random_state=42), 0.6),
                ],
                blend_mode="procrustes",
            ),
        )
    )

    # ==========================================================================
    # 3. PCA(50) → t-SNE Pipeline
    # Rationale: PCA preserves global distances, t-SNE refines clusters
    # ==========================================================================
    hybrids.append(
        (
            "PCA(50)→t-SNE",
            DRPipeline(
                [
                    ("pca", SklearnPCA(n_components=50)),
                    ("tsne", TSNE(n_components=2, random_state=42)),
                ]
            ),
        )
    )

    # ==========================================================================
    # 4. Multi-scale UMAP Ensemble
    # Rationale: Different n_neighbors capture different structure scales
    # ==========================================================================
    hybrids.append(
        (
            "Multi-scale UMAP",
            EnsembleDR(
                methods=[
                    (
                        "local",
                        UMAP(n_components=2, n_neighbors=5, random_state=42),
                        0.5,
                    ),
                    (
                        "global",
                        UMAP(n_components=2, n_neighbors=30, random_state=42),
                        0.5,
                    ),
                ],
                blend_mode="procrustes",
            ),
        )
    )

    # ==========================================================================
    # 5. Progressive: PaCMAP → UMAP
    # Rationale: PaCMAP fast initialization, UMAP fine-tuning
    # ==========================================================================
    hybrids.append(
        (
            "Progressive PaCMAP→UMAP",
            ProgressiveDR(
                coarse=PaCMAP(n_components=2),
                fine=UMAP(n_components=2, random_state=42),
                blend_steps=10,
                blend_function="sigmoid",
            ),
        )
    )

    # ==========================================================================
    # 6. MDS + PaCMAP Ensemble (Global + Fast)
    # Rationale: MDS global structure + PaCMAP speed/local
    # ==========================================================================
    hybrids.append(
        (
            "MDS+PaCMAP Ensemble",
            EnsembleDR(
                methods=[
                    ("mds", MDS(n_components=2), 0.5),
                    ("pacmap", PaCMAP(n_components=2), 0.5),
                ],
                blend_mode="procrustes",
            ),
        )
    )

    # ==========================================================================
    # Bonus hybrids based on analysis
    # ==========================================================================

    # 7. PCA(30) → t-SNE (faster t-SNE)
    hybrids.append(
        (
            "PCA(30)→t-SNE",
            DRPipeline(
                [
                    ("pca", SklearnPCA(n_components=30)),
                    ("tsne", TSNE(n_components=2, random_state=42)),
                ]
            ),
        )
    )

    # 8. Triple ensemble: MDS + UMAP + PaCMAP
    hybrids.append(
        (
            "MDS+UMAP+PaCMAP",
            EnsembleDR(
                methods=[
                    ("mds", MDS(n_components=2), 0.3),
                    ("umap", UMAP(n_components=2, random_state=42), 0.4),
                    ("pacmap", PaCMAP(n_components=2), 0.3),
                ],
                blend_mode="procrustes",
            ),
        )
    )

    return hybrids


def create_baseline_techniques() -> list[tuple[str, Any]]:
    """Create baseline techniques for comparison."""
    import squeeze as sqz

    UMAP, PaCMAP, MDS, TSNE, PCA = sqz.UMAP, sqz.PaCMAP, sqz.MDS, sqz.TSNE, sqz.PCA

    baselines = [
        ("UMAP", UMAP(n_components=2, random_state=42)),
        ("t-SNE", TSNE(n_components=2, random_state=42)),
        ("MDS", MDS(n_components=2)),
        ("PaCMAP", PaCMAP(n_components=2)),
        ("PCA", PCA(n_components=2)),
    ]

    return baselines


def run_techniques(
    X: np.ndarray, techniques: list[tuple[str, Any]]
) -> list[HybridResult]:
    """Run all techniques and collect embeddings."""
    results = []

    for name, reducer in techniques:
        print(f"Running {name}...", end=" ", flush=True)
        try:
            start = time.perf_counter()
            embedding = reducer.fit_transform(X)
            elapsed = time.perf_counter() - start
            print(f"done ({elapsed:.2f}s)")
            results.append(
                HybridResult(name=name, embedding=embedding, fit_time=elapsed)
            )
        except Exception as e:
            print(f"FAILED: {e}")

    return results


def compute_metrics(
    X: np.ndarray, y: np.ndarray, results: list[HybridResult]
) -> pd.DataFrame:
    """Compute all evaluation metrics for each technique."""
    from squeeze.evaluation import (
        trustworthiness,
        continuity,
        co_ranking_quality,
        spearman_distance_correlation,
        global_structure_preservation,
        local_density_preservation,
        reconstruction_error,
        clustering_quality,
        classification_accuracy,
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


def create_heatmap(
    df: pd.DataFrame,
    output_path: str = "hybrid_metrics_heatmap.png",
    title: str = "Hybrid DR Techniques: Algorithm vs Metric Comparison",
) -> None:
    """Create a heatmap visualization."""
    algorithms = df["Algorithm"].tolist()

    # Select metric columns
    metric_cols = [col for col in df.columns if col not in ["Algorithm", "Time (s)"]]
    metrics_df = df[metric_cols].copy()

    # Normalize metrics to [0, 1] for color mapping
    normalized = metrics_df.copy()
    for col in metric_cols:
        values = metrics_df[col].values
        min_val = values.min()
        max_val = values.max()

        if max_val > min_val:
            normalized[col] = (values - min_val) / (max_val - min_val)
        else:
            normalized[col] = 0.5

        # Handle silhouette range [-1, 1]
        if col == "Silhouette":
            shifted = (values + 1) / 2
            min_val = shifted.min()
            max_val = shifted.max()
            if max_val > min_val:
                normalized[col] = (shifted - min_val) / (max_val - min_val)
            else:
                normalized[col] = 0.5

    # Create figure
    fig, ax = plt.subplots(figsize=(18, 10))

    # Custom colormap: red -> yellow -> green
    colors = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]
    cmap = LinearSegmentedColormap.from_list("RdYlGn", colors, N=256)

    # Plot heatmap
    heatmap_data = normalized.values
    im = ax.imshow(heatmap_data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_yticks(np.arange(len(algorithms)))
    ax.set_xticklabels(metric_cols, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(algorithms, fontsize=11)

    # Add text annotations
    for i in range(len(algorithms)):
        for j in range(len(metric_cols)):
            value = metrics_df.iloc[i, j]
            norm_value = normalized.iloc[i, j]
            text_color = "white" if norm_value < 0.5 else "black"

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

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Relative Performance", rotation=-90, va="bottom", fontsize=10)

    # Labels
    ax.set_xlabel("Evaluation Metrics", fontsize=12, fontweight="bold")
    ax.set_ylabel("Algorithm", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{title}\n(Green = Best, Red = Worst)", fontsize=14, fontweight="bold", pad=20
    )

    # Grid
    ax.set_xticks(np.arange(len(metric_cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(algorithms) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nHeatmap saved to: {output_path}")
    plt.close()


def create_comparison_chart(
    df: pd.DataFrame, output_path: str = "hybrid_comparison.png"
) -> None:
    """Create a bar chart comparing key metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    key_metrics = [
        ("Trust. (k=15)", "Trustworthiness (k=15)", "higher"),
        ("Spearman", "Spearman Distance Correlation", "higher"),
        ("Global Struct.", "Global Structure Preservation", "higher"),
        ("Silhouette", "Silhouette Score", "higher"),
        ("Classif. Acc.", "Classification Accuracy", "higher"),
        ("Time (s)", "Execution Time (s)", "lower"),
    ]

    # Color hybrids differently from baselines
    colors = []
    for name in df["Algorithm"]:
        if any(x in name for x in ["→", "Ensemble", "Progressive", "Multi-scale"]):
            colors.append("#2ecc71")  # Green for hybrids
        else:
            colors.append("#3498db")  # Blue for baselines

    for ax, (col, title, direction) in zip(axes, key_metrics):
        values = df[col].values
        names = df["Algorithm"].values

        # Sort by value
        if direction == "higher":
            order = np.argsort(values)[::-1]
        else:
            order = np.argsort(values)

        sorted_values = values[order]
        sorted_names = names[order]
        sorted_colors = [colors[i] for i in order]

        bars = ax.barh(range(len(sorted_names)), sorted_values, color=sorted_colors)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.set_xlabel(col)
        ax.set_title(title, fontweight="bold")
        ax.invert_yaxis()

        # Add value labels
        for bar, val in zip(bars, sorted_values):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}" if val < 10 else f"{val:.1f}",
                va="center",
                fontsize=8,
            )

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2ecc71", label="Hybrid"),
        Patch(facecolor="#3498db", label="Baseline"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.suptitle("Hybrid vs Baseline Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Comparison chart saved to: {output_path}")
    plt.close()


def print_analysis(df: pd.DataFrame) -> None:
    """Print analysis of results."""
    print("\n" + "=" * 80)
    print("HYBRID TECHNIQUES ANALYSIS")
    print("=" * 80)

    # Identify hybrids vs baselines
    hybrids = df[df["Algorithm"].str.contains("→|Ensemble|Progressive|Multi-scale")]
    baselines = df[~df["Algorithm"].str.contains("→|Ensemble|Progressive|Multi-scale")]

    print(f"\nHybrids evaluated: {len(hybrids)}")
    print(f"Baselines evaluated: {len(baselines)}")

    # Key metrics to compare
    key_metrics = [
        "Trust. (k=15)",
        "Spearman",
        "Global Struct.",
        "Silhouette",
        "Classif. Acc.",
    ]

    print("\n" + "-" * 80)
    print("BEST PERFORMERS BY METRIC")
    print("-" * 80)

    for metric in key_metrics:
        best_idx = df[metric].idxmax()
        best_name = df.loc[best_idx, "Algorithm"]
        best_val = df.loc[best_idx, metric]

        # Check if hybrid beat best baseline
        best_baseline_idx = baselines[metric].idxmax()
        best_baseline_name = baselines.loc[best_baseline_idx, "Algorithm"]
        best_baseline_val = baselines.loc[best_baseline_idx, metric]

        hybrid_beat = "→|Ensemble|Progressive|Multi-scale" in best_name

        print(f"\n{metric}:")
        print(f"  Best overall: {best_name} ({best_val:.4f})")
        print(f"  Best baseline: {best_baseline_name} ({best_baseline_val:.4f})")
        if hybrid_beat:
            improvement = ((best_val - best_baseline_val) / best_baseline_val) * 100
            print(f"  ✓ Hybrid wins! (+{improvement:.1f}% improvement)")

    # Overall ranking
    print("\n" + "-" * 80)
    print("OVERALL RANKING (by average rank across key metrics)")
    print("-" * 80)

    avg_ranks = {}
    for alg in df["Algorithm"]:
        ranks = []
        for metric in key_metrics:
            sorted_algs = df.sort_values(metric, ascending=False)["Algorithm"].tolist()
            ranks.append(sorted_algs.index(alg) + 1)
        avg_ranks[alg] = np.mean(ranks)

    sorted_overall = sorted(avg_ranks.items(), key=lambda x: x[1])
    for i, (alg, avg_rank) in enumerate(sorted_overall, 1):
        marker = (
            "★"
            if "→" in alg
            or "Ensemble" in alg
            or "Progressive" in alg
            or "Multi-scale" in alg
            else " "
        )
        print(f"  {i:2d}. {marker} {alg}: avg rank = {avg_rank:.2f}")

    # Speed vs quality tradeoff
    print("\n" + "-" * 80)
    print("SPEED vs QUALITY TRADEOFF (Pareto optimal)")
    print("-" * 80)

    # Compute quality score (average of key metrics excluding time)
    df["Quality"] = df[key_metrics].mean(axis=1)

    # Find Pareto optimal points
    pareto = []
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i != j:
                # Other dominates if faster AND better quality
                if (
                    other["Time (s)"] <= row["Time (s)"]
                    and other["Quality"] >= row["Quality"]
                ):
                    if (
                        other["Time (s)"] < row["Time (s)"]
                        or other["Quality"] > row["Quality"]
                    ):
                        dominated = True
                        break
        if not dominated:
            pareto.append(row["Algorithm"])

    print("\nPareto optimal (best speed-quality tradeoff):")
    for alg in pareto:
        row = df[df["Algorithm"] == alg].iloc[0]
        marker = (
            "★"
            if "→" in alg
            or "Ensemble" in alg
            or "Progressive" in alg
            or "Multi-scale" in alg
            else " "
        )
        print(
            f"  {marker} {alg}: Quality={row['Quality']:.3f}, Time={row['Time (s)']:.2f}s"
        )


def main():
    """Main entry point."""
    print("=" * 80)
    print("HYBRID DIMENSIONALITY REDUCTION BENCHMARK")
    print("=" * 80)
    print()

    # Load data
    X, y = load_data()
    print()

    # Create techniques
    print("Creating hybrid techniques...")
    hybrids = create_hybrid_techniques()
    print(f"  {len(hybrids)} hybrid techniques")

    print("\nCreating baseline techniques...")
    baselines = create_baseline_techniques()
    print(f"  {len(baselines)} baseline techniques")

    all_techniques = baselines + hybrids
    print(f"\nTotal: {len(all_techniques)} techniques to evaluate")
    print()

    # Run all techniques
    print("Running techniques...")
    print("-" * 40)
    results = run_techniques(X, all_techniques)
    print()

    # Compute metrics
    print("Computing evaluation metrics...")
    print("-" * 40)
    df = compute_metrics(X, y, results)
    print()

    # Save results
    df.to_csv("hybrid_results.csv", index=False)
    print("Raw results saved to: hybrid_results.csv")

    # Create visualizations
    print("\nGenerating visualizations...")
    create_heatmap(
        df,
        "hybrid_metrics_heatmap.png",
        "Hybrid DR Techniques: Algorithm vs Metric Comparison",
    )
    create_comparison_chart(df, "hybrid_comparison.png")

    # Print analysis
    print_analysis(df)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
