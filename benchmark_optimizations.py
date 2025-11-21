# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy>=2",
#     "scipy>=1.3.1",
#     "scikit-learn>=1.6",
#     "numba>=0.51.2",
#     "pynndescent>=0.5",
#     "tqdm",
#     "matplotlib",
# ]
# ///
"""Benchmark script for all Squeeze dimensionality reduction algorithms.

This benchmark measures:
1. Execution time for each algorithm
2. Trustworthiness at multiple k values
3. Generates visualization of results
"""

import time
from typing import Any

import matplotlib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import trustworthiness

matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

import squeeze as sqz
from squeeze import STRATEGIES, list_strategies

# Default k values for trustworthiness evaluation
K_VALUES = [5, 10, 15, 20, 30, 50]


def compute_trustworthiness_multi_k(
    X_original: np.ndarray, X_embedded: np.ndarray, k_values: list[int] = K_VALUES
) -> dict[int, float]:
    """Compute trustworthiness for multiple k values."""
    return {k: trustworthiness(X_original, X_embedded, n_neighbors=k) for k in k_values}


def benchmark_strategy(
    X: np.ndarray,
    strategy_name: str,
    k_values: list[int] = K_VALUES,
    param_overrides: dict | None = None,
) -> dict[str, Any]:
    """Benchmark a single strategy."""
    param_overrides = param_overrides or {}

    strategy = STRATEGIES.get(strategy_name)

    print(f"\n{'=' * 60}")
    print(f"Testing: {strategy_name.upper()} - {strategy.description}")
    print(f"{'=' * 60}")

    try:
        # Create reducer with overrides
        reducer = strategy.create(**param_overrides)

        # Time the fit_transform
        start = time.time()
        X_embedded = reducer.fit_transform(X)
        elapsed = time.time() - start

        # Compute quality metrics for all k values
        trust_scores = compute_trustworthiness_multi_k(X, X_embedded, k_values)

        print(f"Time: {elapsed:.2f}s")
        print(
            f"Trustworthiness: "
            + ", ".join(f"k={k}: {v:.3f}" for k, v in trust_scores.items())
        )
        print(f"Embedding shape: {X_embedded.shape}")

        return {
            "name": strategy_name,
            "description": strategy.description,
            "category": strategy.category,
            "time": elapsed,
            "trustworthiness": trust_scores,  # Now a dict of k -> score
            "embedding": X_embedded,
            "success": True,
            "error": None,
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            "name": strategy_name,
            "description": strategy.description,
            "category": strategy.category,
            "time": None,
            "trustworthiness": None,
            "embedding": None,
            "success": False,
            "error": str(e),
        }


def plot_results(
    results: list[dict],
    k_values: list[int] = K_VALUES,
    output_file: str = "benchmark_results.png",
):
    """Generate benchmark visualization with trustworthiness vs k plot."""
    successful = [r for r in results if r["success"]]

    if not successful:
        print("No successful results to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    names = [r["name"] for r in successful]
    times = [r["time"] for r in successful]
    categories = [r["category"] for r in successful]

    # Color by category
    category_colors = {
        "linear": "#1f77b4",
        "nonlinear": "#ff7f0e",
        "graph-based": "#2ca02c",
        "diffusion": "#d62728",
        "other": "#9467bd",
    }
    colors = [category_colors.get(c, "#333333") for c in categories]

    # Plot 1: Execution Time (bar chart)
    ax1 = axes[0]
    bars1 = ax1.barh(names, times, color=colors)
    ax1.set_xlabel("Time (seconds)", fontsize=12)
    ax1.set_title("Execution Time", fontsize=14)
    ax1.invert_yaxis()
    for bar, val in zip(bars1, times):
        ax1.text(
            val + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}s",
            va="center",
            fontsize=9,
        )

    # Plot 2: Trustworthiness at k=15 (bar chart)
    ax2 = axes[1]
    trusts_k15 = [
        r["trustworthiness"].get(15, r["trustworthiness"].get(10, 0))
        for r in successful
    ]
    bars2 = ax2.barh(names, trusts_k15, color=colors)
    ax2.set_xlabel("Trustworthiness (k=15)", fontsize=12)
    ax2.set_title("Embedding Quality", fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    for bar, val in zip(bars2, trusts_k15):
        ax2.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
        )

    # Plot 3: Trustworthiness vs k (line plot)
    ax3 = axes[2]

    # Use a colormap for distinct colors per algorithm
    cmap = plt.colormaps["tab10"]

    for idx, result in enumerate(successful):
        trust_dict = result["trustworthiness"]
        ks = sorted(trust_dict.keys())
        scores = [trust_dict[k] for k in ks]
        color = cmap(idx % 10)
        ax3.plot(
            ks,
            scores,
            marker="o",
            label=result["name"],
            color=color,
            linewidth=2,
            markersize=6,
        )

    ax3.set_xlabel("k (neighborhood size)", fontsize=12)
    ax3.set_ylabel("Trustworthiness", fontsize=12)
    ax3.set_title("Trustworthiness vs k", fontsize=14)
    ax3.set_ylim(0.4, 1.0)
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Add legend for categories
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=color, label=cat)
        for cat, color in category_colors.items()
        if cat in categories
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        fontsize=10,
        title="Category",
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def plot_embeddings(
    results: list[dict],
    labels: np.ndarray,
    output_file: str = "embeddings_comparison.png",
):
    """Plot all embeddings in a grid."""
    successful = [r for r in results if r["success"]]

    if not successful:
        return

    n_plots = len(successful)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, result in enumerate(successful):
        ax = axes[idx]
        embedding = result["embedding"]
        trust_k15 = result["trustworthiness"].get(15, 0)

        ax.scatter(
            embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=5, alpha=0.7
        )
        ax.set_title(
            f"{result['name'].upper()}\n(t={result['time']:.2f}s, trust@15={trust_k15:.3f})"
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for idx in range(len(successful), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Embeddings plot saved to: {output_file}")
    plt.close()


def main():
    """Run comprehensive benchmarks for all strategies."""
    print("=" * 60)
    print("Squeeze Dimensionality Reduction Benchmark")
    print("=" * 60)

    # Print available strategies
    print("\nAvailable strategies:")
    print(STRATEGIES.summary())

    # Load dataset
    print("Loading dataset...")
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Evaluating trustworthiness at k = {K_VALUES}")

    # Strategy-specific parameter overrides for faster benchmarking
    param_overrides = {
        "umap": {"n_epochs": 200, "random_state": 42},
        "tsne": {"n_iter": 500, "random_state": 42},
        "mds": {"n_iter": 100},
        "phate": {"t": 5},
        "trimap": {"n_iter": 400, "random_state": 42},
        "pacmap": {"n_iter": 200, "random_state": 42},
    }

    results = []

    # Benchmark each strategy
    for strategy_name in list_strategies():
        overrides = param_overrides.get(strategy_name, {})
        result = benchmark_strategy(X, strategy_name, K_VALUES, overrides)
        results.append(result)

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}\n")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    # Header
    k_headers = " ".join(f"k={k:<5}" for k in K_VALUES)
    print(f"{'Algorithm':<10} {'Time':>8}  {k_headers}")
    print("-" * (20 + 7 * len(K_VALUES)))

    for r in sorted(successful, key=lambda x: -x["trustworthiness"].get(15, 0)):
        trust_vals = " ".join(
            f"{r['trustworthiness'].get(k, 0):.3f} " for k in K_VALUES
        )
        print(f"{r['name']:<10} {r['time']:>7.2f}s  {trust_vals}")

    if failed:
        print(f"\nFailed algorithms:")
        for r in failed:
            print(f"  {r['name']}: {r['error']}")

    # Find best results
    if successful:
        fastest = min(successful, key=lambda x: x["time"])
        best_quality = max(successful, key=lambda x: x["trustworthiness"].get(15, 0))

        print(f"\n{'=' * 60}")
        print("KEY FINDINGS")
        print(f"{'=' * 60}\n")
        print(f"Fastest: {fastest['name']} ({fastest['time']:.2f}s)")
        print(
            f"Best quality (k=15): {best_quality['name']} (trust={best_quality['trustworthiness'].get(15, 0):.4f})"
        )

    # Generate plots
    plot_results(results, K_VALUES)
    plot_embeddings(results, y)


if __name__ == "__main__":
    main()
