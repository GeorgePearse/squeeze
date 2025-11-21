#!/usr/bin/env python
"""Generate embeddings for the interactive demos.

This script generates 2D embeddings using various Squeeze algorithms
and saves them as JSON files for the web visualization.

Usage:
    python scripts/generate_embeddings.py

Output:
    public/data/{dataset}_{algorithm}.json
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Add parent directory to path for squeeze imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import squeeze as sqz

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "public" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Algorithms to generate
ALGORITHMS = {
    "umap": lambda: sqz.UMAP(n_components=2, random_state=42, n_neighbors=15),
    "tsne": lambda: sqz.TSNE(n_components=2, random_state=42),
    "pacmap": lambda: sqz.PaCMAP(n_components=2, random_state=42),
    "pca": lambda: sqz.PCA(n_components=2),
    "mds": lambda: sqz.MDS(n_components=2),
}


def load_digits_dataset():
    """Load sklearn digits dataset."""
    print("Loading Digits dataset...")
    digits = load_digits()
    return digits.data.astype(np.float64), digits.target.tolist(), "digits"


def load_fashion_mnist():
    """Load Fashion MNIST (subset)."""
    print("Loading Fashion MNIST...")
    try:
        from sklearn.datasets import fetch_openml

        fashion = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
        # Take first 10k samples
        X = fashion.data[:10000].astype(np.float64)
        y = fashion.target[:10000].astype(int).tolist()
        return X, y, "fashion"
    except Exception as e:
        print(f"  Could not load Fashion MNIST: {e}")
        return None, None, None


def load_cifar10():
    """Load CIFAR-10 (subset)."""
    print("Loading CIFAR-10...")
    try:
        from sklearn.datasets import fetch_openml

        cifar = fetch_openml("CIFAR_10", version=1, as_frame=False)
        # Take first 10k samples
        X = cifar.data[:10000].astype(np.float64)
        y = cifar.target[:10000].astype(int).tolist()
        return X, y, "cifar10"
    except Exception as e:
        print(f"  Could not load CIFAR-10: {e}")
        return None, None, None


def load_newsgroups():
    """Load 20 Newsgroups as TF-IDF vectors."""
    print("Loading 20 Newsgroups...")
    try:
        newsgroups = fetch_20newsgroups(
            subset="all", remove=("headers", "footers", "quotes")
        )
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        X = vectorizer.fit_transform(newsgroups.data).toarray().astype(np.float64)
        y = newsgroups.target.tolist()
        return X, y, "newsgroups"
    except Exception as e:
        print(f"  Could not load 20 Newsgroups: {e}")
        return None, None, None


def generate_embedding(X, y, dataset_name, algo_name, algo_factory):
    """Generate and save an embedding."""
    output_file = OUTPUT_DIR / f"{dataset_name}_{algo_name}.json"

    print(f"  Generating {algo_name.upper()} embedding for {dataset_name}...")

    try:
        algo = algo_factory()
        start = time.perf_counter()
        embedding = algo.fit_transform(X)
        elapsed = time.perf_counter() - start

        # Normalize to [-1, 1] range for better visualization
        embedding = embedding - embedding.mean(axis=0)
        scale = np.abs(embedding).max()
        if scale > 0:
            embedding = embedding / scale

        # Save as JSON
        data = {
            "points": embedding.tolist(),
            "labels": y,
            "metadata": {
                "dataset": dataset_name,
                "algorithm": algo_name,
                "n_points": len(y),
                "n_classes": len(set(y)),
                "compute_time": elapsed,
            },
        }

        with open(output_file, "w") as f:
            json.dump(data, f)

        print(f"    Saved to {output_file.name} ({elapsed:.2f}s)")
        return True

    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def main():
    """Generate all embeddings."""
    print("=" * 60)
    print("Squeeze Demo Embedding Generator")
    print("=" * 60)
    print()

    # Load datasets
    datasets = []

    # Always include digits (fast and reliable)
    X, y, name = load_digits_dataset()
    if X is not None:
        datasets.append((X, y, name))

    # Try to load larger datasets
    for loader in [load_fashion_mnist, load_cifar10, load_newsgroups]:
        X, y, name = loader()
        if X is not None:
            datasets.append((X, y, name))

    print(f"\nLoaded {len(datasets)} datasets")
    print()

    # Generate embeddings
    total = len(datasets) * len(ALGORITHMS)
    success = 0

    for X, y, dataset_name in datasets:
        print(f"\nProcessing {dataset_name} ({len(y)} points)...")

        for algo_name, algo_factory in ALGORITHMS.items():
            if generate_embedding(X, y, dataset_name, algo_name, algo_factory):
                success += 1

    print()
    print("=" * 60)
    print(f"Complete: {success}/{total} embeddings generated")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
