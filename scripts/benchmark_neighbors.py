"""Reproducible Digits / Fashion-MNIST comparison of additive Rust graph embeddings.

Run from the repository root with ``python -m scripts.benchmark_neighbors``.
Timing includes the complete fit_transform; quality evaluation is outside timing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding, trustworthiness
from sklearn.metrics import pairwise_distances
from threadpoolctl import threadpool_info, threadpool_limits

import squeeze
from scripts.benchmark_datasets import load_benchmark_data
from squeeze import _hnsw_backend


def specifications(seed: int) -> dict:
    """Return constructors and explicit, serializable parameters."""
    methods = {
        "sklearn-pca": (PCA, {"n_components": 2, "svd_solver": "full"}),
        "sklearn-tsne": (
            TSNE,
            {
                "n_components": 2,
                "random_state": seed,
                "perplexity": 30,
                "max_iter": 1000,
                "n_jobs": 1,
            },
        ),
        "sklearn-isomap": (Isomap, {"n_components": 2, "n_neighbors": 15, "n_jobs": 1}),
        "sklearn-spectral": (
            SpectralEmbedding,
            {"n_components": 2, "n_neighbors": 15, "random_state": seed, "n_jobs": 1},
        ),
        "rust-pca": (squeeze.PCA, {"n_components": 2}),
        "rust-tsne": (squeeze.TSNE, {"random_state": seed, "n_iter": 1000}),
        "rust-pacmap": (squeeze.PaCMAP, {"random_state": seed, "n_iter": 450}),
        "rust-trimap": (squeeze.TriMap, {"random_state": seed, "n_iter": 800}),
        "squeeze-umap": (
            squeeze.UMAP,
            {
                "n_components": 2,
                "n_neighbors": 15,
                "random_state": seed,
                "n_epochs": 200,
                "n_jobs": 1,
            },
        ),
        "spectral-map": (
            squeeze.SpectralMap,
            {"random_state": seed, "n_neighbors": 15, "n_iter": 128},
        ),
    }
    for init in ("pca", "spectral"):
        for epochs in (80, 160, 320):
            methods[f"neighbor-{init}-{epochs}"] = (
                squeeze.NeighborMap,
                {
                    "random_state": seed,
                    "init": init,
                    "n_epochs": epochs,
                    "n_neighbors": 15,
                    "negative_samples": 5,
                    "learning_rate": 1.0,
                },
            )
    return methods


def score(
    data: np.ndarray,
    embedding: np.ndarray,
    reference: np.ndarray,
    pairs: np.ndarray,
) -> dict:
    """Separate rank trustworthiness, neighbor overlap, and global distances."""
    distances = pairwise_distances(embedding)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :15]
    recall = np.mean([len(set(a) & set(b)) / 15 for a, b in zip(reference, neighbors)])
    original = np.linalg.norm(data[pairs[:, 0]] - data[pairs[:, 1]], axis=1)
    projected = np.linalg.norm(embedding[pairs[:, 0]] - embedding[pairs[:, 1]], axis=1)
    correlation = float(spearmanr(original, projected).statistic)
    return {
        **{
            f"trustworthiness_{k}": float(
                trustworthiness(data, embedding, n_neighbors=k),
            )
            for k in (5, 15, 30)
        },
        "neighbor_recall_15": float(recall),
        "distance_spearman": correlation if np.isfinite(correlation) else None,
    }


def main() -> None:  # noqa: PLR0915 - sequential experiment protocol
    """Run warmed, interleaved comparisons and persist every result or failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["digits", "fashion-mnist"],
        default="digits",
    )
    parser.add_argument("--samples", type=int, help="Fashion-MNIST count; default 2000")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 53])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "sklearn-pca",
            "sklearn-tsne",
            "sklearn-isomap",
            "sklearn-spectral",
            "rust-pca",
            "rust-tsne",
            "rust-pacmap",
            "rust-trimap",
            "squeeze-umap",
            "spectral-map",
            "neighbor-pca-160",
            "neighbor-spectral-160",
        ],
    )
    args = parser.parse_args()
    unknown = set(args.methods) - specifications(0).keys()
    if unknown:
        parser.error(f"Unknown methods: {sorted(unknown)}")
    args.output.mkdir(parents=True, exist_ok=True)
    data, labels, dataset_metadata = load_benchmark_data(
        args.dataset,
        args.samples,
        cache_dir=args.cache_dir,
    )
    data = np.ascontiguousarray(data, dtype=np.float64)
    distance = pairwise_distances(data)
    np.fill_diagonal(distance, np.inf)
    reference = np.argsort(distance, axis=1, kind="stable")[:, :15]
    # Fixed pairs shared by every algorithm and seed; no labels enter fitting.
    rng = np.random.default_rng(20261006)
    pairs = rng.integers(0, len(data), size=(50000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    root = Path(__file__).resolve().parents[1]
    sources = [
        *sorted((root / "src").rglob("*.rs")),
        root / "Cargo.lock",
        root / "Cargo.toml",
        root / "pyproject.toml",
        root / "uv.lock",
        *sorted((root / "squeeze").glob("*.py")),
        Path(__file__),
        Path(__file__).with_name("benchmark_datasets.py"),
    ]
    metadata = {
        "dataset": dataset_metadata,
        "data_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
        "base_commit": subprocess.check_output(  # noqa: S603 - fixed read-only git arguments
            [shutil.which("git") or "/usr/bin/git", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "extension_sha256": hashlib.sha256(
            Path(_hnsw_backend.__file__).read_bytes(),
        ).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        ),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "scikit-learn", "numba", "pynndescent")
        },
        "thread_environment": {
            k: os.environ.get(k)
            for k in (
                "RAYON_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "seeds": args.seeds,
        "methods": args.methods,
        "protocol": (
            "One full-data warmup per method (seed 42), then randomized "
            "interleaved full fit_transform runs. BLAS limited to one thread. "
            "Metrics excluded from fit timing. Seeds are optimizer replicates, "
            "not independent datasets."
        ),
        "pair_seed": 20261006,
        "pair_count": len(pairs),
    }
    results, embeddings, warmed = [], {"labels": labels}, set()
    with threadpool_limits(limits=1):
        metadata["threadpools"] = threadpool_info()
        for seed in args.seeds:
            order = np.random.default_rng(seed).permutation(args.methods)
            for name in order:
                constructor, parameters = specifications(seed)[name]
                result = {"method": str(name), "seed": seed, "parameters": parameters}
                try:
                    if name not in warmed:
                        warm_constructor, warm_parameters = specifications(42)[name]
                        start = time.perf_counter()
                        warm_constructor(**warm_parameters).fit_transform(data)
                        result["warmup_seconds"] = time.perf_counter() - start
                        warmed.add(name)
                    model = constructor(**parameters)
                    start = time.perf_counter()
                    embedding = np.asarray(model.fit_transform(data))
                    result["seconds"] = time.perf_counter() - start
                    if (
                        embedding.shape != (len(data), 2)
                        or not np.isfinite(embedding).all()
                    ):
                        message = "Invalid embedding shape or non-finite coordinates"
                        raise ValueError(message)  # noqa: TRY301 - record failure
                    result.update(score(data, embedding, reference, pairs))
                    result["status"] = "ok"
                    embeddings[f"{name}__{seed}"] = embedding
                except Exception:  # noqa: BLE001 - failures are explicit benchmark results
                    result["status"] = "error"
                    result["error"] = traceback.format_exc()
                results.append(result)
                print(json.dumps(result), flush=True)  # noqa: T201 - CLI progress
                (args.output / "results.json").write_text(
                    json.dumps(
                        {"metadata": metadata, "runs": results},
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n",
                )
                np.savez_compressed(args.output / "embeddings.npz", **embeddings)


if __name__ == "__main__":
    main()
