"""CPU vs GPU benchmark for the squeeze compute backends.

Each (dataset, algorithm, device) job runs in a fresh subprocess so wall-clock time and
peak host RSS are per job and GPU state never leaks between jobs. Results are appended as
JSON lines, then summarised into a Markdown table and a plot.

Examples
--------
    python scripts/benchmark_gpu.py --devices cpu,wgpu --datasets digits
    python scripts/benchmark_gpu.py --devices cpu,wgpu --datasets digits,fashion-mnist \
        --knn-rows 1000000 --out /var/tmp/squeeze-gpu/bench-t4

Trustworthiness is sklearn's at k=15, same seeds on every device.

"""

# ruff: noqa: PLC0415, T201, PERF401, E501, N806, C901, S603, RUF059, PLR0912, PLR0915, ICN001, D103, ANN202  # test/benchmark pragmatics: prints are the timing record

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ALGORITHMS = ("UMAP", "TSNE", "MDS", "Isomap", "LLE", "PHATE", "TriMap", "PaCMAP")
SEED = 42
K_TRUST = 15


def _load(dataset: str, samples: int | None) -> tuple[np.ndarray, dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_datasets import load_benchmark_data  # type: ignore[import-not-found]

    if dataset == "digits":
        data, _labels, meta = load_benchmark_data("digits")
    else:
        data, _labels, meta = load_benchmark_data(
            "fashion-mnist",
            samples=samples,
            seed=SEED,
        )
    return np.asarray(data, dtype=np.float64), meta


def _make(name: str, device: str):
    import squeeze

    if name == "UMAP":
        return squeeze.UMAP(n_neighbors=15, random_state=SEED, device=device)
    cls = getattr(squeeze, name)
    kwargs: dict = {"n_components": 2, "device": device}
    if name == "TSNE":
        kwargs.update(perplexity=30.0, random_state=SEED)
    elif name in ("TriMap", "PaCMAP", "PHATE", "MDS"):
        kwargs.update(random_state=SEED)
    return cls(**kwargs)


def run_job(dataset: str, samples: int | None, algorithm: str, device: str) -> dict:
    """Run one algorithm on one device and return its record (executed in a subprocess)."""
    from sklearn.manifold import trustworthiness

    import squeeze

    X, meta = _load(dataset, samples)
    resolved = squeeze.resolve_device(device)
    record = {
        "dataset": dataset,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "algorithm": algorithm,
        "device": device,
        "resolved_device": resolved,
        "host": platform.node(),
        "machine": platform.machine(),
    }
    fallbacks: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        emb = _make(algorithm, device).fit_transform(X)
        elapsed = time.perf_counter() - t0
        fallbacks = [str(w.message) for w in caught if "falling back" in str(w.message)]
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record.update(
        seconds=elapsed,
        peak_rss_mb=rss_kb / 1024.0
        if sys.platform != "darwin"
        else rss_kb / (1024.0 * 1024.0),
        trustworthiness=float(trustworthiness(X, emb, n_neighbors=K_TRUST)),
        fallbacks=fallbacks,
        dataset_meta={k: v for k, v in meta.items() if k != "class_names"},
    )
    return record


def run_knn_job(rows: int, dim: int, device: str) -> dict:
    """Exact-vs-HNSW kNN timing on synthetic data (kNN alone, k=15)."""
    import squeeze
    from squeeze.hnsw_wrapper import HnswIndexWrapper

    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((rows, dim), dtype=np.float32)
    resolved = squeeze.resolve_device(device)
    fallbacks: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        index = HnswIndexWrapper(X, n_neighbors=15, device=device)
        indices, dists = index.neighbor_graph
        elapsed = time.perf_counter() - t0
        fallbacks = [str(w.message) for w in caught if "falling back" in str(w.message)]
    # exact recall on a 200-row sample (brute force in numpy)
    sample = rng.choice(rows, size=min(200, rows), replace=False)
    sq = (X**2).sum(1)
    d2 = sq[sample, None] + sq[None, :] - 2.0 * X[sample] @ X.T
    d2[np.arange(len(sample)), sample] = np.inf
    exact = np.argsort(d2, axis=1)[:, :15]
    recall = float(
        np.mean([len(set(a) & set(b)) / 15 for a, b in zip(indices[sample], exact)]),
    )
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "dataset": f"synthetic-{rows}x{dim}",
        "n": rows,
        "d": dim,
        "algorithm": "kNN(k=15)",
        "device": device,
        "resolved_device": resolved,
        "compute_device": index.compute_device,
        "host": platform.node(),
        "machine": platform.machine(),
        "seconds": elapsed,
        "peak_rss_mb": rss_kb / 1024.0
        if sys.platform != "darwin"
        else rss_kb / (1024.0 * 1024.0),
        "recall_vs_exact": recall,
        "fallbacks": fallbacks,
    }


def summarise(records: list[dict], out: Path) -> str:
    """Write a Markdown table (speedup vs cpu, trustworthiness delta) and a bar plot."""
    by_key: dict[tuple[str, str], dict[str, dict]] = {}
    for r in records:
        by_key.setdefault((r["dataset"], r["algorithm"]), {})[r["device"]] = r
    devices = sorted({r["device"] for r in records}, key=lambda d: (d != "cpu", d))
    lines = [
        "| dataset | algorithm | "
        + " | ".join(f"{d} time (s)" for d in devices)
        + " | "
        + " | ".join(f"{d} speedup" for d in devices if d != "cpu")
        + " | "
        + " | ".join(f"{d} quality" for d in devices)
        + " |",
        "|" + "---|" * (2 + len(devices) + (len(devices) - 1) + len(devices)),
    ]
    rows_for_plot = []
    for (dataset, algorithm), per_dev in sorted(by_key.items()):
        cpu = per_dev.get("cpu")
        cells = [dataset, algorithm]
        for d in devices:
            cells.append(f"{per_dev[d]['seconds']:.2f}" if d in per_dev else "—")
        for d in devices:
            if d == "cpu":
                continue
            if cpu and d in per_dev and per_dev[d]["seconds"] > 0:
                cells.append(f"{cpu['seconds'] / per_dev[d]['seconds']:.2f}x")
            else:
                cells.append("—")
        for d in devices:
            r = per_dev.get(d)
            if r is None:
                cells.append("—")
            elif "trustworthiness" in r:
                q = r["trustworthiness"]
                delta = (
                    f" ({q - cpu['trustworthiness']:+.4f})"
                    if cpu and d != "cpu"
                    else ""
                )
                cells.append(f"T={q:.4f}{delta}")
            else:
                cells.append(f"recall={r['recall_vs_exact']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
        rows_for_plot.append(
            (
                f"{algorithm}\n{dataset}",
                {d: per_dev[d]["seconds"] for d in devices if d in per_dev},
            ),
        )
    table = "\n".join(lines)
    (out / "results.md").write_text(table + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Fixed categorical colours per device kind (blue, orange, aqua, violet).
        colours = {"cpu": "#2a78d6", "wgpu": "#eb6834", "mlx": "#1baf7a", "cuda": "#4a3aa7"}
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(rows_for_plot)), 4.5))
        width = 0.8 / max(1, len(devices))
        xs = np.arange(len(rows_for_plot))
        for i, d in enumerate(devices):
            vals = [row[1].get(d, np.nan) for row in rows_for_plot]
            colour = colours.get(d.split(":")[0], "#52514e")
            ax.bar(xs + i * width, vals, width * 0.92, label=d, color=colour, linewidth=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(xs + width * (len(devices) - 1) / 2)
        ax.set_xticklabels([row[0] for row in rows_for_plot], fontsize=8)
        ax.set_ylabel("wall-clock seconds (log)")
        ax.set_yscale("log")
        ax.set_title(
            f"squeeze CPU vs GPU, {records[0]['host']} ({records[0]['machine']})",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "results.png", dpi=130)
    except Exception as exc:  # noqa: BLE001 - plotting is optional
        print(f"plot skipped: {exc}", file=sys.stderr)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--devices",
        default="cpu,auto",
        help="comma-separated device strings",
    )
    parser.add_argument("--datasets", default="digits", help="digits,fashion-mnist")
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS))
    parser.add_argument("--fashion-samples", type=int, default=2000)
    parser.add_argument(
        "--knn-rows",
        type=int,
        default=0,
        help="synthetic kNN rows (0 = skip)",
    )
    parser.add_argument("--knn-dim", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("benchmark_gpu_results"))
    parser.add_argument("--job", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.job:
        spec = json.loads(args.job)
        if spec["kind"] == "knn":
            rec = run_knn_job(spec["rows"], spec["dim"], spec["device"])
        else:
            rec = run_job(
                spec["dataset"],
                spec.get("samples"),
                spec["algorithm"],
                spec["device"],
            )
        print("RESULT " + json.dumps(rec))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    specs: list[dict] = []
    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        for algorithm in [a.strip() for a in args.algorithms.split(",") if a.strip()]:
            for device in devices:
                specs.append(
                    {
                        "kind": "algo",
                        "dataset": dataset,
                        "samples": args.fashion_samples
                        if dataset == "fashion-mnist"
                        else None,
                        "algorithm": algorithm,
                        "device": device,
                    },
                )
    if args.knn_rows:
        for device in devices:
            specs.append(
                {
                    "kind": "knn",
                    "rows": args.knn_rows,
                    "dim": args.knn_dim,
                    "device": device,
                },
            )

    records: list[dict] = []
    log = args.out / "results.jsonl"
    with log.open("a") as fh:
        for spec in specs:
            print(f"-> {spec}", flush=True)
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            if spec["kind"] == "knn":
                # the synthetic kNN run is the one place the brute-force row cap is lifted
                env.setdefault("SQUEEZE_BRUTEFORCE_MAX_ROWS", str(max(spec["rows"], 500_000)))
            proc = subprocess.run(
                [sys.executable, __file__, "--job", json.dumps(spec)],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            result_lines = [
                line for line in proc.stdout.splitlines() if line.startswith("RESULT ")
            ]
            if proc.returncode != 0 or not result_lines:
                print(f"   failed: {proc.stderr[-2000:]}", flush=True)
                continue
            rec = json.loads(result_lines[-1][len("RESULT ") :])
            rec["spec"] = spec
            records.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            q = rec.get("trustworthiness", rec.get("recall_vs_exact"))
            print(
                f"   {rec['resolved_device']}: {rec['seconds']:.2f}s  quality={q:.4f}  "
                f"peak_rss={rec['peak_rss_mb']:.0f}MB  fallbacks={len(rec['fallbacks'])}",
                flush=True,
            )
    if records:
        print(summarise(records, args.out))


if __name__ == "__main__":
    main()
