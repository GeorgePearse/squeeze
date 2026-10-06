"""Generate a portable HTML explorer and a standard PNG from a benchmark run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    """Aggregate replicate medians without hiding ranges or failed runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    results = json.loads((args.directory / "results.json").read_text())
    dataset = results["metadata"]["dataset"]
    dataset_name = dataset["name"] if isinstance(dataset, dict) else "Digits"
    methods = sorted({row["method"] for row in results["runs"]})
    summary = []
    keys = ["seconds", "trustworthiness_15", "neighbor_recall_15", "distance_spearman"]
    for method in methods:
        rows = [
            r for r in results["runs"] if r["method"] == method and r["status"] == "ok"
        ]
        item = {"method": method, "successful_runs": len(rows)}
        for key in keys:
            values = [r[key] for r in rows if r[key] is not None]
            item[key] = (
                {
                    "median": float(np.median(values)),
                    "min": min(values),
                    "max": max(values),
                }
                if values
                else None
            )
        summary.append(item)
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axis = plt.subplots(figsize=(12, 7), layout="constrained")
    for row in summary:
        if not row["successful_runs"]:
            continue
        runtime, quality = row["seconds"], row["trustworthiness_15"]
        axis.errorbar(
            runtime["median"],
            quality["median"],
            xerr=[
                [runtime["median"] - runtime["min"]],
                [runtime["max"] - runtime["median"]],
            ],
            yerr=[
                [quality["median"] - quality["min"]],
                [quality["max"] - quality["median"]],
            ],
            fmt="o",
            capsize=3,
            label=row["method"],
        )
    axis.set(
        xscale="log",
        xlabel="Full fit_transform seconds (log scale; lower is faster)",
        ylabel="sklearn trustworthiness at k=15 (higher is better)",
        title=f"{dataset_name}: speed / local quality — median and full seed range",
    )
    axis.grid(alpha=0.2)
    axis.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    fig.savefig(args.directory / "speed_quality.png", dpi=160)
    plt.close(fig)
    with np.load(args.directory / "embeddings.npz") as embeddings:
        points = {
            name: np.round(embeddings[name], 5).tolist() for name in embeddings.files
        }
    payload = json.dumps(
        {"summary": summary, "results": results, "points": points},
        allow_nan=False,
    ).replace("<", "\\u003c")
    template = Path(__file__).with_name("neighbor_report.html").read_text()
    (args.directory / "report.html").write_text(
        template.replace("__BENCHMARK_JSON__", payload),
    )


if __name__ == "__main__":
    main()
