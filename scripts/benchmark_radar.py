"""Render radar and fit-time comparisons from saved metrics without refitting."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Deliberately use a small set of complementary axes, not three copies of k.
AXES = (
    ("Trust. (k=15)", "Rank trust\nk=15"),
    ("Neighbor Recall", "Neighbor recall\nk=15"),
    ("Spearman", "Global distance\ncorrelation"),
    ("Silhouette", "Cluster\nseparation"),
    ("Transductive Acc.", "Transductive\naccuracy"),
    ("Time (s)", "Fit speed\n(reversed log time)"),
)
DEFAULT_METHODS = ("UMAP", "t-SNE", "PaCMAP", "NeighborMap")
COLORS = (
    "#0072B2",
    "#5C677D",
    "#D55E00",
    "#6F42C1",
    "#A16207",
    "#CC79A7",
    "#00857D",
    "#8C564B",
    "#B44B86",
    "#009E73",
    "#5454CF",
)


def normalize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Scale each axis over the complete cohort; reverse log runtime before scaling.

    Zero means the cohort's worst observed value, not zero absolute capability.
    Equal-valued axes map to 0.5. Invalid data must not silently disappear.
    """
    required = ["Algorithm", *(column for column, _ in AXES)]
    missing = set(required) - set(frame.columns)
    if missing:
        message = f"Missing radar columns: {sorted(missing)}"
        raise ValueError(message)
    names = frame["Algorithm"]
    if frame.empty or names.isna().any() or names.duplicated().any():
        message = "Radar input needs nonempty, unique algorithm names"
        raise ValueError(message)
    if not all(isinstance(name, str) and name.strip() for name in names):
        message = "Algorithm names must be nonempty strings"
        raise ValueError(message)
    values = frame[[column for column, _ in AXES]].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        message = "Radar metrics must be finite"
        raise ValueError(message)
    if (values["Time (s)"] <= 0).any():
        message = "Radar runtime must be positive"
        raise ValueError(message)
    values["Time (s)"] = -np.log10(values["Time (s)"])
    for column in values:
        low, high = values[column].min(), values[column].max()
        values[column] = (values[column] - low) / (high - low) if high > low else 0.5
    values.index = names.to_numpy()
    return values


def render_static(
    frame: pd.DataFrame,
    scores: pd.DataFrame,
    output: Path,
    title: str,
) -> None:
    """Write a small-multiple PNG and SVG with consistent radial scales."""
    columns = 3
    rows = (len(frame) + columns - 1) // columns
    angles = np.linspace(0, 2 * np.pi, len(AXES), endpoint=False)
    closed_angles = np.append(angles, angles[0])
    with plt.rc_context(
        {"font.family": "DejaVu Sans", "svg.hashsalt": "squeeze-radar"},
    ):
        fig, plots = plt.subplots(
            rows,
            columns,
            figsize=(17, rows * 4.6 + 1.7),
            subplot_kw={"projection": "polar"},
            squeeze=False,
        )
        for index, ax in enumerate(plots.flat):
            if index >= len(frame):
                ax.set_visible(False)
                continue
            name = frame.iloc[index]["Algorithm"]
            color = COLORS[index % len(COLORS)]
            radii = scores.loc[name].to_numpy()
            closed_radii = np.append(radii, radii[0])
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=8, color="#64748b")
            ax.set_xticks(angles)
            ax.set_xticklabels([label for _, label in AXES], fontsize=9)
            ax.tick_params(axis="x", pad=12)
            ax.grid(color="#cbd5e1", linewidth=0.7)
            ax.spines["polar"].set_color("#cbd5e1")
            ax.plot(closed_angles, closed_radii, color=color, linewidth=2.3)
            ax.fill(closed_angles, closed_radii, color=color, alpha=0.15)
            ax.scatter(angles, radii, color=color, s=15, zorder=3)
            seconds = frame.iloc[index]["Time (s)"]
            ax.set_title(
                f"{name}  ·  fit {seconds:.3g}s",
                fontsize=14,
                fontweight="bold",
                color=color,
                pad=35,
            )
        fig.suptitle(
            f"Squeeze · {title}\nAlgorithm performance profiles",
            fontsize=23,
            fontweight="bold",
            y=0.995,
        )
        fig.text(
            0.5,
            0.018,
            "Farther out = better within this dataset. "
            "Each axis spans the full algorithm cohort.\n"
            "0 = worst observed · 1 = best observed · ties = 0.5 · "
            "fit speed uses reversed log time.\n"
            "Shapes are not an overall score. "
            "Single-seed results; see raw metrics and protocol.",
            ha="center",
            fontsize=11,
            color="#475569",
            linespacing=1.6,
        )
        fig.subplots_adjust(
            left=0.075,
            right=0.925,
            top=0.90,
            bottom=0.095,
            hspace=0.85,
            wspace=0.6,
        )
        fig.savefig(output / "metrics_radar.png", dpi=160, facecolor="white")
        fig.savefig(
            output / "metrics_radar.svg",
            facecolor="white",
            metadata={"Date": None},
        )
        plt.close(fig)
        svg_path = output / "metrics_radar.svg"
        svg_path.write_text(
            "\n".join(line.rstrip() for line in svg_path.read_text().splitlines())
            + "\n",
        )


def spatial_preservation(frame: pd.DataFrame) -> pd.Series:
    """Average three local scores and rescaled global distance correlation.

    Use fixed metric ranges, so selecting a different cohort cannot change a
    method's score. This descriptive equal-weight aggregate is not a validated
    universal quality measure.
    """
    columns = ["Trust. (k=15)", "Continuity", "Neighbor Recall", "Spearman"]
    missing = set(columns) - set(frame.columns)
    if missing:
        message = f"Missing spatial preservation columns: {sorted(missing)}"
        raise ValueError(message)
    values = frame[columns].astype(float).copy()
    if not np.isfinite(values.to_numpy()).all():
        message = "Spatial preservation metrics must be finite"
        raise ValueError(message)
    values["Spearman"] = (values["Spearman"] + 1) / 2
    if ((values < 0) | (values > 1)).any().any():
        message = "Spatial preservation metrics are outside their expected ranges"
        raise ValueError(message)
    return values.mean(axis=1)


def render_fit_times(frame: pd.DataFrame, output: Path, title: str) -> None:
    """Plot fit seconds against aggregate spatial preservation; upper-left wins."""
    quality = spatial_preservation(frame)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "svg.hashsalt": "squeeze-fit-time",
            "svg.fonttype": "none",
        },
    ):
        fig, ax = plt.subplots(figsize=(11, 7))
        for index, (_, row) in enumerate(frame.iterrows()):
            ax.scatter(
                row["Time (s)"],
                quality.iloc[index],
                color=COLORS[index % len(COLORS)],
                label=row["Algorithm"],
                s=85,
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )
            ax.annotate(
                row["Algorithm"],
                (row["Time (s)"], quality.iloc[index]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=COLORS[index % len(COLORS)],
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
                zorder=4,
            )
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
        ax.set_xlabel("fit_transform time (seconds, log scale) · lower is better")
        ax.set_ylabel("Spatial preservation (0-1) · higher is better")
        ax.grid(color="#dbe3e8")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("Better trade-off: upper left", loc="left", fontsize=11)
        fig.suptitle(
            f"Squeeze · {title}\nFit time vs spatial preservation",
            fontsize=18,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.035,
            "Spatial preservation = mean(trustworthiness k15, continuity k15, "
            "neighbour recall k15, (Spearman + 1) / 2)\n"
            "Equal-weight descriptive score. Saved single-seed fits; "
            "warmup, loading and scoring excluded.",
            ha="center",
            fontsize=10,
            color="#475569",
        )
        fig.subplots_adjust(left=0.09, right=0.96, top=0.81, bottom=0.2)
        fig.savefig(output / "fit_time.png", dpi=160, facecolor="white")
        fig.savefig(output / "fit_time.svg", facecolor="white", metadata={"Date": None})
        plt.close(fig)
        svg_path = output / "fit_time.svg"
        svg_path.write_text(
            "\n".join(line.rstrip() for line in svg_path.read_text().splitlines())
            + "\n",
        )


def create_radar(frame: pd.DataFrame, output: Path, title: str) -> None:
    """Save static figures and a self-contained selectable HTML comparison."""
    scores = normalize_metrics(frame)
    quality = spatial_preservation(frame)
    output.mkdir(parents=True, exist_ok=True)
    render_static(frame, scores, output, title)
    render_fit_times(frame, output, title)
    defaults = set(DEFAULT_METHODS) & set(frame["Algorithm"])
    selected = defaults or set(frame["Algorithm"].head(4))
    data = {
        "title": title,
        "axes": [
            {"column": column, "label": label.replace("\n", " ")}
            for column, label in AXES
        ],
        "algorithms": [
            {
                "name": row["Algorithm"],
                "color": COLORS[index % len(COLORS)],
                "scores": scores.loc[row["Algorithm"]].tolist(),
                "raw": [float(row[column]) for column, _ in AXES],
                "spatial_preservation": float(quality.iloc[index]),
                "selected": row["Algorithm"] in selected,
            }
            for index, (_, row) in enumerate(frame.iterrows())
        ],
    }
    template = Path(__file__).with_name("benchmark_radar.html").read_text()
    rendered = template.replace("__TITLE__", html.escape(title)).replace(
        "__DATA__",
        json.dumps(data, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c"),
    )
    (output / "metrics_radar.html").write_text(rendered)


def main() -> None:
    """Render existing CSV measurements without loading or fitting an algorithm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Saved metrics_results.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", required=True, help="Dataset and sample count")
    args = parser.parse_args()
    create_radar(pd.read_csv(args.csv), args.output_dir, args.title)


if __name__ == "__main__":
    main()
