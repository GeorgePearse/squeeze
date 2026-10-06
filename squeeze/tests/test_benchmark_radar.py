"""Check radar scaling, invalid measurements, and published result provenance."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_radar import AXES, create_radar, normalize_metrics

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def metrics() -> pd.DataFrame:
    """Three deliberately different profiles; not a performance benchmark."""
    frame = pd.DataFrame({column: [0.1, 0.5, 0.9] for column, _ in AXES})
    frame["Algorithm"] = ["A", "B", "C"]
    frame["Time (s)"] = [1.0, 10.0, 100.0]
    frame["Trust. (k=15)"] = 0.9
    return frame


def test_direction_log_spacing_and_ties(metrics: pd.DataFrame) -> None:
    """A tenfold time change is equally spaced; ties do not divide by zero."""
    before = metrics.copy(deep=True)
    scores = normalize_metrics(metrics)
    np.testing.assert_allclose(scores["Time (s)"], [1, 0.5, 0])
    np.testing.assert_allclose(scores["Neighbor Recall"], [0, 0.5, 1])
    np.testing.assert_allclose(scores["Trust. (k=15)"], [0.5, 0.5, 0.5])
    pd.testing.assert_frame_equal(metrics, before)


@pytest.mark.parametrize("runtime", [0, -1, np.nan, np.inf])
def test_rejects_invalid_runtime(metrics: pd.DataFrame, runtime: float) -> None:
    """Invalid timing cannot turn into a misleading speed advantage."""
    metrics.loc[0, "Time (s)"] = runtime
    with pytest.raises(ValueError, match=r"finite|positive"):
        normalize_metrics(metrics)


def test_rejects_missing_nonfinite_and_duplicate_data(metrics: pd.DataFrame) -> None:
    """A radar must not quietly omit a measurement or overwrite an algorithm."""
    with pytest.raises(ValueError, match="Missing radar columns"):
        normalize_metrics(metrics.drop(columns="Spearman"))
    with pytest.raises(ValueError, match="nonempty"):
        normalize_metrics(metrics.iloc[:0])
    metrics.loc[0, "Spearman"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        normalize_metrics(metrics)
    metrics.loc[0, "Spearman"] = 0.5
    metrics.loc[0, "Algorithm"] = "B"
    with pytest.raises(ValueError, match="unique"):
        normalize_metrics(metrics)


def embedded_data(path: Path) -> dict:
    """Read the chart's machine-readable payload for provenance validation."""
    match = re.search(
        r'<script id="radar-data" type="application/json">(.*?)</script>',
        path.read_text(),
        re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def test_renders_portable_outputs_and_escapes_data(
    metrics: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Render real figure formats and prevent names from breaking the HTML payload."""
    metrics.loc[0, "Algorithm"] = "</script><script>alert(1)</script>"
    create_radar(metrics, tmp_path, "Example <cohort>")
    png = (tmp_path / "metrics_radar.png").read_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert "<svg" in (tmp_path / "metrics_radar.svg").read_text()
    page = (tmp_path / "metrics_radar.html").read_text()
    assert "Example &lt;cohort&gt;" in page
    assert metrics.loc[0, "Algorithm"] not in page
    data = embedded_data(tmp_path / "metrics_radar.html")
    assert data["algorithms"][0]["name"] == metrics.loc[0, "Algorithm"]
    assert all(method["selected"] for method in data["algorithms"])


@pytest.mark.parametrize(
    ("dataset", "csv"),
    [
        ("digits", "metrics_results.csv"),
        (
            "fashion-mnist",
            "working_docs/heatmap_refresh/fashion-mnist/metrics_results.csv",
        ),
    ],
)
def test_published_charts_match_saved_measurements(dataset: str, csv: str) -> None:
    """Committed charts must include every algorithm and its original values."""
    frame = pd.read_csv(ROOT / csv)
    data = embedded_data(
        ROOT / f"docs/assets/benchmarks/{dataset}-radar/metrics_radar.html",
    )
    scores = normalize_metrics(frame)
    assert [method["name"] for method in data["algorithms"]] == frame[
        "Algorithm"
    ].tolist()
    assert [axis["column"] for axis in data["axes"]] == [column for column, _ in AXES]
    for method in data["algorithms"]:
        row = frame.set_index("Algorithm").loc[method["name"]]
        np.testing.assert_allclose(method["raw"], [row[column] for column, _ in AXES])
        np.testing.assert_allclose(method["scores"], scores.loc[method["name"]])
