"""Check benchmark entry points and output routing without measuring performance."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_digits

from scripts import benchmark_metrics_heatmap as benchmark
from scripts.check_docs import check_assets

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("dataset", "custom_output"),
    [("digits", False), ("fashion-mnist", False), ("digits", True)],
)
def test_outputs_stay_in_their_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: str,
    custom_output: bool,
) -> None:
    """Default and custom runs keep plots, data, and provenance off the root."""
    data, labels = load_digits(return_X_y=True)
    measurements = pd.read_csv(
        ROOT / "assets/benchmarks/digits/metrics_results.csv",
    ).query("Algorithm == 'PCA'")
    monkeypatch.chdir(tmp_path)
    argv = ["benchmark", "--dataset", dataset]
    if custom_output:
        argv += ["--output-dir", "custom"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        benchmark,
        "load_benchmark_data",
        lambda *_args, **_kwargs: (data, labels, {"name": dataset}),
    )
    monkeypatch.setattr(benchmark, "get_algorithms", lambda: [("PCA", None)])
    monkeypatch.setattr(
        benchmark,
        "run_algorithms",
        lambda *_args: [benchmark.AlgorithmResult("PCA", data[:, :2], 0.01)],
    )
    monkeypatch.setattr(benchmark, "compute_metrics", lambda *_args: measurements)
    benchmark.main()

    output = tmp_path / ("custom" if custom_output else f"assets/benchmarks/{dataset}")
    evidence = (
        output
        if custom_output
        else tmp_path / f"working_docs/heatmap_refresh/{dataset}"
    )
    assert {path.name for path in output.iterdir()} >= {
        "metrics_results.csv",
        "metrics_heatmap.png",
        "metrics_radar.png",
        "metrics_radar.svg",
        "metrics_radar.html",
        "fit_time.png",
        "fit_time.svg",
    }
    assert (
        json.loads((evidence / "protocol.json").read_text())["dataset"]["name"]
        == dataset
    )
    with np.load(evidence / "embeddings.npz") as embeddings:
        np.testing.assert_array_equal(embeddings["PCA"], data[:, :2])
    assert not any(path.is_file() for path in tmp_path.iterdir())
    if custom_output:
        assert not (tmp_path / "assets").exists()
        assert not (tmp_path / "working_docs").exists()


def test_documentation_assets_match_relocated_sources() -> None:
    """The docs must still show the same immutable benchmark measurements."""
    check_assets()
