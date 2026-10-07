"""Compute-device selection and GPU/CPU agreement tests.

The CPU tests always run. The GPU tests run when ``SQUEEZE_DEVICE`` resolves to a GPU
(locally and in CI that is lavapipe, software Vulkan, with ``SQUEEZE_DEVICE=wgpu``);
otherwise they are skipped with the probe report in the skip reason.
"""

# ruff: noqa: D103, N806, N803, ANN001, T201, PLR2004, PLC0415, E501  # test/benchmark pragmatics: prints are the timing record

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pytest
from sklearn.datasets import load_digits

import squeeze
from squeeze.hnsw_wrapper import HnswIndexWrapper

RUST_ALGORITHMS = ("TSNE", "MDS", "Isomap", "LLE", "PHATE", "TriMap", "PaCMAP")


def _gpu_name() -> str | None:
    """Return the GPU device name when an explicit device request resolves to one."""
    requested = os.environ.get("SQUEEZE_DEVICE")
    if requested in (None, "", "auto", "cpu"):
        resolved = squeeze.default_device()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resolved = squeeze.resolve_device(requested)
    return None if resolved == "cpu" else resolved


GPU = _gpu_name()
needs_gpu = pytest.mark.skipif(
    GPU is None,
    reason=f"no GPU device resolved; probe report:\n{squeeze.devices()}",
)


@pytest.fixture(scope="module")
def digits() -> tuple[np.ndarray, np.ndarray]:
    data = load_digits()
    return np.asarray(data.data, dtype=np.float64), data.target


# ---------------------------------------------------------------------------
# Device API (always)
# ---------------------------------------------------------------------------


def test_devices_report_is_text() -> None:
    report = squeeze.devices()
    assert isinstance(report, str)
    assert "chosen:" in report
    for kind in ("mlx", "cuda", "wgpu", "cpu"):
        assert kind in report


def test_resolve_cpu_and_auto() -> None:
    assert squeeze.resolve_device("cpu") == "cpu"
    assert squeeze.resolve_device("auto") == squeeze.default_device()
    assert squeeze.resolve_device(None) == squeeze.default_device()


def test_unknown_device_raises() -> None:
    with pytest.raises(ValueError, match="unknown device"):
        squeeze.resolve_device("tpu")


def test_unavailable_device_warns_and_falls_back() -> None:
    # No build has every backend; pick one that is certainly absent on this platform.
    missing = "cuda" if "cuda" not in squeeze.default_device() else "mlx"
    with pytest.warns(RuntimeWarning, match="falling back to cpu"):
        resolved = squeeze.resolve_device(missing)
    assert resolved == "cpu"


@pytest.mark.parametrize("name", RUST_ALGORITHMS)
def test_algorithms_accept_device_cpu(name: str, digits) -> None:
    X, _ = digits
    cls = getattr(squeeze, name)
    kwargs = {"device": "cpu"}
    if name == "TSNE":
        kwargs.update(n_iter=60, random_state=0)
    elif name in ("TriMap", "PaCMAP"):
        kwargs.update(n_iter=20, random_state=0)
    elif name == "MDS":
        kwargs.update(n_iter=5, random_state=0)
    elif name == "Isomap":
        kwargs.update(n_neighbors=25)  # 300 digits are disconnected at the default 10
    emb = cls(n_components=2, **kwargs).fit_transform(X[:300])
    assert emb.shape == (300, 2)
    assert np.isfinite(emb).all()


def test_hnsw_wrapper_device_cpu(digits) -> None:
    X, _ = digits
    idx = HnswIndexWrapper(X.astype(np.float32), n_neighbors=10, device="cpu")
    assert idx.compute_device == "cpu"
    indices, dists = idx.neighbor_graph
    assert indices.shape == (X.shape[0], 10)
    assert (indices >= 0).all()
    assert np.isfinite(dists).all()


def test_umap_device_cpu(digits) -> None:
    X, _ = digits
    emb = squeeze.UMAP(
        n_neighbors=10,
        n_epochs=30,
        random_state=0,
        device="cpu",
    ).fit_transform(X[:400])
    assert emb.shape == (400, 2)


# ---------------------------------------------------------------------------
# GPU vs CPU agreement (needs a GPU; lavapipe counts)
# ---------------------------------------------------------------------------


def _exact_knn(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sq = (X**2).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    np.fill_diagonal(d2, np.inf)
    idx = np.argsort(d2, axis=1)[:, :k]
    return idx, np.sqrt(np.maximum(np.take_along_axis(d2, idx, axis=1), 0.0))


@needs_gpu
def test_gpu_knn_matches_exact(digits) -> None:
    X, _ = digits
    X32 = X.astype(np.float32)
    t = time.perf_counter()
    gpu = HnswIndexWrapper(
        X32,
        n_neighbors=15,
        device=os.environ.get("SQUEEZE_DEVICE", "gpu"),
    )
    g_idx, g_dist = gpu.neighbor_graph
    elapsed = time.perf_counter() - t
    assert gpu.compute_device != "cpu", gpu.compute_device
    e_idx, e_dist = _exact_knn(X, 15)
    # Digits has many tied distances (integer pixels), so a neighbour counts as correct when
    # its distance is within tolerance of the exact k-th distance ("identical up to ties").
    recall = float(np.mean(g_dist <= e_dist[:, -1:] + 1e-3))
    strict = np.mean([len(set(a) & set(b)) / 15 for a, b in zip(g_idx, e_idx)])
    print(
        f"\n[{GPU}] kNN digits 1797x64 k=15: {elapsed:.3f}s "
        f"recall vs exact {recall:.4f} (strict index match {strict:.4f})",
    )
    assert recall >= 0.999
    np.testing.assert_allclose(
        np.sort(g_dist, 1),
        np.sort(e_dist, 1),
        rtol=1e-3,
        atol=1e-3,
    )

    # query() path (no self-exclusion) agrees with the brute-force answer too
    q_idx, q_dist = gpu.query(X32[:50], k=5)
    assert q_idx.shape == (50, 5)
    assert (q_idx[:, 0] == np.arange(50)).all()
    assert np.allclose(q_dist[:, 0], 0.0, atol=1e-4)


@needs_gpu
@pytest.mark.parametrize("name", RUST_ALGORITHMS)
def test_gpu_trustworthiness_matches_cpu(name: str, digits) -> None:
    from sklearn.manifold import trustworthiness

    X, _ = digits
    cls = getattr(squeeze, name)
    kwargs: dict = {}
    if name == "TSNE":
        kwargs.update(n_iter=300, random_state=42, perplexity=30.0)
    elif name in ("TriMap", "PaCMAP"):
        kwargs.update(random_state=42)
    elif name == "MDS":
        kwargs.update(n_iter=50, random_state=42)
    elif name == "PHATE":
        kwargs.update(random_state=42)
    device = os.environ.get("SQUEEZE_DEVICE", "gpu")
    results = {}
    for dev in ("cpu", device):
        t = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            emb = cls(n_components=2, device=dev, **kwargs).fit_transform(X)
        fallback = [str(w.message) for w in caught if "falling back" in str(w.message)]
        assert not fallback, fallback
        results[dev] = (
            trustworthiness(X, emb, n_neighbors=15),
            time.perf_counter() - t,
        )
    (t_cpu, s_cpu), (t_gpu, s_gpu) = results["cpu"], results[device]
    print(
        f"\n[{GPU}] {name}: cpu T={t_cpu:.4f} ({s_cpu:.2f}s)  gpu T={t_gpu:.4f} ({s_gpu:.2f}s)",
    )
    assert abs(t_cpu - t_gpu) <= 0.01, (t_cpu, t_gpu)


@needs_gpu
def test_gpu_umap_matches_cpu(digits) -> None:
    from sklearn.manifold import trustworthiness

    X, _ = digits
    device = os.environ.get("SQUEEZE_DEVICE", "gpu")
    scores = {}
    for dev in ("cpu", device):
        emb = squeeze.UMAP(n_neighbors=15, random_state=42, device=dev).fit_transform(X)
        scores[dev] = trustworthiness(X, emb, n_neighbors=15)
    print(f"\n[{GPU}] UMAP: cpu T={scores['cpu']:.4f} gpu T={scores[device]:.4f}")
    assert abs(scores["cpu"] - scores[device]) <= 0.01
