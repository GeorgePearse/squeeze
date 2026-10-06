"""Offline contracts for benchmark data sampling and integrity checks."""

import struct
from pathlib import Path

import numpy as np
import pytest

from scripts import benchmark_datasets as datasets


@pytest.fixture
def fashion_idx(monkeypatch: pytest.MonkeyPatch) -> tuple[bytes, bytes]:
    """Mock IDX data with the official shape, without network downloads."""
    data = np.zeros((10000, 784), dtype=np.uint8)
    data[:, 0] = np.arange(10000) % 256
    images = struct.pack(">IIII", 2051, 10000, 28, 28) + data.tobytes()
    labels = (
        struct.pack(">II", 2049, 10000)
        + np.repeat(np.arange(10, dtype=np.uint8), 1000).tobytes()
    )
    monkeypatch.setattr(
        datasets,
        "read_fashion_file",
        lambda _cache, name: images if "images" in name else labels,
    )
    return images, labels


@pytest.mark.usefixtures("fashion_idx")
def test_balanced_reproducible_sample() -> None:
    """Sampling is deterministic, disjoint by index, balanced and seed-sensitive."""
    data, labels, meta = datasets.load_benchmark_data("fashion-mnist", samples=200)
    again, _, repeated = datasets.load_benchmark_data("fashion-mnist", samples=200)
    _, _, changed = datasets.load_benchmark_data("fashion-mnist", samples=200, seed=17)
    np.testing.assert_array_equal(data, again)
    np.testing.assert_array_equal(np.bincount(labels), np.full(10, 20))
    assert len(set(meta["indices"])) == len(data)
    assert data.shape == (200, 784)
    assert data.dtype == np.float64
    assert meta == repeated
    assert meta["indices"] != changed["indices"]


@pytest.mark.parametrize("count", [0, 99, 123, 10010])
def test_invalid_sample_size(count: int) -> None:
    """Reject unsupported sample sizes before any network request."""
    with pytest.raises(ValueError, match="multiple of 10"):
        datasets.load_benchmark_data("fashion-mnist", samples=count)


def test_idx_payload_validation(fashion_idx: tuple[bytes, bytes]) -> None:
    """Malformed headers and truncated image payloads fail explicitly."""
    images, labels = fashion_idx
    with pytest.raises(ValueError, match="payload"):
        datasets.parse_fashion(images[:-1], labels)
    with pytest.raises(ValueError, match="dimensions"):
        datasets.parse_fashion(b"\0\0\0\0" + images[4:], labels)


def test_corrupt_cache_is_rejected(tmp_path: Path) -> None:
    """Existing corrupt data cannot silently enter a benchmark."""
    name = "t10k-labels-idx1-ubyte.gz"
    (tmp_path / name).write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum mismatch"):
        datasets.read_fashion_file(tmp_path, name)


def test_digits_default_is_unchanged() -> None:
    """Digits retains its full raw-pixel matrix and refuses Fashion-only sizing."""
    data, labels, meta = datasets.load_benchmark_data()
    assert data.shape == (1797, 64)
    assert labels.shape == (1797,)
    assert meta["name"] == "sklearn Digits"
    with pytest.raises(ValueError, match="only to fashion-mnist"):
        datasets.load_benchmark_data(samples=100)
