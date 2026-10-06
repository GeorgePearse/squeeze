"""Reproducible Digits and official Fashion-MNIST benchmark inputs."""

from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits

FASHION_REVISION = "b2617bb6d3ffa2e429640350f613e3291e10b141"
FASHION_URL = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/"
    f"{FASHION_REVISION}/data/fashion/"
)
# Integrity values published by the dataset authors (not authentication hashes).
FASHION_FILES = {
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def read_fashion_file(cache: Path, name: str) -> bytes:
    """Download/cache an official file and verify its published checksum."""
    expected = FASHION_FILES[name]
    path = cache / name
    if path.exists():
        compressed = path.read_bytes()
    else:
        # URL is built from a fixed HTTPS origin and known filenames only.
        with urllib.request.urlopen(FASHION_URL + name, timeout=60) as response:  # noqa: S310
            compressed = response.read()
    if hashlib.md5(compressed, usedforsecurity=False).hexdigest() != expected:
        message = f"Fashion-MNIST checksum mismatch: {path}; remove it and retry"
        raise ValueError(message)
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(compressed)
    return gzip.decompress(compressed)


def parse_fashion(images: bytes, labels: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Validate the official test IDX headers, dimensions and payload lengths."""
    if len(images) < 16 or len(labels) < 8:  # noqa: PLR2004 - IDX header lengths
        message = "Truncated Fashion-MNIST IDX header"
        raise ValueError(message)
    magic, count, rows, columns = struct.unpack(">IIII", images[:16])
    label_magic, label_count = struct.unpack(">II", labels[:8])
    if (magic, count, rows, columns) != (2051, 10000, 28, 28) or (
        label_magic,
        label_count,
    ) != (2049, count):
        message = "Unexpected Fashion-MNIST test IDX dimensions"
        raise ValueError(message)
    if len(images) != 16 + count * rows * columns or len(labels) != 8 + count:
        message = "Truncated or oversized Fashion-MNIST IDX payload"
        raise ValueError(message)
    targets = np.frombuffer(labels, dtype=np.uint8, offset=8)
    if np.any(targets > 9):  # noqa: PLR2004 - official class IDs
        message = "Invalid Fashion-MNIST class label"
        raise ValueError(message)
    return np.frombuffer(images, dtype=np.uint8, offset=16).reshape(count, -1), targets


def load_benchmark_data(
    dataset: str = "digits",
    samples: int | None = None,
    seed: int = 42,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return raw pixels, labels and provenance; labels only stratify sampling."""
    if dataset == "digits":
        if samples is not None:
            message = "--samples applies only to fashion-mnist; Digits uses all rows"
            raise ValueError(message)
        data, labels = load_digits(return_X_y=True)
        metadata = {"name": "sklearn Digits", "split": "all", "samples": len(data)}
    elif dataset == "fashion-mnist":
        samples = 2000 if samples is None else samples
        if samples < 100 or samples > 10000 or samples % 10:  # noqa: PLR2004
            message = "Fashion-MNIST samples must be a multiple of 10 from 100 to 10000"
            raise ValueError(message)
        cache = cache_dir or Path.home() / ".cache" / "squeeze" / "fashion-mnist"
        images, labels = parse_fashion(
            read_fashion_file(cache, "t10k-images-idx3-ubyte.gz"),
            read_fashion_file(cache, "t10k-labels-idx1-ubyte.gz"),
        )
        rng = np.random.default_rng(seed)
        indices = np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(labels == label),
                    samples // 10,
                    replace=False,
                )
                for label in range(10)
            ],
        )
        rng.shuffle(indices)
        data, labels = images[indices], labels[indices]
        metadata = {
            "name": "Fashion-MNIST",
            "split": "official test",
            "samples": samples,
            "sampling_seed": seed,
            "sampling": "equal class counts, without replacement",
            "class_names": [
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ],
            "source": FASHION_URL,
            "source_md5": FASHION_FILES,
            "indices": indices.tolist(),
        }
    else:
        message = f"Unknown benchmark dataset: {dataset}"
        raise ValueError(message)
    data = np.ascontiguousarray(data, dtype=np.float64)
    metadata.update(
        {
            "features": data.shape[1],
            "preprocessing": "raw pixels as float64; no scaling or PCA",
            "data_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(
                np.asarray(labels, dtype=np.int64).tobytes(),
            ).hexdigest(),
        },
    )
    return data, labels, metadata
