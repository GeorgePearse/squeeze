import numpy as np
import pytest


def test_rust_umap_fit_transform_shape() -> None:
    from squeeze import RustUMAP

    if RustUMAP is None:
        pytest.skip("Rust backend not available")

    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 12)).astype(np.float64)

    reducer = RustUMAP(n_components=2, n_neighbors=10, n_epochs=25, random_state=42)
    emb = reducer.fit_transform(X)

    assert emb.shape == (80, 2)
    assert np.isfinite(emb).all()
    # embedding is centered by the backend
    assert np.allclose(emb.mean(axis=0), 0.0, atol=1e-3)

