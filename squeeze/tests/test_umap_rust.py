import numpy as np
import pytest

try:
    from squeeze._hnsw_backend import UMAPRust
except Exception:  # pragma: no cover
    UMAPRust = None


pytestmark = pytest.mark.skipif(UMAPRust is None, reason="Rust backend not available")


def test_umap_rust_fit_transform_runs(iris) -> None:
    reducer = UMAPRust(n_components=2, n_neighbors=15, random_state=42)
    emb = reducer.fit_transform(iris.data)

    assert emb.shape == (iris.data.shape[0], 2)
    assert np.isfinite(emb).all()

    emb2 = reducer.embedding_
    assert emb2.shape == emb.shape


def test_umap_rust_basic_trustworthiness(iris) -> None:
    try:
        from sklearn.manifold import trustworthiness
    except Exception:  # pragma: no cover
        pytest.skip("scikit-learn trustworthiness not available")

    reducer = UMAPRust(n_components=2, n_neighbors=15, random_state=42)
    emb = reducer.fit_transform(iris.data)

    trust = trustworthiness(iris.data, emb, n_neighbors=10)
    assert trust >= 0.75
