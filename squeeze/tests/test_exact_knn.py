"""Exact k-NN backend: agrees with sklearn brute force and is picked automatically."""

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

from squeeze.exact_knn import EXACT_AUTO_MAX_SAMPLES, ExactKnnIndexWrapper
from squeeze.umap_ import _get_nn_backend


@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(6, 40)).astype(np.float32) * 3
    # > 4096 rows so UMAP takes the indexed path rather than its small-data exact branch
    x = np.concatenate([c + rng.normal(size=(900, 40)).astype(np.float32) for c in centres])
    return np.ascontiguousarray(x)


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_neighbor_graph_matches_sklearn(blobs, metric):
    k = 25
    idx, dist = ExactKnnIndexWrapper(blobs, n_neighbors=k, metric=metric, block=64).neighbor_graph
    ref = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute").fit(blobs)
    ref_dist, ref_idx = ref.kneighbors(blobs)
    ref_idx, ref_dist = ref_idx[:, 1:], ref_dist[:, 1:]  # drop self
    assert idx.shape == (blobs.shape[0], k)
    assert not (idx == np.arange(blobs.shape[0])[:, None]).any(), "a row listed itself"
    # identical neighbour sets (ordering may differ only under exact ties)
    same = np.mean([set(a.tolist()) == set(b.tolist()) for a, b in zip(idx, ref_idx)])
    assert same > 0.995
    np.testing.assert_allclose(np.sort(dist, axis=1), np.sort(ref_dist, axis=1), rtol=1e-4, atol=1e-5)


def test_query_new_points(blobs):
    k = 10
    index = ExactKnnIndexWrapper(blobs, n_neighbors=k, metric="cosine")
    q = blobs[:7] + 0.01
    idx, dist = index.query(q, k)
    assert idx.shape == (7, k)
    # a barely perturbed copy of row i finds row i first
    assert (idx[:, 0] == np.arange(7)).all()
    assert (dist[:, 0] < 1e-3).all()


def test_block_size_does_not_change_result(blobs):
    a = ExactKnnIndexWrapper(blobs, n_neighbors=15, metric="euclidean", block=7).neighbor_graph
    b = ExactKnnIndexWrapper(blobs, n_neighbors=15, metric="euclidean", block=500).neighbor_graph
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_backend_selection():
    assert _get_nn_backend("cosine", False, use_hnsw="exact") is ExactKnnIndexWrapper
    assert _get_nn_backend("cosine", False, use_hnsw=None, n_samples=1000) is ExactKnnIndexWrapper
    assert _get_nn_backend("cosine", False, use_hnsw=None, n_samples=EXACT_AUTO_MAX_SAMPLES + 1) is not ExactKnnIndexWrapper
    assert _get_nn_backend("cosine", True, use_hnsw=None, n_samples=1000) is not ExactKnnIndexWrapper
    assert _get_nn_backend("hamming", False, use_hnsw=None, n_samples=1000) is not ExactKnnIndexWrapper


def test_umap_uses_exact_backend_by_default(blobs):
    import squeeze

    model = squeeze.UMAP(n_neighbors=15, n_epochs=50, random_state=0)
    emb = model.fit_transform(blobs)
    assert emb.shape == (blobs.shape[0], 2)
    assert isinstance(getattr(model, "_knn_search_index", None), ExactKnnIndexWrapper)
