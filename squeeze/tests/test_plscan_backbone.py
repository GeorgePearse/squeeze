import numpy as np

from squeeze import PLSCAN, PLSCANBackbone, get_strategy


def test_plscan_cluster_shapes(iris) -> None:
    x = iris.data.astype(np.float64)
    clusterer = PLSCAN(min_samples=5)
    labels = clusterer.fit_predict(x)

    assert labels.shape == (x.shape[0],)
    assert clusterer.probabilities_.shape == (x.shape[0],)
    assert np.all((clusterer.probabilities_ >= 0.0) & (clusterer.probabilities_ <= 1.0))
    assert clusterer.trace_min_size_.shape == clusterer.trace_persistence_.shape


def test_plscan_backbone_fit_transform_shapes(iris) -> None:
    x = iris.data.astype(np.float64)
    reducer = PLSCANBackbone(
        n_components=2,
        min_samples=5,
        rep_strategy="high_prob",
        reps_per_cluster=1,
        must_link_weight=0.1,
        interpolation_k=3,
        restrict_to_cluster=True,
    )
    embedding = reducer.fit_transform(x)

    assert embedding.shape == (x.shape[0], 2)
    assert reducer.labels_.shape == (x.shape[0],)
    assert reducer.probabilities_.shape == (x.shape[0],)
    assert reducer.rep_indices_.ndim == 1
    assert reducer.trace_min_size_.shape == reducer.trace_persistence_.shape


def test_strategy_registry_has_plscan_backbone() -> None:
    strategy = get_strategy("plscan_backbone")
    assert strategy.name == "plscan_backbone"
