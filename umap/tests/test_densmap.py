import pytest

from umap import UMAP

try:
    # works for sklearn>=0.22
    from sklearn.manifold import trustworthiness
except ImportError:
    # this is to comply with requirements (scikit-learn>=0.20)
    # More recent versions of sklearn have exposed trustworthiness
    # in top level module API
    # see: https://github.com/scikit-learn/scikit-learn/pull/15337
    from sklearn.manifold.t_sne import trustworthiness


def test_densmap_trustworthiness(nn_data) -> None:
    data = nn_data[:50]
    embedding, _rad_h, _rad_l = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        n_epochs=100,
        densmap=True,
        output_dens=True,
    ).fit_transform(data)
    trust = trustworthiness(data, embedding, n_neighbors=10)
    assert trust >= 0.72, f"Insufficiently trustworthy embedding fornn dataset: {trust}"


@pytest.mark.skip
def test_densmap_trustworthiness_random_init(nn_data) -> None:  # pragma: no cover
    data = nn_data[:50]
    embedding = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        init="random",
        densmap=True,
    ).fit_transform(data)
    trust = trustworthiness(data, embedding, n_neighbors=10)
    assert trust >= 0.75, f"Insufficiently trustworthy embedding fornn dataset: {trust}"


def test_densmap_trustworthiness_on_iris(iris) -> None:
    densmap_iris_model = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        densmap=True,
        verbose=True,
    ).fit(iris.data)
    embedding = densmap_iris_model.embedding_
    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert trust >= 0.97, (
        f"Insufficiently trustworthy embedding foriris dataset: {trust}"
    )

    with pytest.raises(NotImplementedError):
        densmap_iris_model.transform(iris.data[:10])

    with pytest.raises(ValueError):
        densmap_iris_model.inverse_transform(embedding[:10])


def test_densmap_trustworthiness_on_iris_supervised(iris) -> None:
    densmap_iris_model = UMAP(
        n_neighbors=10,
        min_dist=0.01,
        random_state=42,
        densmap=True,
        verbose=True,
    ).fit(iris.data, y=iris.target)
    embedding = densmap_iris_model.embedding_
    trust = trustworthiness(iris.data, embedding, n_neighbors=10)
    assert trust >= 0.97, (
        f"Insufficiently trustworthy embedding foriris dataset: {trust}"
    )
