"""Tests for out-of-sample extension and streaming DR."""

import numpy as np
import pytest
from sklearn.datasets import load_digits

from squeeze import PCA
from squeeze.extensions import OutOfSampleDR, StreamingDR


@pytest.fixture
def digits_data():
    """Load a subset of digits for testing."""
    X, y = load_digits(return_X_y=True)
    # Use smaller subset for faster tests
    return X[:500].astype(np.float64), y[:500]


class TestOutOfSampleDR:
    """Tests for OutOfSampleDR wrapper."""

    def test_fit_transform(self, digits_data):
        """Test basic fit_transform."""
        X, _ = digits_data
        X_train = X[:400]

        # Wrap PCA
        wrapped = OutOfSampleDR(PCA(n_components=2), n_neighbors=5)
        X_train_emb = wrapped.fit_transform(X_train)

        assert X_train_emb.shape == (400, 2)
        assert hasattr(wrapped, "embedding_")
        assert hasattr(wrapped, "nn_")

    def test_transform(self, digits_data):
        """Test transform on new data."""
        X, _ = digits_data
        X_train, X_test = X[:400], X[400:]

        wrapped = OutOfSampleDR(PCA(n_components=2), n_neighbors=5)
        wrapped.fit(X_train)
        X_test_emb = wrapped.transform(X_test)

        assert X_test_emb.shape == (100, 2)

    def test_weights_uniform(self, digits_data):
        """Test uniform weighting."""
        X, _ = digits_data
        X_train, X_test = X[:400], X[400:410]

        wrapped = OutOfSampleDR(PCA(n_components=2), n_neighbors=3, weights="uniform")
        wrapped.fit(X_train)
        X_test_emb = wrapped.transform(X_test)

        assert X_test_emb.shape == (10, 2)

    def test_weights_distance(self, digits_data):
        """Test distance weighting."""
        X, _ = digits_data
        X_train, X_test = X[:400], X[400:410]

        wrapped = OutOfSampleDR(PCA(n_components=2), n_neighbors=3, weights="distance")
        wrapped.fit(X_train)
        X_test_emb = wrapped.transform(X_test)

        assert X_test_emb.shape == (10, 2)

    def test_not_fitted_error(self, digits_data):
        """Test error when transform called before fit."""
        X, _ = digits_data

        wrapped = OutOfSampleDR(PCA(n_components=2))
        with pytest.raises(Exception):  # sklearn raises NotFittedError
            wrapped.transform(X[:10])


class TestStreamingDR:
    """Tests for StreamingDR wrapper."""

    def test_fit_transform(self, digits_data):
        """Test initial fit_transform."""
        X, _ = digits_data

        streaming = StreamingDR(PCA(n_components=2))
        embedding = streaming.fit_transform(X[:200])

        assert embedding.shape == (200, 2)
        assert streaming.X_all_.shape[0] == 200

    def test_partial_fit(self, digits_data):
        """Test incremental updates."""
        X, _ = digits_data

        streaming = StreamingDR(PCA(n_components=2))
        streaming.fit(X[:200])

        # Add new data
        embedding = streaming.partial_fit_transform(X[200:250])

        assert embedding.shape == (250, 2)
        assert streaming.X_all_.shape[0] == 250

    def test_transform_without_adding(self, digits_data):
        """Test transform without adding to model."""
        X, _ = digits_data

        streaming = StreamingDR(PCA(n_components=2))
        streaming.fit(X[:200])

        # Transform without adding
        X_test_emb = streaming.transform(X[200:210])

        assert X_test_emb.shape == (10, 2)
        assert streaming.X_all_.shape[0] == 200  # Not changed

    def test_multiple_partial_fits(self, digits_data):
        """Test multiple partial_fit calls."""
        X, _ = digits_data

        streaming = StreamingDR(PCA(n_components=2))
        streaming.fit(X[:100])

        for i in range(4):
            start = 100 + i * 50
            end = start + 50
            streaming.partial_fit(X[start:end])

        assert streaming.X_all_.shape[0] == 300
        assert streaming.embedding_.shape == (300, 2)

    def test_embedding_consistency(self, digits_data):
        """Test that original points keep their embeddings after partial_fit."""
        X, _ = digits_data

        streaming = StreamingDR(PCA(n_components=2))
        embedding_initial = streaming.fit_transform(X[:100])

        # Add new data
        streaming.partial_fit(X[100:150])

        # First 100 points should have same embeddings
        np.testing.assert_array_almost_equal(
            embedding_initial, streaming.embedding_[:100]
        )
