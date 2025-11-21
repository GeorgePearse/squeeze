"""Tests for sparse matrix operations."""

import numpy as np
import pytest
import scipy.sparse as sp

from squeeze.sparse_ops import (
    SparseFormatDetector,
    SparseKNNGraph,
    SparseUMAP,
    sparse_cosine,
    sparse_euclidean,
    sparse_jaccard,
    sparse_manhattan,
)


class TestSparseFormatDetector:
    """Tests for sparse format detection and conversion."""

    def test_is_sparse_true(self):
        """Test is_sparse with sparse matrix."""
        X = sp.csr_matrix([[1, 0], [0, 1]])
        assert SparseFormatDetector.is_sparse(X)

    def test_is_sparse_false_array(self):
        """Test is_sparse with dense array."""
        X = np.array([[1, 0], [0, 1]])
        assert not SparseFormatDetector.is_sparse(X)

    def test_is_sparse_false_other(self):
        """Test is_sparse with non-matrix object."""
        assert not SparseFormatDetector.is_sparse([1, 2, 3])

    def test_get_format_sparse(self):
        """Test get_format with sparse matrix."""
        X = sp.csr_matrix([[1, 0], [0, 1]])
        assert SparseFormatDetector.get_format(X) == "csr"

    def test_get_format_dense(self):
        """Test get_format with dense array."""
        X = np.array([[1, 0], [0, 1]])
        assert SparseFormatDetector.get_format(X) is None

    def test_to_canonical_sparse_to_csr(self):
        """Test conversion from sparse to CSR."""
        X = sp.csc_matrix([[1, 0], [0, 1]])
        X_csr = SparseFormatDetector.to_canonical(X, target_format="csr")
        assert X_csr.format == "csr"
        assert X_csr.shape == (2, 2)

    def test_to_canonical_dense_to_csr(self):
        """Test conversion from dense to CSR."""
        X = np.array([[1, 0], [0, 1]])
        X_sparse = SparseFormatDetector.to_canonical(X, target_format="csr")
        assert sp.issparse(X_sparse)
        assert X_sparse.format == "csr"

    def test_to_canonical_invalid_format(self):
        """Test error on invalid format."""
        X = np.array([[1, 0], [0, 1]])
        with pytest.raises(ValueError):
            SparseFormatDetector.to_canonical(X, target_format="invalid")

    def test_get_sparsity_dense(self):
        """Test sparsity computation on dense array."""
        X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        sparsity = SparseFormatDetector.get_sparsity(X)
        assert sparsity == pytest.approx(2.0 / 3.0)

    def test_get_sparsity_sparse(self):
        """Test sparsity computation on sparse matrix."""
        X = sp.csr_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        sparsity = SparseFormatDetector.get_sparsity(X)
        assert sparsity == pytest.approx(2.0 / 3.0)

    def test_get_sparsity_zero_matrix(self):
        """Test sparsity of zero matrix."""
        X = np.zeros((3, 3))
        sparsity = SparseFormatDetector.get_sparsity(X)
        assert sparsity == 1.0

    def test_suggest_format(self):
        """Test format suggestion."""
        X = sp.random(100, 100, density=0.01, format="csr")
        suggested = SparseFormatDetector.suggest_format(X)
        assert suggested in ["csr", "csc", "coo"]


class TestSparseDistances:
    """Tests for sparse distance computations."""

    @pytest.fixture
    def sparse_data(self):
        """Create sparse test data."""
        X = sp.random(20, 50, density=0.1, format="csr", random_state=42)
        Y = sp.random(15, 50, density=0.1, format="csr", random_state=43)
        return X, Y

    def test_sparse_euclidean_shape(self, sparse_data):
        """Test sparse euclidean distance shape."""
        X, Y = sparse_data
        distances = sparse_euclidean(X, Y)
        assert distances.shape == (20, 15)

    def test_sparse_euclidean_self(self, sparse_data):
        """Test sparse euclidean with self (should be identity-like diagonal)."""
        X, _ = sparse_data
        distances = sparse_euclidean(X, X)
        # Diagonal should be nearly zero (allow for numerical error)
        diag = np.diag(distances)
        assert np.allclose(diag, 0, atol=1e-6)

    def test_sparse_euclidean_squared(self, sparse_data):
        """Test squared euclidean distances."""
        X, Y = sparse_data
        dist_squared = sparse_euclidean(X, Y, squared=True)
        dist_normal = sparse_euclidean(X, Y, squared=False)
        # Squared should be sqrt of normal
        assert np.allclose(dist_squared, dist_normal**2)

    def test_sparse_euclidean_symmetry(self, sparse_data):
        """Test that euclidean distances are symmetric."""
        X, Y = sparse_data
        dist_XY = sparse_euclidean(X, Y)
        dist_YX = sparse_euclidean(Y, X)
        assert np.allclose(dist_XY.T, dist_YX)

    def test_sparse_cosine_shape(self, sparse_data):
        """Test sparse cosine distance shape."""
        X, Y = sparse_data
        distances = sparse_cosine(X, Y)
        assert distances.shape == (20, 15)

    def test_sparse_cosine_range(self, sparse_data):
        """Test cosine distances are in [0, 2]."""
        X, Y = sparse_data
        distances = sparse_cosine(X, Y)
        assert np.all(distances >= 0)
        assert np.all(distances <= 2)

    def test_sparse_manhattan_shape(self, sparse_data):
        """Test sparse manhattan distance shape."""
        X, Y = sparse_data
        distances = sparse_manhattan(X, Y)
        assert distances.shape == (20, 15)

    def test_sparse_manhattan_nonnegative(self, sparse_data):
        """Test manhattan distances are non-negative."""
        X, Y = sparse_data
        distances = sparse_manhattan(X, Y)
        assert np.all(distances >= 0)

    def test_sparse_jaccard_shape(self, sparse_data):
        """Test sparse jaccard distance shape."""
        X, Y = sparse_data
        distances = sparse_jaccard(X, Y)
        assert distances.shape == (20, 15)

    def test_sparse_jaccard_range(self, sparse_data):
        """Test jaccard distances are in [0, 1]."""
        X, Y = sparse_data
        distances = sparse_jaccard(X, Y)
        assert np.all(distances >= 0)
        assert np.all(distances <= 1)

    def test_mixed_dense_sparse_euclidean(self):
        """Test euclidean with mixed dense and sparse inputs."""
        X_dense = np.random.randn(10, 20)
        Y_sparse = sp.random(15, 20, density=0.1, format="csr", random_state=42)
        distances = sparse_euclidean(X_dense, Y_sparse)
        assert distances.shape == (10, 15)


class TestSparseKNNGraph:
    """Tests for sparse k-NN graph construction."""

    @pytest.fixture
    def sparse_data(self):
        """Create sparse test data."""
        return sp.random(50, 100, density=0.05, format="csr", random_state=42)

    def test_knn_graph_initialization(self):
        """Test KNN graph initialization."""
        knn = SparseKNNGraph(n_neighbors=15, metric="euclidean")
        assert knn.n_neighbors == 15
        assert knn.metric == "euclidean"

    def test_knn_graph_fit(self, sparse_data):
        """Test KNN graph fitting."""
        knn = SparseKNNGraph(n_neighbors=10)
        knn.fit(sparse_data)
        assert knn.knn_indices_ is not None
        assert knn.knn_distances_ is not None

    def test_knn_graph_shape(self, sparse_data):
        """Test KNN graph output shapes."""
        knn = SparseKNNGraph(n_neighbors=10)
        knn.fit(sparse_data)
        assert knn.knn_indices_.shape == (50, 10)
        assert knn.knn_distances_.shape == (50, 10)

    def test_knn_graph_distances_nonnegative(self, sparse_data):
        """Test KNN distances are non-negative."""
        knn = SparseKNNGraph(n_neighbors=10)
        knn.fit(sparse_data)
        assert np.all(knn.knn_distances_ >= 0)

    def test_knn_graph_valid_indices(self, sparse_data):
        """Test KNN indices are valid."""
        n_samples = sparse_data.shape[0]
        knn = SparseKNNGraph(n_neighbors=10)
        knn.fit(sparse_data)
        # Indices should be in valid range [0, n_samples-1]
        assert np.all(knn.knn_indices_ >= 0)
        assert np.all(knn.knn_indices_ < n_samples)

    def test_knn_graph_fit_predict(self, sparse_data):
        """Test fit_predict method."""
        knn = SparseKNNGraph(n_neighbors=10)
        indices = knn.fit_predict(sparse_data)
        assert indices.shape == (50, 10)

    def test_knn_graph_cosine_metric(self, sparse_data):
        """Test KNN with cosine metric."""
        knn = SparseKNNGraph(n_neighbors=5, metric="cosine")
        knn.fit(sparse_data)
        assert knn.knn_indices_.shape == (50, 5)

    def test_knn_graph_manhattan_metric(self, sparse_data):
        """Test KNN with manhattan metric."""
        knn = SparseKNNGraph(n_neighbors=5, metric="manhattan")
        knn.fit(sparse_data)
        assert knn.knn_indices_.shape == (50, 5)

    def test_knn_graph_invalid_metric(self, sparse_data):
        """Test error on invalid metric."""
        knn = SparseKNNGraph(n_neighbors=5, metric="invalid")
        with pytest.raises(ValueError):
            knn.fit(sparse_data)


class TestSparseUMAP:
    """Tests for sparse UMAP wrapper."""

    @pytest.fixture
    def sparse_data(self):
        """Create sparse test data."""
        return sp.random(50, 100, density=0.05, format="csr", random_state=42)

    def test_sparse_umap_initialization(self):
        """Test SparseUMAP initialization."""
        umap = SparseUMAP(n_components=2, n_neighbors=15)
        assert umap.n_components == 2
        assert umap.n_neighbors == 15

    def test_sparse_umap_fit_transform(self, sparse_data):
        """Test fit_transform on sparse data."""
        umap = SparseUMAP(n_components=2, random_state=42)
        embedding = umap.fit_transform(sparse_data)
        assert embedding.shape == (50, 2)

    def test_sparse_umap_fit_transform_dense(self):
        """Test fit_transform works with dense data too."""
        X = np.random.randn(50, 100)
        umap = SparseUMAP(n_components=2, random_state=42)
        embedding = umap.fit_transform(X)
        assert embedding.shape == (50, 2)

    def test_sparse_umap_fit_and_transform(self, sparse_data):
        """Test separate fit and transform."""
        umap = SparseUMAP(n_components=2, random_state=42)
        umap.fit(sparse_data)

        # Transform should work after fit
        embedding = umap.transform(sparse_data)
        assert embedding.shape == (50, 2)

    def test_sparse_umap_transform_before_fit(self, sparse_data):
        """Test error when transform before fit."""
        umap = SparseUMAP(n_components=2)
        with pytest.raises(ValueError):
            umap.transform(sparse_data)

    def test_sparse_umap_custom_metric(self, sparse_data):
        """Test SparseUMAP with custom metric."""
        umap = SparseUMAP(n_components=2, metric="cosine", random_state=42)
        embedding = umap.fit_transform(sparse_data)
        assert embedding.shape == (50, 2)


class TestSparseIntegration:
    """Integration tests for sparse operations."""

    def test_very_sparse_data(self):
        """Test with very sparse data (95%+ sparse)."""
        # Create 95% sparse data
        X = sp.random(100, 500, density=0.05, format="csr", random_state=42)
        sparsity = SparseFormatDetector.get_sparsity(X)
        assert sparsity >= 0.95

        # Should still compute distances
        distances = sparse_euclidean(X[:10], X[10:20])
        assert distances.shape == (10, 10)

    def test_workflow_sparse_data(self):
        """Test complete workflow with sparse data."""
        # Create sparse data
        X = sp.random(50, 100, density=0.05, format="csr", random_state=42)

        # Detect format
        assert SparseFormatDetector.is_sparse(X)
        fmt = SparseFormatDetector.get_format(X)
        assert fmt == "csr"

        # Build k-NN graph
        knn = SparseKNNGraph(n_neighbors=10)
        knn.fit(X)
        assert knn.knn_indices_.shape == (50, 10)

        # Compute embedding
        umap = SparseUMAP(n_components=2, random_state=42)
        embedding = umap.fit_transform(X)
        assert embedding.shape == (50, 2)

    def test_sparse_vs_dense_equivalence(self):
        """Test that sparse and dense give similar results."""
        # Create small test data
        X_dense = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]).astype(float)
        X_sparse = sp.csr_matrix(X_dense)

        # Compute distances both ways
        # Take subset to make dense computation tractable
        dist_dense = sparse_euclidean(X_dense[:3], X_dense[3:])
        dist_sparse = sparse_euclidean(X_sparse[:3], X_sparse[3:])

        assert np.allclose(dist_dense, dist_sparse, atol=1e-10)
