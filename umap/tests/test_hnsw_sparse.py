import numpy as np
import pytest
from umap.hnsw_wrapper import HnswIndexWrapper

try:
    import scipy.sparse
    SPARSE_AVAILABLE = True
except ImportError:
    SPARSE_AVAILABLE = False

@pytest.mark.skipif(not SPARSE_AVAILABLE, reason="scipy.sparse not available")
def test_sparse_hnsw_euclidean():
    # Create random sparse matrix
    data = scipy.sparse.random(50, 5, density=0.5, format='csr', random_state=42)
    # Ensure float32
    data.data = data.data.astype(np.float32)
    
    index = HnswIndexWrapper(data, n_neighbors=5, metric='euclidean')
    
    # Query with same data
    indices, distances = index.query(data[:3], k=5)
    
    assert indices.shape == (3, 5)
    assert distances.shape == (3, 5)
    # Self should be nearest neighbor (dist ~ 0)
    assert np.all(distances[:, 0] < 1e-5)
    assert np.all(indices[:, 0] == np.arange(3))

@pytest.mark.skipif(not SPARSE_AVAILABLE, reason="scipy.sparse not available")
def test_sparse_hnsw_cosine():
    data = scipy.sparse.random(50, 5, density=0.5, format='csr', random_state=42)
    data.data = data.data.astype(np.float32)
    
    index = HnswIndexWrapper(data, n_neighbors=5, metric='cosine')
    
    indices, distances = index.query(data[:3], k=5)
    
    assert indices.shape == (3, 5)
    assert distances.shape == (3, 5)
    assert np.all(distances[:, 0] < 1e-5)

@pytest.mark.skipif(not SPARSE_AVAILABLE, reason="scipy.sparse not available")
def test_sparse_hnsw_neighbor_graph():
    data = scipy.sparse.random(50, 5, density=0.5, format='csr', random_state=42)
    data.data = data.data.astype(np.float32)
    
    index = HnswIndexWrapper(data, n_neighbors=5, metric='euclidean')
    
    indices, distances = index.neighbor_graph
    
    assert indices.shape == (50, 5)
    assert distances.shape == (50, 5)
