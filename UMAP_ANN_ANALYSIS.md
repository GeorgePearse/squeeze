# UMAP Approximate Nearest Neighbors (ANN) Implementation Analysis

## Executive Summary

UMAP uses **PyNNDescent** as its primary approximate nearest neighbor (ANN) library. PyNNDescent implements the NN-Descent algorithm, which is a general-purpose nearest neighbor descent algorithm designed for approximate nearest neighbor search. The implementation is highly tuned for various distance metrics and supports both dense and sparse data.

---

## 1. ANN Library Used: PyNNDescent

### Key Dependency
- **Library**: `pynndescent >= 0.5` (from pyproject.toml)
- **Purpose**: Efficient approximate nearest neighbor search
- **Location**: Imported in `/home/georgepearse/umap/umap/umap_.py` at line 27
  ```python
  from pynndescent import NNDescent
  from pynndescent.distances import named_distances as pynn_named_distances
  from pynndescent.sparse import sparse_named_distances as pynn_sparse_named_distances
  ```

### Algorithm: NN-Descent
- A greedy nearest neighbor descent algorithm
- Builds an approximate nearest neighbor graph through iterative refinement
- Uses **Random Projection (RP) forests** for initialization
- Performs local neighbor exchange for refinement
- Provides **approximate results** with high accuracy (typically >85%)

---

## 2. Main Entry Point: `nearest_neighbors()` Function

**File**: `/home/georgepearse/umap/umap/umap_.py`, lines 247-340

### Function Signature
```python
def nearest_neighbors(
    X,                    # Input data array (n_samples, n_features)
    n_neighbors,          # Number of neighbors to find
    metric,               # Distance metric (string or callable)
    metric_kwds,          # Keyword arguments for metric
    angular,              # Whether to use angular RP trees
    random_state,         # Random state for reproducibility
    low_memory=True,      # Memory-efficient mode
    use_pynndescent=True, # Use PyNNDescent (always True currently)
    n_jobs=-1,            # Number of parallel jobs
    verbose=False,        # Verbose output
):
    """
    Returns:
    - knn_indices: array of shape (n_samples, n_neighbors)
    - knn_dists: array of shape (n_samples, n_neighbors) 
    - knn_search_index: NNDescent object for later queries
    """
```

### Implementation Details

#### For Precomputed Distances (metric == "precomputed")
- Uses `fast_knn_indices()` (a Numba-JIT'd function)
- Simply sorts precomputed distance matrix to find k smallest values
- No actual ANN search needed
- Location: lines 304-316

#### For Regular Metrics
- **Creates NNDescent object** (lines 322-335):
  ```python
  knn_search_index = NNDescent(
      X,
      n_neighbors=n_neighbors,
      metric=metric,
      metric_kwds=metric_kwds,
      random_state=random_state,
      n_trees=n_trees,           # Dynamic: min(64, 5 + round((X.shape[0]) ** 0.5 / 20.0))
      n_iters=n_iters,           # Dynamic: max(5, round(np.log2(X.shape[0])))
      max_candidates=60,         # Maximum candidates in search
      low_memory=low_memory,     # Memory optimization
      n_jobs=n_jobs,             # Parallelization
      verbose=verbose,
      compressed=False,          # No compression
  )
  ```

- **Key Parameters**:
  - `n_trees`: Number of RP trees (adaptive based on dataset size)
  - `n_iters`: Number of refinement iterations (adaptive based on dataset size)
  - `max_candidates`: Max candidates considered per query (fixed at 60)
  - `low_memory`: Trades speed for reduced memory usage

- **Returns neighbor graph**:
  ```python
  knn_indices, knn_dists = knn_search_index.neighbor_graph
  ```

---

## 3. NNDescent Object Usage

### Build Time Usage (during fit)
**Location**: Called in `UMAP.fit()` and related methods
- Builds the approximate nearest neighbor graph for training data
- Stores the index for later transformation queries

### Query Time Usage (during transform)
**Location**: `/home/georgepearse/umap/umap/umap_.py`, lines 3195-3203

```python
else:  # Large datasets use NNDescent.query()
    angular_trees = getattr(self._knn_search_index, "_angular_trees", False)
    epsilon = 0.24 if angular_trees else 0.12  # Search depth parameter
    indices, dists = self._knn_search_index.query(
        X,                    # New data points to query
        self.n_neighbors,     # Number of neighbors to find
        epsilon=epsilon,      # Search accuracy parameter
    )
```

**Key Query Parameters**:
- `epsilon`: Controls search radius (higher = more thorough but slower)
  - 0.24 for angular metrics
  - 0.12 for other metrics

---

## 4. Distance Metrics Support

### Imported Metrics
UMAP maintains its own implementations of distance metrics in:
- **File**: `/home/georgepearse/umap/umap/distances.py`
- **Technology**: Numba-JIT compiled for performance
- **Examples**: euclidean, manhattan, cosine, correlation, hamming, jaccard, etc.

### Metric Integration
- PyNNDescent supports both:
  - **Named string metrics**: "euclidean", "cosine", "manhattan", etc.
  - **Custom callable metrics**: User-defined or Numba-JIT'd functions

### Sparse Data Support
- PyNNDescent has sparse versions:
  ```python
  from pynndescent.sparse import sparse_named_distances as pynn_sparse_named_distances
  ```
- Used for scipy.sparse matrices (CSR format)

---

## 5. Functionality That Needs to be Replicated in Rust

### Core Functionality (Essential)
1. **NN-Descent Algorithm Implementation**:
   - Random projection forest construction
   - Nearest neighbor descent refinement loop
   - Local graph optimization

2. **Neighbor Graph Output**:
   - Return `(indices, distances)` tuples
   - Support k-nearest neighbors format (n_samples × k)
   - Efficient storage and lookup

3. **Query Interface**:
   - Given trained index, query new points
   - Support k-nearest neighbors queries
   - Return approximate neighbors within acceptable error tolerance

4. **Distance Metrics**:
   - Euclidean distance (essential)
   - Cosine distance (for angular trees)
   - Manhattan distance
   - Minkowski distance
   - Support for custom metrics (callback interface)

### Secondary Functionality (Useful for Compatibility)
1. **Adaptive Parameters**:
   - Compute `n_trees` based on dataset size
   - Compute `n_iters` based on dataset size
   
2. **Query Parameters**:
   - Epsilon/search depth parameter for balancing speed vs accuracy
   - Parallel search across multiple trees

3. **Data Format Support**:
   - Dense numpy arrays (float32, float64)
   - Sparse matrices (CSR format) - optional
   - Integer-like indexing

4. **Options**:
   - `low_memory` mode (uses iterative refinement to reduce memory)
   - `angular` mode (for cosine/angular distances)
   - `n_jobs` parallelization
   - Random seed control for reproducibility

---

## 6. Current Integration Points

### During Model Training (fit)
1. `UMAP.fit()` calls `nearest_neighbors()` with training data
2. Returns `(knn_indices, knn_dists, knn_search_index)`
3. Stores `knn_search_index` as `self._knn_search_index`

### During Inference (transform)
1. `UMAP.transform()` receives new data points X
2. Calls `self._knn_search_index.query(X, n_neighbors, epsilon=epsilon)`
3. Gets neighbors of new points in the training space
4. Uses neighbor relationships to position new points in embedding

### Pre-training Options
1. User can provide `precomputed_knn=(indices, dists, index)` tuple
2. Bypasses nearest neighbor search if already computed

---

## 7. Key Implementation Characteristics

### Data Flow
```
Dense Data → NNDescent(metric) → Neighbor Graph → UMAP Layout Optimization
                                    ↓
                          Used for fuzzy simplicial set construction
```

### Performance Characteristics
- **Time Complexity (Build)**: O(N log N) for NN-Descent with proper parameterization
- **Time Complexity (Query)**: O(log N) average per query with epsilon parameter
- **Space Complexity**: O(k × N) for storing k neighbors per point
- **Approximation Quality**: Typically 85-95% of true k-NN neighbors identified

### Robustness Features
- Handles disconnected components (infinite distances)
- Filters neighbors based on `disconnection_distance` parameter
- Graceful fallback for small datasets (uses pairwise_distances)
- Supports various sparse input formats

---

## 8. Test Files Reference

**Main NN tests**: `/home/georgepearse/umap/umap/tests/test_umap_nn.py`
- Tests for bad metrics handling
- Neighbor accuracy tests (currently skipped)
- Smooth k-NN distance tests
- Both dense and sparse data tests

---

## 9. Alternative Implementations and Fallbacks

### Small Dataset Fallback
When dataset is small enough to fit in memory:
- Uses `sklearn.metrics.pairwise_distances()` for exact computation
- Applies `np.argpartition()` for efficiency
- No approximate algorithm needed

### Precomputed Distance Support
- Allows passing precomputed distance matrices directly
- Uses `fast_knn_indices()` for sorting
- Useful when distance metric cannot be easily computed

---

## Summary: What to Replicate in Rust

### Minimum Viable Implementation
1. NN-Descent algorithm core
2. Basic Euclidean distance
3. Query interface with epsilon parameter
4. Return (indices, distances) tuples
5. Random seed support for reproducibility

### Full Implementation
1. NN-Descent with angular RP trees
2. Multiple distance metrics (euclidean, cosine, manhattan, minkowski)
3. Sparse data support
4. Adaptive parameter computation
5. Parallel search capability
6. Custom metric callback support
7. Low-memory mode implementation

### Integration Points
- Match PyNNDescent's `NNDescent` class interface
- Support `.neighbor_graph` property
- Support `.query(X, k, epsilon=...)` method
- Return NumPy-compatible arrays
- Support Python type annotations/hints
