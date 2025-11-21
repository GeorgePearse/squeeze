# UMAP ANN Implementation - Key Code Locations

## File Structure and Key Locations

### 1. Main UMAP Module
**File**: `/home/georgepearse/umap/umap/umap_.py`

| Component | Lines | Purpose |
|-----------|-------|---------|
| Import NNDescent | 27-28 | Core ANN library import |
| `nearest_neighbors()` function | 247-340 | Main ANN entry point - builds index and returns neighbors |
| NNDescent instantiation | 322-335 | Creates the search index with all parameters |
| `smooth_knn_dist()` function | 165-244 | Smooths k-NN distances for UMAP (used after neighbors found) |
| UMAP.__init__() | ~1700+ | Store ANN parameters |
| UMAP.fit() | ~2100+ | Call nearest_neighbors() during training |
| UMAP.transform() | 3045-3250 | Use knn_search_index.query() for inference |
| Transform small data fallback | 3150-3194 | Uses sklearn pairwise_distances for small datasets |
| Transform large data query | 3195-3203 | **Key query call**: `self._knn_search_index.query(X, self.n_neighbors, epsilon=epsilon)` |

### 2. Utilities Module
**File**: `/home/georgepearse/umap/umap/utils.py`

| Component | Lines | Purpose |
|-----------|-------|---------|
| `fast_knn_indices()` | 22-44 | Numba-JIT'd function to extract k-smallest from sorted array (for precomputed metrics) |
| `submatrix()` | 114-143 | Extract submatrix for sorting operations |

### 3. Distance Metrics
**File**: `/home/georgepearse/umap/umap/distances.py`

| Component | Purpose |
|-----------|---------|
| Various distance functions | Euclidean, Manhattan, Cosine, Correlation, etc. (all Numba-JIT'd) |
| `pairwise_special_metric()` | Compute full pairwise distance matrices |
| `named_distances` dict | Maps metric names to functions |

### 4. Sparse Data Support
**File**: `/home/georgepearse/umap/umap/sparse.py`

| Component | Purpose |
|-----------|---------|
| `sparse_named_distances` dict | Sparse versions of distance functions |
| Sparse metric implementations | For scipy.sparse CSR matrices |

### 5. Tests
**File**: `/home/georgepearse/umap/umap/tests/test_umap_nn.py`

| Test | Purpose |
|------|---------|
| `test_nn_bad_metric()` | Validates metric handling |
| `test_nn_descent_neighbor_accuracy()` | Tests NN-Descent quality (currently skipped) |
| `test_smooth_knn_dist_l1norms()` | Tests sigma/rho smoothing |

---

## Critical Code Snippets

### Snippet 1: Creating NNDescent Index (lines 322-335)
```python
knn_search_index = NNDescent(
    X,                              # Input data
    n_neighbors=n_neighbors,        # k value
    metric=metric,                  # Distance metric (string or callable)
    metric_kwds=metric_kwds,        # Metric parameters
    random_state=random_state,      # Reproducibility
    n_trees=min(64, 5 + round((X.shape[0]) ** 0.5 / 20.0)),     # Adaptive
    n_iters=max(5, round(np.log2(X.shape[0]))),                  # Adaptive
    max_candidates=60,              # Fixed max candidates per iteration
    low_memory=low_memory,          # Memory optimization
    n_jobs=n_jobs,                  # Parallelization
    verbose=verbose,                # Logging
    compressed=False,               # No compression
)
knn_indices, knn_dists = knn_search_index.neighbor_graph
```

### Snippet 2: Querying Index During Transform (lines 3195-3203)
```python
angular_trees = getattr(self._knn_search_index, "_angular_trees", False)
epsilon = 0.24 if angular_trees else 0.12  # Search parameter
indices, dists = self._knn_search_index.query(
    X,                  # New test points
    self.n_neighbors,   # k value
    epsilon=epsilon,    # Search radius/iterations
)
```

### Snippet 3: Small Dataset Fallback (lines 3150-3194)
```python
# Uses exact computation with sklearn
dmat = pairwise_distances(X, self._raw_data, metric=_m, **self._metric_kwds)
indices = np.argpartition(dmat, self._n_neighbors)[:, : self._n_neighbors]
dmat_shortened = submatrix(dmat, indices, self._n_neighbors)
indices_sorted = np.argsort(dmat_shortened)
indices = submatrix(indices, indices_sorted, self._n_neighbors)
dists = submatrix(dmat_shortened, indices_sorted, self._n_neighbors)
```

---

## Data Flow Architecture

```
User Input Data (X)
        ↓
nearest_neighbors(X, metric, ...)
        ↓
    ┌─────────────────┐
    │ Is precomputed? │
    └────┬────────┬───┘
         │        │
        YES      NO
         │        │
         ↓        ↓
    fast_knn_  NNDescent(
    indices()      X,
                   metric=metric,
                   n_trees=...,
                   n_iters=...)
         │        │
         └────┬───┘
              ↓
      (knn_indices, knn_dists, 
       knn_search_index)
              ↓
        UMAP stores in
        self._knn_search_index
              ↓
      ┌──────────────┐
      │  fit() done  │
      └────┬─────────┘
           │
           ↓
      [Training]
      
      During transform(X_new):
           ↓
      knn_search_index.query(
          X_new, 
          k=n_neighbors, 
          epsilon=epsilon)
           ↓
      (new_indices, new_dists)
           ↓
      [Embedding new data]
```

---

## Key Parameter Behavior

### NNDescent Parameters

| Parameter | Value Range | Default in UMAP | Effect |
|-----------|------------|-----------------|--------|
| `n_neighbors` | 2+ | 15 | k value - how many neighbors to find |
| `metric` | string or callable | "euclidean" | Distance function |
| `n_trees` | 1-64 | Dynamic | Number of RP trees (more = slower build, better quality) |
| `n_iters` | 5+ | Dynamic | Refinement iterations (more = better quality) |
| `max_candidates` | 20-200 | 60 | Candidates per iteration (higher = slower, more thorough) |
| `low_memory` | True/False | True | Use less memory at cost of speed |
| `n_jobs` | -1, 1, N | -1 | Parallel jobs (-1 = all cores) |

### Dynamic Parameter Formulas (in UMAP)

```python
n_trees = min(64, 5 + round((X.shape[0]) ** 0.5 / 20.0))
# Examples:
# 100 samples   → n_trees = 5
# 400 samples   → n_trees = 6
# 2500 samples  → n_trees = 10
# 40000 samples → n_trees = 35
# 100000+ samples → n_trees = 64 (capped)

n_iters = max(5, round(np.log2(X.shape[0])))
# Examples:
# 32 samples    → n_iters = 5
# 256 samples   → n_iters = 8
# 1024 samples  → n_iters = 10
# 65536 samples → n_iters = 16
```

### Query Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `epsilon` | 0.12 (default) or 0.24 (angular) | Search depth - higher = more thorough but slower |
| `k` (n_neighbors) | Integer | Number of neighbors to return |

---

## Integration Points Requiring Rust Implementation

### 1. Build-Time (fit)
**Must provide**:
- `__init__(X, n_neighbors, metric, metric_kwds, random_state, n_trees, n_iters, max_candidates, low_memory, n_jobs, verbose, compressed)`
- `neighbor_graph` property that returns `(indices, distances)`

**Called from**: `umap_.py` line 322-336

**Example usage**:
```python
index = NNDescent(data, n_neighbors=15, metric="euclidean", ...)
knn_indices, knn_dists = index.neighbor_graph
```

### 2. Query-Time (transform)  
**Must provide**:
- `query(X, k, epsilon=...)`
- Returns `(indices, distances)` for new points

**Called from**: `umap_.py` line 3199-3203

**Example usage**:
```python
indices, dists = index.query(new_data, 15, epsilon=0.12)
```

### 3. Properties and Attributes
**Must support**:
- `._angular_trees` - Boolean flag for angular metric detection
- `._raw_data` - Access to training data (optional but used in some code paths)
- Standard Python properties via PyO3/PyPEG

---

## Testing Entry Points

### Direct Tests of ANN Functionality
- **File**: `/home/georgepearse/umap/umap/tests/test_umap_nn.py`
- Tests call `nearest_neighbors()` directly
- Validate neighbor accuracy against sklearn's KDTree

### Integration Tests
- **File**: `/home/georgepearse/umap/umap/tests/test_umap_on_iris.py`
- Tests full UMAP workflow including ANN
- Tests transform() with pre-fitted model

---

## Performance Expectations

Based on current PyNNDescent:
- **Build time**: O(N log N) typical, can be O(N^1.5) worst case
- **Query time**: O(log N) per query typical  
- **Memory**: O(k × N) for storing k neighbors per N points
- **Accuracy**: 85-95% of true k-NN identified (approximate, not exact)

For Rust implementation, should match or exceed these characteristics.

