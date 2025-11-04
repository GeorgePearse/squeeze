# UMAP Approximate Nearest Neighbors Implementation - Exploration Summary

## Overview

This document summarizes the exploration of the UMAP codebase to understand its current ANN (Approximate Nearest Neighbors) implementation. The exploration identified what library is used, key files involved, and what functionality needs to be replicated in Rust.

---

## Key Findings

### 1. Primary ANN Library: PyNNDescent

**Library**: `pynndescent >= 0.5`
**Algorithm**: Nearest Neighbor Descent (NN-Descent)
**Description**: A general-purpose algorithm for approximate nearest neighbor search that builds an approximate k-NN graph through iterative refinement using Random Projection (RP) forests.

**Import Location**: `/home/georgepearse/umap/umap/umap_.py`, lines 27-28

```python
from pynndescent import NNDescent
from pynndescent.distances import named_distances as pynn_named_distances
from pynndescent.sparse import sparse_named_distances as pynn_sparse_named_distances
```

---

## Critical Files and Their Roles

### Primary Files

| File | Lines | Role |
|------|-------|------|
| `/home/georgepearse/umap/umap/umap_.py` | 247-340 | Main `nearest_neighbors()` function that wraps NNDescent |
| | 322-335 | NNDescent instantiation with all parameters |
| | 3195-3203 | Query interface during model inference (`transform()`) |
| `/home/georgepearse/umap/umap/distances.py` | Full file | Distance metric implementations (Numba-JIT'd) |
| `/home/georgepearse/umap/umap/utils.py` | 22-44 | Utility functions for k-NN extraction |
| `/home/georgepearse/umap/umap/tests/test_umap_nn.py` | Full file | Tests for nearest neighbor functionality |

### Supporting Files

- `/home/georgepearse/umap/umap/sparse.py` - Sparse data distance metrics
- `/home/georgepearse/umap/umap/spectral.py` - Spectral initialization (uses distances)
- `/home/georgepearse/umap/umap/layouts.py` - Layout optimization (uses distance metrics)

---

## Functional Architecture

### Phase 1: Index Building (during `fit()`)

```python
def nearest_neighbors(X, n_neighbors, metric, ...):
    # Create NNDescent index with computed parameters
    knn_search_index = NNDescent(
        X,
        n_neighbors=n_neighbors,
        metric=metric,                    # "euclidean", "cosine", etc. or callable
        metric_kwds=metric_kwds,          # Optional metric parameters
        random_state=random_state,        # For reproducibility
        n_trees=min(64, 5 + round((X.shape[0]) ** 0.5 / 20.0)),  # Adaptive
        n_iters=max(5, round(np.log2(X.shape[0]))),              # Adaptive
        max_candidates=60,                # Per-iteration candidate limit
        low_memory=low_memory,            # Memory vs. speed trade-off
        n_jobs=n_jobs,                    # Parallelization
        verbose=verbose,
        compressed=False,
    )
    
    # Extract neighbor indices and distances
    knn_indices, knn_dists = knn_search_index.neighbor_graph
    
    return knn_indices, knn_dists, knn_search_index
```

**Output**: 
- `knn_indices`: shape (n_samples, n_neighbors) - indices of k-nearest neighbors
- `knn_dists`: shape (n_samples, n_neighbors) - distances to those neighbors
- `knn_search_index`: NNDescent object for querying new data

### Phase 2: Index Querying (during `transform()`)

```python
# For transforming new data points into existing embedding
indices, dists = self._knn_search_index.query(
    X,                    # New data to find neighbors for
    self.n_neighbors,     # k value
    epsilon=epsilon,      # Search depth (0.12 or 0.24)
)
```

**Output**:
- `indices`: shape (n_test_samples, n_neighbors) - neighbor indices in training data
- `dists`: shape (n_test_samples, n_neighbors) - distances to those neighbors

---

## What Needs to be Replicated in Rust

### Minimum Viable Implementation (MVP)

1. **NN-Descent Algorithm Core**
   - Random Projection forest construction
   - Neighbor descent refinement loop
   - Local graph optimization/merging

2. **Essential Distance Metrics**
   - Euclidean distance (L2)
   - Cosine distance (for angular trees)
   - Basic distance computation infrastructure

3. **Index Interface**
   - Class/struct with `__init__()` accepting all NNDescent parameters
   - `neighbor_graph` property returning (indices, distances) tuple
   - `query(X, k, epsilon=...)` method for querying

4. **Data Format Support**
   - Dense floating-point arrays (float32, float64)
   - Output as NumPy-compatible arrays
   - Integer indexing (i32 or i64)

5. **Reproducibility**
   - Random seed support
   - Deterministic results

### Full Implementation (All Features)

Everything in MVP plus:

6. **Multiple Distance Metrics**
   - Manhattan (L1)
   - Minkowski
   - Correlation
   - Hamming
   - Jaccard
   - Custom callable support

7. **Advanced Features**
   - Angular random projection trees
   - Sparse data support (CSR format)
   - Low-memory mode
   - Parallel search/construction
   - Epsilon/search depth tuning

8. **Robustness**
   - Handling disconnected components (infinite distances)
   - Edge cases (small datasets, duplicate points)
   - Error handling and validation

---

## Critical Integration Requirements

### Python Interface (PyO3/PyPEG)

The Rust implementation must expose:

```python
class NNDescent:
    def __init__(
        self,
        X: np.ndarray,              # Training data
        n_neighbors: int,
        metric: str | Callable,     # "euclidean", "cosine", etc. or function
        metric_kwds: dict | None = None,
        random_state: int | None = None,
        n_trees: int = 10,
        n_iters: int = 7,
        max_candidates: int = 60,
        low_memory: bool = True,
        n_jobs: int = -1,
        verbose: bool = False,
        compressed: bool = False,
    ) -> None:
        ...
    
    @property
    def neighbor_graph(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, distances) of k-nearest neighbors"""
        ...
    
    def query(
        self, 
        X: np.ndarray,      # Query data
        k: int,             # Number of neighbors
        epsilon: float = 0.12,  # Search depth
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, distances) for each query point"""
        ...
    
    # Optional: for angular metric detection
    @property
    def _angular_trees(self) -> bool:
        ...
    
    # Optional: for accessing raw data
    @property
    def _raw_data(self) -> np.ndarray:
        ...
```

### Call Sites in UMAP

1. **Build-time** (line 322 in umap_.py):
   ```python
   index = NNDescent(X, n_neighbors=15, metric="euclidean", ...)
   ```

2. **Query-time** (line 3199 in umap_.py):
   ```python
   indices, dists = index.query(X_new, 15, epsilon=0.12)
   ```

---

## Parameter Semantics

### Build Parameters

| Parameter | UMAP Default/Formula | Semantics |
|-----------|---------------------|-----------|
| `n_neighbors` | 15 | k in k-NN; number of neighbors to find |
| `metric` | "euclidean" | Distance function ("euclidean", "cosine", callable, etc.) |
| `metric_kwds` | {} | Optional parameters for metric function |
| `random_state` | 42 | RNG seed for reproducibility |
| `n_trees` | min(64, 5 + sqrt(N)/20) | Number of RP trees; more = better quality, slower |
| `n_iters` | max(5, log2(N)) | Refinement iterations; more = better quality, slower |
| `max_candidates` | 60 | Max candidates per iteration; affects exploration |
| `low_memory` | True | If True, use less memory (slower) |
| `n_jobs` | -1 | Parallelization (-1 = all cores) |
| `verbose` | False | Print progress |
| `compressed` | False | (Currently always False in UMAP usage) |

### Query Parameters

| Parameter | UMAP Values | Semantics |
|-----------|------------|-----------|
| `epsilon` | 0.12 or 0.24 | Search depth; higher = more thorough but slower |
| `k` | Same as n_neighbors | Number of neighbors to return |

---

## Test Coverage

### Direct ANN Tests

**File**: `/home/georgepearse/umap/umap/tests/test_umap_nn.py`

Tests that validate ANN functionality:
- Metric validation
- Neighbor accuracy (vs. true k-NN from sklearn)
- Sparse data support
- Angular metrics

### Integration Tests

**File**: `/home/georgepearse/umap/umap/tests/test_umap_on_iris.py`

Full UMAP workflow including:
- Training with ANN
- Transforming new data
- Different metrics

---

## Performance Characteristics

### Complexity Analysis

**Build Phase**:
- Time: O(N log N) typical case
- Space: O(k × N) for storing neighbors

**Query Phase**:
- Time: O(log N) per query (binary search-like)
- Space: O(k) result storage per query

### Quality Metrics

- **Accuracy**: Typically 85-95% of true k-NN identified
- **Tradeoff**: Epsilon parameter controls speed vs. accuracy
  - Smaller epsilon: faster but less accurate
  - Larger epsilon: slower but more accurate

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  User Training Data                      │
│              (n_samples × n_features)                    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
      ┌────────────────────────────────────┐
      │   UMAP.fit(X, y=None, ...)        │
      │  (calls nearest_neighbors())      │
      └────────────┬───────────────────────┘
                   │
                   ▼
      ┌────────────────────────────────────┐
      │    NNDescent.__init__(            │
      │      X,                            │
      │      metric=metric,                │
      │      n_neighbors=15,               │
      │      n_trees=..., n_iters=...)    │
      │                                    │
      │    [Build RP Forest]               │
      │    [Run NN-Descent]                │
      │    [Extract neighbor_graph]        │
      └────────────┬───────────────────────┘
                   │
                   ├──▶ knn_indices (n × 15)
                   ├──▶ knn_dists (n × 15)
                   └──▶ knn_search_index (NNDescent object)
                   │
                   ▼
      ┌────────────────────────────────────┐
      │  smooth_knn_dist(knn_dists)       │
      │  compute_membership_strengths()   │
      │  fuzzy_simplicial_set()           │
      │  [Construct graph]                │
      └────────────┬───────────────────────┘
                   │
                   ▼
      ┌────────────────────────────────────┐
      │   optimize_layout()                │
      │   [Layout optimization]            │
      │   [Final embedding]                │
      └────────────┬───────────────────────┘
                   │
                   ▼
      ┌────────────────────────────────────┐
      │   UMAP.transform(X_new)            │
      │                                    │
      │  [For each new point in X_new]    │
      │                                    │
      └────────────┬───────────────────────┘
                   │
                   ▼
      ┌────────────────────────────────────┐
      │   index.query(X_new,              │
      │               k=15,                │
      │               epsilon=0.12)        │
      │                                    │
      │   [Find k-NN in training data]    │
      └────────────┬───────────────────────┘
                   │
                   ├──▶ neighbor_indices (m × 15)
                   └──▶ neighbor_dists (m × 15)
                   │
                   ▼
      ┌────────────────────────────────────┐
      │  smooth_knn_dist()                 │
      │  compute_membership_strengths()   │
      │  [Construct transform graph]       │
      └────────────┬───────────────────────┘
                   │
                   ▼
      ┌────────────────────────────────────┐
      │   optimize_layout_inverse()        │
      │   [Position new points]            │
      │   [Final embedding]                │
      └────────────┬───────────────────────┘
                   │
                   ▼
      ┌────────────────────────────────────┐
      │   Return transformed embedding     │
      │         (m × 2 or m × n_components)│
      └────────────────────────────────────┘
```

---

## Conclusion

The UMAP codebase uses **PyNNDescent** as its exclusive ANN implementation. To replicate this in Rust, the key requirement is to implement:

1. **The NN-Descent algorithm** - the core approximation engine
2. **Distance metrics** - at minimum Euclidean and cosine
3. **Python bindings** - to match the PyNNDescent API
4. **Query functionality** - to support both build-time and query-time operations

The implementation must achieve:
- **Compatibility**: Match PyNNDescent's interface (neighbor_graph property, query method)
- **Performance**: Meet or exceed O(N log N) build and O(log N) query complexity
- **Quality**: Achieve 85-95% accuracy on approximate nearest neighbors
- **Flexibility**: Support various distance metrics and parameters

Success will be measured by being able to replace the PyNNDescent import with the Rust implementation while maintaining all existing UMAP functionality.

