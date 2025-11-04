# UMAP Approximate Nearest Neighbors - Documentation Index

This directory contains comprehensive documentation about the UMAP codebase's Approximate Nearest Neighbors (ANN) implementation, created through detailed exploration of the Python codebase.

## Documentation Files

### 1. UMAP_ANN_EXPLORATION_SUMMARY.md (PRIMARY - Start Here)
**Length**: 392 lines | **Best For**: High-level overview and quick reference

**Contents**:
- Key findings (PyNNDescent library, NN-Descent algorithm)
- Critical files and their roles
- Functional architecture (Phase 1: Index Building, Phase 2: Index Querying)
- What needs to be replicated in Rust (MVP vs. Full implementation)
- Critical integration requirements (Python interface specification)
- Parameter semantics reference
- Test coverage information
- Performance characteristics
- Complete data flow diagram
- Implementation success criteria

**Start here for**: Understanding what needs to be built and how it integrates with UMAP

---

### 2. UMAP_ANN_ANALYSIS.md (DETAILED - Algorithm & Design)
**Length**: 283 lines | **Best For**: Algorithm understanding and implementation details

**Contents**:
1. ANN Library used (PyNNDescent)
2. Main entry point: `nearest_neighbors()` function
3. NNDescent object usage (build-time and query-time)
4. Distance metrics support
5. Functionality that needs replication in Rust
6. Current integration points in UMAP
7. Key implementation characteristics (data flow, complexity, robustness)
8. Test files reference
9. Alternative implementations and fallbacks

**Start here for**: Understanding the algorithm and how PyNNDescent is currently used

---

### 3. UMAP_ANN_CODE_LOCATIONS.md (REFERENCE - Code Maps)
**Length**: 250 lines | **Best For**: Developers working on implementation

**Contents**:
- Detailed file structure and locations
- Line-by-line code locations for all key components
- Critical code snippets with full context
- Data flow architecture diagram
- Key parameter behavior tables
- Dynamic parameter formulas
- Query parameters reference
- Integration points requiring Rust implementation
- Testing entry points
- Performance expectations

**Start here for**: Finding exact code locations and understanding specific implementations

---

## Quick Navigation Guide

### If you want to understand...

| Question | Document | Section |
|----------|----------|---------|
| What library does UMAP use for ANN? | Summary | Key Findings |
| How are neighbors searched in UMAP? | Analysis | NNDescent Object Usage |
| Where is the `nearest_neighbors()` function? | Code Locations | File Structure |
| What Python API must I implement? | Summary | Critical Integration Requirements |
| How does NN-Descent algorithm work? | Analysis | Main Entry Point |
| What parameters are used? | Code Locations | Key Parameter Behavior |
| Where are the tests? | Code Locations | Testing Entry Points |
| What's the data flow? | Summary/Code Locations | Data Flow Diagram |
| Performance targets? | Summary | Performance Characteristics |

---

## Key Findings Summary

**Current Library**: PyNNDescent >= 0.5
**Algorithm**: Nearest Neighbor Descent (NN-Descent)
**Time Complexity**: O(N log N) build, O(log N) query
**Accuracy**: 85-95% of true k-NN identified

**Two Critical Operations**:
1. **Build** (during UMAP.fit()): Create index with training data
2. **Query** (during UMAP.transform()): Find neighbors of new points

**Must Implement in Rust**:
- NN-Descent algorithm core
- Distance metrics (Euclidean, Cosine minimum)
- Python API matching PyNNDescent interface
- NumPy array support

---

## Code Locations Quick Reference

| Component | File | Lines |
|-----------|------|-------|
| NNDescent imports | umap_.py | 27-28 |
| nearest_neighbors() | umap_.py | 247-340 |
| Index creation | umap_.py | 322-335 |
| Query interface | umap_.py | 3195-3203 |
| Distance metrics | distances.py | Full file |
| Tests | test_umap_nn.py | Full file |

---

## Implementation Checklist

### Minimum Viable Product (MVP)
- [ ] NN-Descent algorithm core implementation
- [ ] Euclidean distance metric
- [ ] Cosine distance metric
- [ ] `NNDescent.__init__()` with all parameters
- [ ] `neighbor_graph` property (returns indices, distances)
- [ ] `query(X, k, epsilon)` method
- [ ] Random seed support
- [ ] NumPy array I/O

### Full Implementation
- [ ] All MVP features
- [ ] Additional distance metrics (Manhattan, Minkowski, etc.)
- [ ] Angular RP trees
- [ ] Sparse data support
- [ ] Low-memory mode
- [ ] Parallel operations
- [ ] `_angular_trees` property
- [ ] `_raw_data` property access

### Testing & Integration
- [ ] Unit tests for algorithm
- [ ] Integration tests with UMAP
- [ ] Accuracy validation tests
- [ ] Performance benchmarks
- [ ] Replace PyNNDescent import
- [ ] Full UMAP test suite passing

---

## Python Interface Specification

The Rust implementation must provide this exact interface:

```python
class NNDescent:
    def __init__(
        self,
        X: np.ndarray,                    # Training data (n_samples, n_features)
        n_neighbors: int,                 # k value
        metric: str | Callable,           # Distance metric
        metric_kwds: dict | None = None,  # Metric parameters
        random_state: int | None = None,  # RNG seed
        n_trees: int = 10,               # Number of RP trees
        n_iters: int = 7,                # Refinement iterations
        max_candidates: int = 60,        # Per-iteration limit
        low_memory: bool = True,         # Memory vs. speed
        n_jobs: int = -1,                # Parallelization
        verbose: bool = False,           # Logging
        compressed: bool = False,        # Always False in UMAP
    ) -> None:
        ...
    
    @property
    def neighbor_graph(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (knn_indices, knn_dists) for training data"""
    
    def query(
        self,
        X: np.ndarray,           # Query data
        k: int,                  # Number of neighbors
        epsilon: float = 0.12,   # Search depth
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (neighbor_indices, neighbor_distances)"""
    
    @property
    def _angular_trees(self) -> bool:
        """For angular metric detection"""
    
    @property
    def _raw_data(self) -> np.ndarray:
        """Access to training data (optional but used)"""
```

---

## Performance Targets

**Build Phase**:
- Time: O(N log N) typical case
- Space: O(k × N)

**Query Phase**:
- Time: O(log N) per query
- Space: O(k) per query

**Quality**: 85-95% accuracy (finding that % of true k-NN neighbors)

---

## Document Statistics

- **Total Lines**: 925 lines across 3 documents
- **Total Words**: ~15,000+ words of documentation
- **Code Snippets**: 10+ fully-commented code examples
- **Data Flows**: 3 detailed diagrams
- **Tables**: 20+ reference tables
- **Code Locations**: 50+ specific file:line references

---

## How to Use These Documents

1. **For new readers**: Start with EXPLORATION_SUMMARY.md (392 lines)
2. **For algorithm focus**: Read ANALYSIS.md (283 lines)
3. **For implementation**: Reference CODE_LOCATIONS.md (250 lines)
4. **For quick lookup**: Use this INDEX file and the tables in each document

All documents are written to be:
- Comprehensive but scannable
- Full of concrete examples
- Rich with tables and diagrams
- Linked to specific code locations
- Ready for Rust implementation reference

