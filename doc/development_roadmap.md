================
Development Roadmap
================

This document outlines planned improvements and modernizations for the UMAP codebase.

# Major Initiatives


## 1. Parametric UMAP: TensorFlow → PyTorch Migration


**Status**: Planned

**Motivation**:
- PyTorch has become the standard in modern machine learning
- Better community support and ecosystem for deep learning
- Easier to integrate with modern training frameworks
- TensorFlow 2.x dependency (currently >= 2.1) is outdated and should be >= 2.10+
- PyTorch provides better control over computation graphs and optimization

**Scope**:
- Replace TensorFlow/Keras-based parametric UMAP implementation with PyTorch
- Update parametric_umap.py and related modules
- Migrate all Keras models to PyTorch equivalents
- Update all Parametric UMAP example notebooks to use PyTorch

**Implementation Plan**:
1. Create PyTorch equivalents of all TensorFlow/Keras models
2. Implement PyTorch training loops with equivalent loss functions
3. Add PyTorch as an optional dependency
4. Maintain TensorFlow version temporarily for backward compatibility
5. Add comprehensive testing for PyTorch implementation
6. Update all documentation and examples
7. Deprecate TensorFlow implementation in a future release
8. Remove TensorFlow support in a major version bump

**Files to Modify**:
- umap/parametric_umap.py (core implementation)
- notebooks/Parametric_UMAP/*.ipynb (all examples)
- doc/parametric_umap.rst (documentation)
- pyproject.toml (dependencies)
- tests/ (add PyTorch tests)

**Estimated Effort**: Large (multiple weeks)

**Priority**: High

---

## 2. Remove Conda from Installation and CI/CD


**Status**: Planned

**Motivation**:
- pip and modern Python packaging (PyPI) is the standard
- Removes redundant installation method
- Simplifies CI/CD pipelines
- Reduces maintenance burden
- conda-forge is maintained by volunteers; we should focus on PyPI

**Scope**:
- Remove conda installation instructions from README and documentation
- Remove conda badges from README
- Remove conda-based CI/CD scripts and configurations
- Update all installation documentation
- Remove references to anaconda and conda in setup instructions
- Clean up ci_scripts/install.sh (currently uses conda)

**Implementation Plan**:
1. Update README.rst to remove conda install instructions
2. Update doc/index.rst to remove conda installation section
3. Remove conda references from ci_scripts/
4. Remove .travis.yml and appveyor.yml (legacy CI configurations)
5. Ensure all installation docs reference pip/PyPI only
6. Add documentation for development setup using pip and venv/virtualenv
7. Update any notebooks that show conda install commands

**Files to Modify**:
- README.rst
- doc/index.rst
- doc/parameters.rst
- ci_scripts/install.sh
- .travis.yml (remove or archive)
- appveyor.yml (remove or archive)
- Notebooks showing conda instructions

**Estimated Effort**: Small (1-2 days)

**Priority**: Medium

---

## 3. Remove pynndescent Dependency


**Status**: Planned

**Motivation**:
- pynndescent is an external dependency created for UMAP but not tightly integrated
- Adds a required external dependency that complicates the dependency tree
- Reduces coupling and external dependencies improves maintainability
- pynndescent can remain optional for users who want advanced features, but core UMAP should work with standard libraries
- Industry-standard vector databases like Qdrant use alternative approaches (HNSW, etc.)

**Reference Implementations**:
- **Qdrant**: Production vector database using HNSW
  - See architecture: https://github.com/qdrant/qdrant/tree/master/src
  - Uses HNSW (Hierarchical Navigable Small World) for production-grade nearest neighbor search
  - 10x+ faster than alternatives on large datasets

- **HGG (Hierarchical Greedy Graph)**: Alternative to HNSW from rust-cv
  - See implementation: https://github.com/rust-cv/hgg
  - Data-dependent hierarchy (adapts to local dimensionality)
  - Fully deterministic (unlike HNSW which uses randomness)
  - More efficient edge management and freshening process
  - Written in Rust, but algorithm principles apply to Python implementation

- **ANN-Benchmarks**: Comprehensive benchmark suite for nearest neighbor algorithms
  - See benchmarks: https://ann-benchmarks.com/index.html
  - See source & README: https://github.com/erikbern/ann-benchmarks/?tab=readme-ov-file
  - Real-world datasets and performance comparisons
  - Tracks speed vs recall tradeoffs across all major algorithms
  - Essential resource for validating implementation choices

**Scope**:
- **Replace pynndescent with pluggable ANN backend system** supporting multiple algorithms
- Integrate many ANN algorithms to discover the fastest options for different use cases
- Reference all algorithms from ann-benchmarks.com: HNSW, Annoy, FAISS, NGT, NSW, Vespa, Milvus, etc.
- Implement missing distance metrics that pynndescent provides
- Handle both dense and sparse nearest neighbor computation
- Make pynndescent optional for advanced/performance-critical use cases
- Achieve equivalent or better performance than current pynndescent implementation
- **Goal**: Find the optimal ANN algorithm for different scenarios (speed, memory, accuracy, determinism)

**Alternative Implementations Comparison**:

================ ==================== ==================== ==================== ==================== ====================
Criteria         scikit-learn         HNSWlib              FAISS                Annoy                HGG
================ ==================== ==================== ==================== ==================== ====================
Type             KDTree/BallTree      Graph (HNSW)         Clustering/HNSW      Random projections    Graph (Greedy)
Speed            Good (small data)    Excellent            Excellent            Good                  Excellent
GPU Support      No                   No                   Yes (major feature)  No                    No
Dynamic Updates  Yes                  Yes                  Limited              No (rebuild)          Yes
Distance Metrics Many                 Common               Many                 Limited               Common
Memory Efficient Yes                  Moderate             Yes                  Yes                   Moderate
Deterministic    Yes                  No (randomized)      Yes                  Yes                   Yes (fully)
Maintenance      Well-maintained      Active               Well-maintained      Active                Active
Dependencies     NumPy only           Minimal              Minimal              Minimal               Rust (native)
================ ==================== ==================== ==================== ==================== ====================

**Recommended Approach**:
1. **Default (no new deps)**: Use scikit-learn's KDTree/BallTree for standard cases
   - Sufficient for small to medium datasets
   - No additional dependencies
   - Good for most use cases

2. **Optional (with deps)**: Allow users to opt-in to HNSWlib, FAISS, or HGG for large-scale deployments
   - HNSWlib: Best for production use (Qdrant uses this, industry-standard)
   - FAISS: Best for GPU-accelerated workloads
   - HGG: Best for reproducibility and data-dependent optimization (fully deterministic)
   - All have minimal Python dependencies

3. **Fallback Strategy**: Graceful degradation
   - Use sklearn by default
   - Warn if user tries to use pynndescent features
   - Auto-upgrade to HNSWlib if available and beneficial for large datasets
   - Consider HGG for users requiring deterministic behavior

**Implementation Plan**:
1. Audit all pynndescent usage in the codebase
2. Identify distance metrics that pynndescent provides that sklearn doesn't
3. Implement missing distance metrics using scipy.spatial.distance or native code
4. **Create pluggable ANN backend abstraction layer** (umap/nndescent.py)
   - Design clean interface for swapping different ANN algorithms
   - Support dynamic algorithm selection at runtime
   - Handle algorithm-specific parameters and tuning
5. **Implement support for multiple ANN algorithms** (from ann-benchmarks.com):
   - Default: scikit-learn KDTree/BallTree (no new deps)
   - Optional: HNSWlib, FAISS, Annoy, NGT, NSW, HGG
   - Design: Start with top performers, expand as needed
6. Implement fallback and distance metric handling for all backends
7. **Implement comprehensive benchmark suite with automation**:
   - Set up script → JSON → graph pipeline (rust-analyzer-metrics model)
   - Run full algorithm suite benchmarks on every commit
   - Compare: speed, memory usage, recall accuracy, construction time
   - Validate against ann-benchmarks.com reference datasets and results
   - Track algorithm performance across different data characteristics
   - Generate comparative performance graphs (updated with each commit)
   - Identify optimal algorithms for different use cases
   - Ensure no performance regressions during development
8. Update tests to work without pynndescent as required dependency
9. Make pynndescent optional in pyproject.toml (keep for users who want it)
10. Add algorithm selection guide in documentation
    - Recommendations based on: dataset size, speed requirements, reproducibility needs
    - Performance profiles from benchmarks
    - Trade-off analysis: speed vs memory vs accuracy
11. Add runtime configuration for algorithm selection (config file or environment variables)

**Files to Modify**:
- umap/umap_.py (main implementation - remove pynndescent imports)
- umap/distances.py (implement missing distance metrics)
- umap/sparse.py (handle sparse nearest neighbors)
- umap/nndescent.py (new abstraction layer for NN backends)
- pyproject.toml (move pynndescent to optional, add HNSWlib as optional)
- README.rst (update requirements and performance notes)
- tests/ (ensure all tests pass without pynndescent)
- doc/ (add backend selection guide)
- **benchmarks/** (new: benchmark suite infrastructure)

**Benchmark Infrastructure** (Ideal for Benchmark-on-Commit Discovery Approach):
This task is an excellent candidate for implementing automated benchmarking infrastructure to discover optimal algorithms:

- **Algorithm Suite**: Test all major ANN algorithms from ann-benchmarks.com
  - Start with: sklearn, HNSWlib, FAISS, Annoy, HGG
  - Expand to: NGT, NSW, Vespa, Milvus, others based on availability
- **Metrics Tracking** (JSON output):
  - Speed: query time, construction time
  - Memory: index size, working memory during search
  - Accuracy: recall, precision on different datasets
  - Scalability: performance vs dataset size
- **Automated Benchmarking on Every Commit**:
  - Run full algorithm comparison on each commit to benchmark branch
  - Compare against ann-benchmarks.com reference results
  - JSON → CSV → automatic graph generation
  - Track performance trends over time
- **Discovery & Optimization**:
  - Identify which algorithms are fastest for different scenarios
  - Visualize speed vs memory vs accuracy trade-offs
  - Find algorithm "sweet spots" for common UMAP use cases
  - Auto-suggest best algorithm based on data characteristics

This enables:
- **Real-time algorithm discovery**: See which algorithms perform best as implementations improve
- **Data-driven decisions**: No guessing - use actual performance data
- **Regression detection**: Catch performance regressions immediately
- **Optimization targeting**: Focus effort on algorithms that matter most
- **Historical analysis**: Track which algorithms win under different conditions
- **Publication-ready data**: Comprehensive benchmarks for academic papers

**Estimated Effort**: Large (2-3 weeks for implementation + 2 weeks for benchmarking infrastructure & algorithm integration)

**Priority**: High (reduces coupling and external dependencies)

**Key Considerations**:
- **Algorithm Trade-offs**: Different algorithms excel in different scenarios
  - HNSWlib: 10x+ faster for large datasets, randomized
  - HGG: Fully deterministic, comparable speed, data-dependent
  - FAISS: GPU acceleration, excellent for massive scale
  - Annoy: Memory efficient, fast construction
  - sklearn: Zero new dependencies, good for small-medium data
  - Find optimal algorithm per use case through benchmarking
- **Determinism**: HGG fully deterministic (unlike HNSW which uses randomization), important for reproducible research
- **Sparse matrices**: scipy.sparse has limited distance metric support (critical for sparse UMAP)
- **Distance metrics**: Not all custom metrics supported by all backends - design abstraction carefully
- **Pluggable Architecture**: Design abstraction layer to make adding new algorithms trivial
  - Clean interface for algorithm implementations
  - Dynamic algorithm selection at runtime
  - Per-algorithm configuration and tuning
- **Benchmarking Against Industry Standard**:
  - Use https://ann-benchmarks.com/index.html as reference
  - Validate UMAP results match ann-benchmarks.com reference implementations
  - Ensure UMAP is among fastest implementations for each algorithm
- **Rust-based Options**: HGG and others could be wrapped with PyO3 for native performance
- **Research Reproducibility**: Deterministic algorithms (sklearn, HGG) valuable for scientific use cases
- **Publication Potential**: Comprehensive benchmarks comparing all algorithms is valuable for ML community
- **Performance Goals**:
  - Match or exceed pynndescent speed with fewer dependencies
  - Enable users to find fastest algorithm for their specific use case
  - Make UMAP a reference implementation for ANN algorithm evaluation

---

## 4. Code Modernization and Technical Debt Reduction


**Status**: In Progress

**Completed**:
- Identified outdated patterns (see analysis below)

**Pending**:

### 4.1 Remove Python 2 Artifacts


- Remove `from __future__ import print_function` from:
  - umap/umap_.py
  - umap/sparse.py
- Remove explicit `object` inheritance from old-style classes
- Update any remaining Python 2 compatibility code

**Estimated Effort**: Small (< 1 day)

**Priority**: Medium

---

### 4.2 Fix NumPy Deprecated Type Aliases


- Replace `np.bool_` with `bool` or `np.bool` in:
  - umap/sparse.py
  - umap/aligned_umap.py
- Replace other deprecated NumPy type aliases (np.int_, np.float_, etc.)
- Ensure compatibility with NumPy >= 1.20

**Estimated Effort**: Small (< 1 day)

**Priority**: High

---

### 4.3 Fix Bitwise Operator Misuse


- Replace bitwise AND (`&`) with logical AND (`and`) in boolean contexts
- Affected files:
  - umap/umap_.py (lines 119, 124, 426)
- This improves code clarity and prevents subtle bugs

**Estimated Effort**: Small (< 1 day)

**Priority**: High

---

### 4.4 Optimize Data Structures


- Replace list-based queue with `collections.deque` in breadth_first_search()
- This changes O(n) queue.pop(0) operations to O(1) deque.popleft()
- Location: umap/umap_.py:93

**Estimated Effort**: Small (< 1 day)

**Priority**: Medium

---

### 4.5 Resolve Technical Debt (FIXME/TODO Comments)


- Address FIXME comments in umap/layouts.py regarding uninitialized variables
- Review TODO comments about performance optimizations
- Either implement the optimizations or update comments with justification

**Estimated Effort**: Medium (2-3 days)

**Priority**: Medium

---

### 4.6 Remove Dead Code


- Remove sklearn.externals.joblib fallback code (dead since sklearn >= 1.0)
- Location: umap/umap_.py

**Estimated Effort**: Minimal (< 1 hour)

**Priority**: Low

---

### 4.7 Update Dependency Specifications


- Review and update minimum version constraints in pyproject.toml
- scipy: currently >= 1.3.1 (2019); should be >= 1.8.0 or higher
- numba: currently >= 0.51.2 (2020); should be >= 0.55.0 or higher
- TensorFlow (if keeping): update to >= 2.10.0
- Ensure minimum versions are tested in CI

**Estimated Effort**: Small (< 1 day)

**Priority**: Low

---

## 5. Documentation Updates


**Status**: Pending

### 5.1 Update Installation Instructions


- Remove all conda references
- Clarify Python 3.9+ requirement (currently lists 3.6+)
- Add virtual environment setup instructions
- Document development setup

**Estimated Effort**: Small (1 day)

**Priority**: Medium

---

### 5.2 Clarify Optional Dependencies


- Separate optional dependencies clearly:
  - Plotting: matplotlib, datashader, holoviews
  - Parametric UMAP: PyTorch/TensorFlow (post-migration: PyTorch only)
  - Development: pytest, typing-extensions, etc.

**Estimated Effort**: Small (1 day)

**Priority**: Medium

---

### 5.3 Update Development Status Classifier


- pyproject.toml currently lists "Development Status :: 3 - Alpha"
- Should be updated to "Development Status :: 5 - Production/Stable"
- UMAP has been stable for years

**Estimated Effort**: Minimal (< 1 hour)

**Priority**: Low

---

## 6. Testing Improvements


**Status**: Pending

### 6.1 Add Type Hints


- Add Python type hints throughout the codebase
- Enables better IDE support and static type checking
- Use tools like mypy or pyright for validation
- Priority: Core modules first (umap/umap_.py, umap/sparse.py)

**Estimated Effort**: Large (2-3 weeks for full coverage)

**Priority**: Medium

---

### 6.2 Expand Test Coverage


- Add tests for edge cases and error conditions
- Add performance regression tests
- Ensure PyTorch implementation has equivalent test coverage to current

**Estimated Effort**: Medium

**Priority**: Medium after PyTorch migration

---

# Long-term Considerations


1. **GPU Acceleration**: Consider native GPU support for core UMAP algorithm (not just Parametric UMAP)
2. **API Stability**: Consider stabilizing the public API with semantic versioning
3. **Performance**: Profile and optimize hot paths, especially in KNN computation
4. **Sparse Matrix Support**: Expand and optimize sparse matrix handling
5. **Distributed Computing**: Explore support for large-scale distributed embedding

# Timeline


**Near-term (1-2 months)**:
- Complete code modernization (items 4.1-4.7)
- Remove conda from documentation and CI (item 2) - COMPLETED
- Remove pynndescent dependency (item 3)
- Update documentation (item 5)

**Medium-term (2-3 months)**:
- Begin PyTorch implementation of Parametric UMAP (item 1)
- Add type hints to core modules (item 6.1)

**Long-term (3+ months)**:
- Complete PyTorch migration (item 1)
- Full type hint coverage (item 6.1)
- Deprecate TensorFlow support

# Contributing


Interested in helping with any of these initiatives? Please open an issue or pull request on the GitHub repository.

For major work like the PyTorch migration, please discuss your approach with the maintainers first.
