# Documentation Index - Squeeze Research Platform

## Overview Documents

### 1. **README.md** - Main Project Documentation
The primary entry point for users and developers.

**Contains**:
- Project overview and branding (Squeeze/Reductio)
- Installation instructions
- How to use UMAP basics
- Benefits of the platform
- **Squeeze Research Platform section** with detailed feature documentation:
  - Sequential Composition (DRPipeline)
  - Ensemble Composition (EnsembleDR)
  - Sparse Data Support (SparseUMAP)
  - Comprehensive Evaluation Framework
  - Benchmarking System
- Planned features (Phase 2+)
- Performance recommendations

**Start here**: New users should read this first

---

### 2. **FEATURES.md** - Comprehensive Feature Breakdown
Detailed specification of all Phase 1-3 features.

**Contains**:
- Phase 1 complete features:
  - DRPipeline with 26 tests
  - EnsembleDR with Procrustes alignment
  - ProgressiveDR (progressive refinement)
  - AdaptiveDR (adaptive algorithm selection)
  - 5-metric evaluation framework
  - Sparse data support (SparseUMAP, SparseKNNGraph)
  - Benchmarking system with quality vs speed visualization
- Phase 2 planned optimizations:
  - HNSW-RS parallelization and SIMD
  - GPU acceleration with RAPIDS cuML
  - Hierarchical composition
  - Learned ensemble weights
  - Advanced progressive refinement
  - Co-ranking metrics
- Phase 3 research capabilities:
  - Algorithm comparison suite
  - Parameter optimization
  - Embedding space analysis
- Test coverage summary (93 tests)
- API compatibility matrix
- Known limitations and future work

**Purpose**: Understand all supported features, roadmap, and test coverage

---

### 3. **NAMING_AND_BRANDING.md** - Project Identity Guide
Explains the Squeeze/Reductio names and future transition plan.

**Contains**:
- Primary names (Squeeze preferred, Reductio formal)
- Rationale for each name
- Current state and import patterns
- Future transition plan to `squeeze` package
- Co-branding strategy
- Visual identity suggestions
- Decision timeline

**Purpose**: Understand project naming, branding, and future direction

---

## Implementation Documents

### 4. **PHASE_1_VALIDATION_SUMMARY.md** - Phase 1 Completion Report
Detailed summary of Phase 1 implementation and validation.

**Contains**:
- Overview of all Phase 1 components
- DRPipeline implementation details with validation results:
  - 2-stage pipeline: 0.9997 quality
  - 3-stage pipeline: 0.9978 quality
  - Intermediate access verified
- EnsembleDR implementation:
  - Simple average blending: 0.9969 quality
  - Procrustes alignment: 0.9970 quality
- Sparse Data Support validation:
  - Format detection and conversion
  - Euclidean distance on 95% sparse data
  - k-NN graph construction
  - SparseUMAP embedding generation
- Metrics Evaluation validation:
  - PCA quality: 0.9974
  - UMAP quality: 0.9970
  - All metrics in valid ranges
- Quality vs Speed Benchmarking:
  - Iris dataset: 4 algorithms, 2 runs
  - Digits scaling experiment: 3 sizes
  - Results sorted by quality
- Integration testing results
- Performance notes
- Design decisions
- Known limitations and Phase 2 roadmap

**Purpose**: Understand what was built in Phase 1 and how it was validated

---

## Code Documentation

### 5. **Source Code Docstrings**
All implemented classes and functions have comprehensive docstrings.

**Key modules**:
- `umap/composition.py` (~650 lines)
  - DRPipeline with full docstrings
  - EnsembleDR with examples
  - ProgressiveDR class
  - AdaptiveDR class

- `umap/metrics.py` (~450 lines)
  - trustworthiness() function
  - continuity() function
  - local_continuity_meta_estimate()
  - reconstruction_error()
  - spearman_distance_correlation()
  - DREvaluator class with summary()

- `umap/sparse_ops.py` (~450 lines)
  - SparseFormatDetector class
  - sparse_euclidean()
  - sparse_cosine()
  - sparse_manhattan()
  - sparse_jaccard()
  - SparseKNNGraph class
  - SparseUMAP class

- `umap/benchmark.py` (~350 lines)
  - AlgorithmConfig class
  - BenchmarkResult class
  - DRBenchmark class with all methods

**Purpose**: Understand implementation details and API

---

## Test Documentation

### 6. **Test Files as Examples**
Test files serve as implementation guides and usage examples.

**Files**:
- `umap/tests/test_composition.py` (26 tests)
  - Pipeline functionality tests
  - Ensemble blending tests
  - Parameter passing tests
  - Intermediate access tests

- `umap/tests/test_metrics.py` (26 tests)
  - Trustworthiness tests
  - Continuity tests
  - LCMC tests
  - Reconstruction error tests
  - Spearman correlation tests
  - DREvaluator tests

- `umap/tests/test_sparse_ops.py` (41 tests)
  - Format detection tests
  - Sparse distance tests
  - k-NN graph tests
  - SparseUMAP tests
  - Integration tests

- `umap/tests/test_benchmark.py` (20 tests)
  - Algorithm registration tests
  - Benchmarking tests
  - Scaling experiment tests
  - Visualization tests

**Purpose**: See concrete usage examples and edge case handling

---

## Project Organization

```
Squeeze Research Platform (UMAP-based)
├── README.md                           # Main entry point
├── FEATURES.md                         # Complete feature list
├── NAMING_AND_BRANDING.md             # Project identity
├── PHASE_1_VALIDATION_SUMMARY.md      # Phase 1 report
├── DOCUMENTATION_INDEX.md              # This file
│
├── umap/
│   ├── __init__.py                    # Package exports
│   ├── composition.py                 # Composition patterns (DRPipeline, EnsembleDR, etc.)
│   ├── metrics.py                     # Evaluation metrics
│   ├── sparse_ops.py                  # Sparse data support
│   ├── benchmark.py                   # Benchmarking system
│   │
│   └── tests/
│       ├── test_composition.py        # 26 tests
│       ├── test_metrics.py            # 26 tests
│       ├── test_sparse_ops.py         # 41 tests
│       └── test_benchmark.py          # 20 tests (4 optional)
│
└── [Original UMAP files]
```

---

## How to Use This Documentation

### For New Users
1. Start with **README.md** - understand what Squeeze is
2. Review **NAMING_AND_BRANDING.md** - learn about project identity
3. Read relevant sections of **FEATURES.md** - see what's available

### For Developers
1. Review **PHASE_1_VALIDATION_SUMMARY.md** - understand what's been built
2. Read docstrings in `umap/composition.py`, `umap/metrics.py`, etc.
3. Study test files (`umap/tests/`) for usage examples
4. Check **FEATURES.md** for roadmap and architecture

### For Researchers
1. Read **README.md** section on "Squeeze Research Platform"
2. Review **FEATURES.md** for all available metrics and capabilities
3. Check **PHASE_1_VALIDATION_SUMMARY.md** for validation methodology
4. Study test files for comprehensive examples

### For Contributors
1. Review **FEATURES.md** roadmap (Phase 2 and 3)
2. Read **NAMING_AND_BRANDING.md** for future transition plan
3. Study existing implementation in source files
4. Check test files for quality standards and patterns

---

## Key Features Summary

### Phase 1 (Complete ✓)
- ✓ Sequential composition (DRPipeline)
- ✓ Ensemble blending with Procrustes alignment
- ✓ 5 complementary evaluation metrics
- ✓ Sparse data support (95%+ sparse)
- ✓ Benchmarking with quality vs speed visualization
- ✓ 93 comprehensive tests, all passing

### Phase 2 (Planned)
- HNSW-RS optimization (10-100x speedup)
- GPU acceleration with RAPIDS cuML
- Hierarchical composition
- Learned ensemble weights
- Co-ranking metrics
- Advanced progressive refinement

### Phase 3 (Planned)
- Algorithm comparison suite (20+ algorithms)
- Parameter optimization (grid search + Bayesian)
- Embedding space analysis
- Publication-ready visualization

---

## Import Patterns

### Current (UMAP-based)
```python
from umap import UMAP
from umap.composition import DRPipeline, EnsembleDR
from umap.metrics import DREvaluator, trustworthiness
from umap.sparse_ops import SparseUMAP, SparseKNNGraph
from umap.benchmark import DRBenchmark
```

### Future (Squeeze)
```python
import squeeze as sq

pipeline = sq.DRPipeline([...])
embedding = sq.UMAP(...)
evaluator = sq.metrics.DREvaluator()
sparse_umap = sq.sparse.SparseUMAP()
benchmark = sq.DRBenchmark()
```

---

## Test Coverage

| Component | Tests | Status |
|-----------|-------|--------|
| Composition | 26 | ✓ Passing |
| Metrics | 26 | ✓ Passing |
| Sparse Ops | 41 | ✓ Passing |
| Benchmarking | 20 | ✓ Passing |
| **Total** | **93** | **✓ All Passing** |

---

## Citation

If using Squeeze/Reductio in research, cite:

1. Original UMAP paper:
```bibtex
@article{2018arXivUMAP,
  author = {{McInnes}, L. and {Healy}, J. and {Melville}, J.},
  title = "{UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction}",
  journal = {ArXiv e-prints},
  eprint = {1802.03426},
  year = 2018,
}
```

2. This research platform (when available):
```bibtex
@software{squeeze2025,
  title={Squeeze: A Research Platform for Dimensionality Reduction},
  author={[Authors]},
  year={2025},
  url={https://github.com/[org]/squeeze}
}
```

---

## Questions & Support

- **For feature requests**: See FEATURES.md roadmap
- **For bug reports**: Check existing tests in `umap/tests/`
- **For usage questions**: See README.md examples
- **For implementation details**: Review source docstrings and tests

---

## Document Version

- **Last Updated**: Phase 1 Complete (2025)
- **Next Update**: Phase 2 planning
- **Status**: All Phase 1 documentation complete
