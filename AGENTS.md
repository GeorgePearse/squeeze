# Squeeze - Agent Development Workflow

This document specifies the workflow for implementing tasks and features in this repository.

---

## Project Vision

**Squeeze** is a high-performance library for **all dimensionality reduction techniques**, not just UMAP. The goal is to provide fast, CPU-optimized implementations of:

| Technique | Status | Description |
|-----------|--------|-------------|
| **UMAP** | ✅ Implemented | Uniform Manifold Approximation and Projection |
| **t-SNE** | ✅ Implemented | t-Distributed Stochastic Neighbor Embedding |
| **PCA** | ✅ Implemented | Principal Component Analysis (eigendecomposition) |
| **Isomap** | ✅ Implemented | Isometric Mapping (geodesic + MDS) |
| **LLE** | ✅ Implemented | Locally Linear Embedding |
| **MDS** | ✅ Implemented | Multidimensional Scaling (classical + metric) |
| **PHATE** | ✅ Implemented | Potential of Heat-diffusion for Affinity-based Transition Embedding |
| **TriMap** | ✅ Implemented | Large-scale Dimensionality Reduction Using Triplets |
| **PaCMAP** | ✅ Implemented | Pairwise Controlled Manifold Approximation |

### Why "Squeeze"?

Dimensionality reduction "squeezes" high-dimensional data into lower dimensions while preserving structure. The name reflects:
- **What it does**: Compresses dimensions
- **How it feels**: Fast and efficient (squeezed for performance)
- **Broad scope**: Not tied to any single algorithm

### Core Philosophy

1. **Algorithm Agnostic**: Support multiple DR techniques under one API
2. **Performance First**: SIMD vectorization, Rust backend, optimized algorithms
3. **CPU-Focused**: No GPU dependencies - runs anywhere
4. **Research Platform**: Easy to experiment with new techniques and parameters
5. **Production Ready**: Reliable, tested, well-documented

---

## Core Principles

Every task should follow a structured workflow to ensure code quality, traceability, and easy review.

## Project Scope Constraints

**GPU Implementation Policy:** We are **NOT pursuing GPU implementations** (CUDA/Metal/OpenCL) for this project. The focus is on CPU-based optimizations including SIMD vectorization, improved algorithms (RobustPrune), and better caching strategies. Any roadmap items or documentation mentioning GPU acceleration should be considered **out of scope** and de-prioritized.

**Benchmark Dataset Policy:** For all benchmarking and performance testing, use **ONLY the sklearn Digits dataset** (`sklearn.datasets.load_digits`). This provides a consistent baseline across all optimizations:
- 1,797 samples
- 64 features (8×8 pixel images)
- 10 classes (digits 0-9)
- Sufficient size to show SIMD benefits
- Fast enough for rapid iteration

Do not use larger datasets (MNIST, etc.) or synthetic datasets for standard benchmarking. This keeps results comparable and testing fast.

**Multi-Algorithm Focus:** When implementing features or optimizations, consider how they might benefit multiple DR algorithms:
- Distance computations → shared across UMAP, t-SNE, Isomap
- k-NN graphs → used by UMAP, t-SNE, LLE, Isomap
- Eigensolvers → used by PCA, MDS, spectral methods
- Gradient descent → used by UMAP, t-SNE, MDS

**Working Documentation:** All working documents, design notes, exploration summaries, and internal documentation should be placed in the `working_docs/` directory. This keeps the root directory clean while preserving useful context. Standard files like `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, and `CODE_OF_CONDUCT.md` remain in the root.

Key documents in `working_docs/`:
- `FEATURES.md` - Overview of implemented features
- `GPU_POLICY.md` - Why we're CPU-focused
- `HNSW_*.md` - HNSW implementation details and architecture
- `OPTIMIZATION_*.md` - Performance optimization notes
- `BENCHMARKING*.md` - Benchmarking methodology and results

---

## Common Commands

Use `just` as the command runner. Run `just` to see all available commands.

```bash
# Build the Rust extension
just build

# Run the benchmark (generates graphs)
just benchmark

# Run tests
just test

# Lint and format
just check
just fix

# Install all dependencies
just install
```

### Benchmarking

The benchmark (`just benchmark`) runs all DR algorithms on the sklearn Digits dataset and generates:
- `benchmark_results.png` - Execution time, trustworthiness bar charts, and trustworthiness vs k line plot
- `embeddings_comparison.png` - Visual comparison of all embeddings

Trustworthiness is evaluated at multiple k values: k=5, 10, 15, 20, 30, 50.

---

## Workflow

### 1. Create a Fresh Branch

Before starting any task, create a new branch from `master`:

```bash
git checkout master
git pull origin master
git checkout -b <branch-name>
```

**Branch naming conventions:**
- Feature: `feat/<description>` (e.g., `feat/add-tsne-implementation`)
- Bug fix: `fix/<description>` (e.g., `fix/memory-leak`)
- Documentation: `docs/<description>` (e.g., `docs/update-installation`)
- Refactoring: `refactor/<description>` (e.g., `refactor/shared-distance-metrics`)
- Tests: `test/<description>` (e.g., `test/add-pca-tests`)
- Algorithm: `algo/<description>` (e.g., `algo/implement-trimap`)

### 2. Implement the Task

Work on the implementation in your branch:

- Write code following the project's conventions
- Add tests for new functionality
- Update documentation as needed
- Run pre-commit hooks to ensure code quality
- Commit changes with clear, descriptive messages

**Commit Message Guidelines:**
- Use imperative mood ("add" not "adds", "fix" not "fixes")
- First line should be concise (50 chars or less)
- Provide detailed explanation in the body if needed
- Reference related issues if applicable

Example:
```bash
git add .
git commit -m "Add t-SNE implementation with Barnes-Hut optimization

- Implement core t-SNE algorithm
- Add Barnes-Hut tree for O(N log N) complexity
- Reuse HNSW k-NN graph from shared infrastructure
- Add comprehensive test suite
"
```

### 3. Submit a Draft PR

After completing the task, push your branch and create a **draft pull request**:

```bash
git push -u origin <branch-name>
gh pr create --draft --title "<title>" --body "<description>"
```

**Draft PR Requirements:**
- Clear, descriptive title
- Summary of changes made
- List of key implementation details
- Note any testing performed
- Flag any known issues or TODOs

Example:
```markdown
## Summary
Implements t-SNE with Barnes-Hut optimization, reusing shared k-NN infrastructure.

## Changes
- Added t-SNE algorithm in `squeeze/tsne.py`
- Reused HNSW backend for k-NN computation
- Implemented Barnes-Hut tree for gradient computation
- Added perplexity auto-tuning

## Testing
- Unit tests pass
- Benchmarks show competitive performance with sklearn
- Visual validation on Digits dataset
```

### 4. Code Review

The draft PR allows for:
- Early feedback on approach and design
- Discussion of implementation details
- Identification of issues before final submission
- Iteration based on review comments

When ready for final review, convert the draft to a regular PR or request review.

## Pre-submission Checklist

Before creating a PR, ensure:

- [ ] All tests pass locally
- [ ] Pre-commit hooks pass
- [ ] Code follows project conventions
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with `master`
- [ ] No unrelated changes are included
- [ ] New algorithms include benchmarks against sklearn equivalents

## Important Notes

### No Direct Commits to Master
- **Never commit directly to `master`**
- All changes must go through the branch → draft PR → review → merge workflow

### Keep Branches Fresh
- Rebase on `master` if it diverges significantly
- Keep branch scope focused on a single task
- Delete branch after merge

### Tests Are Required
- All new code must have corresponding tests
- All tests must pass before submitting PR
- Include both unit and integration tests where appropriate

### Documentation
- Update README if adding user-facing features
- Update docstrings for code changes
- Add migration guides for breaking changes

### Shared Infrastructure
When adding new DR algorithms, leverage shared components:
- `squeeze/distances.py` - Distance metrics (Euclidean, Manhattan, Cosine, etc.)
- `squeeze/hnsw_wrapper.py` - Fast approximate k-NN
- `squeeze/spectral.py` - Eigensolvers and spectral methods
- `squeeze/layouts.py` - Gradient descent optimization
- `src/` - Rust SIMD-optimized implementations

---

## Example Workflow

```bash
# 1. Create a branch for new algorithm
git checkout -b algo/implement-trimap

# 2. Implement the algorithm
# ... write code, tests, docs ...

# 3. Test locally
pytest squeeze/tests/ -v

# 4. Commit changes
git add .
git commit -m "Add TriMap dimensionality reduction

- Implement TriMap algorithm with triplet constraints
- Reuse HNSW k-NN from shared infrastructure
- Add automatic weight selection
- Include comprehensive test suite and benchmarks
"

# 5. Push to remote
git push -u origin algo/implement-trimap

# 6. Create draft PR
gh pr create --draft \
  --title "algo: Add TriMap dimensionality reduction" \
  --body "
## Summary
Implements TriMap for large-scale dimensionality reduction using triplets.

## Changes
- Core TriMap algorithm in squeeze/trimap.py
- Reused HNSW for k-NN computation
- Automatic weight selection based on data characteristics

## Testing
- All tests pass
- Benchmarks show 2x speedup vs reference implementation
- Visual validation on standard datasets
  "
```

---

## Roadmap

### Phase 1: Core Algorithms ✅ Complete
- [x] UMAP with HNSW k-NN backend
- [x] t-SNE (gradient descent)
- [x] PCA (eigendecomposition)
- [x] MDS (classical + metric SMACOF)
- [x] Isomap (geodesic + MDS)
- [x] LLE (locally linear embedding)
- [x] PHATE (diffusion-based)
- [x] TriMap (triplet constraints)
- [x] PaCMAP (pairwise controlled)

### Phase 2: Quality & Performance (Current)
- [x] SIMD-optimized distance computations
- [x] Benchmark framework with multi-k trustworthiness
- [ ] Improve Isomap/LLE/TriMap quality (currently ~0.5-0.66 vs sklearn's 0.83-0.91)
- [ ] Add Barnes-Hut tree to t-SNE for O(N log N)
- [ ] Further parallelize with Rayon

### Phase 3: Testing & Documentation
- [ ] Add comprehensive tests for all Rust algorithms
- [ ] API documentation
- [ ] Usage examples

### Phase 4: Advanced Features
- [ ] Streaming/incremental DR
- [ ] Out-of-sample extension
- [ ] Ensemble methods

---

## Questions?

If you're unsure about any aspect of the workflow:
1. Check existing PRs for examples
2. Refer to the project's documentation
3. Open an issue with questions about the process
