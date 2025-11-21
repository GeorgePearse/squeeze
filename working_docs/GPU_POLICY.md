# GPU Implementation Policy

**Date:** November 20, 2025  
**Status:** Official Project Policy  
**Scope:** All UMAP and HNSW backend development

---

## Policy Statement

This project is **NOT pursuing GPU implementations** (CUDA/Metal/OpenCL) for the core UMAP algorithm or HNSW nearest neighbor backend.

---

## Rationale

### 1. Focus on CPU Optimizations

**CPU-based optimizations provide significant gains without GPU complexity:**
- ✅ **SIMD vectorization**: 2-4x speedup potential (AVX2/NEON)
- ✅ **Algorithmic improvements**: RobustPrune, better caching
- ✅ **Multi-threading**: Already implemented with Rayon
- ✅ **Memory optimization**: Reduced cloning, efficient data structures

**Current results:** 1.5-1.7x speedup over PyNNDescent with CPU-only implementation.

### 2. Complexity and Portability

**GPU implementations add significant complexity:**
- ❌ Platform-specific code (CUDA for NVIDIA, Metal for Apple, OpenCL for others)
- ❌ GPU driver dependencies and version compatibility issues
- ❌ Memory management complexity (host-device transfers)
- ❌ Build system complexity (CUDA toolkit, compiler requirements)
- ❌ Testing complexity (requires GPU hardware)
- ❌ Deployment complexity (GPU availability in production)

**CPU implementations are portable:**
- ✅ Runs everywhere (laptops, servers, cloud instances)
- ✅ No special hardware requirements
- ✅ Simple build process (just Rust compiler)
- ✅ Easy testing on any machine
- ✅ Predictable performance characteristics

### 3. Diminishing Returns

**Graph traversal in HNSW is inherently sequential:**
- The hierarchical navigation requires visiting nodes in order
- Cannot be easily parallelized across GPU cores
- Distance computation is already fast with SIMD

**GPU acceleration primarily benefits:**
- Dense matrix multiplication (not our bottleneck)
- Embarrassingly parallel workloads (graph traversal is not)
- Very large batch operations (UMAP is typically online/small-batch)

**Our profiling shows:**
- Distance computation: 60% of time → **SIMD solves this**
- Heap operations: 20% → **Already optimized**
- Graph traversal: 20% → **Difficult to parallelize on GPU**

### 4. Alternative Solutions Exist

**Users needing GPU acceleration can:**
- Use RAPIDS cuML UMAP (separate project, GPU-native)
- Use FAISS with GPU support (for pure k-NN, not full UMAP)
- Batch process on CPU with parallelization (often sufficient)

**Our project fills a different niche:**
- Fast CPU implementation with Rust
- Drop-in replacement for PyNNDescent
- Portable, dependency-free deployment
- Focus on algorithmic quality

---

## Scope Clarifications

### What This Policy Covers

**❌ NOT PLANNED (GPU implementations):**
- CUDA distance metric kernels
- Metal compute shaders for Apple Silicon
- OpenCL implementations
- GPU-accelerated k-NN graph construction
- GPU memory management
- Host-device memory transfers

### What This Policy Does NOT Cover

**✅ ALLOWED (These are CPU-only or unrelated):**
- **Parametric UMAP neural network training**: Uses PyTorch which may use GPU
  - This is PyTorch's responsibility, not ours
  - Users can install GPU PyTorch if desired
  - Core UMAP/HNSW backend remains CPU-only

- **External GPU libraries** (user's choice):
  - If someone wants to use RAPIDS cuML UMAP separately, that's fine
  - If someone wants to use FAISS-GPU for k-NN, that's fine
  - We don't prevent GPU usage, we just don't implement it

---

## Impact on Documentation

### Changes Made

All documentation has been updated to reflect this policy:

1. **AGENTS.md** - Added policy statement in "Project Scope Constraints"
2. **README.md** - Removed GPU from roadmap, clarified PyTorch GPU is for Parametric UMAP only
3. **FEATURES.md** - Marked GPU acceleration as "OUT OF SCOPE"
4. **HNSW_IMPLEMENTATION_REVIEW.md** - Changed GPU section to "OUT OF SCOPE"
5. **PHASE_1_VALIDATION_SUMMARY.md** - Removed GPU from Phase 2 plans

### Old References Removed

❌ "GPU Acceleration: RAPIDS cuML integration"  
❌ "CUDA-based distance metrics"  
❌ "GPU memory management"  
❌ "CPU/GPU hybrid execution"  
❌ "Target: 10-100x speedup with GPU"  

### New Focus Documented

✅ "SIMD Vectorization (HIGH PRIORITY)"  
✅ "CPU-based optimizations"  
✅ "RobustPrune heuristic"  
✅ "Property-based testing"  
✅ "Automated benchmarking"

---

## Developer Guidelines

### For Contributors

**If a task mentions GPU:**
- Stop and reconsider the approach
- Focus on CPU-based alternatives (SIMD, caching, algorithms)
- Reference this policy document if questioned

**Examples:**

❌ **DON'T:** "I'll add CUDA kernels for faster distance computation"  
✅ **DO:** "I'll add AVX2 SIMD for faster distance computation"

❌ **DON'T:** "Let's move graph construction to GPU"  
✅ **DO:** "Let's optimize graph construction with RobustPrune"

❌ **DON'T:** "Use cuBLAS for matrix operations"  
✅ **DO:** "Use BLAS with OpenBLAS/MKL for matrix operations"

### For Reviewers

**If a PR includes GPU code:**
- Request changes to use CPU-based optimizations instead
- Reference this policy document
- Suggest SIMD/algorithmic alternatives

### For Users Asking About GPU

**Response template:**

> This project focuses on CPU-based optimizations for portability and simplicity. We're achieving excellent performance (1.5-1.7x faster than PyNNDescent) through SIMD vectorization, algorithmic improvements, and efficient caching.
>
> If you need GPU acceleration for UMAP, consider:
> - RAPIDS cuML UMAP (GPU-native implementation)
> - FAISS-GPU (for pure k-NN search)
>
> For Parametric UMAP neural network training, GPU support is available through PyTorch (install GPU PyTorch separately).

---

## Exception Process

### If GPU Implementation Is Absolutely Required

**This policy can be reconsidered if:**
1. CPU optimizations are exhausted (SIMD, RobustPrune, etc. all implemented)
2. Profiling shows >80% of time in parallelizable operations
3. User demand is overwhelming (multiple production use cases)
4. Maintainer capacity exists to support GPU code
5. Cross-platform solution is available (not just CUDA)

**Process:**
1. Open an issue titled "Reconsider GPU Policy"
2. Provide profiling data showing CPU bottleneck
3. Demonstrate user demand (links to issues, discussions)
4. Propose cross-platform implementation (CUDA + Metal + OpenCL)
5. Discuss maintenance burden with maintainers
6. Update this policy if approved

**Currently:** None of these conditions are met. CPU optimizations are not exhausted.

---

## Benefits of This Policy

### For Developers

✅ **Simpler codebase** - No GPU-specific code paths  
✅ **Faster development** - No GPU hardware required for testing  
✅ **Easier debugging** - CPU tools are mature and accessible  
✅ **Portable testing** - CI/CD works on any runner  

### For Users

✅ **Simpler installation** - No CUDA toolkit, no driver issues  
✅ **Works everywhere** - Laptops, servers, cloud, containers  
✅ **Predictable performance** - No GPU availability surprises  
✅ **Smaller binary size** - No CUDA libraries bundled  

### For Maintainers

✅ **Lower support burden** - No GPU driver/version issues  
✅ **Easier to review** - No platform-specific code  
✅ **Clear project scope** - Focus on CPU optimization excellence  
✅ **Sustainable** - Don't need GPU hardware to maintain  

---

## Alternatives and Comparisons

### CPU-Only Projects (Like Us)

- **scikit-learn**: CPU-only, highly optimized with OpenMP/BLAS
- **NumPy**: CPU-only, SIMD-optimized BLAS backends
- **Rust ecosystem**: Many high-performance CPU-only libraries

**Philosophy:** Optimize CPU to the fullest before considering GPU.

### GPU-Native Projects (Different Niche)

- **RAPIDS cuML**: GPU-native machine learning, requires CUDA
- **FAISS-GPU**: GPU-accelerated similarity search
- **PyTorch/TensorFlow**: GPU for neural networks

**Philosophy:** Target GPU from the start, require GPU hardware.

### Hybrid Projects (Complex)

- **scikit-learn-intelex**: Attempts CPU + GPU, complex codebase
- **CuPy**: NumPy-like API for GPU, extensive codebase

**Philosophy:** Support both, accept complexity burden.

**Our choice:** CPU-only (simpler, more portable, still fast).

---

## Related Policies

### SIMD is In-Scope (CPU vectorization)

✅ **AVX2** (x86_64)  
✅ **AVX-512** (newer x86_64)  
✅ **NEON** (ARM64)  
✅ **Runtime detection** (fallback to scalar)  

**Rationale:** SIMD is CPU-based, portable across platforms, and provides 2-4x speedup without complexity.

### Multi-threading is In-Scope (CPU parallelism)

✅ **Rayon** (already implemented)  
✅ **Thread pool management**  
✅ **Parallel queries**  

**Rationale:** Multi-threading is standard Rust, works everywhere, excellent for batch queries.

### External Library Optimization is In-Scope

✅ **OpenBLAS/MKL** (CPU BLAS)  
✅ **LAPACK** (linear algebra)  
✅ **ndarray** (Rust arrays)  

**Rationale:** These are mature CPU libraries with excellent performance.

---

## Conclusion

**This policy is FINAL for Phase 1 and Phase 2.**

We focus on:
- ✅ SIMD vectorization (2-4x speedup)
- ✅ RobustPrune heuristic (quality improvement)
- ✅ Property-based testing (robustness)
- ✅ Automated benchmarking (tracking)

We do NOT pursue:
- ❌ GPU implementations
- ❌ CUDA/Metal/OpenCL
- ❌ GPU memory management
- ❌ Host-device transfers

**Rationale:** CPU optimizations provide excellent performance with much lower complexity.

**Result so far:** 1.5-1.7x speedup with Phase 1. Target: 3-5x with Phase 2 (SIMD + RobustPrune).

---

**Approved by:** Project Maintainers  
**Effective Date:** November 20, 2025  
**Next Review:** After Phase 2 completion (only if all CPU optimizations exhausted)
