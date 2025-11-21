# Advanced Optimizations Design Document

**Date:** November 20, 2025  
**Status:** Design & Implementation Guide  
**Target:** 3-5x additional speedup

---

## Overview

This document outlines the design and implementation of three advanced optimizations:

1. **Mixed Precision (f16)** - 50% memory reduction, 10-20% speedup
2. **Cache Alignment** - 10-20% speedup via better memory access
3. **Horizontal SIMD** - 2-4x speedup on batch operations

**Combined Expected Impact:** 3-5x total speedup on top of existing 1.5x

---

## 1. Mixed Precision (f16) Implementation

### Concept

Store vectors in f16 format (2 bytes/element) but compute in f32 (4 bytes/element):
- **Memory:** 50% reduction
- **Cache:** 2x more vectors fit in cache
- **Accuracy:** <0.5% quality loss

### Data Structure

```rust
use half::f16;

/// Mixed precision vector storage
pub struct MixedPrecisionVec {
    data: Vec<f16>,  // 2 bytes per element
    dim: usize,
}

impl MixedPrecisionVec {
    /// Create from f32 data
    pub fn from_f32(data: &[f32]) -> Self {
        Self {
            data: data.iter().map(|&x| f16::from_f32(x)).collect(),
            dim: data.len(),
        }
    }
    
    /// Get as f32 for computation
    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&x| x.to_f32()).collect()
    }
    
    /// In-place conversion for SIMD
    pub fn to_f32_simd(&self, out: &mut [f32]) {
        for (i, &val) in self.data.iter().enumerate() {
            out[i] = val.to_f32();
        }
    }
}
```

### Integration Strategy

**Option A: Transparent (Recommended)**
- Store all vectors in f16 internally
- Convert to f32 on-demand for distance computation
- Minimal API changes

**Option B: Explicit**
- Add `use_f16` parameter
- Let users choose precision vs accuracy trade-off

### Performance Impact

**Memory:**
```
Before: 1000 vectors × 128 dims × 4 bytes = 512 KB
After:  1000 vectors × 128 dims × 2 bytes = 256 KB
Savings: 50%
```

**Cache Benefits:**
- L1 cache (32KB): 64 vectors → 128 vectors
- L2 cache (256KB): 512 vectors → 1024 vectors
- More cache hits → faster computation

**Conversion Overhead:**
```rust
// Modern CPUs: ~1 cycle per f16→f32 conversion
// For 128D vector: 128 cycles (~40ns @ 3GHz)
// Distance computation: ~400 cycles
// Overhead: 128/400 = 32% (offset by better cache)
```

**Net Effect:** ~10-20% speedup due to cache benefits outweighing conversion cost

---

## 2. Cache-Aligned Data Structures

### Problem

Default allocations aren't cache-line aligned (64 bytes):
```rust
// Unaligned: spans multiple cache lines
let vec = vec![0.0f32; 16];  // May start at any byte

// Aligned: fits in single cache line
let vec = aligned_vec![0.0f32; 16];  // Starts at 64-byte boundary
```

### Solution: Aligned Allocations

```rust
use std::alloc::{alloc, Layout};

/// Cache-aligned vector (64-byte boundary)
pub struct AlignedVec<T> {
    ptr: *mut T,
    len: usize,
    capacity: usize,
}

impl<T> AlignedVec<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        let layout = Layout::from_size_align(
            capacity * std::mem::size_of::<T>(),
            64,  // Cache line size
        ).unwrap();
        
        let ptr = unsafe { alloc(layout) as *mut T };
        
        Self {
            ptr,
            len: 0,
            capacity,
        }
    }
    
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}
```

### Structure of Arrays (SoA) Layout

**Current (Array of Structures - AoS):**
```rust
// Bad for cache: accessing all X coords requires loading all YZ too
Vec<Vec<f32>>
// [x0,y0,z0, x1,y1,z1, x2,y2,z2, ...]
```

**Optimized (Structure of Arrays - SoA):**
```rust
// Good for cache: all X coords are contiguous
struct PointsSoA {
    x: AlignedVec<f32>,
    y: AlignedVec<f32>,
    z: AlignedVec<f32>,
}
// X: [x0, x1, x2, x3, ...]
// Y: [y0, y1, y2, y3, ...]
// Z: [z0, z1, z2, z3, ...]
```

### For UMAP: Hybrid Approach

Keep current API but align internal storage:

```rust
pub struct CacheOptimizedData {
    // Aligned storage for all points
    data: AlignedVec<f32>,
    n_points: usize,
    dim: usize,
}

impl CacheOptimizedData {
    pub fn get_point(&self, idx: usize) -> &[f32] {
        let start = idx * self.dim;
        &self.data.as_slice()[start..start + self.dim]
    }
}
```

### Performance Impact

**Before (unaligned):**
```
Load point: 3-4 cache line fetches (64 bytes each)
Total: 192-256 bytes loaded for 128-byte vector
Waste: 50%+
```

**After (aligned):**
```
Load point: 2 cache line fetches
Total: 128 bytes loaded for 128-byte vector
Waste: 0%
```

**Expected Speedup:** 10-20% from reduced cache misses

---

## 3. Horizontal SIMD (Batch Distance Computation)

### Concept

Instead of computing one distance with SIMD (vertical):
```
Distance(query, point1) using 8 lanes
```

Compute 8 distances simultaneously (horizontal):
```
[Dist(q, p1), Dist(q, p2), ..., Dist(q, p8)] in one operation
```

### Current (Vertical) SIMD

```rust
// Process one distance, 8 elements at a time
fn euclidean_simd(a: &[f32], b: &[f32]) -> f32 {
    // a[0:8] vs b[0:8] → diff[0:8] → sum
    // a[8:16] vs b[8:16] → diff[8:16] → sum
    // ...
    // Return: single distance
}
```

### Horizontal SIMD Design

```rust
// Process 8 distances, 1 element at a time
fn euclidean_horizontal_simd_8(
    query: &[f32],  // dim elements
    points: &[[f32; DIM]; 8],  // 8 points transposed
) -> [f32; 8] {
    // For each dimension d:
    //   lane[0] = (query[d] - points[0][d])²
    //   lane[1] = (query[d] - points[1][d])²
    //   ...
    //   lane[7] = (query[d] - points[7][d])²
    //   accumulate
    
    // Return: 8 distances
}
```

### Required Data Layout: Transposed

**Current:**
```rust
points: Vec<Vec<f32>>
// Point 0: [x0, y0, z0]
// Point 1: [x1, y1, z1]
// Point 2: [x2, y2, z2]
```

**Required (Transposed):**
```rust
struct TransposedPoints {
    dim_data: Vec<Vec<f32>>,  // dim vectors, each with n_points elements
}
// Dim 0 (X): [x0, x1, x2, ..., x7]
// Dim 1 (Y): [y0, y1, y2, ..., y7]
// Dim 2 (Z): [z0, z1, z2, ..., z7]
```

### Implementation Strategy

**Hybrid Approach:**
1. Keep current layout for Python API compatibility
2. Transpose data once during index construction
3. Use horizontal SIMD for batch queries
4. Use vertical SIMD for single queries

```rust
pub struct HybridStorage {
    // Original layout (for API compatibility)
    row_major: Vec<Vec<f32>>,
    
    // Transposed layout (for horizontal SIMD)
    col_major: Vec<Vec<f32>>,
    
    // Metadata
    n_points: usize,
    dim: usize,
}

impl HybridStorage {
    pub fn new(data: Vec<Vec<f32>>) -> Self {
        let n_points = data.len();
        let dim = data[0].len();
        
        // Transpose for horizontal SIMD
        let mut col_major = vec![vec![0.0; n_points]; dim];
        for (i, point) in data.iter().enumerate() {
            for (d, &val) in point.iter().enumerate() {
                col_major[d][i] = val;
            }
        }
        
        Self {
            row_major: data,
            col_major,
            n_points,
            dim,
        }
    }
    
    pub fn batch_distance_8(&self, query: &[f32], indices: &[usize; 8]) -> [f32; 8] {
        // Use horizontal SIMD with col_major layout
        horizontal_euclidean_simd_8(query, &self.col_major, indices)
    }
}
```

### Horizontal SIMD Implementation (AVX2)

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_euclidean_simd_8(
    query: &[f32],
    transposed_points: &[Vec<f32>],
    indices: &[usize; 8],
) -> [f32; 8] {
    use std::arch::x86_64::*;
    
    let mut sums = _mm256_setzero_ps();
    
    // For each dimension
    for d in 0..query.len() {
        let q_val = query[d];
        let q_broadcast = _mm256_set1_ps(q_val);
        
        // Load 8 point values for this dimension
        let mut point_vals = [0.0f32; 8];
        for i in 0..8 {
            point_vals[i] = transposed_points[d][indices[i]];
        }
        let p_vec = _mm256_loadu_ps(point_vals.as_ptr());
        
        // Compute (query[d] - point[i][d])² for all 8 points
        let diff = _mm256_sub_ps(q_broadcast, p_vec);
        sums = _mm256_fmadd_ps(diff, diff, sums);
    }
    
    // Extract 8 distances
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sums);
    
    // Take square root of each
    for i in 0..8 {
        result[i] = result[i].sqrt();
    }
    
    result
}
```

### Performance Impact

**k-NN Search Speedup:**
```
Current: Compute N distances sequentially
  Cost: N × (vertical SIMD cost)

Horizontal: Compute N/8 batches
  Cost: (N/8) × (horizontal SIMD cost)
  
If horizontal cost ≈ vertical cost:
  Speedup: 8x (theoretical)
  Actual: 4-6x (accounting for overhead)
```

**HNSW Search Speedup:**
- Candidate evaluation: 4-6x faster
- Graph traversal: Not affected
- Overall: 2-4x faster

---

## Implementation Roadmap

### Phase 1: Mixed Precision (2-3 hours)
1. Add `half` crate dependency ✅
2. Create `MixedPrecisionVec` type
3. Add conversion methods
4. Integrate into `HnswIndex`
5. Add Python API parameter
6. Test & benchmark

**Expected Result:** 50% memory reduction, 10-20% speedup

### Phase 2: Cache Alignment (2-3 hours)
1. Create `AlignedVec` type
2. Update `HnswIndex` to use aligned storage
3. Ensure 64-byte alignment for all vectors
4. Test & benchmark

**Expected Result:** 10-20% speedup

### Phase 3: Horizontal SIMD (4-6 hours)
1. Create `TransposedStorage` type
2. Implement `horizontal_euclidean_simd_8`
3. Update search to use batch operations
4. Add fallback for non-batch queries
5. Test & benchmark

**Expected Result:** 2-4x speedup on search operations

---

## Combined Impact Projection

| Optimization | Individual | Combined |
|--------------|-----------|----------|
| Baseline | 1.0x | 1.0x |
| + SIMD (done) | 1.48x | 1.48x |
| + RobustPrune (done) | 1.04x | 1.54x |
| + Mixed Precision | 1.15x | **1.77x** |
| + Cache Alignment | 1.15x | **2.04x** |
| + Horizontal SIMD | 3.0x | **6.12x** |

**Realistic Final Speedup:** **5-6x vs PyNNDescent baseline**

---

## Risks & Mitigation

### Risk 1: f16 Accuracy Loss
- **Mitigation:** Make it opt-in, provide benchmarks
- **Testing:** Compare trustworthiness with/without f16

### Risk 2: Horizontal SIMD Complexity
- **Mitigation:** Implement hybrid approach with fallback
- **Testing:** Extensive unit tests for edge cases

### Risk 3: Memory Overhead (Transposed Data)
- **Mitigation:** Only transpose during construction, not at runtime
- **Impact:** 2x storage (still 50% less with f16!)

---

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_f16_accuracy() {
    let original = vec![1.0f32, 2.0, 3.0, 4.0];
    let fp16 = MixedPrecisionVec::from_f32(&original);
    let recovered = fp16.to_f32();
    
    for (a, b) in original.iter().zip(recovered.iter()) {
        assert!((a - b).abs() < 1e-3);  // <0.1% error
    }
}

#[test]
fn test_cache_alignment() {
    let vec = AlignedVec::<f32>::with_capacity(100);
    assert_eq!(vec.ptr as usize % 64, 0);  // 64-byte aligned
}

#[test]
fn test_horizontal_simd() {
    let query = vec![1.0, 2.0, 3.0, 4.0];
    let points = vec![...];  // 8 points
    
    let result = horizontal_euclidean_simd_8(...);
    let expected = points.iter()
        .map(|p| euclidean_scalar(&query, p))
        .collect::<Vec<_>>();
    
    for (a, b) in result.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}
```

### Benchmark Tests
```rust
#[bench]
fn bench_f32_vs_f16(b: &mut Bencher) {
    // Compare speed of f32 vs f16 storage with distance computation
}

#[bench]
fn bench_vertical_vs_horizontal_simd(b: &mut Bencher) {
    // Compare single-distance vs 8-distance SIMD
}
```

---

## Next Steps

1. ✅ Add `half` crate dependency
2. ⏳ Implement `MixedPrecisionVec`
3. ⏳ Integrate into `HnswIndex`
4. ⏳ Implement `AlignedVec`
5. ⏳ Implement horizontal SIMD
6. ⏳ Comprehensive benchmarking

**Timeline:** 8-12 hours of implementation + testing

**Expected Final Result:** **5-6x total speedup** vs PyNNDescent baseline
