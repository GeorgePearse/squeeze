# CPU performance and backend selection

Squeeze uses different execution paths for different algorithms. A Rust extension
being installed does not mean an entire UMAP fit executes in Rust.

## UMAP neighbor search

`UMAP(use_hnsw=False)` selects PyNNDescent for the approximate neighbor path.
`use_hnsw=True` requests the Rust HNSW wrapper. Unsupported metrics or a missing
extension can fall back to PyNNDescent with a warning. Small-data and precomputed
paths may bypass approximate search entirely.

`hnsw_prune_strategy="simple"` is the default;
`hnsw_prune_strategy="robust"` selects RobustPrune, with `hnsw_alpha=1.2` by default.
Measure graph quality and end-to-end fit time before changing these settings.
UMAP's subsequent layout is Python/Numba code.

## Rust reducers

The Rust reducers use CPU distance, linear algebra, and optimization routines.
The new graph methods parallelize exact neighbor construction with Rayon and
use runtime-selected SIMD squared-distance kernels where supported. They retain
only the nearest edges, but still compare all pairs: graph construction time is
quadratic in the sample count. They are not an approximate HNSW search mode.

PCA diagonalizes a feature covariance matrix. MDS, Isomap, LLE, PHATE and the
current t-SNE affinity construction can allocate dense matrices. Barnes-Hut
repulsion in t-SNE does not eliminate its dense affinity construction.

## Measure the complete operation

Use release builds and a controlled thread budget. Record fit time, quality,
input shape, backend, and hardware. A distance-kernel speedup may be a small
fraction of total runtime. The [current benchmarks](benchmarking.md) publish
measured results rather than a universal speed multiplier.
