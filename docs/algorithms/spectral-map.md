# SpectralMap

Experimental approximate spectral layout · Rust backend

SpectralMap builds the same exact Euclidean neighbor graph as NeighborMap and
computes a two-dimensional approximate spectral embedding using fixed-budget
block iteration. It omits NeighborMap's sampled layout refinement.

The output always has **two columns**. Inputs must be finite, dense float64 arrays
with at least three rows and one feature; `1 <= n_neighbors < n_samples`.
`n_iter` and `n_neighbors` must be positive, and `random_state` is an unsigned
integer seed.

More iterations spend more work on the approximation, but there is no residual-based
convergence guarantee. Disconnected graphs and repeated eigenvalues can make a
layout ambiguous. The published results show lower neighborhood quality than
NeighborMap in exchange for lower runtime on these datasets.

Graph construction is exact and quadratic; the method only exposes
`fit_transform(X)`. It is not a general sparse eigensolver or a learned transform.

## Constructor

```text
squeeze.SpectralMap(n_neighbors=15, n_iter=128, random_state=42)
```

## Example

```python
import numpy as np
from sklearn.datasets import load_digits
from squeeze import SpectralMap

X = np.ascontiguousarray(load_digits().data[:100], dtype=np.float64)
embedding = SpectralMap(n_neighbors=15, n_iter=128, random_state=42).fit_transform(X)
assert embedding.shape == (100, 2)
assert np.isfinite(embedding).all()
```

The example uses a small Digits subset to check the API. See the
[full benchmark](../benchmarking.md) for measured quality and runtime, and
[data requirements](../data.md) before substituting your own input.

[Source](https://github.com/GeorgePearse/squeeze/blob/main/src/neighbor_map.rs)
