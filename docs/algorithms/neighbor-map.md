# NeighborMap

Experimental sampled neighbor layout · Rust backend

NeighborMap is an experimental Squeeze algorithm. It constructs an exact
Euclidean neighbor graph, starts from PCA or a sparse spectral layout, and refines
the coordinates with weighted attraction and sampled repulsion. It is inspired
by neighbor/negative-sampling layouts but is not an implementation of UMAP.

The output always has **two columns**. There is no `n_components` or `metric`
argument. Inputs must be finite, dense float64 matrices with at least three rows
and one feature; `1 <= n_neighbors < n_samples`.

`init` accepts `"pca"` or `"spectral"`. `n_epochs`, `negative_samples`, and
`n_neighbors` must be positive. `learning_rate` must be positive and finite.
`random_state` is an unsigned integer seed. The implementation returns an error
if optimization produces non-finite coordinates.

Exact graph search is quadratic in sample count even though only nearest edges
are retained. Rayon and SIMD improve constants, not that asymptotic cost. Only
`fit_transform(X)` is exposed. Use an explicit interpolation wrapper if you need
an approximate placement of new rows.

## Constructor

```text
squeeze.NeighborMap(n_neighbors=15, n_epochs=160, negative_samples=5, learning_rate=1.0, init='pca', random_state=42)
```

## Example

```python
import numpy as np
from sklearn.datasets import load_digits
from squeeze import NeighborMap

X = np.ascontiguousarray(load_digits().data[:100], dtype=np.float64)
embedding = NeighborMap(n_neighbors=15, n_epochs=160, init="pca", random_state=42).fit_transform(X)
assert embedding.shape == (100, 2)
assert np.isfinite(embedding).all()
```

The example uses a small Digits subset to check the API. See the
[full benchmark](../benchmarking.md) for measured quality and runtime, and
[data requirements](../data.md) before substituting your own input.

[Source](https://github.com/GeorgePearse/squeeze/blob/main/src/neighbor_map.rs)
