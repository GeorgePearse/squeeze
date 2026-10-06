# UMAP

Fuzzy neighbor graph and optimized layout · Python/Numba with optional Rust search

Squeeze's UMAP implementation descends from umap-learn. It builds a weighted
neighborhood graph and optimizes coordinates to represent that graph. Unlike the
fit-only Rust reducers, it implements sklearn-style parameter access and fitted
transforms. Its broader interface does not apply automatically to other algorithms.

## Main controls

| Parameter | Default | Role |
| --- | --- | --- |
| `n_neighbors` | 15 | Neighborhood scale |
| `n_components` | 2 | Output dimension |
| `metric` | `"euclidean"` | Original-space distance |
| `min_dist` | 0.1 | Low-dimensional packing |
| `spread` | 1.0 | Scale paired with min_dist |
| `n_epochs` | `None` | Automatic optimization budget unless set |
| `random_state` | `None` | Reproducible seed when supplied |
| `n_jobs` | -1 | Worker request; seeded execution can restrict parallelism |
| `use_hnsw` | `None` | Neighbor-backend selection on approximate-search paths |

For the complete constructor and its additional fields, use the [API reference](../api.md).

## Fit and transform

```python
import numpy as np
from sklearn.datasets import load_digits
from squeeze import UMAP

X = np.asarray(load_digits().data[:120], dtype=np.float64)
model = UMAP(n_neighbors=10, n_epochs=50, random_state=42, n_jobs=1)
training = model.fit_transform(X[:100])
held_out = model.transform(X[100:])
assert training.shape == (100, 2)
assert held_out.shape == (20, 2)
```

`fit(X)` returns the estimator; `embedding_` holds the fitted coordinates.
`fit_transform(X, y)` can use labels for supervised embedding. Record supervised
fits separately from unsupervised benchmark results. `inverse_transform` approximates
original features for supported modes; it does not recover discarded information.

## Backend selection and variants

`use_hnsw=False` selects PyNNDescent for approximate search. `True` requests Rust
HNSW but can fall back for an unavailable backend or unsupported metric. Small-data
paths may use direct pairwise calculations. See [performance](../performance.md).

UMAP also exposes density, precomputed-neighbor, and supervised options. Related
aligned and parametric classes are covered under [UMAP variants](variants.md).
Use the installed API's capabilities rather than assuming every upstream example
or third-party UMAP integration works with Squeeze.
