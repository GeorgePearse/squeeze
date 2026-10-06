# MDS

Pairwise distance fitting · Rust backend

MDS embeds pairwise distances. `metric=True` uses iterative metric stress
optimization, with `n_iter` controlling the budget and `random_state` controlling
initialization. In this implementation, **`metric=False` selects classical MDS**;
it does not mean sklearn-style nonmetric MDS.

`fit_transform(X)` builds Euclidean distances from feature rows.
`fit_transform_from_distances(D)` accepts a precomputed dense distance matrix.
Supply a finite symmetric square matrix with a zero diagonal; the method does not
provide complete input validation. `stress_` exposes the stored optional stress
value for the applicable optimization path.

Both routes require dense pairwise storage. There is no native out-of-sample transform.

## Constructor

```text
squeeze.MDS(n_components=2, metric=True, n_iter=300, random_state=None)
```

## Example

```python
import numpy as np
from sklearn.datasets import load_digits
from squeeze import MDS

X = np.ascontiguousarray(load_digits().data[:100], dtype=np.float64)
embedding = MDS(n_components=2, metric=False).fit_transform(X)
assert embedding.shape == (100, 2)
assert np.isfinite(embedding).all()
```

The example uses a small Digits subset to check the API. See the
[full benchmark](../benchmarking.md) for measured quality and runtime, and
[data requirements](../data.md) before substituting your own input.

[Source](https://github.com/GeorgePearse/squeeze/blob/main/src/mds.rs)
