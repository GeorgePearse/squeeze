# LLE

Local linear reconstruction · Rust backend

LLE finds weights that reconstruct each observation from its neighbors, then
solves for coordinates preserving those weights. `n_neighbors` selects the local
neighborhood and `reg` regularizes the local covariance systems.

`error_on_singular=False` permits the implementation's singular-system fallback;
set it to `True` to surface those cases as errors. Duplicate observations and
locally low-rank neighborhoods deserve particular attention.

The embedding solve involves dense matrices. Only `fit_transform(X)` is exposed;
it does not retain a model for out-of-sample inference.

## Constructor

```text
squeeze.LLE(n_components=2, n_neighbors=12, reg=0.001, error_on_singular=False)
```

## Example

```python
import numpy as np
from sklearn.datasets import load_digits
from squeeze import LLE

X = np.ascontiguousarray(load_digits().data[:100], dtype=np.float64)
embedding = LLE(n_components=2, n_neighbors=12, reg=0.001).fit_transform(X)
assert embedding.shape == (100, 2)
assert np.isfinite(embedding).all()
```

The example uses a small Digits subset to check the API. See the
[full benchmark](../benchmarking.md) for measured quality and runtime, and
[data requirements](../data.md) before substituting your own input.

[Source](https://github.com/GeorgePearse/squeeze/blob/main/src/lle.rs)
