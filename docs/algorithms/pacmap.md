# PaCMAP

Near, mid-near and far pairs · Rust backend

PaCMAP combines neighbor, mid-near, and far-pair objectives over optimization
phases. `n_neighbors` controls the local pairs, while `mn_ratio` and `fp_ratio`
set the relative mid-near and far-pair sampling budgets.

`n_iter`, `learning_rate`, and `random_state` control optimization and repeatability.
Use the recorded benchmark configuration for comparisons; changing the iteration
budget changes the result as well as runtime.

Only `fit_transform(X)` is exposed. Its Python constructor is Squeeze's Rust
binding, not a drop-in promise for the original PaCMAP package. Measure sample-size
scaling before choosing it for a large dataset.

## Constructor

```text
squeeze.PaCMAP(n_components=2, n_neighbors=10, mn_ratio=0.5, fp_ratio=2.0, n_iter=450, learning_rate=1.0, random_state=None)
```

## Example

```python
import numpy as np
from sklearn.datasets import load_digits
from squeeze import PaCMAP

X = np.ascontiguousarray(load_digits().data[:100], dtype=np.float64)
embedding = PaCMAP(n_components=2, n_iter=100, random_state=42).fit_transform(X)
assert embedding.shape == (100, 2)
assert np.isfinite(embedding).all()
```

The example uses a small Digits subset to check the API. See the
[full benchmark](../benchmarking.md) for measured quality and runtime, and
[data requirements](../data.md) before substituting your own input.

[Source](https://github.com/GeorgePearse/squeeze/blob/main/src/pacmap.rs)
