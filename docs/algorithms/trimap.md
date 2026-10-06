# TriMap

Triplet constraints · Rust backend

TriMap optimizes relative-distance constraints involving an anchor, an inlier,
and an outlier. `n_inliers`, `n_outliers`, and `n_random` control constraint
sampling; `weight_adj` affects their weighting. `learning_rate` and `n_iter`
control the optimizer.

The current Squeeze implementation has weak neighborhood preservation in the
published Digits and Fashion-MNIST snapshots. Treat it as an implementation to
evaluate and improve, not evidence against the original TriMap algorithm or a
recommended large-data default.

Only `fit_transform(X)` is exposed. A small iteration budget in this example is
an API smoke check, not the configuration used for the published heatmap.

## Constructor

```text
squeeze.TriMap(n_components=2, n_inliers=12, n_outliers=4, n_random=3, n_iter=800, learning_rate=0.1, weight_adj=50.0, random_state=None)
```

## Example

```python
import numpy as np
from sklearn.datasets import load_digits
from squeeze import TriMap

X = np.ascontiguousarray(load_digits().data[:100], dtype=np.float64)
embedding = TriMap(n_components=2, n_iter=100, random_state=42).fit_transform(X)
assert embedding.shape == (100, 2)
assert np.isfinite(embedding).all()
```

The example uses a small Digits subset to check the API. See the
[full benchmark](../benchmarking.md) for measured quality and runtime, and
[data requirements](../data.md) before substituting your own input.

[Source](https://github.com/GeorgePearse/squeeze/blob/main/src/trimap.rs)
