# PCA

Linear variance projection · Rust backend

PCA centers the feature matrix, forms its covariance matrix, and diagonalizes it.
It is a useful linear baseline with a reusable transform. It does not automatically
standardize features, whiten coordinates, or use a randomized SVD.

`n_components` selects the retained directions and must not exceed the feature
count. Fitted attributes are `components_`, `explained_variance_`, and
`explained_variance_ratio_`. `fit(X)` returns `None`; `transform(X)` uses the stored
mean and components. No public `inverse_transform` or sklearn `get_params` is exposed.

The covariance matrix is feature-by-feature. Very wide data can therefore be
expensive even when the number of observations is modest.

## Constructor

```text
squeeze.PCA(n_components=2)
```

## Example

```python
import numpy as np
from sklearn.datasets import load_digits
from squeeze import PCA

X = np.ascontiguousarray(load_digits().data[:100], dtype=np.float64)
embedding = PCA(n_components=2).fit_transform(X)
assert embedding.shape == (100, 2)
assert np.isfinite(embedding).all()
```

The example uses a small Digits subset to check the API. See the
[full benchmark](../benchmarking.md) for measured quality and runtime, and
[data requirements](../data.md) before substituting your own input.

[Source](https://github.com/GeorgePearse/squeeze/blob/main/src/pca.rs)
