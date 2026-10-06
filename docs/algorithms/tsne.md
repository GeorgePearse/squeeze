# TSNE

Local probability layout · Rust backend

t-SNE constructs high-dimensional neighbor probabilities and optimizes a
low-dimensional layout. Local groupings are its primary interpretation; distances
between separate islands are not a calibrated measure of original separation.

`perplexity` controls neighborhood scale. `early_exaggeration` adjusts early
attraction; `learning_rate` and `n_iter` control optimization. The implementation
requires at least four samples; choose perplexity below your sample count.

`use_barnes_hut=None` selects Barnes-Hut automatically for more than 1,000 samples
when `n_components=2`. `theta` controls that approximation. Other output dimensions
use exact repulsion. `min_grad_norm` and `n_iter_without_progress` control early
stopping. Input pairwise distances and affinities are still dense, so Barnes-Hut
does not make this implementation an end-to-end linear-memory algorithm.

Only `fit_transform(X)` is exposed; there is no learned transform for new rows.

## Constructor

```text
squeeze.TSNE(n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, early_exaggeration=12.0, random_state=None, theta=0.5, use_barnes_hut=None, min_grad_norm=1e-07, n_iter_without_progress=300)
```

## Example

```python
import numpy as np
from sklearn.datasets import load_digits
from squeeze import TSNE

X = np.ascontiguousarray(load_digits().data[:100], dtype=np.float64)
embedding = TSNE(perplexity=15.0, n_iter=300, random_state=42).fit_transform(X)
assert embedding.shape == (100, 2)
assert np.isfinite(embedding).all()
```

The example uses a small Digits subset to check the API. See the
[full benchmark](../benchmarking.md) for measured quality and runtime, and
[data requirements](../data.md) before substituting your own input.

[Source](https://github.com/GeorgePearse/squeeze/blob/main/src/tsne.rs)
