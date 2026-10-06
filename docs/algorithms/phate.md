# PHATE

Diffusion potential embedding · Rust backend

This Rust PHATE implementation builds neighborhood affinities, diffuses them,
and embeds potential distances. It provides a diffusion-based comparison for
structure that may not be well represented by compact cluster islands.

`k` sets the neighbor count, `t` is a fixed diffusion-time budget, and `decay`
controls the affinity kernel. `random_state` controls stochastic initialization
where used. This interface does not expose automatic diffusion-time selection or
all parameters from the authors' reference package.

Dense diffusion and distance operations limit practical sample size. Only
`fit_transform(X)` is exposed. Compare the actual Squeeze output rather than
assuming reference-package behavior from the algorithm name.

## Constructor

```text
squeeze.PHATE(n_components=2, k=15, t=5, decay=2.0, random_state=None)
```

## Example

```python
import numpy as np
from sklearn.datasets import load_digits
from squeeze import PHATE

X = np.ascontiguousarray(load_digits().data[:100], dtype=np.float64)
embedding = PHATE(n_components=2, k=15, t=5, random_state=42).fit_transform(X)
assert embedding.shape == (100, 2)
assert np.isfinite(embedding).all()
```

The example uses a small Digits subset to check the API. See the
[full benchmark](../benchmarking.md) for measured quality and runtime, and
[data requirements](../data.md) before substituting your own input.

[Source](https://github.com/GeorgePearse/squeeze/blob/main/src/phate.rs)
