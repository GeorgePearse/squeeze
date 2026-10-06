# Isomap

Geodesic graph distances · Rust backend

Isomap forms a nearest-neighbor graph, estimates geodesic distances using paths
through that graph, and embeds those distances with classical MDS.

`n_neighbors` changes graph connectivity. Too few neighbors can disconnect the
graph; too many can introduce shortcuts across a curved manifold. Inspect your
result and neighborhood quality instead of treating either setting as universally
better. `n_components` sets the output dimensions.

The implementation uses dense distance/path calculations and exposes only
`fit_transform(X)`. A conceptual sparse neighborhood graph does not imply sparse
end-to-end memory use or a reusable transform.

## Constructor

```text
squeeze.Isomap(n_components=2, n_neighbors=10)
```

## Example

```python
import numpy as np
from sklearn.datasets import load_digits
from squeeze import Isomap

X = np.ascontiguousarray(load_digits().data[:100], dtype=np.float64)
embedding = Isomap(n_components=2, n_neighbors=15).fit_transform(X)
assert embedding.shape == (100, 2)
assert np.isfinite(embedding).all()
```

The example uses a small Digits subset to check the API. See the
[full benchmark](../benchmarking.md) for measured quality and runtime, and
[data requirements](../data.md) before substituting your own input.

[Source](https://github.com/GeorgePearse/squeeze/blob/main/src/isomap.rs)
