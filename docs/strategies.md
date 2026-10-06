# Strategy registry

The registry lets applications create an established algorithm from a string.
Available names depend on whether the Rust extension imports successfully.

```python
from squeeze import create_reducer, get_strategy, list_strategies

print(list_strategies())
reducer = create_reducer("pca", n_components=2)
assert reducer is not None
assert get_strategy("PCA").name == "pca"
```

The built-in names are `umap` and, with the extension, `pca`, `tsne`, `mds`,
`isomap`, `lle`, `phate`, `trimap`, and `pacmap`. **NeighborMap and SpectralMap
are not registered by default**; instantiate their exported classes directly.

`get_strategy(name)` returns a `Strategy` with `name`, `algorithm_class`,
`default_params`, `description`, and `category`. `.create(**kwargs)` overrides
its defaults. Unknown names raise `KeyError`.

For a local registry, create `StrategyRegistry()` and call
`register(Strategy(name="...", algorithm_class=..., default_params={...}))`.
Names are case-insensitive. `names()`, `categories()`, `by_category(category)`,
and iteration support discovery. `STRATEGIES` is the shared default instance;
use a separate instance when registrations should remain local to an application.
