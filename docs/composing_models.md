# Composition

Squeeze includes wrappers for sequential reductions and experimental combinations.
Their fit-time convenience does not give every underlying method a native transform.

## Sequential reductions

```python
import numpy as np
from sklearn.datasets import load_digits
from squeeze import DRPipeline, PCA, NeighborMap

X = np.asarray(load_digits().data[:100], dtype=np.float64)
pipeline = DRPipeline([
    ("linear", PCA(n_components=16)),
    ("layout", NeighborMap(n_epochs=40, random_state=42)),
])
embedding = pipeline.fit_transform(X)
assert embedding.shape == (100, 2)
```

This changes the data presented to NeighborMap and is a different experiment from
the raw-pixel benchmarks. For inference, each step must have a suitable transform;
do not use a fit-only final reducer as though it were a fitted projection.

## Ensemble and progressive wrappers

`EnsembleDR(methods=[(name, reducer, weight), ...])` fits each method. Its default
`weighted_average` blend requires weights summing to one. `blend_mode="procrustes"`
uses an alignment-based blend. Arbitrary rotations and scales make direct coordinate
averaging difficult to interpret; validate the output rather than assuming a gain.
The `alignment` constructor field alone does not change the default blend mode.

`ProgressiveDR(coarse, fine, blend_steps=10, blend_function="linear")` fits both
reducers and blends their coordinates. It does not use the coarse coordinates to
initialize the fine optimizer. For new data, these wrappers may call `fit_transform`
again when a component lacks `transform`, producing a newly fitted coordinate system.
Use [OutOfSampleDR](transform.md) explicitly when interpolation is what you intend.

## Adaptive selection

`AdaptiveDR(method_map, strategy="size")` selects `size:small` below 1,000 rows,
`size:medium` below 100,000, and `size:large` otherwise. The `dimensionality` strategy
uses `dim:low` below 50 features, `dim:medium` below 1,000, and `dim:high` otherwise.
These are routing thresholds, not measured scalability guarantees. Selected
estimators must accept `fit(X, y)` and provide `transform`; raw Rust fit-only
reducers do not satisfy that contract.
