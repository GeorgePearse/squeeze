# Transforming new data

A transform should place new rows in an existing coordinate system. Fitting an
embedding again on a new batch does not provide that guarantee.

## Native transforms

`PCA` stores its fitted mean and components. Its Rust `fit(X)` returns `None`,
so call it separately instead of chaining `.fit(X).transform(...)`.

```python
import numpy as np
from sklearn.datasets import load_digits
from squeeze import PCA

X = np.asarray(load_digits().data, dtype=np.float64)
model = PCA(n_components=2)
model.fit(X[:1000])
new_coordinates = model.transform(X[1000:])
assert new_coordinates.shape == (797, 2)
```

`UMAP` also supplies `transform` using the fitted model. Some modes have additional
restrictions, including densMAP; do not assume every UMAP option supports inference.
Other documented Rust reducers have no native out-of-sample transform.

## Interpolation for fit-only methods

`OutOfSampleDR` fits the base reducer once, retains the training data, and interpolates
new coordinates from nearest training neighbors. This is a wrapper approximation,
not an inference method learned by NeighborMap or t-SNE.

```python
import numpy as np
from sklearn.datasets import load_digits
from squeeze import NeighborMap, OutOfSampleDR

X = np.asarray(load_digits().data[:200], dtype=np.float64)
model = OutOfSampleDR(NeighborMap(random_state=42), n_neighbors=5, weights="distance")
training_coordinates = model.fit_transform(X[:150])
new_coordinates = model.transform(X[150:])
assert training_coordinates.shape == (150, 2)
assert new_coordinates.shape == (50, 2)
```

Weights may be `"distance"` or `"uniform"`. Novel clusters outside the training
support can interpolate poorly; evaluate on held-out data before using this route.
UMAP's `inverse_transform` is likewise an approximation, not lossless reconstruction.
Rust PCA does not expose an `inverse_transform` method.
