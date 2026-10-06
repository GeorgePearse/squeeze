# Streaming observations

`StreamingDR` embeds the initial batch with a base reducer. Later batches use
nearest-neighbor interpolation; they do not retrain the base algorithm or optimize
old coordinates again.

```python
import numpy as np
from sklearn.datasets import load_digits
from squeeze import PCA, StreamingDR

X = np.asarray(load_digits().data[:200], dtype=np.float64)
model = StreamingDR(PCA(n_components=2), n_neighbors=5)
initial = model.fit_transform(X[:100])
updated = model.partial_fit_transform(X[100:150])
assert initial.shape == (100, 2)
assert updated.shape == (150, 2)  # All rows accumulated so far.
preview = model.transform(X[150:])
assert preview.shape == (50, 2)   # Does not append these rows.
assert model.embedding_.shape == (150, 2)
```

`partial_fit` appends the new rows and their interpolated coordinates, then rebuilds
the nearest-neighbor index. The wrapper retains all observations in `X_all_`, so
memory grows with the stream. It is not a bounded-memory online optimizer.
For drift or a new population, evaluate whether a complete refit is more appropriate.
