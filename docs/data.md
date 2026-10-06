# Data and preprocessing

## Dense arrays

For the Rust algorithms, supply a finite NumPy `float64` matrix of shape
`(n_samples, n_features)`. Use at least a few samples and choose a neighbor count
smaller than the sample count. Validation differs between older implementations;
check inputs before calling the backend.

```python
import numpy as np
from sklearn.datasets import load_digits

X = np.ascontiguousarray(load_digits().data, dtype=np.float64)
assert X.ndim == 2 and X.shape[0] > 2 and X.shape[1] > 0
assert np.isfinite(X).all()
```

Feature scales influence Euclidean distance. If you standardize or normalize data,
fit that preprocessing on the training split and reuse it for evaluation. The
published Digits and Fashion-MNIST runs intentionally use raw pixel features;
preprocessed results belong in a separately labeled comparison.

## Sparse input

`UMAP` has sparse input paths. The public Rust reducers documented here consume
dense matrices. A sparse HNSW implementation also exists internally, but that does
not make every reducer sparse-compatible. Estimate dense memory before calling
`.toarray()` on a large matrix.

## Labels

Benchmark labels stratify Fashion-MNIST sampling and support evaluation. They do
not enter the unsupervised embedding fit. UMAP separately accepts supervised
labels via `fit_transform(X, y)`; record that explicitly when comparing results.

## Missing values and unusual magnitudes

Resolve NaN and infinity before embedding. NeighborMap and SpectralMap reject
non-finite inputs and rescale by a single maximum absolute value internally.
That preserves Euclidean neighbor ordering; it does not standardize each feature.
The distance kernels use float32 internally in several methods despite float64
Python input, so precision requirements should be validated for your data.
