# Quick start

Start with a small, real dataset and an explicit dtype. Dense Rust methods expect
a two-dimensional NumPy `float64` array with rows as samples and columns as features.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import trustworthiness
from squeeze import NeighborMap, PCA

X = np.ascontiguousarray(load_digits().data, dtype=np.float64)
linear = PCA(n_components=2).fit_transform(X)
embedding = NeighborMap(random_state=42).fit_transform(X)
assert embedding.shape == (1797, 2)
assert np.isfinite(embedding).all()
print("PCA:", trustworthiness(X, linear, n_neighbors=15))
print("NeighborMap:", trustworthiness(X, embedding, n_neighbors=15))
```

This evaluates rank-based trustworthiness on the fitted samples. It does not test
out-of-sample performance. The similarly named `squeeze.trustworthiness` currently
computes neighbor overlap; see [metric definitions](evaluation_metrics.md).

## Compare a UMAP embedding

```python
import numpy as np
from sklearn.datasets import load_digits
from squeeze import UMAP

X = np.asarray(load_digits().data[:100], dtype=np.float64)
model = UMAP(n_neighbors=15, n_epochs=50, random_state=42, n_jobs=1)
embedding = model.fit_transform(X)
assert embedding.shape == (100, 2)
```

This shortened example checks the API. Use the [benchmark protocol](benchmarking.md)
for performance comparisons; its full sample counts and optimization budgets differ.

## Choose the next step

- [Plot and save coordinates](plotting.md).
- [Choose an algorithm](algorithms/index.md) based on the structure you want to retain.
- [Transform unseen rows](transform.md) using a fitted transform or an explicit approximation.
- [Reproduce the heatmaps](benchmarking.md) on Digits or Fashion-MNIST.
