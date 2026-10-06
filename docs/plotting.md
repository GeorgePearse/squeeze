# Plotting and saved embeddings

Coordinates are ordinary NumPy arrays. Keep sample order, labels, and dataset
provenance alongside them so a scatter plot can be traced back to its inputs.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from squeeze import PCA

digits = load_digits()
X = np.asarray(digits.data, dtype=np.float64)
embedding = PCA(n_components=2).fit_transform(X)
fig, ax = plt.subplots(figsize=(7, 5))
points = ax.scatter(embedding[:, 0], embedding[:, 1], c=digits.target,
                    cmap="tab10", s=8, alpha=0.7)
fig.colorbar(points, ax=ax, label="Digit class")
ax.set(title="Squeeze PCA · Digits", xlabel="Component 1", ylabel="Component 2")
fig.tight_layout()
fig.savefig("digits-pca.png", dpi=150)
np.savez_compressed("digits-pca.npz", embedding=embedding, labels=digits.target)
plt.close(fig)
```

For nonlinear layouts, axis directions and cluster positions can change between
runs. A visually clean separation does not establish predictive performance or
prove that the same distances hold in the original feature space.

## Interactive benchmark report

The [repeated-seed runner](benchmarking.md) saves embeddings and measurements.
Generate its HTML explorer with:

```bash
uv run --no-sync python -m scripts.report_neighbors working_docs/neighbor_benchmarks/local
```

The report provides a method/seed selector, class labels, and quality/runtime
plots. It is a standalone HTML artifact you can open locally or share. The
Fashion-MNIST report uses clothing class names rather than digit labels.
