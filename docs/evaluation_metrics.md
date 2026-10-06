# Evaluation metrics

Measure several properties of an embedding. No single score captures neighborhood
fidelity, global geometry, cluster separation, and runtime.

## Rank trustworthiness versus neighbor overlap

The refreshed heatmaps use **scikit-learn rank trustworthiness**. It penalizes
embedded neighbors that were far away in the original ranking. Rank continuity
is computed by reversing the original and embedded spaces.

The public `squeeze.trustworthiness`, `squeeze.continuity`, and
`squeeze.co_ranking_quality` functions currently use neighborhood-overlap
calculations. Those names are retained for compatibility. `quick_evaluate` and
`DREvaluator` inherit those definitions. Do not compare their values directly
with rank trustworthiness as if they were the same metric.

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import trustworthiness as rank_trustworthiness
from squeeze import PCA, trustworthiness as neighbor_overlap

X = np.asarray(load_digits().data[:100], dtype=np.float64)
Y = PCA(n_components=2).fit_transform(X)
rank_score = rank_trustworthiness(X, Y, n_neighbors=15)
recall = neighbor_overlap(X, Y, k=15)
assert 0 <= rank_score <= 1 and 0 <= recall <= 1
print({"rank_trustworthiness": rank_score, "neighbor_recall": recall})
```

## Reading the heatmap

| Metric | Interpretation and limitation |
| --- | --- |
| Rank trustworthiness, k=5/15/30; continuity, k=15 | Local rank penalties in each direction; higher is better |
| Neighbor recall, k=15 | Fraction of original neighbors recovered; higher is better |
| Spearman | Correlation of pairwise distances; higher is better |
| Global structure | Correlation of distances between class centroids; requires labels |
| Density | Correlation of local density estimates |
| Reconstruction R² | Linear reconstruction from coordinates; not a reducer inverse |
| Silhouette | Separation of KMeans-assigned clusters in the embedding |
| ARI / NMI | Agreement between KMeans assignments and supplied labels |
| Transductive accuracy | RandomForest cross-validation on embeddings already fitted to all rows |
| Time | Full fit after warmup; lower is better; metric computation excluded |

Heatmap colors are normalized within each column. Time uses a reversed log scale;
cell labels retain the raw values. A green cell is relative to the displayed methods.

## Evaluation API

`DREvaluator(X_original, X_reduced, labels=None, reducer=None, method_name="Unknown")`
provides local/global structure, reconstruction, clustering, classification, and
optional stability/noise evaluations. `evaluate_all(include_stability=False)`
avoids repeated stability fits. Its `EvaluationReport` has `summary()` and
`to_dict()` methods. [API details](api.md) describe the exported metric functions.

For predictive evaluation, split data before fitting the embedding. Cross-validation
of a classifier on a full-data embedding is transductive, even when no labels were
used to fit the embedding itself.
