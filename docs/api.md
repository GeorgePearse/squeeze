# Python API reference

All names below are exported from `squeeze`. Constructor signatures are recorded
from the built package and checked by the documentation smoke command. This is
an inventory of the current interface, not a promise of sklearn compatibility.

## Core reducers

Rust methods consume dense float64 matrices. Except PCA, they primarily return
a new embedding from `fit_transform(X)`. See the [algorithm capability table](algorithms/index.md).

### UMAP

```text
squeeze.UMAP(n_neighbors=15, n_components=2, metric='euclidean', metric_kwds=None, output_metric='euclidean', output_metric_kwds=None, n_epochs=None, learning_rate=1.0, init='spectral', min_dist=0.1, spread=1.0, low_memory=True, n_jobs=-1, set_op_mix_ratio=1.0, local_connectivity=1.0, repulsion_strength=1.0, negative_sample_rate=5, transform_queue_size=4.0, a=None, b=None, random_state=None, angular_rp_forest=False, target_n_neighbors=-1, target_metric='categorical', target_metric_kwds=None, target_weight=0.5, transform_seed=42, transform_mode='embedding', force_approximation_algorithm=False, verbose=False, tqdm_kwds=None, unique=False, densmap=False, dens_lambda=2.0, dens_frac=0.3, dens_var_shift=0.1, output_dens=False, disconnection_distance=None, precomputed_knn=(None, None, None), use_hnsw=None, hnsw_prune_strategy='simple', hnsw_alpha=1.2) -> None
```

[Usage, parameters and limitations](algorithms/umap.md).

### PCA

```text
squeeze.PCA(n_components=2)
```

[Usage, parameters and limitations](algorithms/pca.md).

### TSNE

```text
squeeze.TSNE(n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, early_exaggeration=12.0, random_state=None, theta=0.5, use_barnes_hut=None, min_grad_norm=1e-07, n_iter_without_progress=300)
```

[Usage, parameters and limitations](algorithms/tsne.md).

### MDS

```text
squeeze.MDS(n_components=2, metric=True, n_iter=300, random_state=None)
```

[Usage, parameters and limitations](algorithms/mds.md).

### Isomap

```text
squeeze.Isomap(n_components=2, n_neighbors=10)
```

[Usage, parameters and limitations](algorithms/isomap.md).

### LLE

```text
squeeze.LLE(n_components=2, n_neighbors=12, reg=0.001, error_on_singular=False)
```

[Usage, parameters and limitations](algorithms/lle.md).

### PHATE

```text
squeeze.PHATE(n_components=2, k=15, t=5, decay=2.0, random_state=None)
```

[Usage, parameters and limitations](algorithms/phate.md).

### TriMap

```text
squeeze.TriMap(n_components=2, n_inliers=12, n_outliers=4, n_random=3, n_iter=800, learning_rate=0.1, weight_adj=50.0, random_state=None)
```

[Usage, parameters and limitations](algorithms/trimap.md).

### PaCMAP

```text
squeeze.PaCMAP(n_components=2, n_neighbors=10, mn_ratio=0.5, fp_ratio=2.0, n_iter=450, learning_rate=1.0, random_state=None)
```

[Usage, parameters and limitations](algorithms/pacmap.md).

### NeighborMap

```text
squeeze.NeighborMap(n_neighbors=15, n_epochs=160, negative_samples=5, learning_rate=1.0, init='pca', random_state=42)
```

[Usage, parameters and limitations](algorithms/neighbor-map.md).

### SpectralMap

```text
squeeze.SpectralMap(n_neighbors=15, n_iter=128, random_state=42)
```

[Usage, parameters and limitations](algorithms/spectral-map.md).

## UMAP-family interfaces

`AlignedUMAP` fits related datasets with explicit row relations. `ParametricUMAP`
requires its optional TensorFlow training stack; its dependency-error placeholder
is not the actual model signature. See [UMAP variants](algorithms/variants.md).

## Composition and extensions

### DRPipeline

```text
squeeze.DRPipeline(steps: 'list[tuple[str, Any]]') -> 'None'
```

### EnsembleDR

```text
squeeze.EnsembleDR(methods: 'list[tuple[str, Any, float]]', blend_mode: 'str' = 'weighted_average', alignment: 'str' = 'procrustes') -> 'None'
```

### ProgressiveDR

```text
squeeze.ProgressiveDR(coarse: 'Any', fine: 'Any', blend_steps: 'int' = 10, blend_function: 'str' = 'linear') -> 'None'
```

### AdaptiveDR

```text
squeeze.AdaptiveDR(method_map: 'dict[str, Any]', strategy: 'str' = 'size') -> 'None'
```

### OutOfSampleDR

```text
squeeze.OutOfSampleDR(base_reducer: 'Any', n_neighbors: 'int' = 5, weights: 'str' = 'distance') -> 'None'
```

### StreamingDR

```text
squeeze.StreamingDR(base_reducer: 'Any', n_neighbors: 'int' = 5) -> 'None'
```

Read [composition](composing_models.md), [transforms](transform.md), and
[streaming](streaming.md) for fit/refit behavior and inference limitations.

## Evaluation

`DREvaluator` evaluates a pair of original/reduced arrays. `EvaluationReport` stores
its outputs and exposes `summary()` and `to_dict()`. Overlap-based legacy names
are distinct from sklearn rank trustworthiness; see [metric definitions](evaluation_metrics.md).

### DREvaluator

Evaluation entry point; expensive stability checks can be disabled.

```text
squeeze.DREvaluator(X_original: 'np.ndarray', X_reduced: 'np.ndarray', labels: 'np.ndarray | None' = None, reducer: 'BaseEstimator | None' = None, method_name: 'str' = 'Unknown') -> 'None'
```

### quick_evaluate

Returns overlap-based trustworthiness/continuity plus Spearman correlation.

```text
squeeze.quick_evaluate(X_original: 'np.ndarray', X_reduced: 'np.ndarray', k: 'int' = 15) -> 'dict[str, float]'
```

### trustworthiness

Legacy name for neighbor-overlap quality at k; not a rank penalty.

```text
squeeze.trustworthiness(X_original: 'np.ndarray', X_reduced: 'np.ndarray', k: 'int' = 15) -> 'float'
```

### continuity

Legacy overlap-based continuity calculation at k.

```text
squeeze.continuity(X_original: 'np.ndarray', X_reduced: 'np.ndarray', k: 'int' = 15) -> 'float'
```

### co_ranking_quality

Overlap-based neighborhood quality, not a full co-ranking matrix.

```text
squeeze.co_ranking_quality(X_original: 'np.ndarray', X_reduced: 'np.ndarray', k: 'int' = 15) -> 'float'
```

### spearman_distance_correlation

Correlates pairwise distances; optionally caps sample count.

```text
squeeze.spearman_distance_correlation(X_original: 'np.ndarray', X_reduced: 'np.ndarray', max_samples: 'int | None' = 5000) -> 'float'
```

### global_structure_preservation

Compares distances between class centroids; requires labels.

```text
squeeze.global_structure_preservation(X_original: 'np.ndarray', X_reduced: 'np.ndarray', labels: 'np.ndarray') -> 'float'
```

### local_density_preservation

Compares local density estimates at k.

```text
squeeze.local_density_preservation(X_original: 'np.ndarray', X_reduced: 'np.ndarray', k: 'int' = 15) -> 'float'
```

### reconstruction_error

Returns reconstruction error measures; only the linear method is implemented.

```text
squeeze.reconstruction_error(X_original: 'np.ndarray', X_reduced: 'np.ndarray', method: 'str' = 'linear') -> 'dict[str, float]'
```

### clustering_quality

Scores clustering in the embedding; labels enable agreement metrics.

```text
squeeze.clustering_quality(X_reduced: 'np.ndarray', labels_true: 'np.ndarray | None' = None, n_clusters: 'int | None' = None) -> 'dict[str, float]'
```

### classification_accuracy

Cross-validates a classifier on the supplied embedding; it does not refit the embedding per fold.

```text
squeeze.classification_accuracy(X_reduced: 'np.ndarray', labels: 'np.ndarray', cv: 'int' = 5, classifier: 'BaseEstimator | None' = None) -> 'dict[str, float]'
```

### bootstrap_stability

Refits on resampled data and compares stability; additional fit cost.

```text
squeeze.bootstrap_stability(X: 'np.ndarray', reducer: 'BaseEstimator', n_bootstrap: 'int' = 10, sample_fraction: 'float' = 0.8, random_state: 'int | None' = None) -> 'dict[str, float]'
```

### noise_robustness

Refits with noise at each supplied level.

```text
squeeze.noise_robustness(X: 'np.ndarray', reducer: 'BaseEstimator', noise_levels: 'list[float] | None' = None, random_state: 'int | None' = None) -> 'dict[float, float]'
```

### parameter_sensitivity

Refits a reducer class across a parameter grid.

```text
squeeze.parameter_sensitivity(X: 'np.ndarray', reducer_class: 'type', parameters_to_vary: 'dict[str, list[Any]] | None' = None, base_params: 'dict[str, Any] | None' = None, random_state: 'int | None' = None) -> 'dict[str, dict[str, Any]]'
```

## Strategy registry

`STRATEGIES` is the shared instance of `StrategyRegistry`.
`Strategy` stores a name, algorithm class, default parameters, description and category.
`StrategyRegistry` supplies `register`, `get`, `create`, `names`, `categories`,
`by_category`, `summary`, and iteration. See [registry usage](strategies.md).

```text
squeeze.get_strategy(name: 'str') -> 'Strategy'
```

```text
squeeze.list_strategies() -> 'list[str]'
```

```text
squeeze.create_reducer(name: 'str', **kwargs)
```

## Version and source

`squeeze.__version__` reports installed package metadata, falling back to a development
version when metadata is unavailable. Record the Git commit as well for experiments.
