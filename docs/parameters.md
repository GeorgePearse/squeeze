# Parameters and experiment design

Tune one question at a time. Changing preprocessing, sample selection, neighborhood
size, and optimization budget together makes it difficult to explain the result.

| Question | Controls |
| --- | --- |
| How local should the neighborhood be? | `n_neighbors` in graph methods, `perplexity` in t-SNE, `k` in PHATE |
| How much optimization work is allowed? | `n_epochs` in UMAP/NeighborMap, `n_iter` in several Rust methods |
| How tightly should UMAP pack nearby points? | `min_dist` and `spread` |
| How does a graph layout start? | `NeighborMap(init="pca" or "spectral")`; UMAP has its own initialization options |
| How repeatable is the experiment? | Seed, input order, backend, threads, dependency versions |

These controls are not interchangeable across methods. A budget of 160 epochs in
NeighborMap has no direct equivalence to 160 iterations in t-SNE.

## Compare several seeds

Use the [repeated-seed runner](benchmarking.md) to retain each result. Compare
runtime alongside neighbor recall and rank trustworthiness, and inspect the
coordinates. A smaller runtime is not an improvement if the layout loses the
structure your task needs.

## Components and supported metrics

NeighborMap and SpectralMap produce exactly two dimensions and use Euclidean
distance; they have no `n_components` or `metric` argument. Most older Rust
constructors accept `n_components`, but they do not expose UMAP's broad metric
selection. Refer to each [algorithm page](algorithms/index.md) for its exact signature.
