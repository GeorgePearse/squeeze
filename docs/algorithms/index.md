# Choose an algorithm

Squeeze has eleven core reduction methods. Compare an inexpensive linear baseline
with methods matching the structure you care about, then measure quality and runtime.
The API differences below are part of the contract.

| Method | Implementation | Fit new coordinates | Native transform | Main purpose |
| --- | --- | --- | --- | --- |
| [PCA](pca.md) | Rust | `fit_transform`; `fit` | Yes | Linear variance baseline |
| [UMAP](umap.md) | Python/Numba + optional Rust search | `fit_transform`; `fit` | Yes, mode-dependent | Neighbor graph layout |
| [t-SNE](tsne.md) | Rust | `fit_transform` | No | Local probability layout |
| [MDS](mds.md) | Rust | `fit_transform`; from distances | No | Distance fitting |
| [Isomap](isomap.md) | Rust | `fit_transform` | No | Graph geodesics |
| [LLE](lle.md) | Rust | `fit_transform` | No | Local linear relations |
| [PHATE](phate.md) | Rust | `fit_transform` | No | Diffusion geometry |
| [TriMap](trimap.md) | Rust | `fit_transform` | No | Triplet constraints |
| [PaCMAP](pacmap.md) | Rust | `fit_transform` | No | Pairwise layout |
| [NeighborMap](neighbor-map.md) | Experimental Rust | `fit_transform` | No | Sampled graph refinement |
| [SpectralMap](spectral-map.md) | Experimental Rust | `fit_transform` | No | Approximate spectral layout |

NeighborMap and SpectralMap are Euclidean, two-dimensional methods. The other
constructors expose `n_components`, but that does not establish identical behavior
or performance at every dimension. Rust reducers do not implement sklearn's full
parameter/cloning interface. [Composition wrappers](../composing_models.md) have
additional limits when they contain fit-only components.

## Start from your requirement

- For a reusable linear projection, use PCA as a baseline.
- For neighborhood visualization, compare UMAP, t-SNE, PaCMAP and NeighborMap.
- For global distances or graph geometry, inspect MDS, Isomap, LLE and PHATE.
- For an inexpensive approximate graph layout, evaluate SpectralMap's quality tradeoff.

There is no supported claim here that every Rust method is faster than every
alternative. The [Digits and Fashion-MNIST heatmaps](../benchmarking.md) show
measured configurations, including weak results for the current TriMap implementation.
Do not infer large-dataset scalability from these small benchmarks.
