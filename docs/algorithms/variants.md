# Additional UMAP interfaces

Squeeze retains UMAP-family functionality inherited from umap-learn. These
interfaces are distinct from the new Rust algorithms and are not part of the
11-method heatmap comparison.

## Density-aware UMAP

`UMAP(densmap=True)` adds density-oriented optimization. Relevant controls are
`dens_lambda=2.0`, `dens_frac=0.3`, and `dens_var_shift=0.1`.
`output_dens=True` changes the fit-transform result to
`(embedding, radii_original, radii_embedding)`; the radii are log-transformed.
It can also be enabled without densMAP. A tuple return is therefore possible
instead of the ordinary coordinate array. densMAP does not support the standard
UMAP out-of-sample transform path.

## AlignedUMAP

`AlignedUMAP` fits a list of related datasets. Its `fit` and `fit_transform` calls
require `relations=[mapping_0_to_1, mapping_1_to_2, ...]`, with each mapping a
dictionary of corresponding row indices in consecutive datasets. There must be
one fewer relation than datasets. `alignment_regularisation` controls coupling;
`alignment_window_size` controls the alignment window. The result is a list of
embeddings, not a single matrix. `update` supports adding a related dataset;
there is no ordinary independent-row `transform` method.

## ParametricUMAP

`ParametricUMAP` uses TensorFlow/Keras to train a neural encoder. The optional
`parametric` dependency is required; without it, construction raises `ImportError`.
The module also imports additional optional training/export packages, so availability
must be checked in the target environment. This path has not been validated by the
core documentation smoke run and is not a Rust or PyTorch backend.

Its constructor takes `batch_size`, `dims`, `encoder`, `decoder`,
`parametric_reconstruction`, reconstruction-loss options, global-correlation and
landmark-loss options, `keras_fit_kwargs`, and inherited UMAP keyword arguments.
It provides model saving through `.save(...)` and loading via
`squeeze.parametric_umap.load_ParametricUMAP(...)`.

## Precomputed neighbors

`UMAP(precomputed_knn=(indices, distances, search_index))` accepts a compatible
neighbor graph. Indices and distances must match input row order and neighborhood
size. New-row transformation requires an appropriate search index, not merely the
two arrays. Retain graph construction parameters and provenance alongside the data.

See the implementation docstrings for detailed mode restrictions. These optional
interfaces should be tested against the intended workload before deployment.
