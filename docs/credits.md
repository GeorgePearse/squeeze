# Credits and licensing

Squeeze is maintained by George Pearse. It combines an inherited UMAP implementation
with additional Rust dimensionality-reduction methods and benchmark tooling.

## UMAP lineage

The UMAP code derives from [umap-learn](https://github.com/lmcinnes/umap), created by
Leland McInnes and collaborators. Rewriting Squeeze's documentation does not remove
that authorship or change the source license. See the repository's
[license](https://github.com/GeorgePearse/squeeze/blob/main/LICENSE.txt) for the retained terms and notices.

When publishing work using UMAP, cite the original algorithm and software:

- McInnes, Healy and Melville, *UMAP: Uniform Manifold Approximation and Projection
  for Dimension Reduction*, [arXiv:1802.03426](https://arxiv.org/abs/1802.03426).
- McInnes, Healy, Saul and Grossberger, *UMAP: Uniform Manifold Approximation and
  Projection*, [JOSS 3(29), 861](https://doi.org/10.21105/joss.00861).

Squeeze's Rust implementations should be distinguished from the original authors'
reference implementations. Matching an algorithm name does not imply identical
numerics, parameters, behavior, or measured results.

## Datasets and dependencies

Digits is loaded through scikit-learn. Fashion-MNIST is provided by Zalando Research;
see its [repository and citation instructions](https://github.com/zalandoresearch/fashion-mnist).
Squeeze depends on NumPy, SciPy, scikit-learn, Numba, PyNNDescent, PyO3, ndarray,
Rayon and other projects listed in the lockfiles.

For a Squeeze experiment, record the repository commit alongside the dataset,
algorithm references and protocol. NeighborMap and SpectralMap are experimental
methods in this repository, not established substitutes for a cited UMAP result.
