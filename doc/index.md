<!-- .. umap documentation master file, created by -->
   sphinx-quickstart on Fri Jun  8 10:09:40 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

![Image](logo_large.png)

  :width: 600
  :align: center

# UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction


Uniform Manifold Approximation and Projection (UMAP) is a dimension reduction
technique that can be used for visualisation similarly to t-SNE, but also for
general non-linear dimension reduction. The algorithm is founded on three
assumptions about the data

1. The data is uniformly distributed on Riemannian manifold;
2. The Riemannian metric is locally constant (or can be approximated as such);
3. The manifold is locally connected.

From these assumptions it is possible to model the manifold with a fuzzy
topological structure. The embedding is found by searching for a low dimensional
projection of the data that has the closest possible equivalent fuzzy
topological structure.

The details for the underlying mathematics can be found in
[our paper on ArXiv](https://arxiv.org/abs/1802.03426):

McInnes, L, Healy, J, *UMAP: Uniform Manifold Approximation and Projection
for Dimension Reduction*, ArXiv e-prints 1802.03426, 2018

You can find the software [on github](https://github.com/lmcinnes/umap).

**Installation**

Install UMAP via pip:

```bash
pip install umap

```

This will install UMAP and all required dependencies. For development installation, see the README.


## User Guide / Tutorial:

- [basic_usage](basic_usage)
- [parameters](parameters)
- [plotting](plotting)
- [reproducibility](reproducibility)
- [transform](transform)
- [inverse_transform](inverse_transform)
- [parametric_umap](parametric_umap)
- [transform_landmarked_pumap](transform_landmarked_pumap)
- [sparse](sparse)
- [supervised](supervised)
- [clustering](clustering)
- [outliers](outliers)
- [composing_models](composing_models)
- [densmap_demo](densmap_demo)
- [mutual_nn_umap](mutual_nn_umap)
- [document_embedding](document_embedding)
- [embedding_space](embedding_space)
- [aligned_umap_basic_usage](aligned_umap_basic_usage)
- [aligned_umap_politics_demo](aligned_umap_politics_demo)
- [precomputed_k-nn](precomputed_k-nn)
- [benchmarking](benchmarking)
- [release_notes](release_notes)
- [faq](faq)

## Background on UMAP:

- [how_umap_works](how_umap_works)
- [performance](performance)

## Examples of UMAP usage

- [interactive_viz](interactive_viz)
- [exploratory_analysis](exploratory_analysis)
- [scientific_papers](scientific_papers)
- [nomic_atlas_umap_of_text_embeddings](nomic_atlas_umap_of_text_embeddings)
- [nomic_atlas_visualizing_mnist_training](nomic_atlas_visualizing_mnist_training)

## API Reference:

- [api](api)

## Development:

- [development_roadmap](development_roadmap)

# Indices and tables


* `genindex`
* `modindex`
* `search`
