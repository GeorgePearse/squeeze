# Dimensionality Reduction Research & Implementation Guide

This page compiles a comprehensive collection of research papers on dimensionality reduction techniques. Our goal is to implement most of these methods in the UMAP ecosystem, providing a unified interface for dimensionality reduction across all major techniques.

**Last Updated:** November 2025
**Total Papers Cataloged:** 150+
**Implementation Status:** In Progress

---

## Table of Contents

1. [Foundational Papers (1901-2006)](#foundational-papers)
2. [Classical Linear Methods](#classical-linear-methods)
3. [Spectral & Graph Methods](#spectral--graph-methods)
4. [Local Geometry Methods](#local-geometry-methods)
5. [Distance Preservation Methods](#distance-preservation-methods)
6. [Probability-Based Methods](#probability-based-methods)
7. [Topology-Based Methods](#topology-based-methods)
8. [UMAP Ecosystem](#umap-ecosystem)
9. [Modern Manifold Learning](#modern-manifold-learning)
10. [Deep Learning Methods](#deep-learning-methods)
11. [Contrastive Learning](#contrastive-learning-for-dr)
12. [Graph Embedding Methods](#graph-embedding-methods)
13. [Accelerated Implementations](#accelerated-implementations)
14. [Time-Series DR](#time-series-dimensionality-reduction)
15. [Interpretable & Explainable DR](#interpretable--explainable-dr)
16. [Information-Theoretic Methods](#information-theoretic-dr)
17. [Domain-Specific Methods](#domain-specific-methods)
18. [Comparison & Benchmarks](#comparison--benchmark-papers)
19. [Theoretical Papers](#theoretical-papers)
20. [Implementation Resources](#implementation-resources)

---

## Foundational Papers

### Principal Component Analysis (PCA)

**Title:** On lines and planes of closest fit to systems of points in space
**Authors:** Karl Pearson
**Year:** 1901
**Publication:** Philosophical Magazine, Series 6, Vol. 2, No. 11, pp. 559-572
**Description:** Pearson's foundational work on finding lines and planes that best fit a set of points in p-dimensional space.
**Implementation:** Scikit-learn, NumPy

---

**Title:** Analysis of a Complex of Statistical Variables Into Principal Components
**Authors:** Harold Hotelling
**Year:** 1933
**Publication:** Journal of Educational Psychology, Vol. 24
**Description:** Hotelling's algebraic approach to PCA, introducing the concept of fundamental independent variables.

---

### Early Factor Analysis

**Title:** General Intelligence, Objectively Determined and Measured
**Authors:** Charles Spearman
**Year:** 1904
**Description:** Early application of factor analysis to model human intelligence (the g-factor).

---

### Multidimensional Scaling (MDS)

**Title:** Multidimensional scaling: I. Theory and method
**Authors:** Warren S. Torgerson
**Year:** 1952
**Publication:** Psychometrika, Vol. 17, pp. 401-419
**Description:** Foundational paper introducing classical MDS procedures.
**Implementation:** Scikit-learn (`sklearn.manifold.MDS`)

---

**Title:** Nonmetric multidimensional scaling: A numerical method
**Authors:** Joseph B. Kruskal
**Year:** 1964
**Publication:** Psychometrika
**Description:** Introduced nonmetric MDS approach using monotonic relationships.

---

## Classical Linear Methods

### Kernel PCA

**Title:** Nonlinear Component Analysis as a Kernel Eigenvalue Problem
**Authors:** Bernhard Schölkopf, Alexander Smola, Klaus-Robert Müller
**Year:** 1997
**Description:** Extends PCA to nonlinear data using the kernel trick.
**Implementation:** Scikit-learn (`sklearn.decomposition.KernelPCA`)

---

### Independent Component Analysis (ICA)

**Title:** FastICA Algorithm
**Authors:** Aapo Hyvärinen, Erkki Oja
**Description:** Separates multivariate signals into independent components.
**Implementation:** Scikit-learn (`sklearn.decomposition.FastICA`)

---

### Non-negative Matrix Factorization (NMF)

**Title:** Learning the parts of objects by non-negative matrix factorization
**Authors:** Daniel D. Lee, H. Sebastian Seung
**Year:** 1999
**Publication:** Nature, Vol. 401, pp. 788-791
**Description:** Approximates nonnegative data matrix by product of two lower-rank nonnegative matrices.
**Implementation:** Scikit-learn (`sklearn.decomposition.NMF`)

---

### Random Projections

**Title:** Johnson-Lindenstrauss Lemma and Applications
**Authors:** William B. Johnson, Joram Lindenstrauss
**Year:** 1984
**Description:** Foundation for random projection-based dimensionality reduction.
**Implementation:** Scikit-learn (`sklearn.random_projection`)

---

## Spectral & Graph Methods

### Spectral Clustering and Embedding

**Title:** Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering
**Authors:** Mikhail Belkin, Partha Niyogi
**Year:** 2001
**Publication:** NIPS 2001
**Description:** Geometrically motivated algorithm using the graph Laplacian.
**Implementation:** Scikit-learn (`sklearn.manifold.SpectralEmbedding`)

---

### Diffusion Maps

**Title:** Diffusion maps
**Authors:** Ronald R. Coifman, Stéphane Lafon
**Year:** 2006
**Publication:** Applied and Computational Harmonic Analysis
**DOI:** 10.1016/j.acha.2006.04.006
**Description:** Uses eigenfunctions of Markov matrices to construct diffusion maps.

---

### Graph-Based Methods

**Title:** DeepWalk: Online Learning of Social Representations
**Authors:** Bryan Perozzi, Rami Al-Rfou, Steven Skiena
**Year:** 2014
**Link:** https://arxiv.org/abs/1403.6652
**Description:** Uses random walks on graphs with Word2Vec to learn node representations.

---

**Title:** node2vec: Scalable Feature Learning for Networks
**Authors:** Aditya Grover, Jure Leskovec
**Year:** 2016
**Link:** https://arxiv.org/abs/1607.00653
**Description:** Extends DeepWalk with biased random walks.
**Implementation:** https://github.com/aditya-grover/node2vec

---

**Title:** struc2vec: Learning Node Representations from Structural Identity
**Authors:** Leonardo F. R. Ribeiro, Pedro H. P. Saverese, Daniel R. Figueiredo
**Year:** 2017
**Link:** https://arxiv.org/abs/1704.03165
**Description:** Models structural identity via multi-layer weighted graphs.

---

**Title:** GraphWave: Learning to Generate Networks from Continuous-Time Diffusion
**Authors:** Claire Donnat, Marinka Zitnik, David Hallac, Jure Leskovec
**Year:** 2018
**Link:** https://arxiv.org/abs/1810.02771
**Description:** Uses heat wavelet diffusion patterns for node embedding.

---

**Title:** Inductive Representation Learning on Large Graphs (GraphSAGE)
**Authors:** William L. Hamilton, Rex Ying, Jure Leskovec
**Year:** 2017
**Link:** https://arxiv.org/abs/1706.02216
**Description:** Inductive framework for generating node embeddings.
**Implementation:** PyTorch Geometric

---

## Local Geometry Methods

### Isomap

**Title:** A Global Geometric Framework for Nonlinear Dimensionality Reduction
**Authors:** Joshua B. Tenenbaum, Vin de Silva, John C. Langford
**Year:** 2000
**Publication:** Science, Vol. 290, No. 5500, pp. 2319-2323
**DOI:** 10.1126/science.290.5500.2319
**Description:** Uses geodesic distances in a neighborhood graph to extend MDS.
**Implementation:** Scikit-learn (`sklearn.manifold.Isomap`)

---

### Locally Linear Embedding (LLE)

**Title:** Nonlinear Dimensionality Reduction by Locally Linear Embedding
**Authors:** Sam T. Roweis, Lawrence K. Saul
**Year:** 2000
**Publication:** Science, Vol. 290, No. 5500, pp. 2323-2326
**DOI:** 10.1126/science.290.5500.2323
**Description:** Computes neighborhood-preserving embeddings using local linear reconstructions.
**Implementation:** Scikit-learn (`sklearn.manifold.LocallyLinearEmbedding`)

---

**Title:** Variants of LLE: Tutorial and Survey
**Authors:** Benyamin Ghojogh, et al.
**Year:** 2020
**Link:** https://arxiv.org/abs/2011.10925
**Description:** Comprehensive tutorial covering kernel LLE, inverse LLE, incremental LLE, landmark LLE, supervised LLE, and robust variants.

---

### Hessian Eigenmaps

**Title:** Hessian eigenmaps: New locally linear embedding techniques for high-dimensional data
**Authors:** David L. Donoho, Carrie Grimes
**Year:** 2003
**Publication:** PNAS USA, Vol. 100, pp. 5591-5596
**Description:** Modification of LLE using Hessian-based quadratic form.

---

### Local Tangent Space Alignment (LTSA)

**Title:** Principal Manifolds and Nonlinear Dimension Reduction via Local Tangent Space Alignment
**Authors:** Zhenyue Zhang, Hongyuan Zha
**Year:** 2004
**Publication:** SIAM Journal on Scientific Computing, Vol. 26, No. 1, pp. 313-338
**ArXiv:** cs/0212008
**Description:** Preserves local geometry by aligning local tangent spaces.

---

### Laplacian Eigenmaps Tutorial

**Title:** A Tutorial on Spectral Clustering
**Authors:** Ulrike von Luxburg
**Year:** 2007
**Publication:** Statistics and Computing, Vol. 17, No. 4
**Description:** Comprehensive tutorial on spectral methods including Laplacian eigenmaps.

---

**Title:** Laplacian-Based Dimensionality Reduction Methods: A Tutorial
**Authors:** Benyamin Ghojogh, et al.
**Year:** 2021
**Link:** https://arxiv.org/abs/2106.02154
**Description:** Tutorial covering spectral clustering, Laplacian eigenmap, LPP, graph embedding, diffusion maps.

---

## Distance Preservation Methods

### Classical MDS Connections

**Title:** Connection between Kernel PCA and MDS
**Year:** 2015+
**Link:** https://link.springer.com/article/10.1023/A:1012485807823
**Description:** Establishes theoretical connections between kernel PCA and metric MDS.

---

## Probability-Based Methods

### LargeVis (Large-scale Visualization)

**Title:** Visualizing Large-scale and High-dimensional Data
**Authors:** Jian Tang, Jingzhou Liu, Ming Zhang, Qiaozhu Mei
**Year:** 2016
**ArXiv:** 1602.00370
**Publication:** Proceedings of the 25th International Conference on World Wide Web (WWW 2016)
**Link:** https://arxiv.org/abs/1602.00370
**Description:** Constructs accurately approximated k-nearest neighbor graph and optimizes layout in low-dimensional space using linear time complexity. Scales to millions of high-dimensional data points with significantly reduced computational cost compared to t-SNE.
**Implementation:** Official C++ implementation available on GitHub

---

### Stochastic Neighbor Embedding (SNE)

**Title:** Stochastic Neighbor Embedding
**Authors:** Geoffrey Hinton, Sam T. Roweis
**Year:** 2002
**Publication:** NIPS 2002
**Description:** Probabilistic approach using Gaussian distributions and KL divergences.

---

### t-Distributed Stochastic Neighbor Embedding (t-SNE)

**Title:** Visualizing Data using t-SNE
**Authors:** Laurens van der Maaten, Geoffrey Hinton
**Year:** 2008
**Publication:** Journal of Machine Learning Research, Vol. 9, pp. 2579-2605
**Link:** https://www.jmlr.org/papers/v9/vandermaaten08a.html
**Description:** t-SNE uses t-distribution in low-dimensional space, much easier to optimize than SNE.
**Implementation:** Scikit-learn, RAPIDS cuML

---

### Barnes-Hut t-SNE

**Title:** Accelerating t-SNE using Tree-Based Algorithms
**Authors:** Laurens van der Maaten
**Year:** 2014
**Publication:** Journal of Machine Learning Research, Vol. 15, pp. 3221-3245
**ArXiv:** 1301.3342
**Description:** O(N log N) implementation using Barnes-Hut algorithm for force approximation.
**Implementation:** OpenTSNE (recommended)

---

### FIt-SNE (Fast Interpolation-based t-SNE)

**Title:** Fast interpolation-based t-SNE for improved visualization of single-cell RNA-seq data
**Authors:** George C. Linderman, Manas Rachh, Jeremy G. Hoskins, Stefan Steinerberger, Yuval Kluger
**Year:** 2019
**Publication:** Nature Methods, Vol. 16, pp. 243-245
**Link:** https://github.com/KlugerLab/FIt-SNE
**Description:** FFT-based interpolation achieving O(N) scaling. Dramatically accelerates t-SNE for large datasets.

---

### Multicore t-SNE

**Title:** Multicore t-SNE
**Authors:** Dmitry Ulyanov
**Year:** 2016-2020
**Link:** https://github.com/DmitryUlyanov/Multicore-TSNE
**Description:** Parallel t-SNE with n_jobs parameter for multi-threaded processing.

---

### openTSNE

**Title:** openTSNE: A Modular Python Library for t-SNE
**Authors:** Pavlin G. Poličar, Martin Stražar, Blaž Zupan
**Year:** 2019+
**Link:** https://opentsne.readthedocs.io/
**Description:** Modular implementation with latest improvements and extensible architecture.

---

### RAPIDS cuML t-SNE

**Title:** t-SNE with GPUs
**Authors:** NVIDIA RAPIDS Team
**Year:** 2019-2024
**Link:** https://medium.com/rapids-ai/tsne-with-gpus-hours-to-seconds-9d9c17c941db
**Description:** GPU-accelerated t-SNE running up to 2,000x faster than scikit-learn.

---

### Parametric t-SNE

**Title:** Learning a Parametric Embedding by Preserving Local Structure
**Authors:** Laurens van der Maaten
**Year:** 2009
**Publication:** Proceedings of the Twelfth International Conference on Artificial Intelligence & Statistics (AISTATS)
**Link:** https://lvdmaaten.github.io/software/
**Description:** Extends t-SNE with neural network-based parametric mapping. Enables generalization to new data points without recalculation. Uses multilayer neural network trained with t-SNE objective function.
**Implementation:** Official Matlab implementation, various Python implementations available

---

### Perplexity-free Parametric t-SNE

**Title:** Perplexity-free Parametric t-SNE
**Year:** 2020
**Link:** https://arxiv.org/abs/1906.00297
**Description:** Improves parametric t-SNE by removing dependence on perplexity hyperparameter. Multiscale-aware variant for better global structure preservation.
**Implementation:** Keras/Scikit-learn compatible (FrancescoCrecchi/Multiscale-Parametric-t-SNE)

---

### t-SNE-CUDA (GPU-Accelerated t-SNE)

**Title:** t-SNE-CUDA: GPU-Accelerated t-SNE and its Applications to Modern Data
**Authors:** CannyLab
**Year:** 2018
**ArXiv:** 1807.11824
**Link:** https://github.com/CannyLab/tsne-cuda
**Description:** GPU-accelerated t-SNE implementation using CUDA. Achieves 50-700x speedups on CIFAR-10 and MNIST. Up to 1200x faster than scikit-learn, 50x faster than Multicore-TSNE.
**Implementation:** CUDA C++ with Python bindings

---

### NeuralTSNE

**Title:** NeuralTSNE: A Python Package for the Dimensionality Reduction of Molecular Dynamics Data Using Neural Networks
**Year:** 2025
**ArXiv:** 2505.16476
**Description:** Modern parametric t-SNE implementation for molecular dynamics and scientific computing. Neural network-based parametric mapping for fast inference.
**Implementation:** Python package

---

### Contrastive Learning SNE Variants

**Title:** Supervised SNE with Contrastive Learning
**Year:** 2023
**Link:** https://arxiv.org/abs/2309.08077
**Description:** Incorporates contrastive learning framework into SNE for supervised DR.

---

## Topology-Based Methods

### UMAP (Uniform Manifold Approximation and Projection)

**Title:** UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction
**Authors:** Leland McInnes, John Healy, James Melville
**Year:** 2018
**Publication:** ArXiv preprint
**Link:** https://arxiv.org/abs/1802.03426
**JOSS:** https://joss.theoj.org/papers/10.21105/joss.00861
**Description:** Constructed from Riemannian geometry and algebraic topology. Competitive with t-SNE but preserves more global structure.
**Implementation:** umap-learn (Python), RAPIDS cuML (GPU)

---

**Title:** Nature Review Methods Primers: Uniform manifold approximation and projection
**Authors:** John Healy, Leland McInnes
**Year:** 2024
**Publication:** Nature Reviews Methods Primers, Vol. 4, No. 82
**Link:** https://doi.org/10.1038/s43586-024-00363-x
**Description:** Comprehensive introduction and tutorial for UMAP.

---

## UMAP Ecosystem

### Parametric UMAP

**Title:** Parametric UMAP Embeddings for Representation and Semisupervised Learning
**Authors:** Tim Sainburg, Leland McInnes, Timothy Q. Gentner
**Year:** 2021
**Publication:** Neural Computation, Vol. 33, No. 11, pp. 2881-2907
**ArXiv:** 2009.12981
**Link:** https://direct.mit.edu/neco/article/33/11/2881/107068
**Description:** Extends UMAP with parametric optimization over neural network weights. Enables fast inference on new data.
**Implementation:** umap-learn with PyTorch backend

---

### densMAP

**Title:** Density-Preserving Data Visualization Unveils Dynamic Patterns of Single-Cell Transcriptomic Variability
**Authors:** Ashwin Narayan, Bonnie Berger, Hyunghoon Cho
**Year:** 2021
**Publication:** Nature Biotechnology, Vol. 39, pp. 765-774
**DOI:** 10.1038/s41587-020-00801-7
**bioRxiv:** 10.1101/2020.05.12.077776
**Description:** Augments UMAP to preserve local density information.
**Implementation:** umap-learn with densMAP option

---

### Aligned UMAP

**Title:** Aligned UMAP for Longitudinal Data
**Authors:** UMAP Contributors
**Year:** 2023
**Link:** https://umap-learn.readthedocs.io/en/latest/aligned_umap_basic_usage.html
**Description:** Extension for visualizing longitudinal/time-series data by aligning multiple embeddings.
**Implementation:** umap-learn

---

### UMAP Supervised/Semi-supervised

**Title:** UMAP: Uniform Manifold Approximation and Projection (with supervised features)
**Authors:** Leland McInnes
**Year:** 2018+
**Description:** UMAP supports supervised and semi-supervised modes using target vectors.
**Implementation:** umap-learn with `y` parameter

---

## Modern Manifold Learning

### PHATE (Potential of Heat-diffusion for Affinity-based Transition Embedding)

**Title:** Visualizing structure and transitions in high-dimensional biological data
**Authors:** Kevin R. Moon, David van Dijk, et al.
**Year:** 2019
**Publication:** Nature Biotechnology, Vol. 37, pp. 1482-1492
**bioRxiv:** 10.1101/120378
**Link:** https://github.com/KrishnaswamyLab/PHATE
**Description:** Captures local and global structure using information-geometric distance. Naturally discovers branching structures.
**Implementation:** phate (Python)

---

### TriMap (Large-scale Dimensionality Reduction Using Triplets)

**Title:** TriMap: Large-scale Dimensionality Reduction Using Triplets
**Authors:** Ehsan Amid, Manfred K. Warmuth
**Year:** 2019
**Link:** https://arxiv.org/abs/1910.00204
**Description:** Uses triplet constraints sampled from high-dimensional space. Preserves global structure better than t-SNE/UMAP. Creates triplets predominantly from nearest neighbors with fraction of random samples for robust global structure preservation.
**Implementation:** trimap (Python)

**Note on Comparison:** While TriMap excels at global structure preservation, it has trade-offs in local structure vs. global structure preservation compared to newer methods like PaCMAP.

---

### PaCMAP (Pairwise Controlled Manifold Approximation)

**Title:** Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP
**Authors:** Yingfan Wang, Haiyang Huang, Cynthia Rudin, Yaron Shaposhnik
**Year:** 2021
**Publication:** Journal of Machine Learning Research, Vol. 22, No. 201, pp. 1-73
**Link:** https://arxiv.org/abs/2012.04456
**Description:** Uses three kinds of pairs to dynamically capture global structure first, then refine local structure.
**Implementation:** pacmap (Python)

---

### LocalMAP (2024)

**Title:** LocalMAP: Locally adaptive dimensionality reduction
**Authors:** Recent Development
**Year:** 2024
**Link:** https://arxiv.org/abs/2412.15426
**Description:** Dynamically discovers untrustworthy graph regions, cleaning data during execution.

---

### DREAMS (2024)

**Title:** DREAMS: Deep Representation Embedding for Adaptive Manifold Structure
**Year:** 2024
**Link:** https://arxiv.org/abs/2508.13747
**Description:** Combines interpretability and global structure of PCA with local sensitivity of t-SNE.

---

### FeatMAP (2022)

**Title:** FeatMAP: Feature-preserving Manifold Approximation and Projection
**Authors:** Various
**Year:** 2022
**Link:** https://arxiv.org/abs/2211.09321
**Description:** Interpretable DR preserving source features via tangent space embedding.

---

## Deep Learning Methods

### Autoencoders

**Title:** Reducing the Dimensionality of Data with Neural Networks
**Authors:** Geoffrey E. Hinton, Ruslan R. Salakhutdinov
**Year:** 2006
**Publication:** Science, Vol. 313, No. 5786, pp. 504-507
**DOI:** 10.1126/science.1127647
**Description:** Neural networks with small central layer for DR and reconstruction.
**Implementation:** PyTorch, TensorFlow

---

### Variational Autoencoders (VAE)

**Title:** Auto-Encoding Variational Bayes
**Authors:** Diederik P. Kingma, Max Welling
**Year:** 2013
**Link:** https://arxiv.org/abs/1312.6114
**Description:** Combines autoencoders with probabilistic modeling. Learns latent distributions.
**Implementation:** PyTorch, TensorFlow

---

### Deep Autoencoder Review (2025)

**Title:** A Review of Deep Learning: Recent Advances and Prospects for Applied Deep Learning
**Year:** 2025
**Link:** https://link.springer.com/article/10.1007/s11831-025-10260-5
**Description:** Comprehensive review of autoencoder architectures from basic to advanced (adversarial, convolutional, variational).

---

### Neural ODEs for DR (2025)

**Title:** Latent Neural ODEs with Learnable Kernels
**Authors:** Various
**Year:** 2025
**Link:** https://arxiv.org/abs/2502.08683
**Description:** Couples DR with Neural ODEs to learn PDE solution operators in reduced space.

---

### Normalizing Flows

**Title:** Normalizing Flows as Dimensionality Reduction
**Year:** 2019-2024
**Link:** https://arxiv.org/abs/2311.01404
**Description:** Continuous-time normalizing flows using neural ODEs for invertible mappings.

---

### scVI/scANVI (Single-cell Analysis)

**Title:** Deep generative modeling for interpretable classifiers of intracellular localization
**Authors:** Romain Lopez, Jeffrey Regier, et al.
**Year:** 2018-2021
**Link:** https://www.nature.com/articles/s41592-018-0229-2
**Description:** Deep generative models for probabilistic single-cell analysis.
**Implementation:** scvi-tools

---

## Contrastive Learning for DR

### SimCLR (A Simple Framework for Contrastive Learning)

**Title:** A Simple Framework for Contrastive Learning of Visual Representations
**Authors:** Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
**Year:** 2020
**Publication:** ICML 2020
**Link:** https://arxiv.org/abs/2002.05709
**Description:** Self-supervised framework maximizing similarity between augmented image pairs.
**Implementation:** PyTorch, TensorFlow

---

### SimSiam (Simple Siamese Networks)

**Title:** Exploring Simple Siamese Representation Learning
**Authors:** Facebook AI Research
**Year:** 2020
**Link:** https://learnopencv.com/simsiam/
**Description:** Self-supervised learning without negative samples using stop-gradient mechanism.

---

### NCLDR (Nearest-Neighbor Contrastive Learning)

**Title:** NCLDR: Nearest-Neighbor Contrastive Learning with Dual Correlation Loss
**Year:** 2024
**Link:** https://www.sciencedirect.com/science/article/abs/pii/S0925231224006192
**Description:** Contrastive learning framework for neighbor embedding.

---

### Siamese Prototypical Contrastive Learning

**Title:** Siamese Prototypical Contrastive Learning
**Year:** 2022
**Link:** https://arxiv.org/abs/2208.08819
**Description:** Combines Siamese networks with prototypical learning.

---

## Graph Embedding Methods

### ForceAtlas2 (Network Visualization)

**Title:** ForceAtlas2: A Continuous Graph Layout Algorithm for Handy Network Visualization
**Authors:** Mathieu Jacomy, Tommaso Venturini, Sebastien Heymann, Mathieu Bastian
**Year:** 2014
**Publication:** PLOS ONE, Vol. 9, No. 6, e98679
**DOI:** 10.1371/journal.pone.0098679
**Description:** Force-directed layout integrating Barnes-Hut simulation. Default in Gephi.
**Implementation:** Gephi, PyVis

---

### GNN-Based Graph Embedding

**Title:** Recent Advances in Graph Neural Networks
**Year:** 2023-2024
**Description:** GNNs learn node embeddings capturing connectivity. Graph pooling methods reduce dimensionality.
**Implementation:** PyTorch Geometric, DGL

---

## Accelerated Implementations

### RAPIDS cuML UMAP

**Title:** UMAP on GPU with RAPIDS cuML
**Authors:** NVIDIA RAPIDS Team
**Year:** 2020-2025
**Link:** https://developer.nvidia.com/blog/even-faster-and-more-scalable-umap-on-the-gpu-with-rapids-cuml/
**Description:** Batched approximate nearest neighbor algorithm for UMAP. Up to 311x speedup.
**Implementation:** RAPIDS cuML

---

### RAPIDS cuML Spectral Embedding

**Title:** GPU-Accelerated Spectral Embedding
**Authors:** NVIDIA RAPIDS Team
**Year:** 2025
**Link:** https://developer.nvidia.com/blog/nvidia-rapids-25-08/
**Description:** GPU-accelerated spectral embedding using eigenvalue decomposition.

---

### Panorama (ML-Driven DR)

**Title:** Panorama: ML-Driven Orthogonal Transforms for Data Compression
**Year:** 2024
**Link:** https://arxiv.org/abs/2510.00566
**Description:** Data-adaptive learned orthogonal transforms. 2-30x speedup with HNSW/MRPT/Annoy.

---

### Dimensionality-Reduction for ANN Survey

**Title:** Survey of DR Techniques for Approximate Nearest Neighbor Search
**Year:** 2024
**Link:** https://arxiv.org/abs/2403.13491
**Description:** Evaluates DR impact on ANN. Shows 6.3x improvement with 98% recall using DWT, ADSampling, OPQ.

---

## Time-Series Dimensionality Reduction

### SAX_CP (SAX with Change Points)

**Title:** SAX_CP: Novel Trend-Based SAX Reduction
**Year:** 2019
**Link:** https://www.sciencedirect.com/science/article/abs/pii/S0957417419302568
**Description:** Captures trends via abrupt change points, improving on standard SAX.

---

### Signal2Vec

**Title:** Signal2Vec: Time Series Embedding Representation
**Authors:** Various
**Year:** 2019
**Link:** https://www.researchgate.net/publication/333086784
**Description:** Unsupervised framework for time-series vector space representation.

---

### SPARTAN (2025)

**Title:** SPARTAN: Data-Adaptive Symbolic Time-Series Approximation
**Year:** 2025
**Link:** https://dl.acm.org/doi/10.1145/3725357
**Description:** Improves upon traditional SAX approaches.

---

### TimeCluster

**Title:** TimeCluster: Dimension Reduction Applied to Temporal Data for Visual Analytics
**Year:** 2019
**Link:** https://link.springer.com/article/10.1007/s00371-019-01673-y
**Description:** Applies DR and clustering to sliding windows of time-series.

---

## Interpretable & Explainable DR

### LXDR (Local eXplanation of DR)

**Title:** LXDR: Local eXplanation of Dimensionality Reduction
**Year:** 2024
**Link:** https://www.sciencedirect.com/science/article/abs/pii/S0957417424009400
**Description:** Model-agnostic technique for local interpretations of any DR method.

---

### Interpretable Discriminative DR

**Title:** Interpretable Discriminative Dimensionality Reduction
**Year:** 2019
**Link:** https://arxiv.org/abs/1909.09218
**Description:** Framework for interpretable DR with feature selection on manifolds.

---

### FeatMAP (Feature Importance)

**Title:** FeatMAP: Feature-preserving Manifold Approximation
**Year:** 2022
**Link:** https://arxiv.org/abs/2211.09321
**Description:** Preserves source features with local feature importance visualization.

---

## Information-Theoretic DR

### Maximally Informative Dimensions (MID)

**Title:** Maximally Informative Dimensions
**Year:** 2015
**Link:** https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004141
**Description:** Information-theoretic method for encoding model dimensionality reduction.

---

### Information-Theoretic DR Approaches

**Title:** Information-theoretic Approaches to Dimensionality Reduction
**Authors:** Sambriddhi Mainali, Max Garzon, et al.
**Year:** 2021
**Link:** https://link.springer.com/article/10.1007/s41060-021-00272-2
**Description:** Nonlinear alternatives using mutual information and conditional entropy.

---

### Probabilistic Perspective on UMAP and t-SNE

**Title:** A Probabilistic Perspective on UMAP and t-SNE
**Year:** 2024
**Link:** https://arxiv.org/abs/2405.17412
**Description:** Recasts UMAP and t-SNE as MAP inference methods with Wishart distributions.

---

## Domain-Specific Methods

### Word2Vec/Doc2Vec (Text)

**Title:** Efficient Estimation of Word Representations in Vector Space
**Authors:** Tomas Mikolov, et al.
**Year:** 2013
**Description:** Skip-gram and CBOW for text embeddings (100-1000 dims). Doc2Vec extends to documents.
**Implementation:** Gensim

---

### BERT/Sentence-BERT (Text)

**Title:** Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
**Authors:** Nils Reimers, Iryna Gurevych
**Year:** 2019
**Link:** https://sbert.net/
**Description:** Fine-tuned BERT for efficient sentence embeddings. 768 dims → 128 dims with minimal loss.
**Implementation:** sentence-transformers

---

### Single-Cell RNA-seq Methods

**Title:** Nonlinear dimensionality reduction based visualization of single-cell RNA sequencing data
**Year:** 2023
**Link:** https://jast.springeropen.com/articles/10.1186/s40759-023-00082-w
**Description:** Reviews t-SNE (best accuracy) and UMAP (highest stability) for scRNA-seq visualization.

---

**Title:** Structure-preserving visualization for single-cell RNA-Seq with deep manifold transformation
**Year:** 2023
**Link:** https://www.nature.com/articles/s42003-023-05046-z
**Description:** DV method preserves local/global structure while handling batch effects.

---

**Title:** DREAM: Improved VAE for scRNA-seq
**Year:** 2023
**Link:** https://academic.oup.com/bib/article-abstract/24/6/bbad341/7268652
**Description:** Combines VAE with Gaussian mixture for cell type identification.

---

### Topological Data Analysis

**Title:** Topological Data Analysis: Mapper, Persistence and Applications
**Description:** Comprehensive tutorial on TDA including Mapper algorithm and persistent homology.
**Implementation:** Kepler Mapper, ripser

---

**Title:** A Survey on Mapper Algorithm
**Authors:** Various
**Year:** 2025
**Link:** https://www.researchgate.net/publication/390772164
**Description:** Comprehensive review of Mapper from 2007-2025 across fields.

---

### Population Genetics

**Title:** Revealing Multi-scale Population Structure in Large Cohorts
**Year:** 2019
**Link:** https://www.biorxiv.org/content/10.1101/423632v2
**Description:** Uses UMAP for population genetics visualization with novel RGB projection.

---

## Metric Learning for DR

### Hierarchical Triplet Loss

**Title:** Hierarchical Triplet Loss for Metric Learning
**Year:** 2018
**Link:** https://arxiv.org/abs/1810.06951
**Description:** Automatically collects informative samples via hierarchical tree. 1-18% improvement over standard triplet loss.

---

### TripletPCA and ContrastivePCA

**Title:** TripletPCA and ContrastivePCA for Metric Learning
**Year:** 2023
**Link:** https://www.sciencedirect.com/science/article/abs/pii/S0957417423025666
**Description:** First attempts incorporating DR into pair-based metric learning. 10.55% average improvement.

---

## Advanced Topics

### Online/Incremental DR

**Title:** Xtreaming: Incremental Dimensionality Reduction
**Year:** 2021
**Link:** https://www.sciencedirect.com/science/article/abs/pii/S0097849321001734
**Description:** Continuous updating for streaming data without revisiting high-dimensional data.

---

**Title:** MOSES: Memory-limited Online Subspace Estimation
**Year:** 2018
**Link:** https://arxiv.org/abs/1806.01304
**Description:** Single-pass subspace estimation for streaming data.

---

### Privacy-Preserving DR

**Title:** Privacy-Preserving Federated DR
**Year:** 2025
**Link:** https://www.mdpi.com/2079-9292/14/16/3182
**Description:** Multi-stage DR framework for vertical federated learning.

---

### Quantum DR

**Title:** Quantum Resonant Dimensionality Reduction
**Year:** 2024
**Link:** https://arxiv.org/abs/2405.12625
**Description:** Novel quantum algorithm reducing ancilla qubits for quantum PCA.

---

**Title:** Quantum LDA for Dimensionality Reduction
**Year:** 2021-2023
**Link:** https://arxiv.org/abs/2103.03131
**Description:** Quantum algorithm for linear discriminant analysis.

---

## Theoretical Papers

### Dimensionality Reduction and Wasserstein Stability

**Title:** Dimensionality Reduction and Wasserstein Stability
**Year:** 2022
**Link:** https://arxiv.org/abs/2203.09347
**Publication:** JMLR 2024
**Description:** Stability analysis for kernel regression after dimensionality reduction.

---

### Convergence Guarantees for Network Time Series

**Title:** Convergence Guarantees for Network Time Series Embedding
**Year:** 2025
**Link:** https://arxiv.org/abs/2501.08456
**Description:** Theoretical convergence analysis for raw stress embedding.

---

### Riemannian Geometric Framework

**Title:** Riemannian Geometric Framework for Manifold Learning
**Year:** 2020
**Link:** https://link.springer.com/article/10.1007/s11634-020-00426-3
**Description:** Framework for manifold learning of non-Euclidean data.

---

### Geometric Viewpoint of Manifold Learning

**Title:** A Geometric Viewpoint on Manifold Learning
**Year:** 2015
**Link:** https://applied-informatics-j.springeropen.com/articles/10.1186/s40535-015-0006-6
**Description:** Survey with geometric perspective on ISOMAP, LLE, Laplacian eigenmaps.

---

## Comparison & Benchmark Papers

### Comprehensive Evaluation of DR Methods for Transcriptomic Data (2022)

**Title:** Towards a comprehensive evaluation of dimension reduction methods for transcriptomic data visualization
**Authors:** Haiyang Huang, Yingfan Wang, Cynthia Rudin, Edward P. Browne
**Year:** 2022
**Publication:** Communications Biology, Vol. 5, Article 719
**DOI:** 10.1038/s42003-022-03628-x
**Link:** https://www.nature.com/articles/s42003-022-03628-x
**Description:** Comprehensive evaluation framework assessing PCA, t-SNE, UMAP, TriMap, PaCMAP, ForceAtlas2, and PHATE across five criteria: local structure preservation, global structure preservation, parameter sensitivity, preprocessing sensitivity, and computational efficiency.
**Key Findings:**
- t-SNE, UMAP excel at local structure but fail at global structure
- TriMap, PaCMAP, ForceAtlas2 preserve both local and global structure
- PaCMAP is fastest, especially for large datasets (>10,000 samples)
- t-SNE and UMAP are highly sensitive to parameter choices

---

### Comprehensive Review (2025)

**Title:** Comprehensive review of dimensionality reduction algorithms
**Year:** 2025
**Publication:** PeerJ Computer Science
**Link:** https://peerj.com/articles/cs-2254
**Description:** Reviews PCA, t-SNE, UMAP, autoencoders with interpretability and fairness analysis.

---

### Dimensionality Reduction: A Comparative Review

**Title:** Dimensionality Reduction: A Comparative Review
**Authors:** Laurens van der Maaten, Eric Postma, Jaap van den Herik
**Year:** 2009
**Publication:** Technical Report TiCC TR 2009-005
**Description:** Comprehensive comparison of spectral, probabilistic, and neural methods.

---

### CyTOF Data Comparative Analysis

**Title:** Comparative analysis of dimension reduction methods for cytometry by time-of-flight data
**Year:** 2023
**Publication:** Nature Communications
**Volume:** 14, Article 6968
**DOI:** 10.1038/s41467-023-37478-w
**Description:** Comprehensive benchmarking of 21 DR methods on 110 real and 425 synthetic single-cell datasets. Evaluates local structure, global structure, parameter sensitivity, and computational efficiency on cytometry data.
**Key Methods Evaluated:** t-SNE, UMAP, TriMap, PaCMAP, ForceAtlas2, PHATE, and others
**Findings:** All methods preserve local structure well, while only TriMap, PaCMAP, and ForceAtlas2 preserve global structure reliably.

---

### Time-Series DR Survey

**Title:** Survey on Dimensionality Reduction for Time-Series
**Year:** 2023
**Publication:** IEEE
**Description:** 12 DR algorithms for time-series, categorized by complexity and properties.

---

### Manifold Learning Comparative Study

**Title:** Comparative Analysis of Manifold Learning-Based DR Methods
**Year:** 2024
**Publication:** MDPI Mathematics, Vol. 12, No. 15
**Link:** https://mdpi.com/2227-7390/12/15/2388
**Description:** Mathematical comparison of Isomap, LLE, LE, HE, LTSA, MVU.

---

### Quality Assessment of DR

**Title:** Quality assessment of dimensionality reduction: Rank-based criteria
**Year:** 2009
**Publication:** Neurocomputing
**Description:** Introduces trustworthiness and continuity metrics.

---

### pyDRMetrics Toolkit

**Title:** pyDRMetrics - Python Toolkit for DR Quality Assessment
**Year:** 2021
**Publication:** Heliyon
**Description:** Python toolkit for evaluation metrics (reconstruction error, distance matrix, LCMC, etc).

---

### Metric Selection Bias Reduction

**Title:** Metric Design != Metric Behavior: Unbiased Evaluation of DR
**Year:** 2025
**Link:** https://arxiv.org/abs/2507.02225
**Description:** Proposes workflow reducing bias in evaluation metric selection.

---

### Critical Perspective: Why You Should Not Rely on t-SNE, UMAP or TriMAP

**Title:** Why you should not rely on t-SNE, UMAP or TriMAP
**Author:** Mathias Gruber
**Publication:** Towards Data Science (Medium)
**Year:** 2019-2021
**Link:** https://towardsdatascience.com/why-you-should-not-rely-on-t-sne-umap-or-trimap-f8f5dc333e59
**Description:** Critical analysis of popular DR methods' limitations. Identifies that t-SNE and UMAP focus on local structure preservation while TriMAP focuses on global structure, creating a fundamental trade-off. Demonstrates that PaCMAP achieves better balance between local and global structure preservation while maintaining faster computational performance.
**Key Points:**
- t-SNE (69s), UMAP (24s), TriMAP (12.56s) vs. PaCMAP (8.4s)
- PaCMAP manages local vs. global trade-off better than alternatives
- Parameter sensitivity is often overlooked by practitioners
- Advocates for understanding algorithm mechanics before interpretation

---

## Implementation Resources

### Core Libraries

| Library | Language | Method | Status |
|---------|----------|--------|--------|
| **umap-learn** | Python | UMAP, Parametric UMAP, densMAP | ✅ Mature |
| **scikit-learn** | Python | PCA, Isomap, LLE, t-SNE, MDS | ✅ Mature |
| **RAPIDS cuML** | Python/CUDA | GPU-accelerated UMAP, t-SNE, PCA | ✅ Production |
| **openTSNE** | Python | Optimized t-SNE variants | ✅ Active |
| **PHATE** | Python | PHATE algorithm | ✅ Active |
| **TriMap** | Python | TriMap algorithm | ✅ Active |
| **PaCMAP** | Python | PaCMAP algorithm | ✅ Active |
| **PyTorch** | Python | Autoencoders, VAE, Neural ODE | ✅ Mature |
| **TensorFlow** | Python | Deep learning DR | ✅ Mature |
| **PyTorch Geometric** | Python | Graph embedding, GNN | ✅ Active |
| **sentence-transformers** | Python | BERT embeddings | ✅ Active |
| **scvi-tools** | Python | scVI/scANVI for single-cell | ✅ Active |
| **Kepler Mapper** | Python | Topological data analysis | ✅ Active |
| **Gephi** | Java | Network visualization | ✅ Mature |

### Datasets for Benchmarking

- MNIST (70,000 images, 784 dimensions)
- Fashion-MNIST (70,000 images, 784 dimensions)
- CIFAR-10 (60,000 images, 3,072 dimensions)
- Single-cell RNA-seq (10,000-1,000,000 cells, 20,000 genes)
- UCI Shuttle (43,500 samples, 8 dimensions)
- Word2Vec (millions of word embeddings, 300 dimensions)
- Single-cell benchmarks (scRNA-seq, scATAC-seq, CyTOF)

### Evaluation Metrics

- **Trustworthiness:** Preservation of k nearest neighbors
- **Continuity:** Inverse relationship (neighbors remain neighbors)
- **LCMC:** Local Correlation Integral Dimension
- **Co-ranking Matrix:** Assesses local and global structure
- **Reconstruction Error:** Autoencoder loss
- **Silhouette Score:** Cluster cohesion
- **Davies-Bouldin Index:** Cluster separation

---

## Future Implementation Plans

### Phase 1: Core Methods (Months 1-3)
- [ ] Optimize HNSW-RS backend integration
- [ ] Implement GPU acceleration for all core algorithms
- [ ] Add Parametric UMAP with PyTorch
- [ ] Implement densMAP variants

### Phase 2: Expand Ecosystem (Months 4-6)
- [ ] Add PHATE, TriMap, PaCMAP
- [ ] Implement spectral methods (LE, Isomap, LLE variants)
- [ ] GPU-accelerated t-SNE improvements
- [ ] Add hierarchical DR methods

### Phase 3: Advanced Features (Months 7-12)
- [ ] Deep learning methods (autoencoders, VAE, Neural ODE)
- [ ] Graph embedding methods (Node2Vec, GraphSAGE)
- [ ] Time-series specific methods
- [ ] Interpretable DR wrappers
- [ ] Topological data analysis integration

### Phase 4: Enterprise Features (Months 13+)
- [ ] Streaming/online DR
- [ ] Privacy-preserving DR (federated learning)
- [ ] Multi-modal DR
- [ ] Interactive visualizations
- [ ] Quantum DR algorithms

---

## References

### Foundational Texts
- Manifold Learning (Choi, 2018)
- Deep Learning (Goodfellow et al., 2016)
- The Elements of Statistical Learning (Hastie et al., 2009)

### Survey Papers
- Dimensionality Reduction: A Comparative Review (van der Maaten et al., 2009)
- Manifold Learning: What, How, and Why (Annual Reviews)
- Deep Learning for Representation Learning (Bengio et al., 2013)

### Code Repositories
- https://github.com/lmcinnes/umap
- https://github.com/scikit-learn/scikit-learn
- https://github.com/KlugerLab/FIt-SNE
- https://github.com/DmitryUlyanov/Multicore-TSNE
- https://github.com/KrishnaswamyLab/PHATE
- https://github.com/aditya-grover/node2vec

---

**Last Updated:** November 2025
**Maintainers:** UMAP Development Team
**Contributing:** Contributions welcome! Submit papers via GitHub Issues.

---

*This research page is a living document. New papers are regularly added as they are published. For the latest papers, see the ArXiv preprint archive in dimensionality reduction and machine learning categories.*
