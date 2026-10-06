# Frozen validation protocol (2026-10-06)

Exploration uses Digits and seed 42. Validation uses seeds 17, 29, 53 on the
same Digits data; these are optimizer replicates, **not held-out datasets**.
No labels enter training or hyperparameter selection. All 1,797 samples and
64 raw pixel features are used for every timed fit.

Freeze after exploring PCA/spectral initializations at 80/160/320 epochs:
- NeighborMap PCA 80, 160 (default), 320 epochs: retain speed/quality choices.
- NeighborMap spectral 160: comparison of initialization tradeoffs.
- SpectralMap 128 block iterations: cheap approximate spectral alternative.
- Baselines: existing Rust PCA, t-SNE, PaCMAP, TriMap; Squeeze Python UMAP;
  sklearn PCA, t-SNE, Isomap, SpectralEmbedding.
- One thread per BLAS/Numba/Rayon/OpenMP library; full-data warmup per method.
- Interleave method order deterministically for each seed.
- Record full fit_transform runtime (graph + initialization + layout), exact
  sklearn trustworthiness at k=5/15/30, neighbor recall at k=15, Spearman distance
  correlation on fixed pairs (actual count is persisted in results).
- Record failures, library versions, data/source/extension hashes, raw embeddings.

The exploratory squared-distance optimization opts into a new AVX2 multiply/add
kernel, defers sqrt until after top-k selection, and keeps the old metric entry
points and baseline algorithms unchanged. The first optimized exploration
ran alongside a Rust test compilation; its timing is preliminary. Validation
must run with no other test/build/benchmark started by this task.

No claim of generalization beyond Digits, statistical significance from three
seeds, convergence of approximate spectral iteration, or equivalence to UMAP.
The machine is shared; timing is wall-clock and susceptible to unrelated load.
