# Reproducibility

A seed is one part of an experiment. Preserve the input rows and order,
preprocessing, constructor parameters, dependency versions, code revision, and
thread settings as well.

## Control threads before starting Python

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 uv run --no-sync python benchmark_metrics_heatmap.py
```

Numba compilation and other one-time work can dominate a first call. The heatmap
runner performs one full-data warmup per method, excluded from measured fitting.
Its metric scoring is also outside the fit timer. The repeated-seed runner keeps
its own recorded protocol; do not combine timings from different protocols.

## Save coordinates as well as scores

The runners save NPZ embeddings and JSON provenance. Fashion-MNIST also records
selected test-set indices and checksums. The sample selection seed is separate
from each algorithm seed, so changing the layout seed does not change the cohort.

Floating-point reduction order, CPU instruction selection, BLAS implementation,
and dependency changes can affect results. Compare quality distributions over
seeds rather than requiring bitwise-identical layouts across machines.

## Separate exploration from evaluation

Tuning algorithms against the same benchmark repeatedly makes that benchmark an
exploration set. Report these results as such. Hold out a separate evaluation
cohort for a generalization claim, and fit preprocessing and transforms only on
the training split when assessing downstream inference.
