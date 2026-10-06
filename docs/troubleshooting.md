# Troubleshooting

## A Rust reducer is None

Check `squeeze.__file__`, then rebuild the release extension using the
[installation commands](installation.md). `squeeze.__init__` exposes unavailable
Rust classes as `None`; an older extension may provide PCA but lack NeighborMap.
Restart an existing Python process after rebuilding a loaded extension.

## NumPy dtype or dimensionality errors

Pass a two-dimensional `float64` NumPy array to the Rust methods. Ensure all rows
have the same number of features. Convert sparse inputs only after estimating
memory needs. Check for NaN and infinity and keep neighbor counts below sample count.

## BLAS linking or build failure

Install OpenBLAS development libraries, a Fortran compiler, pkg-config, and OpenSSL
headers. Use the documented `ndarray-linalg/openblas-system` feature. A Python-only
sync does not rebuild the compiled extension. Avoid mixing extensions from another
Python ABI or machine.

## The first fit is slow

UMAP invokes Numba compilation. Compare release builds after warmup, with identical
thread settings and input rows. Metric scoring can itself be quadratic and should
be timed separately. Check [performance notes](performance.md).

## A picture changes between runs

Set a supported `random_state`, retain input order, and record versions and thread
counts. Rotation and reflection can change without changing local relationships.
Compare metrics and aligned coordinates, not screen positions alone.

## A wrapper has transform but a reducer does not

Some composition wrappers refit fit-only components on new data. That is a new
coordinate system. Use a native transform or explicit interpolation as described
in [Transforming new data](transform.md).

## Fashion-MNIST checksum mismatch

The loader rejects corrupt cached files instead of benchmarking them. Remove the
identified corrupt download from the configured cache and retry with network access.
Do not disable the checksum to get a run through.

## Report a reproducible issue

Open an issue in [GeorgePearse/squeeze](https://github.com/GeorgePearse/squeeze/issues).
Include code revision, Python/uv versions, platform, constructor arguments, input
shape/dtype, thread settings, and the traceback. Prefer a small Digits example that
reproduces the problem without private data.
