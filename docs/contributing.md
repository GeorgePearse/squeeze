# Contributing

Start from `main`, make a focused branch, and open a draft pull request against
[GeorgePearse/squeeze](https://github.com/GeorgePearse/squeeze). Preserve existing
implementations when adding an algorithm or backend option.

## Local setup and checks

Follow [Installation](installation.md) to build the extension. Then run targeted
checks for the area changed:

```bash
uv run --no-sync pytest squeeze/tests/test_neighbor_map.py squeeze/tests/test_neighbor_benchmark.py squeeze/tests/test_benchmark_datasets.py
cargo test --release --lib --features ndarray-linalg/openblas-system
prek run --from-ref origin/main --to-ref HEAD
```

Use Digits as the default performance dataset. Fashion-MNIST is the supported
additional dataset. Retain exact sample selection, raw results, coordinates,
source revision and thread settings when reporting a performance improvement.
Working experiments belong in `working_docs/`; maintained user documentation
belongs in `docs/`.

## Build documentation without the Rust toolchain

```bash
uv venv .venv-docs --python 3.10
uv pip install --python .venv-docs/bin/python -r docs_requirements.txt
.venv-docs/bin/mkdocs build --strict
.venv-docs/bin/mkdocs serve
```

The build uses MkDocs Material with pinned direct dependencies. The site has one
source tree and no Sphinx build. Local serving uses the configured `/squeeze/` base
path. `just docs` and `just docs-serve` provide equivalent shortcuts after setup.

## Verify examples and artifacts

After building the extension, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
uv run --no-sync python scripts/check_docs.py
```

Every `python` code fence in the maintained docs is an independent executable
example. The check runs them in a temporary directory and checks constructor
signatures, public API coverage, and copied benchmark assets. Optional TensorFlow
examples are described separately rather than presented as tested core examples.

The Documentation workflow builds the site strictly on pull requests, stores the
rendered site as an artifact, and publishes main through GitHub Pages. The Rust
graph workflow runs the executable examples against its fresh extension build.
