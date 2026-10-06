# Installation

Build from this repository to get the current Rust algorithms and benchmark tools.
The instructions below are the Linux source-build path used by CI. Other platforms
may require different BLAS and compiler setup; these instructions do not establish
Windows or macOS support.

## Toolchain

Use Python 3.10 for the CI-tested environment, a stable Rust toolchain, and
**uv 0.12.23**. The version pin lives in `.uv-version` and `pyproject.toml`.
If uv manages its own installation, `uv self update 0.12.23` selects that version.

On Debian or Ubuntu, install the native prerequisites:

```bash
sudo apt-get install libopenblas-dev libssl-dev gfortran pkg-config
```

Then clone and build:

```bash
git clone https://github.com/GeorgePearse/squeeze.git
cd squeeze
uv sync --frozen --extra dev --extra benchmark --no-install-project
uv run --no-sync maturin develop --release --features extension-module,ndarray-linalg/openblas-system
```

The two-step build installs locked Python dependencies before compiling the local
extension against system OpenBLAS. Subsequent `uv run --no-sync` commands use that
environment without replacing the manually built extension.

## Check the installed backend

```python
import squeeze

assert squeeze.PCA is not None, "Build the Rust extension first"
assert squeeze.NeighborMap is not None, "Rebuild for the new graph methods"
print(squeeze.__version__)
print(squeeze.__file__)
```

Run examples from the checkout with `uv run --no-sync python your_script.py`.
If `NeighborMap` or `SpectralMap` is `None`, an old or missing compiled extension
is being imported. Rebuild and check `squeeze.__file__` before debugging parameters.

## Optional dependencies

| Extra | Purpose |
| --- | --- |
| `benchmark` | Matplotlib and pandas for benchmark charts and result tables |
| `dev` | Maturin, pytest, linting, and development utilities |
| `plot` | Additional Python plotting integrations |
| `parametric` | TensorFlow for the inherited ParametricUMAP implementation |
| `test` | Python test tooling |

The core install does not enable TensorFlow. Documentation builds use a separate,
small environment described in [Contributing](contributing.md).
