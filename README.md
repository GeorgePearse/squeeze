# Squeeze

Squeeze is one focused Rust library for CPU-based dimensionality reduction.
It contains several algorithms behind a small native API, plus the distance
and nearest-neighbor primitives they share.

## Algorithms

- PCA
- t-SNE with optional Barnes-Hut approximation
- classical and metric MDS
- Isomap
- locally linear embedding (LLE)
- PHATE
- TriMap
- PaCMAP

## Usage

```rust
use ndarray::array;
use squeeze::algorithms::PCA;

let samples = array![
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 4.0, 6.0],
];

let embedding = PCA::new(2).fit_transform(&samples)?;
# Ok::<(), squeeze::Error>(())
```

All algorithm inputs and outputs use `ndarray::Array2`. Shared functionality
is exposed under:

- `squeeze::algorithms`
- `squeeze::neighbors`
- `squeeze::distance`

## Development

```bash
cargo fmt --check
cargo clippy --all-targets
cargo test --all-targets
cargo bench
```

The project intentionally has no Python package, extension-module, GPU, or
generated-site layer. Language bindings can live in separate repositories and
depend on this crate.
