# Squeeze development guide

Squeeze is a single Rust library for CPU-based dimensionality reduction.

## Scope

- Keep algorithm implementations in `src/` and expose them through
  `squeeze::algorithms`.
- Put shared nearest-neighbor code under `squeeze::neighbors` and distance
  primitives under `squeeze::distance`.
- Keep the core crate independent from Python, JavaScript, and GPU runtimes.
- Prefer shared numerical primitives when multiple algorithms need the same
  operation.
- Add focused unit tests with algorithm changes.
- Use the sklearn Digits dataset only when external benchmark data is needed.

## Commands

Run these before submitting a change:

```bash
cargo fmt --check
cargo clippy --all-targets
cargo test --all-targets
```

Use `cargo bench` for performance-sensitive work.

## Workflow

Work on a scoped branch from `main`. Do not commit directly to `main`. Keep
generated output and local build artifacts out of Git, and document public API
changes in the README or rustdoc.
