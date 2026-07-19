# Contributing

Create a branch from `main`, keep the change focused, and include tests for
behavior changes.

Before opening a pull request, run:

```bash
cargo fmt --check
cargo clippy --all-targets
cargo test --all-targets
```

Performance changes should include a Criterion benchmark or before/after
results from an existing benchmark.
