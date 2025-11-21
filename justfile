# Justfile for squeeze development
# Run `just` to see available commands

# Default recipe - show help
default:
    @just --list

# Install dependencies and build the Rust extension
install:
    uv sync --extra dev --extra benchmark
    uv run maturin develop --release

# Build the Rust extension in release mode
build:
    uv run maturin develop --release

# Build the Rust extension in debug mode (faster compilation)
build-debug:
    uv run maturin develop

# Run the Python k-NN benchmark (compares PyNNDescent vs HNSW backends)
benchmark:
    uv sync --extra benchmark
    uv run python benchmark_optimizations.py

# Run Rust benchmarks with Criterion
benchmark-rust:
    cargo bench

# Run all tests
test:
    uv run pytest squeeze/tests/ -v

# Run tests with testmon (only changed tests)
test-fast:
    uv run pytest squeeze/tests/ --testmon

# Run a specific test file
test-file FILE:
    uv run pytest {{FILE}} -v

# Run linting with ruff
lint:
    uv run ruff check .

# Fix linting issues automatically
lint-fix:
    uv run ruff check --fix .

# Format code with ruff
format:
    uv run ruff format .

# Check formatting without making changes
format-check:
    uv run ruff format --check .

# Run both linting and formatting
check: lint format-check

# Fix all linting and formatting issues
fix: lint-fix format

# Clean build artifacts
clean:
    rm -rf target/
    rm -rf *.egg-info/
    rm -rf dist/
    rm -rf build/
    rm -rf .pytest_cache/
    rm -rf __pycache__/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Watch tests and re-run on changes
test-watch:
    uv run pytest-watch -- squeeze/tests/ -v

# Run pre-commit hooks on all files
pre-commit:
    uv run pre-commit run --all-files

# Install pre-commit hooks
pre-commit-install:
    uv run pre-commit install

# Build documentation
docs:
    uv run mkdocs build

# Serve documentation locally
docs-serve:
    uv run mkdocs serve

# Run cargo tests for Rust code
test-rust:
    cargo test

# Check Rust code
check-rust:
    cargo check
    cargo clippy

# Full CI check (lint, format, test)
ci: check test test-rust
