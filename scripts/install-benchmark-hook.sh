#!/bin/bash
# Install the benchmark pre-commit hook

set -e

HOOK_DIR=".git/hooks"
HOOK_FILE="$HOOK_DIR/pre-commit"

# Create hooks directory if it doesn't exist
mkdir -p "$HOOK_DIR"

# Check if hook already exists
if [ -f "$HOOK_FILE" ]; then
    echo "Pre-commit hook already exists. Backing up to pre-commit.backup"
    cp "$HOOK_FILE" "$HOOK_FILE.backup"
fi

# Create or update pre-commit hook
cat > "$HOOK_FILE" << 'EOF'
#!/bin/bash
# Pre-commit hook for UMAP
# Runs performance benchmarks and blocks commit if performance regresses

# Check if we should skip benchmarks
if [ "$SKIP_BENCHMARKS" = "1" ]; then
    echo "Skipping benchmarks (SKIP_BENCHMARKS=1)"
    exit 0
fi

# Check if any Rust files changed
RUST_FILES_CHANGED=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(rs|toml)$' || true)

if [ -z "$RUST_FILES_CHANGED" ]; then
    echo "No Rust files changed, skipping benchmarks"
    exit 0
fi

echo "Rust files changed, running performance benchmarks..."

# Run benchmark script
if ! ./scripts/benchmark-pre-commit.sh; then
    echo ""
    echo "To skip benchmarks (not recommended):"
    echo "  SKIP_BENCHMARKS=1 git commit"
    echo "  or: git commit --no-verify"
    exit 1
fi

exit 0
EOF

chmod +x "$HOOK_FILE"

echo "✓ Benchmark pre-commit hook installed successfully!"
echo ""
echo "The hook will:"
echo "  1. Run benchmarks when Rust files are modified"
echo "  2. Compare performance against baseline"
echo "  3. Block commit if performance regresses"
echo "  4. Update baseline if performance improves or stays same"
echo ""
echo "To skip benchmarks: SKIP_BENCHMARKS=1 git commit"
echo "To bypass hook entirely: git commit --no-verify"
echo ""
echo "Run benchmarks manually: cargo bench"
