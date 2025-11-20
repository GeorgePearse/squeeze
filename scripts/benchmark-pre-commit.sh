#!/bin/bash
# Pre-commit benchmark hook
# Ensures that code changes don't regress performance

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running performance benchmarks...${NC}"

# Create benchmark directory if it doesn't exist
mkdir -p .benchmark-baselines

# Baseline file
BASELINE_FILE=".benchmark-baselines/baseline.json"

# Run benchmarks and capture results
echo "Building optimized binary..."
cargo build --release --quiet

echo "Running benchmarks (this may take 1-2 minutes)..."
cargo bench --bench hnsw_benchmarks -- --save-baseline current --quiet > /tmp/bench_output.txt 2>&1

# Check if baseline exists
if [ ! -f "$BASELINE_FILE" ]; then
    echo -e "${YELLOW}No baseline found. Creating initial baseline...${NC}"
    cp target/criterion/*/base/estimates.json "$BASELINE_FILE" 2>/dev/null || true
    echo -e "${GREEN}✓ Baseline created. Future commits will be compared against this.${NC}"
    exit 0
fi

# Compare with baseline
echo "Comparing performance with baseline..."

# Extract key metrics from benchmark output
# This is a simple version - Criterion provides more detailed comparison
if grep -q "Performance has regressed" /tmp/bench_output.txt; then
    echo -e "${RED}✗ PERFORMANCE REGRESSION DETECTED!${NC}"
    echo ""
    echo "Benchmark comparison:"
    grep -A 5 "Performance has regressed" /tmp/bench_output.txt || true
    echo ""
    echo -e "${RED}Commit blocked. Please optimize your changes before committing.${NC}"
    echo ""
    echo "Tips:"
    echo "  1. Run 'cargo bench' to see detailed performance comparison"
    echo "  2. Profile your code with 'cargo flamegraph' to find bottlenecks"
    echo "  3. Use 'cargo bench -- --baseline current' to compare iterations"
    echo ""
    echo "To bypass this check (not recommended):"
    echo "  git commit --no-verify"
    exit 1
fi

# Check for significant improvements
if grep -q "Performance has improved" /tmp/bench_output.txt; then
    echo -e "${GREEN}✓ PERFORMANCE IMPROVED!${NC}"
    grep -A 3 "Performance has improved" /tmp/bench_output.txt || true
fi

# Check if tests still pass
echo "Verifying tests still pass..."
if ! cargo test --release --quiet 2>&1 | grep -q "test result: ok"; then
    echo -e "${RED}✗ TESTS FAILED!${NC}"
    echo "Benchmarks passed but tests failed. Please fix your tests."
    exit 1
fi

echo -e "${GREEN}✓ All performance checks passed!${NC}"

# Update baseline if performance is same or better
echo "Updating baseline..."
cp target/criterion/*/base/estimates.json "$BASELINE_FILE" 2>/dev/null || true

exit 0
