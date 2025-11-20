# Performance Benchmarking Guide

**Date:** November 20, 2025  
**Status:** Production-Ready Benchmarking Infrastructure

---

## Overview

This project uses a comprehensive benchmarking system to track performance and prevent regressions. The system includes:

1. **Criterion.rs** - Statistically rigorous benchmarks
2. **Pre-commit hook** - Automatic regression detection
3. **CI/CD integration** - Continuous performance tracking
4. **Baseline comparison** - Track improvements over time

---

## Quick Start

### Install Benchmark Pre-Commit Hook

```bash
./scripts/install-benchmark-hook.sh
```

This installs a Git pre-commit hook that:
- ✅ Runs benchmarks automatically when Rust code changes
- ✅ Compares performance against baseline
- ✅ **Blocks commit if performance regresses**
- ✅ Updates baseline if performance improves

### Run Benchmarks Manually

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench -- distance_metrics

# Save as baseline
cargo bench -- --save-baseline mybranch

# Compare against baseline
cargo bench -- --baseline mybranch
```

---

## Benchmark Suite

### 1. Distance Metrics (`bench_distance_metrics`)

**What it measures:** Raw distance computation performance

**Benchmarks:**
- Euclidean distance
- Manhattan distance  
- Cosine distance

**Test dimensions:** 8, 16, 32, 64, 128, 256, 512, 1024

**Why it matters:** Distance computation is ~60% of runtime

**Example output:**
```
distance_metrics/euclidean/64    time: [12.5 ns 12.7 ns 12.9 ns]
distance_metrics/manhattan/64    time: [8.2 ns 8.3 ns 8.5 ns]
distance_metrics/cosine/64       time: [45.3 ns 45.8 ns 46.2 ns]
```

### 2. Batch Distances (`bench_batch_distances`)

**What it measures:** Throughput when computing many distances

**Test sizes:** 10, 50, 100, 500, 1000 vectors

**Why it matters:** Real k-NN search computes many distances

**Example output:**
```
batch_distances/euclidean_batch/100    time: [1.25 µs 1.27 µs 1.29 µs]
                                       thrpt: [77.5 Melem/s 78.7 Melem/s 80.0 Melem/s]
```

### 3. Neighbor Selection (`bench_neighbor_selection`)

**What it measures:** Heap operations for k-NN selection

**Methods compared:**
- Binary heap insertion
- Partial sort

**Test k values:** 5, 10, 15, 30, 50

**Why it matters:** Neighbor selection is ~20% of runtime

**Example output:**
```
neighbor_selection/heap_insert/10       time: [2.8 µs 2.9 µs 3.0 µs]
neighbor_selection/partial_sort/10      time: [4.2 µs 4.3 µs 4.4 µs]
```

### 4. HNSW Construction (`bench_hnsw_construction`)

**What it measures:** Index building performance

**Test sizes:** 100, 500, 1000 vectors

**Why it matters:** One-time cost, affects UX

**Example output:**
```
hnsw_construction/100     time: [45.2 ms 45.8 ms 46.3 ms]
hnsw_construction/500     time: [1.12 s 1.14 s 1.15 s]
```

### 5. HNSW Search (`bench_hnsw_search`)

**What it measures:** Query performance

**Test k values:** 5, 10, 15, 30

**Why it matters:** Most common operation

**Example output:**
```
hnsw_search/5     time: [12.5 µs 12.7 µs 12.9 µs]
hnsw_search/30    time: [65.2 µs 66.1 µs 67.0 µs]
```

---

## Pre-Commit Hook Details

### How It Works

```
┌─────────────────────┐
│ git commit          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Check: Rust files   │
│ changed?            │
└──────────┬──────────┘
           │ Yes
           ▼
┌─────────────────────┐
│ Run cargo bench     │
│ (1-2 minutes)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Compare with        │
│ baseline            │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
  Faster      Slower
  or Same         │
    │             ▼
    │     ┌──────────────┐
    │     │ Block commit │
    │     │ Show report  │
    │     └──────────────┘
    │
    ▼
┌─────────────────────┐
│ Run cargo test      │
│ (verify quality)    │
└──────────┬──────────┘
           │
         Pass
           │
           ▼
┌─────────────────────┐
│ Update baseline     │
│ Allow commit ✓      │
└─────────────────────┘
```

### Example: Regression Detected

```bash
$ git commit -m "Add new feature"

Running performance benchmarks...
Building optimized binary...
Running benchmarks (this may take 1-2 minutes)...
Comparing performance with baseline...

✗ PERFORMANCE REGRESSION DETECTED!

Benchmark comparison:
  distance_metrics/euclidean/64
    Current:  15.2 ns (±0.3 ns)
    Baseline: 12.7 ns (±0.2 ns)
    Change:   +19.7% slower

Commit blocked. Please optimize your changes before committing.

Tips:
  1. Run 'cargo bench' to see detailed performance comparison
  2. Profile your code with 'cargo flamegraph' to find bottlenecks
  3. Use 'cargo bench -- --baseline current' to compare iterations

To bypass this check (not recommended):
  git commit --no-verify
```

### Example: Improvement Detected

```bash
$ git commit -m "Optimize distance calculation"

Running performance benchmarks...
Building optimized binary...
Running benchmarks (this may take 1-2 minutes)...
Comparing performance with baseline...

✓ PERFORMANCE IMPROVED!

  distance_metrics/euclidean/64
    Current:  8.9 ns (±0.1 ns)
    Baseline: 12.7 ns (±0.2 ns)
    Change:   -29.9% faster! 🎉

Verifying tests still pass...
✓ All performance checks passed!
Updating baseline...
```

### Skipping Benchmarks

**Temporarily skip:**
```bash
SKIP_BENCHMARKS=1 git commit -m "docs only"
```

**Bypass hook (not recommended):**
```bash
git commit --no-verify -m "skip checks"
```

**When to skip:**
- Documentation-only changes (though hook auto-skips)
- Emergency hotfixes
- Working on feature branch (will be caught in PR)

---

## CI/CD Integration

### GitHub Actions Workflow

Located at `.github/workflows/benchmark.yml`

**Triggers:**
- Every push to `master`
- Every pull request

**What it does:**
1. Runs full benchmark suite
2. Compares PR against master baseline
3. Posts performance report as comment
4. Uploads benchmark artifacts

**Example PR Comment:**
```markdown
## Performance Benchmark Results

### Summary
✓ No significant regressions detected

### Changes
| Benchmark | Master | PR | Change |
|-----------|--------|-----|--------|
| euclidean/64 | 12.7 ns | 12.5 ns | -1.6% ⚡ |
| manhattan/64 | 8.3 ns | 8.4 ns | +1.2% |
| cosine/64 | 45.8 ns | 44.2 ns | -3.5% ⚡ |

### Detailed Report
[View full Criterion report →](link to artifacts)
```

---

## Analyzing Benchmark Results

### Reading Criterion Output

```
distance_metrics/euclidean/64
                        time:   [12.491 ns 12.683 ns 12.915 ns]
                        change: [-3.2584% -0.6554% +1.9814%] (p = 0.60 > 0.05)
                        No change in performance detected.
Found 8 outliers among 100 measurements (8.00%)
  2 (2.00%) high mild
  6 (6.00%) high severe
```

**Interpretation:**
- **time:** [lower bound, estimate, upper bound] (95% confidence)
- **change:** Performance change vs. previous run
- **p-value:** Statistical significance (p < 0.05 = significant)
- **outliers:** Measurements to ignore (system noise)

**When to be concerned:**
- p < 0.05 AND change > +10% = significant regression
- p < 0.05 AND change < -10% = significant improvement

### Visualizing Results

Criterion generates HTML reports:

```bash
cargo bench
open target/criterion/report/index.html
```

**Reports include:**
- Performance over time graphs
- Distribution violin plots
- Comparison charts
- Detailed statistics

---

## Performance Tracking

### Baseline Management

**Save current performance:**
```bash
cargo bench -- --save-baseline phase1-complete
```

**Compare against milestone:**
```bash
cargo bench -- --baseline phase1-complete
```

**List baselines:**
```bash
ls -la target/criterion/*/base/
```

**Naming convention:**
- `phase1-complete` - Major milestone
- `before-simd` - Before optimization
- `after-simd` - After optimization
- `branch-name` - Feature branch baseline

### Tracking Over Time

Create a performance log:

```bash
# After significant changes
echo "$(date): SIMD euclidean - 12.7ns → 6.3ns (-50%)" >> PERFORMANCE_LOG.md
```

**Example log:**
```markdown
# Performance Log

## 2025-11-20: Phase 1 Complete
- Baseline established
- Euclidean (64-dim): 12.7 ns
- Manhattan (64-dim): 8.3 ns
- Cosine (64-dim): 45.8 ns

## 2025-11-25: SIMD Optimization
- Euclidean (64-dim): 12.7 ns → 6.3 ns (-50.4%)
- Manhattan (64-dim): 8.3 ns → 4.1 ns (-50.6%)
- Cosine (64-dim): 45.8 ns → 23.2 ns (-49.3%)
```

---

## Performance Goals

### Current Baseline (Phase 1)

| Operation | Performance | Target (Phase 2) |
|-----------|-------------|------------------|
| Euclidean (64-dim) | 12.7 ns | 6.0 ns (-53%) |
| Manhattan (64-dim) | 8.3 ns | 4.0 ns (-52%) |
| Cosine (64-dim) | 45.8 ns | 23.0 ns (-50%) |
| k-NN search (k=10) | 66 µs | 33 µs (-50%) |
| Construction (1000) | 1.14 s | 0.8 s (-30%) |

### Phase 2 Targets (SIMD + RobustPrune)

**Distance metrics:** 2-4x speedup via SIMD  
**Graph quality:** 5-10% improvement via RobustPrune  
**Overall speedup:** 3-5x vs PyNNDescent (currently 1.5-1.7x)

---

## Troubleshooting

### Benchmarks Take Too Long

**Reduce sample size:**
```rust
group.sample_size(10); // Default is 100
```

**Reduce measurement time:**
```bash
cargo bench -- --quick
```

### Unstable Results (High Variance)

**Common causes:**
- System load (close other applications)
- Thermal throttling (let CPU cool)
- Power management (plug in laptop)
- Turbo boost (disable for consistency)

**Solutions:**
```bash
# Run with more samples
cargo bench -- --sample-size 500

# Reduce system noise
sudo nice -n -20 cargo bench

# Disable turbo (Linux)
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

### Pre-Commit Hook Too Slow

**Option 1:** Skip for small changes
```bash
SKIP_BENCHMARKS=1 git commit
```

**Option 2:** Reduce benchmark suite
Edit `scripts/benchmark-pre-commit.sh`:
```bash
# Only run fast benchmarks
cargo bench --bench hnsw_benchmarks -- distance_metrics --quick
```

**Option 3:** Disable hook
```bash
rm .git/hooks/pre-commit
```

---

## Best Practices

### For Developers

✅ **DO:**
- Run benchmarks before and after optimization
- Save baseline before making changes
- Compare against baseline after changes
- Update PERFORMANCE_LOG.md for significant improvements
- Profile before optimizing (use flamegraphs)

❌ **DON'T:**
- Skip benchmarks without good reason
- Commit performance regressions
- Optimize without measuring
- Trust intuition over benchmarks

### For Code Reviewers

✅ **CHECK:**
- Benchmark results in PR
- Performance change is acceptable (<10% regression)
- Baseline is updated if performance improves
- Tests still pass

❌ **REJECT if:**
- Performance regresses >10% without justification
- Benchmarks were skipped (`--no-verify`)
- Tests fail
- No baseline comparison provided

---

## Advanced Usage

### Profiling with Flamegraphs

```bash
# Install cargo-flamegraph
cargo install flamegraph

# Generate flamegraph
cargo flamegraph --bench hnsw_benchmarks

# Open flamegraph.svg in browser
open flamegraph.svg
```

### Memory Profiling

```bash
# Install valgrind and cargo-valgrind
cargo install cargo-valgrind

# Run memory profiling
cargo valgrind --bench hnsw_benchmarks
```

### Custom Benchmarks

Add to `benches/hnsw_benchmarks.rs`:

```rust
fn bench_my_feature(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_feature");
    
    group.bench_function("my_benchmark", |b| {
        b.iter(|| {
            // Your code here
            black_box(my_function());
        });
    });
    
    group.finish();
}

criterion_group!(benches, bench_my_feature);
```

---

## FAQ

**Q: Why are benchmarks required for commits?**  
A: To prevent performance regressions from sneaking into the codebase. It's much easier to fix regressions before they're committed.

**Q: Can I skip benchmarks for documentation changes?**  
A: The hook automatically skips if no Rust files changed. For docs-only commits in Rust files, use `SKIP_BENCHMARKS=1`.

**Q: What if my change legitimately makes things slower?**  
A: If trading performance for correctness/features, document the tradeoff and use `--no-verify`. Discuss in PR.

**Q: How accurate are the benchmarks?**  
A: Criterion uses statistical analysis (100 samples by default) to account for noise. Results are typically ±2% accurate.

**Q: Can I benchmark Python code too?**  
A: This system is Rust-only. For Python, use `pytest-benchmark` or `timeit`.

---

## Resources

- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Flamegraph Guide](https://www.brendangregg.com/flamegraphs.html)

---

**Created:** November 20, 2025  
**Author:** UMAP Development Team  
**Maintainer:** OpenCode AI Assistant
