# Numerical Stability

## Overview

Squeeze includes a numerical utilities module (`numerical.rs`) that provides standardized constants and helper functions for maintaining numerical stability across all dimensionality reduction algorithms. This ensures consistent behavior and prevents common numerical issues like division by zero, log of zero, and overflow.

## Constants

The following constants are used throughout the codebase:

| Constant | Value | Description |
|----------|-------|-------------|
| `MIN_PROBABILITY` | 1e-12 | Minimum value for probability computations |
| `MIN_DISTANCE` | 1e-10 | Minimum distance to prevent division by zero |
| `MIN_VARIANCE` | 1e-10 | Minimum variance for numerical stability |
| `DEFAULT_CONVERGENCE_TOL` | 1e-7 | Default tolerance for convergence checks |
| `SAFE_EPSILON` | 1e-8 | Safe epsilon for general numerical operations |

## Usage

These constants ensure consistency across algorithms:

### Probability Clamping

All probability distributions (e.g., in t-SNE, PHATE) clamp values to prevent log(0):

```rust
// Instead of:
let p = compute_probability();

// We use:
let p = compute_probability().max(MIN_PROBABILITY);
```

### Safe Division

When dividing by potentially small values:

```rust
// Instead of:
let result = a / b;

// We use:
let result = safe_div(a, b);  // Returns 0 if b is too small
```

### Safe Logarithm

For logarithms of values that might be zero:

```rust
// Instead of:
let log_p = p.ln();

// We use:
let log_p = safe_log(p);  // Clamps p to MIN_PROBABILITY first
```

## Algorithms Using Numerical Utilities

### t-SNE

- **P and Q distributions**: Clamped to `MIN_PROBABILITY` (1e-12)
- **Gradient computation**: Safe division in Barnes-Hut Z normalization
- **Early stopping**: Uses `DEFAULT_CONVERGENCE_TOL` for gradient norm threshold

### PHATE

- **Diffusion probabilities**: Row-normalized with minimum probability
- **Potential distances**: Log transform with safe logarithm
- **Affinity matrices**: Gaussian kernel with clamped distances

### PaCMAP

- **Distance computations**: Minimum distance for numerical stability
- **Weight schedules**: Bounded to prevent overflow

### LLE

- **Regularization**: Added to covariance matrix diagonal
- **Weight normalization**: Safe division by weight sum

### Isomap

- **Geodesic distances**: Infinity for disconnected components
- **MDS centering**: Numerical stability in double-centering

## Best Practices

### For Users

1. **Data preprocessing**: Standardize your data to have similar scales across features
2. **Check for duplicates**: Near-duplicate points can cause numerical issues
3. **Monitor warnings**: Pay attention to fallback warnings (e.g., LLE uniform weights)

### For Developers

1. **Use provided constants**: Don't hardcode numerical thresholds
2. **Prefer helper functions**: Use `safe_div`, `safe_log`, `clamp_probability`
3. **Validate inputs**: Use `validate_input_data` to check for NaN/Inf
4. **Document edge cases**: Note when fallback behavior might occur

## Validation Functions

### `validate_input_data`

Checks input arrays for common issues:

```rust
pub fn validate_input_data(data: &Array2<f64>) -> Result<(), String> {
    // Checks for:
    // - NaN values
    // - Infinite values
    // - Empty arrays
}
```

### `is_safe`

Quick check if a value is usable:

```rust
pub fn is_safe(x: f64) -> bool {
    x.is_finite() && !x.is_nan()
}
```

## Troubleshooting

### "NaN in embedding"

**Cause**: Numerical overflow or division by zero
**Solution**:
1. Standardize your input data
2. Reduce learning rate (for iterative methods)
3. Check for outliers in your data

### "Singular matrix" errors

**Cause**: Colinearity or near-duplicate points
**Solution**:
1. Increase regularization parameter
2. Remove duplicate points
3. Add small noise to data

### "Disconnected graph" errors

**Cause**: Sparse data with insufficient neighbors
**Solution**:
1. Increase n_neighbors parameter
2. Check for outliers
3. Verify data doesn't have disconnected clusters

## Implementation Notes

The numerical module is implemented in Rust (`src/numerical.rs`) and used internally by all algorithms. While not directly exposed to Python users, its effects are visible in:

- Consistent probability bounds across algorithms
- Predictable handling of edge cases
- Clear error messages when issues occur
- Warnings when fallback behavior is triggered
