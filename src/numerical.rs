//! Numerical stability constants and utilities for DR algorithms.
//!
//! This module provides standardized constants and helper functions to ensure
//! consistent numerical stability across all dimensionality reduction algorithms.

/// Minimum probability to avoid log(0) in entropy calculations
pub const MIN_PROBABILITY: f64 = 1e-12;

/// Minimum distance to avoid division by zero in distance-based calculations
pub const MIN_DISTANCE: f64 = 1e-10;

/// Minimum variance/eigenvalue to consider non-zero
pub const MIN_VARIANCE: f64 = 1e-10;

/// Default convergence tolerance for iterative algorithms
pub const DEFAULT_CONVERGENCE_TOL: f64 = 1e-6;

/// Machine epsilon scaled for f64 operations
pub const SAFE_EPSILON: f64 = 1e-15;

/// Safe division that avoids division by zero
#[inline]
pub fn safe_div(numerator: f64, denominator: f64) -> f64 {
    numerator / denominator.max(MIN_DISTANCE)
}

/// Safe logarithm that avoids -infinity
#[inline]
pub fn safe_log(x: f64) -> f64 {
    if x > MIN_PROBABILITY {
        x.ln()
    } else {
        MIN_PROBABILITY.ln()
    }
}

/// Clamp a probability to valid range [MIN_PROBABILITY, 1.0]
#[inline]
pub fn clamp_probability(p: f64) -> f64 {
    p.max(MIN_PROBABILITY).min(1.0)
}

/// Clamp a distance to avoid zero
#[inline]
pub fn clamp_distance(d: f64) -> f64 {
    d.max(MIN_DISTANCE)
}

/// Check if a value is numerically safe (not NaN or Inf)
#[inline]
pub fn is_safe(x: f64) -> bool {
    x.is_finite()
}

/// Validate that an array contains no NaN or Inf values
pub fn validate_array(arr: &ndarray::Array2<f64>) -> Result<(), &'static str> {
    for &val in arr.iter() {
        if !val.is_finite() {
            return Err("Array contains NaN or Inf values");
        }
    }
    Ok(())
}

/// Validate input data for DR algorithms
/// Returns Ok if valid, Err with description if invalid
pub fn validate_input_data(
    n_samples: usize,
    n_features: usize,
    n_components: usize,
) -> Result<(), String> {
    if n_samples < 2 {
        return Err(format!(
            "Need at least 2 samples, got {}",
            n_samples
        ));
    }

    if n_features < 1 {
        return Err("Need at least 1 feature".to_string());
    }

    if n_components < 1 {
        return Err("Need at least 1 component".to_string());
    }

    if n_components > n_samples {
        return Err(format!(
            "n_components ({}) cannot exceed n_samples ({})",
            n_components, n_samples
        ));
    }

    if n_components > n_features {
        return Err(format!(
            "n_components ({}) cannot exceed n_features ({})",
            n_components, n_features
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_safe_div_normal() {
        assert_relative_eq!(safe_div(10.0, 2.0), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_safe_div_zero_denominator() {
        // Should not panic, should return large but finite value
        let result = safe_div(1.0, 0.0);
        assert!(result.is_finite());
        assert!(result > 0.0);
    }

    #[test]
    fn test_safe_div_tiny_denominator() {
        let result = safe_div(1.0, 1e-20);
        assert!(result.is_finite());
        // Should clamp to MIN_DISTANCE
        assert_relative_eq!(result, 1.0 / MIN_DISTANCE, epsilon = 1e-5);
    }

    #[test]
    fn test_safe_log_normal() {
        assert_relative_eq!(safe_log(1.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(safe_log(std::f64::consts::E), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_safe_log_zero() {
        // Should not panic, should return finite negative value
        let result = safe_log(0.0);
        assert!(result.is_finite());
        assert!(result < 0.0);
    }

    #[test]
    fn test_safe_log_tiny() {
        let result = safe_log(1e-100);
        assert!(result.is_finite());
        // Should clamp to MIN_PROBABILITY
        assert_relative_eq!(result, MIN_PROBABILITY.ln(), epsilon = 1e-5);
    }

    #[test]
    fn test_clamp_probability() {
        assert_relative_eq!(clamp_probability(0.5), 0.5, epsilon = 1e-10);
        assert_relative_eq!(clamp_probability(0.0), MIN_PROBABILITY, epsilon = 1e-15);
        assert_relative_eq!(clamp_probability(-1.0), MIN_PROBABILITY, epsilon = 1e-15);
        assert_relative_eq!(clamp_probability(2.0), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_clamp_distance() {
        assert_relative_eq!(clamp_distance(5.0), 5.0, epsilon = 1e-10);
        assert_relative_eq!(clamp_distance(0.0), MIN_DISTANCE, epsilon = 1e-15);
        assert_relative_eq!(clamp_distance(-1.0), MIN_DISTANCE, epsilon = 1e-15);
    }

    #[test]
    fn test_is_safe() {
        assert!(is_safe(0.0));
        assert!(is_safe(1.0));
        assert!(is_safe(-1e308));
        assert!(!is_safe(f64::NAN));
        assert!(!is_safe(f64::INFINITY));
        assert!(!is_safe(f64::NEG_INFINITY));
    }

    #[test]
    fn test_validate_input_data_valid() {
        assert!(validate_input_data(100, 10, 2).is_ok());
        assert!(validate_input_data(10, 5, 5).is_ok());
    }

    #[test]
    fn test_validate_input_data_too_few_samples() {
        assert!(validate_input_data(1, 10, 2).is_err());
    }

    #[test]
    fn test_validate_input_data_components_exceed_samples() {
        assert!(validate_input_data(5, 10, 10).is_err());
    }

    #[test]
    fn test_validate_input_data_components_exceed_features() {
        assert!(validate_input_data(100, 3, 5).is_err());
    }
}
