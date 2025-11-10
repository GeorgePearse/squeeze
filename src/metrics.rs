use std::fmt;

/// Result type returned by metric helpers.
pub type MetricResult<T> = Result<T, MetricError>;

/// Errors that can arise while evaluating a metric.
#[derive(Debug, PartialEq, Eq)]
pub enum MetricError {
    DimensionMismatch { left: usize, right: usize },
}

impl fmt::Display for MetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricError::DimensionMismatch { left, right } => {
                write!(f, "vector length mismatch: left={} right={}", left, right)
            }
        }
    }
}

impl std::error::Error for MetricError {}

#[inline]
fn ensure_same_length(a: &[f32], b: &[f32]) -> MetricResult<()> {
    if a.len() != b.len() {
        Err(MetricError::DimensionMismatch {
            left: a.len(),
            right: b.len(),
        })
    } else {
        Ok(())
    }
}

/// Euclidean distance (L2 norm)
///
/// sqrt(sum((x_i - y_i)^2))
pub fn euclidean(a: &[f32], b: &[f32]) -> MetricResult<f32> {
    ensure_same_length(a, b)?;
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt())
}

/// Manhattan distance (L1 norm)
///
/// sum(|x_i - y_i|)
pub fn manhattan(a: &[f32], b: &[f32]) -> MetricResult<f32> {
    ensure_same_length(a, b)?;
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum())
}

/// Cosine distance (1 - cosine similarity)
///
/// 1 - (dot(x, y) / (||x|| * ||y||))
///
/// Returns distance in [0, 2], where 0 means identical direction
pub fn cosine(a: &[f32], b: &[f32]) -> MetricResult<f32> {
    ensure_same_length(a, b)?;

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    let denom = norm_a * norm_b;
    if denom < 1e-10 {
        Ok(1.0) // Undefined for zero vectors, return max distance
    } else {
        Ok(1.0 - (dot / denom))
    }
}

/// Chebyshev distance (L∞ norm or max norm)
///
/// max(|x_i - y_i|)
pub fn chebyshev(a: &[f32], b: &[f32]) -> MetricResult<f32> {
    ensure_same_length(a, b)?;
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max))
}

/// Minkowski distance with parameter p
///
/// (sum(|x_i - y_i|^p))^(1/p)
pub fn minkowski(a: &[f32], b: &[f32], p: f32) -> MetricResult<f32> {
    ensure_same_length(a, b)?;

    if p == 1.0 {
        return manhattan(a, b);
    } else if p == 2.0 {
        return euclidean(a, b);
    } else if p == f32::INFINITY {
        return chebyshev(a, b);
    }

    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs().powf(p))
        .sum::<f32>()
        .powf(1.0 / p))
}

/// Hamming distance (number of differing bits)
///
/// Counts the number of positions where the float bits differ.
/// Note: This is approximate for floating-point values.
pub fn hamming(a: &[f32], b: &[f32]) -> MetricResult<f32> {
    ensure_same_length(a, b)?;
    Ok(a.iter()
        .zip(b.iter())
        .filter(|(x, y)| x != y)
        .count() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        assert_relative_eq!(euclidean(&a, &b).unwrap(), 5.0);
    }

    #[test]
    fn test_euclidean_zero_distance() {
        let a = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(euclidean(&a, &a).unwrap(), 0.0);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        assert_relative_eq!(manhattan(&a, &b).unwrap(), 2.0);
    }

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(cosine(&a, &a).unwrap(), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert_relative_eq!(cosine(&a, &b).unwrap(), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert_relative_eq!(cosine(&a, &b).unwrap(), 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_chebyshev_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert_relative_eq!(chebyshev(&a, &b).unwrap(), 4.0);
    }

    #[test]
    fn test_minkowski_p1() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        assert_relative_eq!(minkowski(&a, &b, 1.0).unwrap(), 2.0);
    }

    #[test]
    fn test_minkowski_p2() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert_relative_eq!(minkowski(&a, &b, 2.0).unwrap(), 5.0);
    }

    #[test]
    fn test_hamming_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(hamming(&a, &a).unwrap(), 0.0);
    }

    #[test]
    fn test_hamming_different() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 5.0, 3.0];
        assert_relative_eq!(hamming(&a, &b).unwrap(), 1.0);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        let err = euclidean(&a, &b).unwrap_err();
        assert_eq!(err, MetricError::DimensionMismatch { left: 2, right: 1 });
    }
}
