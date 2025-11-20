/// SIMD-optimized distance metrics
///
/// This module provides vectorized implementations of distance metrics using:
/// - AVX2 for x86/x64 CPUs (8 floats per instruction)
/// - NEON for ARM CPUs (4 floats per instruction)
/// - Scalar fallback for all other platforms
///
/// Runtime CPU feature detection automatically selects the best implementation.

use crate::metrics::{MetricError, MetricResult};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Checks if AVX2 is available on x86_64 at runtime
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

/// Checks if NEON is available on ARM (always true on aarch64)
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn has_neon() -> bool {
    true // NEON is always available on aarch64
}

/// Returns true if SIMD is available on this platform
#[inline]
pub fn has_simd() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        has_avx2()
    }
    #[cfg(target_arch = "aarch64")]
    {
        has_neon()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

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

// =============================================================================
// EUCLIDEAN DISTANCE
// =============================================================================

/// Euclidean distance with automatic SIMD detection
#[inline]
pub fn euclidean(a: &[f32], b: &[f32]) -> MetricResult<f32> {
    ensure_same_length(a, b)?;

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return Ok(unsafe { euclidean_avx2(a, b) });
        }
        return Ok(euclidean_scalar(a, b));
    }

    #[cfg(target_arch = "aarch64")]
    {
        return Ok(unsafe { euclidean_neon(a, b) });
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Ok(euclidean_scalar(a, b))
    }
}

/// AVX2 implementation (8 floats at once)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn euclidean_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let len = a.len();
    let chunks = len / 8;

    // Process 8 floats at a time
    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum); // diff * diff + sum
    }

    // Horizontal sum of the 8 lanes
    let mut result = horizontal_sum_avx2(sum);

    // Handle remainder elements
    let remainder = len % 8;
    if remainder > 0 {
        for i in (len - remainder)..len {
            let diff = a[i] - b[i];
            result += diff * diff;
        }
    }

    result.sqrt()
}

/// NEON implementation (4 floats at once)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn euclidean_neon(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = vdupq_n_f32(0.0);
    let len = a.len();
    let chunks = len / 4;

    // Process 4 floats at a time
    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        let diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff); // diff * diff + sum
    }

    // Horizontal sum of the 4 lanes
    let mut result = vaddvq_f32(sum);

    // Handle remainder elements
    let remainder = len % 4;
    if remainder > 0 {
        for i in (len - remainder)..len {
            let diff = a[i] - b[i];
            result += diff * diff;
        }
    }

    result.sqrt()
}

/// Scalar fallback
#[inline]
fn euclidean_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// =============================================================================
// MANHATTAN DISTANCE
// =============================================================================

/// Manhattan distance with automatic SIMD detection
#[inline]
pub fn manhattan(a: &[f32], b: &[f32]) -> MetricResult<f32> {
    ensure_same_length(a, b)?;

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return Ok(unsafe { manhattan_avx2(a, b) });
        }
        return Ok(manhattan_scalar(a, b));
    }

    #[cfg(target_arch = "aarch64")]
    {
        return Ok(unsafe { manhattan_neon(a, b) });
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Ok(manhattan_scalar(a, b))
    }
}

/// AVX2 implementation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn manhattan_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let len = a.len();
    let chunks = len / 8;

    // Mask for absolute value (clear sign bit)
    let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));

    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        let diff = _mm256_sub_ps(va, vb);
        let abs_diff = _mm256_and_ps(diff, sign_mask); // abs(diff)
        sum = _mm256_add_ps(sum, abs_diff);
    }

    let mut result = horizontal_sum_avx2(sum);

    // Handle remainder
    let remainder = len % 8;
    if remainder > 0 {
        for i in (len - remainder)..len {
            result += (a[i] - b[i]).abs();
        }
    }

    result
}

/// NEON implementation
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn manhattan_neon(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = vdupq_n_f32(0.0);
    let len = a.len();
    let chunks = len / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        let diff = vsubq_f32(va, vb);
        let abs_diff = vabsq_f32(diff);
        sum = vaddq_f32(sum, abs_diff);
    }

    let mut result = vaddvq_f32(sum);

    // Handle remainder
    let remainder = len % 4;
    if remainder > 0 {
        for i in (len - remainder)..len {
            result += (a[i] - b[i]).abs();
        }
    }

    result
}

/// Scalar fallback
#[inline]
fn manhattan_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum()
}

// =============================================================================
// COSINE DISTANCE
// =============================================================================

/// Cosine distance with automatic SIMD detection
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> MetricResult<f32> {
    ensure_same_length(a, b)?;

    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return Ok(unsafe { cosine_avx2(a, b) });
        }
        return Ok(cosine_scalar(a, b));
    }

    #[cfg(target_arch = "aarch64")]
    {
        return Ok(unsafe { cosine_neon(a, b) });
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Ok(cosine_scalar(a, b))
    }
}

/// AVX2 implementation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cosine_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = _mm256_setzero_ps();
    let mut norm_a = _mm256_setzero_ps();
    let mut norm_b = _mm256_setzero_ps();
    
    let len = a.len();
    let chunks = len / 8;

    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        
        dot = _mm256_fmadd_ps(va, vb, dot);
        norm_a = _mm256_fmadd_ps(va, va, norm_a);
        norm_b = _mm256_fmadd_ps(vb, vb, norm_b);
    }

    let mut dot_result = horizontal_sum_avx2(dot);
    let mut norm_a_result = horizontal_sum_avx2(norm_a);
    let mut norm_b_result = horizontal_sum_avx2(norm_b);

    // Handle remainder
    let remainder = len % 8;
    if remainder > 0 {
        for i in (len - remainder)..len {
            dot_result += a[i] * b[i];
            norm_a_result += a[i] * a[i];
            norm_b_result += b[i] * b[i];
        }
    }

    let denom = norm_a_result.sqrt() * norm_b_result.sqrt();
    if denom < 1e-10 {
        1.0
    } else {
        1.0 - (dot_result / denom)
    }
}

/// NEON implementation
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn cosine_neon(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = vdupq_n_f32(0.0);
    let mut norm_a = vdupq_n_f32(0.0);
    let mut norm_b = vdupq_n_f32(0.0);
    
    let len = a.len();
    let chunks = len / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let va = vld1q_f32(a.as_ptr().add(idx));
        let vb = vld1q_f32(b.as_ptr().add(idx));
        
        dot = vfmaq_f32(dot, va, vb);
        norm_a = vfmaq_f32(norm_a, va, va);
        norm_b = vfmaq_f32(norm_b, vb, vb);
    }

    let mut dot_result = vaddvq_f32(dot);
    let mut norm_a_result = vaddvq_f32(norm_a);
    let mut norm_b_result = vaddvq_f32(norm_b);

    // Handle remainder
    let remainder = len % 4;
    if remainder > 0 {
        for i in (len - remainder)..len {
            dot_result += a[i] * b[i];
            norm_a_result += a[i] * a[i];
            norm_b_result += b[i] * b[i];
        }
    }

    let denom = norm_a_result.sqrt() * norm_b_result.sqrt();
    if denom < 1e-10 {
        1.0
    } else {
        1.0 - (dot_result / denom)
    }
}

/// Scalar fallback
#[inline]
fn cosine_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    let denom = norm_a * norm_b;
    if denom < 1e-10 {
        1.0
    } else {
        1.0 - (dot / denom)
    }
}

// =============================================================================
// SIMD HELPER FUNCTIONS
// =============================================================================

/// Horizontal sum of 8 floats in AVX2 register
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    // Extract high and low 128-bit halves
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    
    // Add the two halves
    let sum128 = _mm_add_ps(hi, lo);
    
    // Horizontal add within 128-bit register
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    let sums = _mm_add_ss(sums, shuf);
    
    _mm_cvtss_f32(sums)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_detection() {
        // Just verify detection doesn't panic
        let _ = has_simd();
        
        #[cfg(target_arch = "x86_64")]
        {
            let _ = has_avx2();
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            assert!(has_neon());
        }
    }

    #[test]
    fn test_euclidean_simd() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        assert_relative_eq!(euclidean(&a, &b).unwrap(), 5.0, epsilon = 1e-5);
    }

    #[test]
    fn test_euclidean_simd_zero() {
        let a = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(euclidean(&a, &a).unwrap(), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_euclidean_simd_large() {
        // Test with 64 dimensions to exercise SIMD paths
        let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..64).map(|i| (i + 1) as f32).collect();
        
        // Compute expected value
        let expected: f32 = 64.0_f32.sqrt(); // sqrt(64 * 1^2)
        
        assert_relative_eq!(euclidean(&a, &b).unwrap(), expected, epsilon = 1e-5);
    }

    #[test]
    fn test_manhattan_simd() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        assert_relative_eq!(manhattan(&a, &b).unwrap(), 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_manhattan_simd_large() {
        let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..64).map(|i| (i + 1) as f32).collect();
        
        let expected: f32 = 64.0; // 64 * 1
        
        assert_relative_eq!(manhattan(&a, &b).unwrap(), expected, epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_simd_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert_relative_eq!(cosine(&a, &a).unwrap(), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_simd_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert_relative_eq!(cosine(&a, &b).unwrap(), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_simd_large() {
        // Test with 64 dimensions
        let a: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..64).map(|i| (i as f32).cos()).collect();
        
        // Compute scalar version for comparison
        let result_simd = cosine(&a, &b).unwrap();
        let result_scalar = cosine_scalar(&a, &b);
        
        assert_relative_eq!(result_simd, result_scalar, epsilon = 1e-5);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        
        assert!(euclidean(&a, &b).is_err());
        assert!(manhattan(&a, &b).is_err());
        assert!(cosine(&a, &b).is_err());
    }

    // Consistency tests: SIMD should match scalar
    #[test]
    fn test_simd_scalar_consistency_euclidean() {
        for len in [7, 15, 31, 63, 127] {
            let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();
            let b: Vec<f32> = (0..len).map(|i| (len - i) as f32 * 0.1).collect();
            
            let simd = euclidean(&a, &b).unwrap();
            let scalar = euclidean_scalar(&a, &b);
            
            assert_relative_eq!(simd, scalar, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_simd_scalar_consistency_manhattan() {
        for len in [7, 15, 31, 63, 127] {
            let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();
            let b: Vec<f32> = (0..len).map(|i| (len - i) as f32 * 0.1).collect();
            
            let simd = manhattan(&a, &b).unwrap();
            let scalar = manhattan_scalar(&a, &b);
            
            assert_relative_eq!(simd, scalar, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_simd_scalar_consistency_cosine() {
        for len in [7, 15, 31, 63, 127] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin()).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).cos()).collect();
            
            let simd = cosine(&a, &b).unwrap();
            let scalar = cosine_scalar(&a, &b);
            
            assert_relative_eq!(simd, scalar, epsilon = 1e-4);
        }
    }
}
