//! Squared Euclidean distances for neighbor selection (sqrt only selected edges).
//!
//! AVX2 multiply/add needs no FMA feature. Keep the established metric entry points
//! unchanged so experimental graph builders can opt into this kernel separately.

use crate::metrics::MetricResult;

#[inline]
pub fn squared_euclidean(a: &[f32], b: &[f32]) -> MetricResult<f32> {
    super::ensure_same_length(a, b)?;
    #[cfg(target_arch = "x86_64")]
    if super::has_avx2() {
        // SAFETY: lengths checked above; CPU feature detected at runtime.
        return Ok(unsafe { avx2(a, b) });
    }
    Ok(a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();
    let end = a.len() / 8 * 8;
    for i in (0..end).step_by(8) {
        let difference = _mm256_sub_ps(
            _mm256_loadu_ps(a.as_ptr().add(i)),
            _mm256_loadu_ps(b.as_ptr().add(i)),
        );
        sum = _mm256_add_ps(sum, _mm256_mul_ps(difference, difference));
    }
    let mut lanes = [0.0_f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    lanes.iter().sum::<f32>()
        + a[end..]
            .iter()
            .zip(&b[end..])
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_f64_reference_with_tail_and_unaligned_slices() {
        for len in 0..100 {
            let a: Vec<_> = (0..len + 1).map(|i| (i as f32 * 0.13).sin()).collect();
            let b: Vec<_> = (0..len + 1).map(|i| (i as f32 * 0.21).cos()).collect();
            let expected: f64 = a[1..]
                .iter()
                .zip(&b[1..])
                .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
                .sum();
            let actual = squared_euclidean(&a[1..], &b[1..]).unwrap() as f64;
            assert!((actual - expected).abs() <= 1e-5 * expected.max(1.0));
            assert_eq!(squared_euclidean(&a, &a).unwrap(), 0.0);
        }
    }

    #[test]
    fn mismatched_dimensions_return_error() {
        assert!(squared_euclidean(&[1.0], &[1.0, 2.0]).is_err());
    }
}
