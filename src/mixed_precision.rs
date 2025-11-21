/// Mixed precision vector storage and operations
///
/// This module provides f16 storage with f32 computation for:
/// - 50% memory reduction
/// - Better cache utilization (2x more vectors in cache)
/// - Minimal accuracy loss (<0.5%)
///
/// Conversion overhead is offset by cache benefits.

use half::f16;
use serde::{Serialize, Deserialize};

/// Vector stored in f16 format (2 bytes per element)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MixedPrecisionVec {
    data: Vec<f16>,
}

impl MixedPrecisionVec {
    /// Create from f32 data
    #[inline]
    pub fn from_f32(data: &[f32]) -> Self {
        Self {
            data: data.iter().map(|&x| f16::from_f32(x)).collect(),
        }
    }
    
    /// Convert to f32 for computation
    #[inline]
    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&x| x.to_f32()).collect()
    }
    
    /// In-place conversion to f32 (avoids allocation)
    #[inline]
    pub fn to_f32_into(&self, out: &mut [f32]) {
        assert_eq!(self.data.len(), out.len());
        for (i, &val) in self.data.iter().enumerate() {
            out[i] = val.to_f32();
        }
    }
    
    /// Get length
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Get element at index as f32
    #[inline]
    pub fn get_f32(&self, index: usize) -> f32 {
        self.data[index].to_f32()
    }
    
    /// Memory usage in bytes
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f16>()
    }
}

/// Storage for multiple vectors with mixed precision
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MixedPrecisionStorage {
    vectors: Vec<MixedPrecisionVec>,
}

impl MixedPrecisionStorage {
    /// Create from Vec<Vec<f32>>
    pub fn from_f32_vecs(data: Vec<Vec<f32>>) -> Self {
        Self {
            vectors: data.into_iter()
                .map(|v| MixedPrecisionVec::from_f32(&v))
                .collect(),
        }
    }
    
    /// Get vector at index as f32
    pub fn get_vec_f32(&self, index: usize) -> Vec<f32> {
        self.vectors[index].to_f32()
    }
    
    /// Get vector at index and write to buffer
    pub fn get_vec_f32_into(&self, index: usize, out: &mut [f32]) {
        self.vectors[index].to_f32_into(out);
    }
    
    /// Number of vectors
    pub fn len(&self) -> usize {
        self.vectors.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
    
    /// Total memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.vectors.iter().map(|v| v.memory_bytes()).sum()
    }
    
    /// Memory savings vs f32 storage
    pub fn memory_savings_percent(&self) -> f32 {
        50.0  // f16 is 50% smaller than f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_f16_roundtrip() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, -1.5];
        let fp16 = MixedPrecisionVec::from_f32(&original);
        let recovered = fp16.to_f32();
        
        for (a, b) in original.iter().zip(recovered.iter()) {
            // f16 has ~3 decimal digits precision
            assert!((a - b).abs() < 1e-3, "Expected {}, got {}", a, b);
        }
    }
    
    #[test]
    fn test_f16_accuracy() {
        // Test that f16 maintains sufficient accuracy for distance computation
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        // Compute distance with f32
        let dist_f32: f32 = a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
            .sqrt();
        
        // Compute distance with f16 storage
        let a_f16 = MixedPrecisionVec::from_f32(&a);
        let b_f16 = MixedPrecisionVec::from_f32(&b);
        let a_recovered = a_f16.to_f32();
        let b_recovered = b_f16.to_f32();
        
        let dist_f16: f32 = a_recovered.iter()
            .zip(b_recovered.iter())
            .map(|(&x, &y)| {
                let diff = x - y;
                diff * diff
            })
            .sum::<f32>()
            .sqrt();
        
        // Distance should be very close (<0.5% error)
        let error = (dist_f32 - dist_f16).abs() / dist_f32;
        assert!(error < 0.005, "Distance error too large: {}%", error * 100.0);
    }
    
    #[test]
    fn test_memory_savings() {
        let data: Vec<f32> = (0..1000).map(|x| x as f32).collect();
        let fp16 = MixedPrecisionVec::from_f32(&data);
        
        let f32_bytes = data.len() * std::mem::size_of::<f32>();
        let f16_bytes = fp16.memory_bytes();
        
        assert_eq!(f16_bytes, f32_bytes / 2);
    }
    
    #[test]
    fn test_into_conversion() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        let fp16 = MixedPrecisionVec::from_f32(&original);
        
        let mut buffer = vec![0.0f32; 4];
        fp16.to_f32_into(&mut buffer);
        
        for (a, b) in original.iter().zip(buffer.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
    }
}
