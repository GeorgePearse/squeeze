//! TriMap (Large-scale Dimensionality Reduction Using Triplets)
//!
//! TriMap uses triplet constraints to preserve both local and global structure.
//! For each anchor point, it tries to keep similar points closer than dissimilar ones.

use ndarray::{Array2, Axis};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray};
use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::Normal;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use ordered_float::OrderedFloat;

use crate::metrics_simd;
use crate::mds::compute_distance_matrix;

/// TriMap dimensionality reduction
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct TriMap {
    n_components: usize,
    n_inliers: usize,     // Number of nearest neighbors (similar points)
    n_outliers: usize,    // Number of far points (dissimilar points)
    n_random: usize,      // Number of random triplets
    n_iter: usize,
    learning_rate: f64,
    weight_adj: f64,
    random_state: Option<u64>,
}

#[pymethods]
impl TriMap {
    #[new]
    #[pyo3(signature = (n_components=2, n_inliers=12, n_outliers=4, n_random=3, n_iter=800, learning_rate=0.1, weight_adj=50.0, random_state=None))]
    pub fn new(
        n_components: usize,
        n_inliers: usize,
        n_outliers: usize,
        n_random: usize,
        n_iter: usize,
        learning_rate: f64,
        weight_adj: f64,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_components,
            n_inliers,
            n_outliers,
            n_random,
            n_iter,
            learning_rate,
            weight_adj,
            random_state,
        }
    }

    /// Fit and transform data using TriMap
    pub fn fit_transform<'py>(&self, py: Python<'py>, data: PyReadonlyArray2<f64>) 
        -> PyResult<Bound<'py, PyArray2<f64>>> 
    {
        let x = data.as_array();
        let n_samples = x.nrows();

        if self.n_inliers + self.n_outliers >= n_samples {
            return Err(PyValueError::new_err(
                "n_inliers + n_outliers must be less than n_samples"
            ));
        }

        // Convert to f32 for distance computation
        let x_f32: Vec<Vec<f32>> = x.rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect();

        // Compute pairwise distances
        let distances = compute_distance_matrix(&x_f32);

        // Generate triplets: (anchor, positive, negative)
        let triplets = self.generate_triplets(&distances, n_samples);
        let weights = self.compute_weights(&distances, &triplets);

        // Initialize embedding with PCA
        let mut embedding = self.initialize_embedding(&x.to_owned(), n_samples)?;

        // Optimize using gradient descent
        self.optimize(&mut embedding, &triplets, &weights, n_samples);

        Ok(embedding.into_pyarray_bound(py))
    }
}

impl TriMap {
    fn generate_triplets(&self, distances: &Array2<f64>, n_samples: usize) -> Vec<(usize, usize, usize)> {
        let mut rng: StdRng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_seed(rand::random()),
        };

        let mut triplets = Vec::new();

        for i in 0..n_samples {
            // Find k nearest neighbors (inliers)
            let mut heap: BinaryHeap<(OrderedFloat<f64>, usize)> = BinaryHeap::new();
            for j in 0..n_samples {
                if i != j {
                    heap.push((OrderedFloat(-distances[[i, j]]), j));
                    if heap.len() > self.n_inliers {
                        heap.pop();
                    }
                }
            }
            let inliers: Vec<usize> = heap.into_iter().map(|(_, j)| j).collect();

            // Find k farthest neighbors (outliers)  
            let mut heap: BinaryHeap<(OrderedFloat<f64>, usize)> = BinaryHeap::new();
            for j in 0..n_samples {
                if i != j && !inliers.contains(&j) {
                    heap.push((OrderedFloat(distances[[i, j]]), j));
                    if heap.len() > self.n_outliers {
                        heap.pop();
                    }
                }
            }
            let outliers: Vec<usize> = heap.into_iter().map(|(_, j)| j).collect();

            // Create triplets: anchor=i, positive=inlier, negative=outlier
            for &pos in &inliers {
                for &neg in &outliers {
                    triplets.push((i, pos, neg));
                }
            }

            // Add random triplets
            for _ in 0..self.n_random {
                if let Some(&pos) = inliers.choose(&mut rng) {
                    let neg = loop {
                        let candidate = rng.gen_range(0..n_samples);
                        if candidate != i && candidate != pos {
                            break candidate;
                        }
                    };
                    triplets.push((i, pos, neg));
                }
            }
        }

        triplets
    }

    fn compute_weights(&self, distances: &Array2<f64>, triplets: &[(usize, usize, usize)]) -> Vec<f64> {
        // Weight based on distance difference
        triplets.par_iter().map(|&(i, j, k)| {
            let d_ij = distances[[i, j]];
            let d_ik = distances[[i, k]];
            let margin = d_ik - d_ij;
            
            // Higher weight for triplets with larger margin
            if margin > 0.0 {
                1.0 + self.weight_adj / (1.0 + d_ij)
            } else {
                1.0
            }
        }).collect()
    }

    fn initialize_embedding(&self, x: &Array2<f64>, n_samples: usize) -> PyResult<Array2<f64>> {
        // Simple PCA initialization
        let mean = x.mean_axis(Axis(0)).unwrap();
        let mut x_centered = x.clone();
        for mut row in x_centered.rows_mut() {
            row -= &mean;
        }

        // Use first n_components of centered data (simplified)
        let mut rng: StdRng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_seed(rand::random()),
        };
        
        let normal = Normal::new(0.0, 0.01).unwrap();
        let mut embedding = Array2::zeros((n_samples, self.n_components));
        
        // Random initialization with reasonable scale
        for mut row in embedding.rows_mut() {
            for v in row.iter_mut() {
                *v = normal.sample(&mut rng);
            }
        }

        // Scale based on data spread - use a reasonable scale for optimization
        let std: f64 = x_centered.mapv(|v| v * v).mean().unwrap().sqrt().max(1.0);
        embedding *= std * 0.1;

        Ok(embedding)
    }

    fn optimize(
        &self,
        embedding: &mut Array2<f64>,
        triplets: &[(usize, usize, usize)],
        weights: &[f64],
        n_samples: usize
    ) {
        let mut velocity = Array2::zeros((n_samples, self.n_components));
        let momentum = 0.5;

        for iter in 0..self.n_iter {
            let mut grad = Array2::zeros((n_samples, self.n_components));

            // Compute gradients for all triplets
            for (idx, &(i, j, k)) in triplets.iter().enumerate() {
                let weight = weights[idx];

                // Compute distances in embedding space
                let mut d_ij_sq = 0.0;
                let mut d_ik_sq = 0.0;
                for c in 0..self.n_components {
                    let diff_ij = embedding[[i, c]] - embedding[[j, c]];
                    let diff_ik = embedding[[i, c]] - embedding[[k, c]];
                    d_ij_sq += diff_ij * diff_ij;
                    d_ik_sq += diff_ik * diff_ik;
                }

                // Triplet loss gradient
                // We want d_ij < d_ik, so loss = max(0, d_ij - d_ik + margin)
                let margin = 1.0;
                let loss = d_ij_sq - d_ik_sq + margin;
                
                if loss > 0.0 {
                    let scale = 2.0 * weight / (triplets.len() as f64);
                    
                    for c in 0..self.n_components {
                        let diff_ij = embedding[[i, c]] - embedding[[j, c]];
                        let diff_ik = embedding[[i, c]] - embedding[[k, c]];
                        
                        // Gradient w.r.t. anchor
                        grad[[i, c]] += scale * (diff_ij - diff_ik);
                        // Gradient w.r.t. positive
                        grad[[j, c]] -= scale * diff_ij;
                        // Gradient w.r.t. negative
                        grad[[k, c]] += scale * diff_ik;
                    }
                }
            }

            // Update with momentum and adaptive learning rate
            let lr = self.learning_rate * (1.0 - iter as f64 / self.n_iter as f64).max(0.01);
            velocity = momentum * &velocity - lr * &grad;
            *embedding = &*embedding + &velocity;

            // Center embedding
            let mean = embedding.mean_axis(Axis(0)).unwrap();
            for mut row in embedding.rows_mut() {
                row -= &mean;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_test_data() -> Array2<f64> {
        let mut data = Array2::zeros((30, 5));
        for i in 0..30 {
            for j in 0..5 {
                data[[i, j]] = (i as f64) * 0.3 + (j as f64) * 0.05;
            }
        }
        data
    }

    fn create_test_distances(n: usize) -> Array2<f64> {
        let mut distances = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                distances[[i, j]] = ((i as f64) - (j as f64)).abs();
            }
        }
        distances
    }

    #[test]
    fn test_triplet_generation_not_empty() {
        let trimap = TriMap::new(2, 3, 2, 1, 100, 0.1, 50.0, Some(42));
        let distances = create_test_distances(30);

        let triplets = trimap.generate_triplets(&distances, 30);

        assert!(!triplets.is_empty(), "Should generate triplets");
    }

    #[test]
    fn test_triplet_generation_distinct_indices() {
        let trimap = TriMap::new(2, 3, 2, 1, 100, 0.1, 50.0, Some(42));
        let distances = create_test_distances(30);

        let triplets = trimap.generate_triplets(&distances, 30);

        // Each triplet should have distinct indices
        for &(anchor, pos, neg) in &triplets {
            assert_ne!(anchor, pos, "Anchor and positive should be different");
            assert_ne!(anchor, neg, "Anchor and negative should be different");
            assert_ne!(pos, neg, "Positive and negative should be different");
        }
    }

    #[test]
    fn test_triplet_generation_valid_indices() {
        let trimap = TriMap::new(2, 3, 2, 1, 100, 0.1, 50.0, Some(42));
        let distances = create_test_distances(30);

        let triplets = trimap.generate_triplets(&distances, 30);

        // All indices should be within bounds
        for &(anchor, pos, neg) in &triplets {
            assert!(anchor < 30, "Anchor index out of bounds");
            assert!(pos < 30, "Positive index out of bounds");
            assert!(neg < 30, "Negative index out of bounds");
        }
    }

    #[test]
    fn test_triplet_structure() {
        // Test that triplets are structured correctly (inliers paired with outliers)
        let trimap = TriMap::new(2, 3, 2, 0, 100, 0.1, 50.0, Some(42));
        let distances = create_test_distances(20);

        let triplets = trimap.generate_triplets(&distances, 20);

        // Each triplet should have (anchor, positive_from_inliers, negative_from_outliers)
        // The algorithm pairs each inlier with each outlier
        // Expected count: n_samples * n_inliers * n_outliers = 20 * 3 * 2 = 120
        assert_eq!(triplets.len(), 20 * 3 * 2, "Should have n_samples * n_inliers * n_outliers triplets");
    }

    #[test]
    fn test_weights_positive() {
        let trimap = TriMap::new(2, 5, 3, 1, 100, 0.1, 50.0, Some(42));
        let distances = create_test_distances(30);
        let triplets = trimap.generate_triplets(&distances, 30);

        let weights = trimap.compute_weights(&distances, &triplets);

        // All weights should be positive
        for (i, &w) in weights.iter().enumerate() {
            assert!(w > 0.0, "Weight {} should be positive, got {}", i, w);
        }
    }

    #[test]
    fn test_weights_count_matches_triplets() {
        let trimap = TriMap::new(2, 5, 3, 1, 100, 0.1, 50.0, Some(42));
        let distances = create_test_distances(30);
        let triplets = trimap.generate_triplets(&distances, 30);

        let weights = trimap.compute_weights(&distances, &triplets);

        assert_eq!(weights.len(), triplets.len(), "Should have one weight per triplet");
    }

    #[test]
    fn test_weights_higher_for_good_triplets() {
        let trimap = TriMap::new(2, 5, 3, 0, 100, 0.1, 50.0, Some(42));
        let distances = create_test_distances(30);
        let triplets = trimap.generate_triplets(&distances, 30);

        let weights = trimap.compute_weights(&distances, &triplets);

        // Triplets with positive margin (d_neg > d_pos) should have higher weight
        let mut good_weights = Vec::new();
        let mut bad_weights = Vec::new();

        for (idx, &(anchor, pos, neg)) in triplets.iter().enumerate() {
            let d_pos = distances[[anchor, pos]];
            let d_neg = distances[[anchor, neg]];
            if d_neg > d_pos {
                good_weights.push(weights[idx]);
            } else {
                bad_weights.push(weights[idx]);
            }
        }

        if !good_weights.is_empty() && !bad_weights.is_empty() {
            let avg_good: f64 = good_weights.iter().sum::<f64>() / good_weights.len() as f64;
            let avg_bad: f64 = bad_weights.iter().sum::<f64>() / bad_weights.len() as f64;
            assert!(avg_good >= avg_bad, "Good triplets should have higher average weight");
        }
    }

    #[test]
    fn test_initialization_shape() {
        let trimap = TriMap::new(2, 5, 3, 1, 100, 0.1, 50.0, Some(42));
        let data = create_test_data();

        let embedding = trimap.initialize_embedding(&data, 30).unwrap();

        assert_eq!(embedding.shape(), &[30, 2]);
    }

    #[test]
    fn test_initialization_reproducible() {
        let trimap1 = TriMap::new(2, 5, 3, 1, 100, 0.1, 50.0, Some(42));
        let trimap2 = TriMap::new(2, 5, 3, 1, 100, 0.1, 50.0, Some(42));
        let data = create_test_data();

        let emb1 = trimap1.initialize_embedding(&data, 30).unwrap();
        let emb2 = trimap2.initialize_embedding(&data, 30).unwrap();

        for i in 0..30 {
            for j in 0..2 {
                assert_relative_eq!(emb1[[i, j]], emb2[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_initialization_different_seeds() {
        let trimap1 = TriMap::new(2, 5, 3, 1, 100, 0.1, 50.0, Some(42));
        let trimap2 = TriMap::new(2, 5, 3, 1, 100, 0.1, 50.0, Some(123));
        let data = create_test_data();

        let emb1 = trimap1.initialize_embedding(&data, 30).unwrap();
        let emb2 = trimap2.initialize_embedding(&data, 30).unwrap();

        // Different seeds should give different embeddings
        let mut different = false;
        for i in 0..30 {
            for j in 0..2 {
                if (emb1[[i, j]] - emb2[[i, j]]).abs() > 1e-10 {
                    different = true;
                    break;
                }
            }
        }
        assert!(different, "Different seeds should produce different initializations");
    }

    #[test]
    fn test_triplet_generation_reproducible() {
        let trimap1 = TriMap::new(2, 5, 3, 1, 100, 0.1, 50.0, Some(42));
        let trimap2 = TriMap::new(2, 5, 3, 1, 100, 0.1, 50.0, Some(42));
        let distances = create_test_distances(30);

        let triplets1 = trimap1.generate_triplets(&distances, 30);
        let triplets2 = trimap2.generate_triplets(&distances, 30);

        // With same seed, should generate same triplets
        assert_eq!(triplets1.len(), triplets2.len());
        for (t1, t2) in triplets1.iter().zip(triplets2.iter()) {
            assert_eq!(t1, t2);
        }
    }

    #[test]
    fn test_triplet_count() {
        let n_inliers = 5;
        let n_outliers = 3;
        let n_random = 2;
        let n_samples = 30;
        let trimap = TriMap::new(2, n_inliers, n_outliers, n_random, 100, 0.1, 50.0, Some(42));
        let distances = create_test_distances(n_samples);

        let triplets = trimap.generate_triplets(&distances, n_samples);

        // Expected: for each sample, n_inliers * n_outliers structured triplets + n_random random triplets
        let expected = n_samples * (n_inliers * n_outliers + n_random);
        assert_eq!(triplets.len(), expected, "Expected {} triplets", expected);
    }

    #[test]
    fn test_initialization_finite() {
        let trimap = TriMap::new(2, 5, 3, 1, 100, 0.1, 50.0, Some(42));
        let data = create_test_data();

        let embedding = trimap.initialize_embedding(&data, 30).unwrap();

        // All values should be finite
        for &val in embedding.iter() {
            assert!(val.is_finite(), "Embedding should have finite values");
        }
    }

    #[test]
    fn test_initialization_scaled() {
        let trimap = TriMap::new(2, 5, 3, 1, 100, 0.1, 50.0, Some(42));
        let data = create_test_data();

        let embedding = trimap.initialize_embedding(&data, 30).unwrap();

        // Embedding should have reasonable scale (not too large, not too small)
        let max_val: f64 = embedding.iter().map(|&v| v.abs()).fold(0.0, f64::max);
        assert!(max_val > 1e-10, "Embedding should not be all zeros");
        assert!(max_val < 1e6, "Embedding should not have extreme values");
    }
}
