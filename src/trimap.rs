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
