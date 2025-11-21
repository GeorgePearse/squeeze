//! t-SNE (t-Distributed Stochastic Neighbor Embedding) implementation.
//!
//! t-SNE is a nonlinear dimensionality reduction technique that models
//! pairwise similarities in high-dimensional space and finds a low-dimensional
//! embedding that preserves these similarities.

use ndarray::{Array1, Array2, Axis};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray};
use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::Normal;
use rayon::prelude::*;

use crate::metrics_simd;

/// t-SNE dimensionality reduction
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct TSNE {
    n_components: usize,
    perplexity: f64,
    learning_rate: f64,
    n_iter: usize,
    early_exaggeration: f64,
    random_state: Option<u64>,
}

#[pymethods]
impl TSNE {
    #[new]
    #[pyo3(signature = (n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, early_exaggeration=12.0, random_state=None))]
    pub fn new(
        n_components: usize,
        perplexity: f64,
        learning_rate: f64,
        n_iter: usize,
        early_exaggeration: f64,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_components,
            perplexity,
            learning_rate,
            n_iter,
            early_exaggeration,
            random_state,
        }
    }

    /// Fit and transform data using t-SNE
    pub fn fit_transform<'py>(&self, py: Python<'py>, data: PyReadonlyArray2<f64>) 
        -> PyResult<Bound<'py, PyArray2<f64>>> 
    {
        let x = data.as_array();
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples < 4 {
            return Err(PyValueError::new_err("t-SNE requires at least 4 samples"));
        }

        // Convert to f32 for distance computation
        let x_f32: Vec<Vec<f32>> = x.rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect();

        // Compute pairwise distances
        let distances = self.compute_pairwise_distances(&x_f32);

        // Compute P (joint probabilities in high-dimensional space)
        let p = self.compute_joint_probabilities(&distances, n_samples);

        // Initialize embedding randomly
        let mut rng: StdRng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_seed(rand::random()),
        };
        
        let normal = Normal::new(0.0, 1e-4).unwrap();
        let mut y: Array2<f64> = Array2::zeros((n_samples, self.n_components));
        for mut row in y.rows_mut() {
            for v in row.iter_mut() {
                *v = normal.sample(&mut rng);
            }
        }

        // Gradient descent
        let mut gains = Array2::ones((n_samples, self.n_components));
        let mut y_incs = Array2::zeros((n_samples, self.n_components));
        let momentum = 0.5;
        let final_momentum = 0.8;
        let momentum_switch_iter = 250;

        for iter in 0..self.n_iter {
            // Apply early exaggeration for first 250 iterations
            let p_scaled = if iter < 250 {
                &p * self.early_exaggeration
            } else {
                p.clone()
            };

            // Compute Q (joint probabilities in low-dimensional space)
            let q = self.compute_q(&y);

            // Compute gradients
            let grad = self.compute_gradient(&p_scaled, &q, &y);

            // Update gains (adaptive learning rate)
            for i in 0..n_samples {
                for j in 0..self.n_components {
                    let sign_match = (grad[[i, j]] > 0.0) == (y_incs[[i, j]] > 0.0);
                    gains[[i, j]] = if sign_match {
                        f64::max(gains[[i, j]] * 0.8, 0.01)
                    } else {
                        gains[[i, j]] + 0.2
                    };
                }
            }

            // Update momentum
            let current_momentum = if iter < momentum_switch_iter {
                momentum
            } else {
                final_momentum
            };

            // Update positions
            y_incs = current_momentum * &y_incs - self.learning_rate * (&gains * &grad);
            y = &y + &y_incs;

            // Center the embedding
            let mean = y.mean_axis(Axis(0)).unwrap();
            for mut row in y.rows_mut() {
                row -= &mean;
            }
        }

        Ok(y.into_pyarray_bound(py))
    }
}

impl TSNE {
    fn compute_pairwise_distances(&self, data: &[Vec<f32>]) -> Array2<f64> {
        let n = data.len();
        let mut distances = Array2::zeros((n, n));

        // Parallel distance computation
        let dists: Vec<Vec<f64>> = (0..n).into_par_iter().map(|i| {
            let mut row = vec![0.0; n];
            for j in 0..n {
                if i != j {
                    let d = metrics_simd::euclidean(&data[i], &data[j]).unwrap_or(0.0) as f64;
                    row[j] = d * d; // Squared distance
                }
            }
            row
        }).collect();

        for (i, row) in dists.into_iter().enumerate() {
            for (j, d) in row.into_iter().enumerate() {
                distances[[i, j]] = d;
            }
        }

        distances
    }

    fn compute_joint_probabilities(&self, distances: &Array2<f64>, n_samples: usize) -> Array2<f64> {
        let target_entropy = (self.perplexity).ln();
        
        // Parallel computation of conditional probabilities P(j|i)
        let p_rows: Vec<Vec<f64>> = (0..n_samples).into_par_iter().map(|i| {
            let mut p_row = vec![0.0; n_samples];
            let mut beta = 1.0; // 1 / (2 * sigma^2)
            let mut beta_min = f64::NEG_INFINITY;
            let mut beta_max = f64::INFINITY;

            // Binary search for sigma that gives target perplexity
            for _ in 0..50 {
                // Compute P(j|i) for current beta
                let mut sum_p = 0.0;
                for j in 0..n_samples {
                    if i != j {
                        let pij = (-beta * distances[[i, j]]).exp();
                        p_row[j] = pij;
                        sum_p += pij;
                    }
                }

                // Normalize
                if sum_p > 1e-10 {
                    for j in 0..n_samples {
                        p_row[j] /= sum_p;
                    }
                }

                // Compute entropy
                let mut entropy = 0.0;
                for j in 0..n_samples {
                    if p_row[j] > 1e-10 {
                        entropy -= p_row[j] * p_row[j].ln();
                    }
                }

                // Adjust beta based on entropy
                let entropy_diff = entropy - target_entropy;
                if entropy_diff.abs() < 1e-5 {
                    break;
                }

                if entropy_diff > 0.0 {
                    beta_min = beta;
                    beta = if beta_max.is_infinite() { beta * 2.0 } else { (beta + beta_max) / 2.0 };
                } else {
                    beta_max = beta;
                    beta = if beta_min.is_infinite() { beta / 2.0 } else { (beta + beta_min) / 2.0 };
                }
            }
            
            p_row
        }).collect();
        
        // Assemble P matrix from parallel results
        let mut p = Array2::zeros((n_samples, n_samples));
        for (i, row) in p_rows.into_iter().enumerate() {
            for (j, val) in row.into_iter().enumerate() {
                p[[i, j]] = val;
            }
        }

        // Symmetrize: P = (P + P^T) / (2n)
        let mut p_sym = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                p_sym[[i, j]] = (p[[i, j]] + p[[j, i]]) / (2.0 * n_samples as f64);
            }
        }

        // Ensure minimum probability
        p_sym.mapv_inplace(|v| v.max(1e-12));

        p_sym
    }

    fn compute_q(&self, y: &Array2<f64>) -> Array2<f64> {
        let n = y.nrows();
        let mut q = Array2::zeros((n, n));
        let mut sum_q = 0.0;

        // Compute Student t-distribution kernel
        for i in 0..n {
            for j in (i + 1)..n {
                let mut dist_sq = 0.0;
                for k in 0..self.n_components {
                    let diff = y[[i, k]] - y[[j, k]];
                    dist_sq += diff * diff;
                }
                let qij = 1.0 / (1.0 + dist_sq);
                q[[i, j]] = qij;
                q[[j, i]] = qij;
                sum_q += 2.0 * qij;
            }
        }

        // Normalize
        if sum_q > 0.0 {
            q /= sum_q;
        }

        // Ensure minimum probability
        q.mapv_inplace(|v| v.max(1e-12));

        q
    }

    fn compute_gradient(&self, p: &Array2<f64>, q: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        let n = y.nrows();
        
        // Compute (P - Q) * kernel
        let pq = p - q;
        
        // Parallel gradient computation - each thread computes gradients for a subset of points
        let grad_rows: Vec<Vec<f64>> = (0..n).into_par_iter().map(|i| {
            let mut grad_row = vec![0.0; self.n_components];
            
            for j in 0..n {
                if i != j {
                    let mut dist_sq = 0.0;
                    for k in 0..self.n_components {
                        let diff = y[[i, k]] - y[[j, k]];
                        dist_sq += diff * diff;
                    }
                    let kernel = 1.0 / (1.0 + dist_sq);
                    let mult = 4.0 * pq[[i, j]] * kernel;

                    for k in 0..self.n_components {
                        grad_row[k] += mult * (y[[i, k]] - y[[j, k]]);
                    }
                }
            }
            
            grad_row
        }).collect();
        
        // Assemble the gradient matrix from parallel results
        let mut grad = Array2::zeros((n, self.n_components));
        for (i, row) in grad_rows.into_iter().enumerate() {
            for (j, val) in row.into_iter().enumerate() {
                grad[[i, j]] = val;
            }
        }

        grad
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_two_clusters() -> Vec<Vec<f32>> {
        let mut data = Vec::new();
        // Cluster 1 around origin
        for i in 0..20 {
            let mut point = vec![0.0; 10];
            for j in 0..10 {
                point[j] = (i as f32) * 0.01 + (j as f32) * 0.001;
            }
            data.push(point);
        }
        // Cluster 2 offset by 10
        for i in 0..20 {
            let mut point = vec![10.0; 10];
            for j in 0..10 {
                point[j] = 10.0 + (i as f32) * 0.01 + (j as f32) * 0.001;
            }
            data.push(point);
        }
        data
    }

    #[test]
    fn test_pairwise_distances_symmetric() {
        let tsne = TSNE::new(2, 5.0, 200.0, 100, 12.0, Some(42));
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        
        let distances = tsne.compute_pairwise_distances(&data);
        
        // Check symmetry
        for i in 0..4 {
            for j in 0..4 {
                assert_relative_eq!(distances[[i, j]], distances[[j, i]], epsilon = 1e-10);
            }
        }
        
        // Check diagonal is zero
        for i in 0..4 {
            assert_eq!(distances[[i, i]], 0.0);
        }
        
        // Check known distances (squared)
        assert_relative_eq!(distances[[0, 1]], 1.0, epsilon = 1e-5); // (1-0)^2 + (0-0)^2
        assert_relative_eq!(distances[[0, 2]], 1.0, epsilon = 1e-5); // (0-0)^2 + (1-0)^2
        assert_relative_eq!(distances[[0, 3]], 2.0, epsilon = 1e-5); // (1-0)^2 + (1-0)^2
    }

    #[test]
    fn test_joint_probabilities_properties() {
        let tsne = TSNE::new(2, 5.0, 200.0, 100, 12.0, Some(42));
        let data = create_two_clusters();
        let distances = tsne.compute_pairwise_distances(&data);
        let p = tsne.compute_joint_probabilities(&distances, 40);
        
        // P should be symmetric
        for i in 0..40 {
            for j in 0..40 {
                assert_relative_eq!(p[[i, j]], p[[j, i]], epsilon = 1e-10,
                    "P matrix not symmetric at [{}, {}]", i, j);
            }
        }
        
        // P should sum to 1
        let sum: f64 = p.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6, "P doesn't sum to 1");
        
        // All probabilities should be non-negative
        for &val in p.iter() {
            assert!(val >= 0.0, "Found negative probability");
        }
        
        // Diagonal should be zero (no self-similarity)
        for i in 0..40 {
            assert!(p[[i, i]] < 1e-10, "Diagonal should be ~0");
        }
    }

    #[test]
    fn test_q_distribution_properties() {
        let tsne = TSNE::new(2, 5.0, 200.0, 100, 12.0, Some(42));
        
        // Create a simple embedding
        let mut y = Array2::zeros((10, 2));
        for i in 0..10 {
            y[[i, 0]] = (i as f64) * 0.5;
            y[[i, 1]] = (i as f64).sin();
        }
        
        let q = tsne.compute_q(&y);
        
        // Q should be symmetric
        for i in 0..10 {
            for j in 0..10 {
                assert_relative_eq!(q[[i, j]], q[[j, i]], epsilon = 1e-10);
            }
        }
        
        // Q should sum to 1
        let sum: f64 = q.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        
        // All values should be non-negative
        for &val in q.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_gradient_computation() {
        let tsne = TSNE::new(2, 5.0, 200.0, 100, 12.0, Some(42));
        
        // Simple test case
        let p = Array2::from_shape_vec((3, 3), vec![
            0.0, 0.3, 0.2,
            0.3, 0.0, 0.2,
            0.2, 0.2, 0.0,
        ]).unwrap();
        
        let q = Array2::from_shape_vec((3, 3), vec![
            0.0, 0.25, 0.25,
            0.25, 0.0, 0.25,
            0.25, 0.25, 0.0,
        ]).unwrap();
        
        let y = Array2::from_shape_vec((3, 2), vec![
            0.0, 0.0,
            1.0, 0.0,
            0.5, 0.866,
        ]).unwrap();
        
        let grad = tsne.compute_gradient(&p, &q, &y);
        
        // Gradient should have correct shape
        assert_eq!(grad.shape(), &[3, 2]);
        
        // Gradient should not be all zeros (unless at optimum)
        let grad_norm: f64 = grad.iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!(grad_norm > 1e-10, "Gradient should be non-zero");
    }

    #[test]
    fn test_perplexity_binary_search() {
        let tsne = TSNE::new(2, 5.0, 200.0, 100, 12.0, Some(42));
        
        // Test that binary search finds reasonable sigmas
        let distances = Array2::from_shape_vec((3, 3), vec![
            0.0, 1.0, 4.0,
            1.0, 0.0, 1.0,
            4.0, 1.0, 0.0,
        ]).unwrap();
        
        let p = tsne.compute_joint_probabilities(&distances, 3);
        
        // Check that closer points have higher probability
        assert!(p[[0, 1]] > p[[0, 2]], "Closer points should have higher probability");
        assert!(p[[1, 0]] > p[[2, 0]], "Closer points should have higher probability");
    }

    #[test]
    fn test_reproducibility_with_seed() {
        // Same seed should give same initialization
        let tsne1 = TSNE::new(2, 5.0, 200.0, 100, 12.0, Some(42));
        let tsne2 = TSNE::new(2, 5.0, 200.0, 100, 12.0, Some(42));
        
        let data = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0],
        ];
        
        let distances1 = tsne1.compute_pairwise_distances(&data);
        let distances2 = tsne2.compute_pairwise_distances(&data);
        
        // Should be identical
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(distances1[[i, j]], distances2[[i, j]]);
            }
        }
    }
}
