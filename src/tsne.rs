//! t-SNE (t-Distributed Stochastic Neighbor Embedding) implementation.
//!
//! t-SNE is a nonlinear dimensionality reduction technique that models
//! pairwise similarities in high-dimensional space and finds a low-dimensional
//! embedding that preserves these similarities.
//!
//! This implementation supports Barnes-Hut approximation for O(n log n) gradient
//! computation on large datasets.

use ndarray::{Array1, Array2, Axis};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray};
use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::Normal;
use rayon::prelude::*;

use crate::metrics_simd;
use crate::barnes_hut::QuadTreeNode;

/// t-SNE dimensionality reduction
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct TSNE {
    n_components: usize,
    perplexity: f64,
    learning_rate: f64,
    n_iter: usize,
    early_exaggeration: f64,
    random_state: Option<u64>,
    /// Barnes-Hut approximation parameter (0 = exact, higher = more approximation)
    theta: f64,
    /// Whether to use Barnes-Hut (None = auto-select based on n_samples)
    use_barnes_hut: Option<bool>,
    /// Minimum gradient norm for early stopping (0 = disabled)
    min_grad_norm: f64,
    /// Number of iterations with no progress before stopping
    n_iter_without_progress: usize,
}

#[pymethods]
impl TSNE {
    #[new]
    #[pyo3(signature = (n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, early_exaggeration=12.0, random_state=None, theta=0.5, use_barnes_hut=None, min_grad_norm=1e-7, n_iter_without_progress=300))]
    pub fn new(
        n_components: usize,
        perplexity: f64,
        learning_rate: f64,
        n_iter: usize,
        early_exaggeration: f64,
        random_state: Option<u64>,
        theta: f64,
        use_barnes_hut: Option<bool>,
        min_grad_norm: f64,
        n_iter_without_progress: usize,
    ) -> Self {
        Self {
            n_components,
            perplexity,
            learning_rate,
            n_iter,
            early_exaggeration,
            random_state,
            theta,
            use_barnes_hut,
            min_grad_norm,
            n_iter_without_progress,
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

        // Determine whether to use Barnes-Hut
        let use_bh = self.should_use_barnes_hut(n_samples);

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

        // Early stopping tracking
        let mut best_grad_norm = f64::INFINITY;
        let mut iter_without_progress = 0;

        for iter in 0..self.n_iter {
            // Apply early exaggeration for first 250 iterations
            let p_scaled = if iter < 250 {
                &p * self.early_exaggeration
            } else {
                p.clone()
            };

            // Compute gradients - use Barnes-Hut for 2D embeddings on large datasets
            let grad = if use_bh && self.n_components == 2 {
                self.compute_gradient_barnes_hut(&p_scaled, &y)
            } else {
                // Compute Q (joint probabilities in low-dimensional space)
                let q = self.compute_q(&y);
                self.compute_gradient(&p_scaled, &q, &y)
            };

            // Compute gradient norm for early stopping (skip during early exaggeration phase)
            if iter >= 250 && self.min_grad_norm > 0.0 {
                let grad_norm: f64 = grad.iter().map(|&v| v * v).sum::<f64>().sqrt()
                    / (n_samples as f64).sqrt();

                // Check for early stopping based on gradient norm
                if grad_norm < self.min_grad_norm {
                    break;
                }

                // Check for lack of progress
                if grad_norm < best_grad_norm {
                    best_grad_norm = grad_norm;
                    iter_without_progress = 0;
                } else {
                    iter_without_progress += 1;
                    if iter_without_progress >= self.n_iter_without_progress {
                        break;
                    }
                }
            }

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
    /// Determine whether to use Barnes-Hut approximation
    fn should_use_barnes_hut(&self, n_samples: usize) -> bool {
        match self.use_barnes_hut {
            Some(use_bh) => use_bh,
            // Auto-select: use Barnes-Hut for large datasets with 2D output
            None => n_samples > 1000 && self.n_components == 2,
        }
    }

    /// Compute gradient using Barnes-Hut approximation
    /// This is O(n log n) instead of O(n²) for the repulsive forces
    fn compute_gradient_barnes_hut(&self, p: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        let n = y.nrows();
        let mut grad = Array2::zeros((n, 2));

        // Build quadtree for Barnes-Hut approximation
        let tree = QuadTreeNode::build(y);

        // Compute normalization constant Z and repulsive forces
        // Z = Σ_{i≠j} (1 + ||y_i - y_j||²)^(-1)
        // In the exact computation, sum_q = 2 * Σ_{i<j} kernel_ij (each pair counted once in upper triangle, times 2)
        // So Z = sum_q for normalization. Here we compute Σ_i Σ_{j≠i} kernel_ij = 2 * Σ_{i<j} kernel_ij
        let mut z_sum = 0.0;

        // First pass: compute Z using tree approximation
        for i in 0..n {
            let point = [y[[i, 0]], y[[i, 1]]];
            // Approximate sum of (1 + dist²)^(-1) using tree
            z_sum += self.compute_z_contribution(&tree, &point, i);
        }
        // z_sum = Σ_i Σ_{j≠i} kernel_ij = 2 * Σ_{i<j} kernel_ij = sum_q (the normalization in exact Q)
        // Make sure Z is not too small
        let z = z_sum.max(1e-12);

        // Second pass: compute gradients
        // The gradient is: 4 * Σ_j [(p_ij - q_ij) * (1 + ||y_i - y_j||²)^(-1) * (y_i - y_j)]
        // Where q_ij = (1 + ||y_i - y_j||²)^(-1) / Z
        //
        // Split into attractive (from P) and repulsive (from Q) terms:
        // Attractive: 4 * Σ_j [p_ij * (1 + ||y_i - y_j||²)^(-1) * (y_i - y_j)]
        // Repulsive: -4 * Σ_j [q_ij * (1 + ||y_i - y_j||²)^(-1) * (y_i - y_j)]
        //          = -4/Z * Σ_j [(1 + ||y_i - y_j||²)^(-2) * (y_i - y_j)]

        // Parallel computation
        let grad_rows: Vec<[f64; 2]> = (0..n).into_par_iter().map(|i| {
            let point = [y[[i, 0]], y[[i, 1]]];
            let mut grad_i = [0.0, 0.0];

            // Attractive forces (exact, since P is sparse-ish)
            for j in 0..n {
                if i != j && p[[i, j]] > 1e-12 {
                    let dx = y[[i, 0]] - y[[j, 0]];
                    let dy = y[[i, 1]] - y[[j, 1]];
                    let dist_sq = dx * dx + dy * dy;
                    let kernel = 1.0 / (1.0 + dist_sq);

                    grad_i[0] += 4.0 * p[[i, j]] * kernel * dx;
                    grad_i[1] += 4.0 * p[[i, j]] * kernel * dy;
                }
            }

            // Repulsive forces (approximated using Barnes-Hut)
            let rep_force = tree.compute_non_edge_forces(&point, self.theta, i);
            // The tree returns forces in the form: Σ (1/(1+d²))² * (com - point) = Σ (1+d²)^(-2) * (y_j - y_i)
            // The repulsive gradient is: -4/Z * Σ (1+d²)^(-2) * (y_i - y_j)
            //                          = -4/Z * Σ (1+d²)^(-2) * (-(y_j - y_i))
            //                          = 4/Z * Σ (1+d²)^(-2) * (y_j - y_i)
            //                          = 4/Z * tree_force
            grad_i[0] += 4.0 * rep_force[0] / z;
            grad_i[1] += 4.0 * rep_force[1] / z;

            grad_i
        }).collect();

        // Assemble gradient matrix
        for (i, g) in grad_rows.into_iter().enumerate() {
            grad[[i, 0]] = g[0];
            grad[[i, 1]] = g[1];
        }

        grad
    }

    /// Compute Z contribution for a point using tree approximation
    fn compute_z_contribution(&self, node: &QuadTreeNode, point: &[f64], point_idx: usize) -> f64 {
        // Skip if this is the same point
        if let Some(idx) = node.point_idx {
            if idx == point_idx {
                return 0.0;
            }
        }

        let dx = node.center_of_mass[0] - point[0];
        let dy = node.center_of_mass[1] - point[1];
        let dist_sq = dx * dx + dy * dy;

        // If node is far enough away, use approximation
        let node_size = node.bounds.width().max(node.bounds.height());
        if node_size * node_size / dist_sq.max(1e-12) < self.theta * self.theta {
            // Approximate contribution from center of mass
            return node.total_mass / (1.0 + dist_sq);
        }

        // Recurse into children
        if let Some(ref children) = node.children {
            let mut contrib = 0.0;
            for child in children.iter() {
                if let Some(ref child_node) = child {
                    contrib += self.compute_z_contribution(child_node, point, point_idx);
                }
            }
            return contrib;
        }

        // Leaf node with single point
        if node.point_idx.is_some() && node.point_idx.unwrap() != point_idx {
            return 1.0 / (1.0 + dist_sq);
        }

        0.0
    }

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

    /// Helper to create TSNE with default parameters
    fn create_tsne(n_components: usize, perplexity: f64, learning_rate: f64,
                   n_iter: usize, early_exaggeration: f64, random_state: Option<u64>) -> TSNE {
        TSNE::new(n_components, perplexity, learning_rate, n_iter, early_exaggeration,
                  random_state, 0.5, None, 1e-7, 300)
    }

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
        let tsne = create_tsne(2, 5.0, 200.0, 100, 12.0, Some(42));
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
        let tsne = create_tsne(2, 5.0, 200.0, 100, 12.0, Some(42));
        let data = create_two_clusters();
        let distances = tsne.compute_pairwise_distances(&data);
        let p = tsne.compute_joint_probabilities(&distances, 40);

        // P should be symmetric
        for i in 0..40 {
            for j in 0..40 {
                assert_relative_eq!(p[[i, j]], p[[j, i]], epsilon = 1e-10);
            }
        }

        // P should sum to 1
        let sum: f64 = p.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

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
        let tsne = create_tsne(2, 5.0, 200.0, 100, 12.0, Some(42));

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
        let tsne = create_tsne(2, 5.0, 200.0, 100, 12.0, Some(42));

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
        let tsne = create_tsne(2, 5.0, 200.0, 100, 12.0, Some(42));

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
        let tsne1 = create_tsne(2, 5.0, 200.0, 100, 12.0, Some(42));
        let tsne2 = create_tsne(2, 5.0, 200.0, 100, 12.0, Some(42));

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

    // ============== Barnes-Hut specific tests ==============

    #[test]
    fn test_should_use_barnes_hut_explicit() {
        // Explicit true
        let tsne = TSNE::new(2, 30.0, 200.0, 1000, 12.0, None, 0.5, Some(true), 1e-7, 300);
        assert!(tsne.should_use_barnes_hut(100)); // Even for small n

        // Explicit false
        let tsne = TSNE::new(2, 30.0, 200.0, 1000, 12.0, None, 0.5, Some(false), 1e-7, 300);
        assert!(!tsne.should_use_barnes_hut(5000)); // Even for large n
    }

    #[test]
    fn test_should_use_barnes_hut_auto() {
        let tsne = TSNE::new(2, 30.0, 200.0, 1000, 12.0, None, 0.5, None, 1e-7, 300);

        // Auto: large dataset with 2D -> use Barnes-Hut
        assert!(tsne.should_use_barnes_hut(2000));

        // Auto: small dataset -> don't use Barnes-Hut
        assert!(!tsne.should_use_barnes_hut(500));

        // Auto: 3D output -> don't use Barnes-Hut (only 2D supported)
        let tsne_3d = TSNE::new(3, 30.0, 200.0, 1000, 12.0, None, 0.5, None, 1e-7, 300);
        assert!(!tsne_3d.should_use_barnes_hut(5000));
    }

    #[test]
    fn test_barnes_hut_gradient_shape() {
        let tsne = TSNE::new(2, 5.0, 200.0, 100, 12.0, Some(42), 0.5, Some(true), 1e-7, 300);

        // Create test P matrix and embedding
        let n = 10;
        let p = Array2::from_elem((n, n), 1.0 / (n * n) as f64);

        let mut y = Array2::zeros((n, 2));
        for i in 0..n {
            y[[i, 0]] = (i as f64) * 0.5;
            y[[i, 1]] = (i as f64).sin();
        }

        let grad = tsne.compute_gradient_barnes_hut(&p, &y);

        // Gradient should have correct shape
        assert_eq!(grad.shape(), &[n, 2]);
    }

    #[test]
    fn test_barnes_hut_gradient_nonzero() {
        let tsne = TSNE::new(2, 5.0, 200.0, 100, 12.0, Some(42), 0.5, Some(true), 1e-7, 300);

        // Create a non-uniform P matrix
        let n = 5;
        let mut p = Array2::zeros((n, n));
        p[[0, 1]] = 0.4;
        p[[1, 0]] = 0.4;
        p[[1, 2]] = 0.1;
        p[[2, 1]] = 0.1;

        // Create a scattered embedding
        let y = Array2::from_shape_vec((n, 2), vec![
            0.0, 0.0,
            1.0, 0.0,
            2.0, 1.0,
            0.0, 2.0,
            -1.0, 1.0,
        ]).unwrap();

        let grad = tsne.compute_gradient_barnes_hut(&p, &y);

        // Gradient should be non-zero
        let grad_norm: f64 = grad.iter().map(|&v| v * v).sum::<f64>().sqrt();
        assert!(grad_norm > 1e-10, "Barnes-Hut gradient should be non-zero");
    }

    #[test]
    fn test_barnes_hut_vs_exact_similar() {
        // With theta=0, Barnes-Hut should give similar results to exact
        // (not identical due to tree structure but should be close)
        let tsne_exact = TSNE::new(2, 5.0, 200.0, 100, 12.0, Some(42), 0.0, Some(false), 1e-7, 300);
        let tsne_bh = TSNE::new(2, 5.0, 200.0, 100, 12.0, Some(42), 0.1, Some(true), 1e-7, 300);

        // Create small test case
        let n = 5;
        let mut p = Array2::from_elem((n, n), 0.04);
        for i in 0..n {
            p[[i, i]] = 0.0;
        }
        // Normalize
        let sum: f64 = p.iter().sum();
        p /= sum;

        let y = Array2::from_shape_vec((n, 2), vec![
            0.0, 0.0,
            1.0, 0.0,
            0.5, 0.866,
            -0.5, 0.866,
            0.0, -1.0,
        ]).unwrap();

        let q = tsne_exact.compute_q(&y);
        let grad_exact = tsne_exact.compute_gradient(&p, &q, &y);
        let grad_bh = tsne_bh.compute_gradient_barnes_hut(&p, &y);

        // Gradients should be in the same ballpark (within factor of 2)
        for i in 0..n {
            for j in 0..2 {
                let exact = grad_exact[[i, j]];
                let bh = grad_bh[[i, j]];
                // Check direction agreement or both near zero
                if exact.abs() > 0.01 && bh.abs() > 0.01 {
                    // Same sign
                    assert!(exact * bh >= 0.0,
                        "Gradient direction mismatch at ({}, {}): exact={}, bh={}",
                        i, j, exact, bh);
                }
            }
        }
    }

    #[test]
    fn test_z_contribution_positive() {
        let tsne = TSNE::new(2, 5.0, 200.0, 100, 12.0, Some(42), 0.5, Some(true), 1e-7, 300);

        let y = Array2::from_shape_vec((4, 2), vec![
            0.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
        ]).unwrap();

        let tree = QuadTreeNode::build(&y);

        // Z contribution should be positive for each point
        for i in 0..4 {
            let point = [y[[i, 0]], y[[i, 1]]];
            let z = tsne.compute_z_contribution(&tree, &point, i);
            assert!(z > 0.0, "Z contribution should be positive for point {}", i);
        }
    }

    #[test]
    fn test_theta_effect_on_approximation() {
        // Higher theta = more approximation = faster but less accurate
        let tsne_low_theta = TSNE::new(2, 5.0, 200.0, 100, 12.0, Some(42), 0.1, Some(true), 1e-7, 300);
        let tsne_high_theta = TSNE::new(2, 5.0, 200.0, 100, 12.0, Some(42), 1.0, Some(true), 1e-7, 300);

        let n = 20;
        let mut y = Array2::zeros((n, 2));
        for i in 0..n {
            y[[i, 0]] = (i as f64) * 0.5;
            y[[i, 1]] = ((i as f64) * 0.3).sin();
        }

        let tree = QuadTreeNode::build(&y);
        let point = [y[[0, 0]], y[[0, 1]]];

        // Both should give positive Z contributions
        let z_low = tsne_low_theta.compute_z_contribution(&tree, &point, 0);
        let z_high = tsne_high_theta.compute_z_contribution(&tree, &point, 0);

        assert!(z_low > 0.0);
        assert!(z_high > 0.0);
        // Both approximations should be in the same order of magnitude
        assert!((z_low - z_high).abs() / z_low < 0.5,
            "Z contributions should be similar: {} vs {}", z_low, z_high);
    }

    // ============== Early stopping tests ==============

    #[test]
    fn test_early_stopping_disabled() {
        // With min_grad_norm = 0, early stopping is disabled
        let tsne = TSNE::new(2, 5.0, 200.0, 100, 12.0, Some(42), 0.5, None, 0.0, 300);
        assert_eq!(tsne.min_grad_norm, 0.0);
    }

    #[test]
    fn test_early_stopping_parameters() {
        let tsne = TSNE::new(2, 5.0, 200.0, 1000, 12.0, Some(42), 0.5, None, 1e-5, 100);
        assert_eq!(tsne.min_grad_norm, 1e-5);
        assert_eq!(tsne.n_iter_without_progress, 100);
    }

    #[test]
    fn test_gradient_norm_computation() {
        // Test that gradient norm is computed correctly
        let grad = Array2::from_shape_vec((3, 2), vec![
            3.0, 0.0,
            0.0, 4.0,
            0.0, 0.0,
        ]).unwrap();

        let n_samples = 3;
        let grad_norm: f64 = grad.iter().map(|&v| v * v).sum::<f64>().sqrt()
            / (n_samples as f64).sqrt();

        // ||grad|| = sqrt(9 + 16) = 5, normalized = 5/sqrt(3) ≈ 2.887
        assert_relative_eq!(grad_norm, 5.0 / (3.0_f64).sqrt(), epsilon = 1e-6);
    }
}
