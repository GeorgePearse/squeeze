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
        let mut p = Array2::zeros((n_samples, n_samples));

        // Compute conditional probabilities P(j|i) using binary search for sigma
        for i in 0..n_samples {
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
                        p[[i, j]] = pij;
                        sum_p += pij;
                    }
                }

                // Normalize
                if sum_p > 1e-10 {
                    for j in 0..n_samples {
                        p[[i, j]] /= sum_p;
                    }
                }

                // Compute entropy
                let mut entropy = 0.0;
                for j in 0..n_samples {
                    if p[[i, j]] > 1e-10 {
                        entropy -= p[[i, j]] * p[[i, j]].ln();
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
        let mut grad = Array2::zeros((n, self.n_components));

        // Compute (P - Q) * kernel
        let pq = p - q;
        
        for i in 0..n {
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
                        grad[[i, k]] += mult * (y[[i, k]] - y[[j, k]]);
                    }
                }
            }
        }

        grad
    }
}
