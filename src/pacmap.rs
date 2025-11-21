//! PaCMAP (Pairwise Controlled Manifold Approximation)
//!
//! PaCMAP uses three types of pairs (near, mid-near, far) to preserve
//! both local and global structure during optimization.

use ndarray::{Array2, Axis};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray};
use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::Normal;
use std::collections::BinaryHeap;
use ordered_float::OrderedFloat;

use crate::metrics_simd;
use crate::mds::compute_distance_matrix;

/// PaCMAP dimensionality reduction
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct PaCMAP {
    n_components: usize,
    n_neighbors: usize,     // Near pairs
    mn_ratio: f64,          // Mid-near ratio (multiplier for n_neighbors)
    fp_ratio: f64,          // Far pair ratio
    n_iter: usize,
    learning_rate: f64,
    random_state: Option<u64>,
}

#[pymethods]
impl PaCMAP {
    #[new]
    #[pyo3(signature = (n_components=2, n_neighbors=10, mn_ratio=0.5, fp_ratio=2.0, n_iter=450, learning_rate=1.0, random_state=None))]
    pub fn new(
        n_components: usize,
        n_neighbors: usize,
        mn_ratio: f64,
        fp_ratio: f64,
        n_iter: usize,
        learning_rate: f64,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_components,
            n_neighbors,
            mn_ratio,
            fp_ratio,
            n_iter,
            learning_rate,
            random_state,
        }
    }

    /// Fit and transform data using PaCMAP
    pub fn fit_transform<'py>(&self, py: Python<'py>, data: PyReadonlyArray2<f64>) 
        -> PyResult<Bound<'py, PyArray2<f64>>> 
    {
        let x = data.as_array();
        let n_samples = x.nrows();

        if self.n_neighbors >= n_samples {
            return Err(PyValueError::new_err(format!(
                "n_neighbors ({}) must be less than n_samples ({})",
                self.n_neighbors, n_samples
            )));
        }

        // Convert to f32 for distance computation
        let x_f32: Vec<Vec<f32>> = x.rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect();

        // Compute pairwise distances
        let distances = compute_distance_matrix(&x_f32);

        // Generate three types of pairs
        let (near_pairs, mid_near_pairs, far_pairs) = self.generate_pairs(&distances, n_samples);

        // Initialize embedding
        let mut embedding = self.initialize_embedding(n_samples)?;

        // Three-phase optimization
        self.optimize(&mut embedding, &near_pairs, &mid_near_pairs, &far_pairs, n_samples);

        Ok(embedding.into_pyarray_bound(py))
    }
}

impl PaCMAP {
    fn generate_pairs(&self, distances: &Array2<f64>, n_samples: usize) 
        -> (Vec<(usize, usize, f64)>, Vec<(usize, usize, f64)>, Vec<(usize, usize)>) 
    {
        let mut rng: StdRng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_seed(rand::random()),
        };

        let n_mn = (self.n_neighbors as f64 * self.mn_ratio) as usize;
        let n_fp = (self.n_neighbors as f64 * self.fp_ratio) as usize;

        let mut near_pairs = Vec::new();
        let mut mid_near_pairs = Vec::new();
        let mut far_pairs = Vec::new();

        for i in 0..n_samples {
            // Sort distances to find neighbors
            let mut dist_idx: Vec<(f64, usize)> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| (distances[[i, j]], j))
                .collect();
            dist_idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Near pairs: k nearest neighbors
            for &(d, j) in dist_idx.iter().take(self.n_neighbors) {
                near_pairs.push((i, j, d));
            }

            // Mid-near pairs: neighbors around 6*k position
            let mid_start = (6 * self.n_neighbors).min(dist_idx.len());
            let mid_end = (mid_start + n_mn).min(dist_idx.len());
            for &(d, j) in dist_idx.iter().skip(mid_start).take(mid_end - mid_start) {
                mid_near_pairs.push((i, j, d));
            }

            // Far pairs: randomly sampled
            for _ in 0..n_fp {
                let j = loop {
                    let candidate = rng.gen_range(0..n_samples);
                    if candidate != i {
                        break candidate;
                    }
                };
                far_pairs.push((i, j));
            }
        }

        (near_pairs, mid_near_pairs, far_pairs)
    }

    fn initialize_embedding(&self, n_samples: usize) -> PyResult<Array2<f64>> {
        let mut rng: StdRng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_seed(rand::random()),
        };
        
        let normal = Normal::new(0.0, 1e-4).unwrap();
        let mut embedding = Array2::zeros((n_samples, self.n_components));
        
        for mut row in embedding.rows_mut() {
            for v in row.iter_mut() {
                *v = normal.sample(&mut rng) * 100.0; // PaCMAP uses larger initial scale
            }
        }

        Ok(embedding)
    }

    fn optimize(
        &self,
        embedding: &mut Array2<f64>,
        near_pairs: &[(usize, usize, f64)],
        mid_near_pairs: &[(usize, usize, f64)],
        far_pairs: &[(usize, usize)],
        n_samples: usize
    ) {
        // PaCMAP uses three phases with different weight schedules
        // Phase 1 (0-100): Focus on mid-near and far
        // Phase 2 (100-200): Transition
        // Phase 3 (200-450): Focus on near

        for iter in 0..self.n_iter {
            let mut grad = Array2::zeros((n_samples, self.n_components));

            // Compute phase-dependent weights
            let (w_near, w_mn, w_fp) = self.get_weights(iter);

            // Near pair gradients: attract
            for &(i, j, d_orig) in near_pairs {
                let d_emb = self.embedding_distance(embedding, i, j);
                let d_emb_safe = d_emb.max(1e-10);
                
                // Attractive force: minimize (d_emb^2) / (10 + d_emb^2)
                let coeff = w_near * 2.0 * 10.0 / ((10.0 + d_emb * d_emb).powi(2));
                
                for c in 0..self.n_components {
                    let diff = embedding[[i, c]] - embedding[[j, c]];
                    grad[[i, c]] += coeff * diff;
                    grad[[j, c]] -= coeff * diff;
                }
            }

            // Mid-near pair gradients: attract then repel
            for &(i, j, d_orig) in mid_near_pairs {
                let d_emb = self.embedding_distance(embedding, i, j);
                
                // Attractive force similar to near pairs
                let coeff = w_mn * 2.0 * 10000.0 / ((10000.0 + d_emb * d_emb).powi(2));
                
                for c in 0..self.n_components {
                    let diff = embedding[[i, c]] - embedding[[j, c]];
                    grad[[i, c]] += coeff * diff;
                    grad[[j, c]] -= coeff * diff;
                }
            }

            // Far pair gradients: repel
            for &(i, j) in far_pairs {
                let d_emb = self.embedding_distance(embedding, i, j);
                let d_emb_safe = d_emb.max(1e-10);
                
                // Repulsive force: maximize 1 / (1 + d_emb^2)
                // Gradient pushes points apart
                let coeff = w_fp * 2.0 / ((1.0 + d_emb * d_emb).powi(2));
                
                for c in 0..self.n_components {
                    let diff = embedding[[i, c]] - embedding[[j, c]];
                    grad[[i, c]] -= coeff * diff;
                    grad[[j, c]] += coeff * diff;
                }
            }

            // Apply gradient with learning rate decay
            let lr = self.learning_rate / (1.0 + iter as f64 / 100.0);
            *embedding = &*embedding - lr * &grad;

            // Center embedding
            let mean = embedding.mean_axis(Axis(0)).unwrap();
            for mut row in embedding.rows_mut() {
                row -= &mean;
            }
        }
    }

    fn get_weights(&self, iter: usize) -> (f64, f64, f64) {
        // PaCMAP weight schedule
        if iter < 100 {
            // Phase 1: Focus on structure
            let t = iter as f64 / 100.0;
            (2.0, 3.0 * (1.0 - t) + 3.0 * t, 1.0)
        } else if iter < 200 {
            // Phase 2: Transition
            let t = (iter - 100) as f64 / 100.0;
            (3.0 * (1.0 - t) + 1.0 * t, 3.0 * (1.0 - t) + 0.0 * t, 1.0)
        } else {
            // Phase 3: Fine-tune local structure
            (1.0, 0.0, 1.0)
        }
    }

    fn embedding_distance(&self, embedding: &Array2<f64>, i: usize, j: usize) -> f64 {
        let mut dist_sq = 0.0;
        for c in 0..self.n_components {
            let diff = embedding[[i, c]] - embedding[[j, c]];
            dist_sq += diff * diff;
        }
        dist_sq.sqrt()
    }
}
