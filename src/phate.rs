//! PHATE (Potential of Heat-diffusion for Affinity-based Trajectory Embedding)
//!
//! PHATE uses diffusion maps to compute potential distances, which better
//! capture global structure than standard distances.

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{Eigh, UPLO};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use ordered_float::OrderedFloat;

use crate::metrics_simd;
use crate::mds::{compute_distance_matrix, classical_mds};

/// PHATE dimensionality reduction
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct PHATE {
    n_components: usize,
    k: usize,           // k for k-NN
    t: usize,           // diffusion time
    decay: f64,         // alpha decay for kernel
    random_state: Option<u64>,
}

#[pymethods]
impl PHATE {
    #[new]
    #[pyo3(signature = (n_components=2, k=15, t=5, decay=2.0, random_state=None))]
    pub fn new(
        n_components: usize,
        k: usize,
        t: usize,
        decay: f64,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_components,
            k,
            t,
            decay,
            random_state,
        }
    }

    /// Fit and transform data using PHATE
    pub fn fit_transform<'py>(&self, py: Python<'py>, data: PyReadonlyArray2<f64>) 
        -> PyResult<Bound<'py, PyArray2<f64>>> 
    {
        let x = data.as_array();
        let n_samples = x.nrows();

        if self.k >= n_samples {
            return Err(PyValueError::new_err(format!(
                "k ({}) must be less than n_samples ({})",
                self.k, n_samples
            )));
        }

        // Convert to f32 for distance computation
        let x_f32: Vec<Vec<f32>> = x.rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect();

        // Compute pairwise distances
        let distances = compute_distance_matrix(&x_f32);

        // Step 1: Compute adaptive kernel bandwidth (local scaling)
        let sigmas = self.compute_local_sigmas(&distances, n_samples);

        // Step 2: Build affinity matrix with adaptive kernel
        let affinity = self.compute_affinity(&distances, &sigmas, n_samples);

        // Step 3: Compute diffusion operator (row-normalize to get Markov matrix)
        let diffusion_op = self.compute_diffusion_operator(&affinity, n_samples);

        // Step 4: Power the diffusion operator (diffusion time t)
        let diffused = self.power_matrix(&diffusion_op, self.t, n_samples);

        // Step 5: Compute potential distances
        let potential_distances = self.compute_potential_distances(&diffused, n_samples);

        // Step 6: Apply MDS to potential distances
        let embedding = classical_mds(&potential_distances, self.n_components)?;
        Ok(embedding.into_pyarray_bound(py))
    }
}

impl PHATE {
    fn compute_local_sigmas(&self, distances: &Array2<f64>, n_samples: usize) -> Vec<f64> {
        // Sigma for each point is the distance to the k-th neighbor
        (0..n_samples).into_par_iter().map(|i| {
            let mut heap: BinaryHeap<OrderedFloat<f64>> = BinaryHeap::new();
            
            for j in 0..n_samples {
                if i != j {
                    heap.push(OrderedFloat(-distances[[i, j]]));
                    if heap.len() > self.k {
                        heap.pop();
                    }
                }
            }

            // Return the k-th smallest distance
            -heap.peek().map(|v| v.into_inner()).unwrap_or(1.0)
        }).collect()
    }

    fn compute_affinity(&self, distances: &Array2<f64>, sigmas: &[f64], n_samples: usize) -> Array2<f64> {
        let mut affinity = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    let d = distances[[i, j]];
                    // Adaptive kernel: K(i,j) = exp(-d^alpha / (sigma_i * sigma_j)^(alpha/2))
                    // For alpha=2: exp(-d^2 / (sigma_i * sigma_j))
                    let sigma_ij = (sigmas[i] * sigmas[j]).powf(self.decay / 2.0);
                    if sigma_ij > 1e-10 {
                        affinity[[i, j]] = (-d.powf(self.decay) / sigma_ij).exp();
                    }
                }
            }
        }

        // Symmetrize
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let avg = (affinity[[i, j]] + affinity[[j, i]]) / 2.0;
                affinity[[i, j]] = avg;
                affinity[[j, i]] = avg;
            }
        }

        affinity
    }

    fn compute_diffusion_operator(&self, affinity: &Array2<f64>, n_samples: usize) -> Array2<f64> {
        let mut diffusion = affinity.clone();

        // Row-normalize to create Markov transition matrix
        for i in 0..n_samples {
            let row_sum: f64 = diffusion.row(i).sum();
            if row_sum > 1e-10 {
                for j in 0..n_samples {
                    diffusion[[i, j]] /= row_sum;
                }
            }
        }

        diffusion
    }

    fn power_matrix(&self, matrix: &Array2<f64>, power: usize, n_samples: usize) -> Array2<f64> {
        if power == 0 {
            return Array2::eye(n_samples);
        }
        if power == 1 {
            return matrix.clone();
        }

        // Use repeated squaring for efficiency
        let mut result = matrix.clone();
        let mut current = matrix.clone();
        let mut p = power - 1;

        while p > 0 {
            if p % 2 == 1 {
                result = result.dot(&current);
            }
            current = current.dot(&current);
            p /= 2;
        }

        result
    }

    fn compute_potential_distances(&self, diffused: &Array2<f64>, n_samples: usize) -> Array2<f64> {
        // Potential distance: sqrt(sum_k (P^t[i,k] - P^t[j,k])^2)
        // This is the Euclidean distance in the diffusion space
        let mut pot_dist = Array2::zeros((n_samples, n_samples));

        // Log transform for better separation (PHATE uses log potential)
        let log_diffused = diffused.mapv(|v| if v > 1e-10 { v.ln() } else { -23.0 }); // ln(1e-10) ≈ -23

        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let mut dist_sq = 0.0;
                for k in 0..n_samples {
                    let diff = log_diffused[[i, k]] - log_diffused[[j, k]];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                pot_dist[[i, j]] = dist;
                pot_dist[[j, i]] = dist;
            }
        }

        pot_dist
    }
}
