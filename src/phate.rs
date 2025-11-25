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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn create_two_clusters() -> Array2<f64> {
        let mut data = Array2::zeros((40, 5));
        // Cluster 1: points near origin
        for i in 0..20 {
            for j in 0..5 {
                data[[i, j]] = (i as f64) * 0.01 + (j as f64) * 0.001;
            }
        }
        // Cluster 2: points offset by 10
        for i in 20..40 {
            for j in 0..5 {
                data[[i, j]] = 10.0 + (i as f64) * 0.01 + (j as f64) * 0.001;
            }
        }
        data
    }

    fn create_test_distances() -> Array2<f64> {
        // Create a simple 10x10 distance matrix
        let mut distances = Array2::zeros((10, 10));
        for i in 0..10 {
            for j in 0..10 {
                distances[[i, j]] = ((i as f64) - (j as f64)).abs();
            }
        }
        distances
    }

    #[test]
    fn test_local_sigmas_positive() {
        let phate = PHATE::new(2, 5, 5, 2.0, Some(42));
        let data = create_two_clusters();
        let x_f32: Vec<Vec<f32>> = data.rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect();
        let distances = compute_distance_matrix(&x_f32);

        let sigmas = phate.compute_local_sigmas(&distances, 40);

        // All sigmas should be positive
        for (i, &sigma) in sigmas.iter().enumerate() {
            assert!(sigma > 0.0, "Sigma at index {} should be positive, got {}", i, sigma);
        }
    }

    #[test]
    fn test_local_sigmas_length() {
        let phate = PHATE::new(2, 3, 5, 2.0, Some(42));
        let distances = create_test_distances();

        let sigmas = phate.compute_local_sigmas(&distances, 10);

        assert_eq!(sigmas.len(), 10, "Should have sigma for each sample");
    }

    #[test]
    fn test_affinity_matrix_symmetric() {
        let phate = PHATE::new(2, 5, 5, 2.0, Some(42));
        let data = create_two_clusters();
        let x_f32: Vec<Vec<f32>> = data.rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect();
        let distances = compute_distance_matrix(&x_f32);
        let sigmas = phate.compute_local_sigmas(&distances, 40);

        let affinity = phate.compute_affinity(&distances, &sigmas, 40);

        // Affinity should be symmetric
        for i in 0..40 {
            for j in 0..40 {
                assert_relative_eq!(
                    affinity[[i, j]],
                    affinity[[j, i]],
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_affinity_matrix_non_negative() {
        let phate = PHATE::new(2, 5, 5, 2.0, Some(42));
        let data = create_two_clusters();
        let x_f32: Vec<Vec<f32>> = data.rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect();
        let distances = compute_distance_matrix(&x_f32);
        let sigmas = phate.compute_local_sigmas(&distances, 40);

        let affinity = phate.compute_affinity(&distances, &sigmas, 40);

        // All affinities should be non-negative
        for &val in affinity.iter() {
            assert!(val >= 0.0, "Affinity should be non-negative");
        }
    }

    #[test]
    fn test_affinity_diagonal_zero() {
        let phate = PHATE::new(2, 5, 5, 2.0, Some(42));
        let data = create_two_clusters();
        let x_f32: Vec<Vec<f32>> = data.rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect();
        let distances = compute_distance_matrix(&x_f32);
        let sigmas = phate.compute_local_sigmas(&distances, 40);

        let affinity = phate.compute_affinity(&distances, &sigmas, 40);

        // Diagonal should be 0 (no self-affinity)
        for i in 0..40 {
            assert_eq!(affinity[[i, i]], 0.0, "Diagonal should be 0");
        }
    }

    #[test]
    fn test_diffusion_operator_row_sums() {
        let phate = PHATE::new(2, 5, 5, 2.0, Some(42));
        let data = create_two_clusters();
        let x_f32: Vec<Vec<f32>> = data.rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect();
        let distances = compute_distance_matrix(&x_f32);
        let sigmas = phate.compute_local_sigmas(&distances, 40);
        let affinity = phate.compute_affinity(&distances, &sigmas, 40);

        let diffusion = phate.compute_diffusion_operator(&affinity, 40);

        // Each row should sum to 1 (Markov matrix)
        for i in 0..40 {
            let row_sum: f64 = diffusion.row(i).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_diffusion_operator_non_negative() {
        let phate = PHATE::new(2, 5, 5, 2.0, Some(42));
        let data = create_two_clusters();
        let x_f32: Vec<Vec<f32>> = data.rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect();
        let distances = compute_distance_matrix(&x_f32);
        let sigmas = phate.compute_local_sigmas(&distances, 40);
        let affinity = phate.compute_affinity(&distances, &sigmas, 40);

        let diffusion = phate.compute_diffusion_operator(&affinity, 40);

        // All values should be non-negative
        for &val in diffusion.iter() {
            assert!(val >= 0.0, "Diffusion operator should be non-negative");
        }
    }

    #[test]
    fn test_potential_distances_symmetric() {
        let phate = PHATE::new(2, 5, 5, 2.0, Some(42));
        let data = create_two_clusters();
        let x_f32: Vec<Vec<f32>> = data.rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect();
        let distances = compute_distance_matrix(&x_f32);
        let sigmas = phate.compute_local_sigmas(&distances, 40);
        let affinity = phate.compute_affinity(&distances, &sigmas, 40);
        let diffusion = phate.compute_diffusion_operator(&affinity, 40);
        let diffused = phate.power_matrix(&diffusion, 5, 40);

        let potential = phate.compute_potential_distances(&diffused, 40);

        // Potential distances should be symmetric
        for i in 0..40 {
            for j in 0..40 {
                assert_relative_eq!(
                    potential[[i, j]],
                    potential[[j, i]],
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_potential_distances_diagonal_zero() {
        let phate = PHATE::new(2, 5, 5, 2.0, Some(42));
        let data = create_two_clusters();
        let x_f32: Vec<Vec<f32>> = data.rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect();
        let distances = compute_distance_matrix(&x_f32);
        let sigmas = phate.compute_local_sigmas(&distances, 40);
        let affinity = phate.compute_affinity(&distances, &sigmas, 40);
        let diffusion = phate.compute_diffusion_operator(&affinity, 40);
        let diffused = phate.power_matrix(&diffusion, 5, 40);

        let potential = phate.compute_potential_distances(&diffused, 40);

        // Diagonal should be 0
        for i in 0..40 {
            assert_relative_eq!(potential[[i, i]], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_power_matrix_identity() {
        let phate = PHATE::new(2, 5, 5, 2.0, Some(42));
        let matrix = Array2::from_shape_vec((3, 3), vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]).unwrap();

        // Identity^n = Identity
        let powered = phate.power_matrix(&matrix, 5, 3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(powered[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_power_matrix_zero_power() {
        let phate = PHATE::new(2, 5, 5, 2.0, Some(42));
        let matrix = Array2::from_shape_vec((3, 3), vec![
            0.5, 0.3, 0.2,
            0.1, 0.6, 0.3,
            0.2, 0.2, 0.6,
        ]).unwrap();

        // Any matrix^0 = Identity
        let powered = phate.power_matrix(&matrix, 0, 3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(powered[[i, j]], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_power_matrix_one_power() {
        let phate = PHATE::new(2, 5, 5, 2.0, Some(42));
        let matrix = Array2::from_shape_vec((3, 3), vec![
            0.5, 0.3, 0.2,
            0.1, 0.6, 0.3,
            0.2, 0.2, 0.6,
        ]).unwrap();

        // Matrix^1 = Matrix
        let powered = phate.power_matrix(&matrix, 1, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(powered[[i, j]], matrix[[i, j]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_power_matrix_square() {
        let phate = PHATE::new(2, 5, 5, 2.0, Some(42));
        let matrix = Array2::from_shape_vec((2, 2), vec![
            0.5, 0.5,
            0.5, 0.5,
        ]).unwrap();

        // For this matrix, M^2 = M (it's idempotent)
        let squared = phate.power_matrix(&matrix, 2, 2);
        let expected = matrix.dot(&matrix);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(
                    squared[[i, j]],
                    expected[[i, j]],
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_potential_distances_non_negative() {
        let phate = PHATE::new(2, 5, 5, 2.0, Some(42));
        let data = create_two_clusters();
        let x_f32: Vec<Vec<f32>> = data.rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect();
        let distances = compute_distance_matrix(&x_f32);
        let sigmas = phate.compute_local_sigmas(&distances, 40);
        let affinity = phate.compute_affinity(&distances, &sigmas, 40);
        let diffusion = phate.compute_diffusion_operator(&affinity, 40);
        let diffused = phate.power_matrix(&diffusion, 5, 40);

        let potential = phate.compute_potential_distances(&diffused, 40);

        // All distances should be non-negative
        for &val in potential.iter() {
            assert!(val >= 0.0, "Potential distance should be non-negative");
        }
    }
}
