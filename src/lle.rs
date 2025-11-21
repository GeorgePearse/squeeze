//! Locally Linear Embedding (LLE) implementation.
//!
//! LLE finds a low-dimensional embedding by preserving local linear
//! relationships between neighboring points.

use ndarray::{Array1, Array2, Axis, s};
use ndarray_linalg::{Eigh, Solve, UPLO};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use ordered_float::OrderedFloat;

use crate::metrics_simd;
use crate::mds::compute_distance_matrix;

/// Locally Linear Embedding
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct LLE {
    n_components: usize,
    n_neighbors: usize,
    reg: f64,
}

#[pymethods]
impl LLE {
    #[new]
    #[pyo3(signature = (n_components=2, n_neighbors=12, reg=1e-3))]
    pub fn new(n_components: usize, n_neighbors: usize, reg: f64) -> Self {
        Self {
            n_components,
            n_neighbors,
            reg,
        }
    }

    /// Fit and transform data using LLE
    pub fn fit_transform<'py>(&self, py: Python<'py>, data: PyReadonlyArray2<f64>) 
        -> PyResult<Bound<'py, PyArray2<f64>>> 
    {
        let x = data.as_array();
        let n_samples = x.nrows();
        let n_features = x.ncols();

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

        // Compute pairwise distances and find k-NN
        let distances = compute_distance_matrix(&x_f32);
        let neighbors = self.find_neighbors(&distances, n_samples);

        // Step 1: Compute reconstruction weights
        let weights = self.compute_weights(&x.to_owned(), &neighbors, n_samples)?;

        // Step 2: Compute embedding via eigendecomposition of (I-W)^T(I-W)
        let embedding = self.compute_embedding(&weights, n_samples)?;

        Ok(embedding.into_pyarray_bound(py))
    }
}

impl LLE {
    fn find_neighbors(&self, distances: &Array2<f64>, n_samples: usize) -> Vec<Vec<usize>> {
        (0..n_samples).into_par_iter().map(|i| {
            let mut heap: BinaryHeap<(OrderedFloat<f64>, usize)> = BinaryHeap::new();
            
            for j in 0..n_samples {
                if i != j {
                    heap.push((OrderedFloat(-distances[[i, j]]), j));
                    if heap.len() > self.n_neighbors {
                        heap.pop();
                    }
                }
            }

            heap.into_iter().map(|(_, j)| j).collect()
        }).collect()
    }

    fn compute_weights(&self, x: &Array2<f64>, neighbors: &[Vec<usize>], n_samples: usize) 
        -> PyResult<Array2<f64>> 
    {
        let n_features = x.ncols();
        let mut weights = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let k = neighbors[i].len();
            
            // Build local covariance matrix
            // C[j,k] = (x[i] - x[neighbors[j]]) . (x[i] - x[neighbors[k]])
            let mut c = Array2::zeros((k, k));
            
            for (j_idx, &j) in neighbors[i].iter().enumerate() {
                for (l_idx, &l) in neighbors[i].iter().enumerate() {
                    let mut dot = 0.0;
                    for f in 0..n_features {
                        let diff_j = x[[i, f]] - x[[j, f]];
                        let diff_l = x[[i, f]] - x[[l, f]];
                        dot += diff_j * diff_l;
                    }
                    c[[j_idx, l_idx]] = dot;
                }
            }

            // Regularization
            let trace: f64 = (0..k).map(|j| c[[j, j]]).sum();
            let reg_val = self.reg * trace / k as f64;
            for j in 0..k {
                c[[j, j]] += reg_val;
            }

            // Solve C * w = 1 for weights
            let ones = Array1::ones(k);
            let w = match c.solve(&ones) {
                Ok(w) => w,
                Err(_) => {
                    // Fallback: uniform weights
                    Array1::from_elem(k, 1.0 / k as f64)
                }
            };

            // Normalize weights
            let w_sum: f64 = w.sum();
            for (j_idx, &j) in neighbors[i].iter().enumerate() {
                weights[[i, j]] = w[j_idx] / w_sum;
            }
        }

        Ok(weights)
    }

    fn compute_embedding(&self, weights: &Array2<f64>, n_samples: usize) -> PyResult<Array2<f64>> {
        // Compute M = (I - W)^T (I - W)
        let mut m = Array2::zeros((n_samples, n_samples));
        
        // M = I - W - W^T + W^T W
        for i in 0..n_samples {
            m[[i, i]] += 1.0;
            for j in 0..n_samples {
                m[[i, j]] -= weights[[i, j]];
                m[[j, i]] -= weights[[i, j]];
                for k in 0..n_samples {
                    m[[i, j]] += weights[[k, i]] * weights[[k, j]];
                }
            }
        }

        // Eigendecomposition - find smallest non-zero eigenvalues
        let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = m.eigh(UPLO::Upper)
            .map_err(|e| PyValueError::new_err(format!("Eigendecomposition failed: {}", e)))?;

        // Sort by eigenvalue (ascending) and skip the first (zero eigenvalue)
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

        // Select eigenvectors corresponding to smallest non-zero eigenvalues
        // Skip index 0 (zero eigenvalue corresponding to constant eigenvector)
        let mut embedding = Array2::zeros((n_samples, self.n_components));
        for (comp, &idx) in indices.iter().skip(1).take(self.n_components).enumerate() {
            for i in 0..n_samples {
                embedding[[i, comp]] = eigenvectors[[i, idx]];
            }
        }

        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_find_neighbors() {
        let lle = LLE::new(2, 2, 1e-3);
        
        let distances = Array2::from_shape_vec((4, 4), vec![
            0.0, 1.0, 2.0, 3.0,
            1.0, 0.0, 1.0, 2.0,
            2.0, 1.0, 0.0, 1.0,
            3.0, 2.0, 1.0, 0.0,
        ]).unwrap();
        
        let neighbors = lle.find_neighbors(&distances, 4);
        
        // Each point should have 2 neighbors
        for n in &neighbors {
            assert_eq!(n.len(), 2);
        }
        
        // Point 0's neighbors should be 1 and 2 (closest)
        assert!(neighbors[0].contains(&1));
        assert!(neighbors[0].contains(&2));
    }

    #[test]
    fn test_reconstruction_weights_sum_to_one() {
        let lle = LLE::new(2, 3, 1e-3);
        
        // Simple test data
        let x = Array2::from_shape_vec((5, 2), vec![
            0.0, 0.0,
            1.0, 0.0,
            2.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
        ]).unwrap();
        
        let neighbors = vec![
            vec![1, 3, 4],
            vec![0, 2, 3],
            vec![1, 3, 4],
            vec![0, 1, 2],
            vec![0, 1, 3],
        ];
        
        let weights = lle.compute_weights(&x, &neighbors, 5).unwrap();
        
        // Each row should sum to 1 (weights for reconstructing each point)
        for i in 0..5 {
            let row_sum: f64 = (0..5).map(|j| weights[[i, j]]).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_embedding_eigendecomposition() {
        let lle = LLE::new(2, 2, 1e-3);
        
        // Simple weight matrix
        let weights = Array2::from_shape_vec((3, 3), vec![
            0.0, 0.5, 0.5,
            0.5, 0.0, 0.5,
            0.5, 0.5, 0.0,
        ]).unwrap();
        
        let embedding = lle.compute_embedding(&weights, 3).unwrap();
        
        // Should return 2D embedding
        assert_eq!(embedding.shape(), &[3, 2]);
    }
}
