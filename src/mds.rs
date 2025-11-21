//! Multidimensional Scaling (MDS) implementation.
//!
//! MDS finds a low-dimensional embedding that preserves pairwise distances
//! from the original high-dimensional space.

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{Eigh, UPLO};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray};
use rand::prelude::*;
use rand_distr::Normal;
use rayon::prelude::*;

use crate::metrics_simd;

/// Classical MDS (using eigendecomposition)
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct MDS {
    n_components: usize,
    metric: bool,
    n_iter: usize,
    random_state: Option<u64>,
    stress: Option<f64>,
}

#[pymethods]
impl MDS {
    #[new]
    #[pyo3(signature = (n_components=2, metric=true, n_iter=300, random_state=None))]
    pub fn new(
        n_components: usize,
        metric: bool,
        n_iter: usize,
        random_state: Option<u64>,
    ) -> Self {
        Self {
            n_components,
            metric,
            n_iter,
            random_state,
            stress: None,
        }
    }

    /// Fit and transform data using MDS
    pub fn fit_transform<'py>(&mut self, py: Python<'py>, data: PyReadonlyArray2<f64>) 
        -> PyResult<Bound<'py, PyArray2<f64>>> 
    {
        let x = data.as_array();
        let n_samples = x.nrows();

        // Convert to f32 for distance computation
        let x_f32: Vec<Vec<f32>> = x.rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect();

        // Compute pairwise distances
        let distances = compute_distance_matrix(&x_f32);

        // Apply MDS
        let embedding = if self.metric {
            self.metric_mds(&distances, n_samples)?
        } else {
            self.classical_mds_internal(&distances, n_samples)?
        };

        Ok(embedding.into_pyarray_bound(py))
    }

    /// Fit and transform from precomputed distance matrix
    pub fn fit_transform_from_distances<'py>(
        &mut self, 
        py: Python<'py>, 
        distances: PyReadonlyArray2<f64>
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let d = distances.as_array().to_owned();
        let n_samples = d.nrows();

        let embedding = if self.metric {
            self.metric_mds(&d, n_samples)?
        } else {
            self.classical_mds_internal(&d, n_samples)?
        };

        Ok(embedding.into_pyarray_bound(py))
    }

    #[getter]
    pub fn stress_(&self) -> Option<f64> {
        self.stress
    }
}

/// Standalone classical MDS function for internal use
pub fn classical_mds(distances: &Array2<f64>, n_components: usize) -> PyResult<Array2<f64>> {
    let n_samples = distances.nrows();
    
    // Double center the squared distance matrix
    let d_sq = distances.mapv(|v| v * v);
    
    // Compute row means, column means, and grand mean
    let row_mean = d_sq.mean_axis(Axis(1)).unwrap();
    let col_mean = d_sq.mean_axis(Axis(0)).unwrap();
    let grand_mean = d_sq.mean().unwrap();

    // B = -0.5 * (D^2 - row_mean - col_mean + grand_mean)
    let mut b = Array2::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in 0..n_samples {
            b[[i, j]] = -0.5 * (d_sq[[i, j]] - row_mean[i] - col_mean[j] + grand_mean);
        }
    }

    // Eigendecomposition
    let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = b.eigh(UPLO::Upper)
        .map_err(|e| PyValueError::new_err(format!("Eigendecomposition failed: {}", e)))?;

    // Sort by eigenvalue (descending)
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

    // Select top n_components positive eigenvalues
    let mut embedding = Array2::zeros((n_samples, n_components));
    for (comp, &idx) in indices.iter().take(n_components).enumerate() {
        let lambda = eigenvalues[idx].max(0.0).sqrt();
        for i in 0..n_samples {
            embedding[[i, comp]] = eigenvectors[[i, idx]] * lambda;
        }
    }

    Ok(embedding)
}

impl MDS {
    /// Classical MDS using eigendecomposition of the double-centered distance matrix
    fn classical_mds_internal(&mut self, distances: &Array2<f64>, n_samples: usize) -> PyResult<Array2<f64>> {
        // Double center the squared distance matrix
        let d_sq = distances.mapv(|v| v * v);
        
        // Compute row means, column means, and grand mean
        let row_mean = d_sq.mean_axis(Axis(1)).unwrap();
        let col_mean = d_sq.mean_axis(Axis(0)).unwrap();
        let grand_mean = d_sq.mean().unwrap();

        // B = -0.5 * (D^2 - row_mean - col_mean + grand_mean)
        let mut b = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                b[[i, j]] = -0.5 * (d_sq[[i, j]] - row_mean[i] - col_mean[j] + grand_mean);
            }
        }

        // Eigendecomposition
        let (eigenvalues, eigenvectors) = b.eigh(UPLO::Upper)
            .map_err(|e| PyValueError::new_err(format!("Eigendecomposition failed: {}", e)))?;

        // Sort by eigenvalue (descending)
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

        // Select top n_components positive eigenvalues
        let mut embedding = Array2::zeros((n_samples, self.n_components));
        for (comp, &idx) in indices.iter().take(self.n_components).enumerate() {
            let lambda = eigenvalues[idx].max(0.0).sqrt();
            for i in 0..n_samples {
                embedding[[i, comp]] = eigenvectors[[i, idx]] * lambda;
            }
        }

        // Compute stress
        self.stress = Some(self.compute_stress(distances, &embedding));

        Ok(embedding)
    }

    /// Metric MDS using SMACOF algorithm
    fn metric_mds(&mut self, distances: &Array2<f64>, n_samples: usize) -> PyResult<Array2<f64>> {
        // Initialize with classical MDS
        let mut embedding = self.classical_mds_internal(distances, n_samples)?;
        
        // SMACOF iterations
        for _ in 0..self.n_iter {
            let new_embedding = self.smacof_iteration(distances, &embedding, n_samples);
            
            // Check for convergence
            let diff: f64 = (&new_embedding - &embedding)
                .mapv(|v| v * v)
                .sum()
                .sqrt();
            
            embedding = new_embedding;
            
            if diff < 1e-6 {
                break;
            }
        }

        self.stress = Some(self.compute_stress(distances, &embedding));
        Ok(embedding)
    }

    fn smacof_iteration(&self, distances: &Array2<f64>, embedding: &Array2<f64>, n_samples: usize) -> Array2<f64> {
        // Compute current distances in embedding space
        let mut d_emb = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let mut dist_sq = 0.0;
                for k in 0..self.n_components {
                    let diff = embedding[[i, k]] - embedding[[j, k]];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                d_emb[[i, j]] = dist;
                d_emb[[j, i]] = dist;
            }
        }

        // Compute B matrix
        let mut b = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j && d_emb[[i, j]] > 1e-10 {
                    b[[i, j]] = -distances[[i, j]] / d_emb[[i, j]];
                }
            }
        }
        
        // Set diagonal
        for i in 0..n_samples {
            let row_sum: f64 = b.row(i).iter().sum();
            b[[i, i]] = -row_sum;
        }

        // New embedding = (1/n) * B * current_embedding
        let new_embedding = b.dot(embedding) / n_samples as f64;
        
        new_embedding
    }

    fn compute_stress(&self, target_distances: &Array2<f64>, embedding: &Array2<f64>) -> f64 {
        let n = embedding.nrows();
        let mut stress_num = 0.0;
        let mut stress_denom = 0.0;

        for i in 0..n {
            for j in (i + 1)..n {
                let mut dist_sq = 0.0;
                for k in 0..self.n_components {
                    let diff = embedding[[i, k]] - embedding[[j, k]];
                    dist_sq += diff * diff;
                }
                let d_emb = dist_sq.sqrt();
                let d_orig = target_distances[[i, j]];
                
                let diff = d_orig - d_emb;
                stress_num += diff * diff;
                stress_denom += d_orig * d_orig;
            }
        }

        if stress_denom > 0.0 {
            (stress_num / stress_denom).sqrt()
        } else {
            0.0
        }
    }
}

/// Compute pairwise distance matrix
pub fn compute_distance_matrix(data: &[Vec<f32>]) -> Array2<f64> {
    let n = data.len();
    let mut distances = Array2::zeros((n, n));

    let dists: Vec<Vec<f64>> = (0..n).into_par_iter().map(|i| {
        let mut row = vec![0.0; n];
        for j in 0..n {
            if i != j {
                let d = metrics_simd::euclidean(&data[i], &data[j]).unwrap_or(0.0) as f64;
                row[j] = d;
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
