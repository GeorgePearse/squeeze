//! Principal Component Analysis (PCA) implementation.
//!
//! PCA finds orthogonal directions of maximum variance in the data
//! and projects data onto a lower-dimensional subspace.

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{Eigh, UPLO};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray};
use rayon::prelude::*;

/// PCA dimensionality reduction
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct PCA {
    n_components: usize,
    components: Option<Array2<f64>>,
    mean: Option<Array1<f64>>,
    explained_variance: Option<Array1<f64>>,
    explained_variance_ratio: Option<Array1<f64>>,
}

#[pymethods]
impl PCA {
    #[new]
    #[pyo3(signature = (n_components=2))]
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            components: None,
            mean: None,
            explained_variance: None,
            explained_variance_ratio: None,
        }
    }

    /// Fit PCA to the data
    pub fn fit(&mut self, py: Python<'_>, data: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x = data.as_array();
        let (n_samples, n_features) = (x.nrows(), x.ncols());

        if self.n_components > n_features {
            return Err(PyValueError::new_err(format!(
                "n_components ({}) cannot exceed n_features ({})",
                self.n_components, n_features
            )));
        }

        // Compute mean
        let mean = x.mean_axis(Axis(0)).unwrap();
        
        // Center the data
        let mut x_centered = x.to_owned();
        for mut row in x_centered.rows_mut() {
            row -= &mean;
        }

        // Compute covariance matrix: X^T X / (n - 1)
        let cov = x_centered.t().dot(&x_centered) / (n_samples - 1) as f64;

        // Eigendecomposition
        let (eigenvalues, eigenvectors) = cov.eigh(UPLO::Upper)
            .map_err(|e| PyValueError::new_err(format!("Eigendecomposition failed: {}", e)))?;

        // Sort by eigenvalue (descending)
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

        // Select top n_components
        let mut components = Array2::zeros((self.n_components, n_features));
        let mut explained_variance = Array1::zeros(self.n_components);
        
        for (i, &idx) in indices.iter().take(self.n_components).enumerate() {
            components.row_mut(i).assign(&eigenvectors.column(idx));
            explained_variance[i] = eigenvalues[idx].max(0.0);
        }

        // Compute explained variance ratio
        let total_variance: f64 = eigenvalues.iter().filter(|&&v| v > 0.0).sum();
        let explained_variance_ratio = &explained_variance / total_variance;

        self.components = Some(components);
        self.mean = Some(mean);
        self.explained_variance = Some(explained_variance);
        self.explained_variance_ratio = Some(explained_variance_ratio);

        Ok(())
    }

    /// Transform data using fitted PCA
    pub fn transform<'py>(&self, py: Python<'py>, data: PyReadonlyArray2<f64>) 
        -> PyResult<Bound<'py, PyArray2<f64>>> 
    {
        let components = self.components.as_ref()
            .ok_or_else(|| PyValueError::new_err("PCA not fitted. Call fit() first."))?;
        let mean = self.mean.as_ref().unwrap();

        let x = data.as_array();
        
        // Center and project
        let mut x_centered = x.to_owned();
        for mut row in x_centered.rows_mut() {
            row -= mean;
        }

        let transformed = x_centered.dot(&components.t());
        Ok(transformed.into_pyarray_bound(py))
    }

    /// Fit and transform in one step
    pub fn fit_transform<'py>(&mut self, py: Python<'py>, data: PyReadonlyArray2<f64>) 
        -> PyResult<Bound<'py, PyArray2<f64>>> 
    {
        self.fit(py, data.clone())?;
        self.transform(py, data)
    }

    /// Get principal components
    #[getter]
    pub fn components_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let components = self.components.as_ref()
            .ok_or_else(|| PyValueError::new_err("PCA not fitted"))?;
        Ok(components.clone().into_pyarray_bound(py))
    }

    /// Get explained variance
    #[getter]
    pub fn explained_variance_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let ev = self.explained_variance.as_ref()
            .ok_or_else(|| PyValueError::new_err("PCA not fitted"))?;
        Ok(ev.clone().into_pyarray_bound(py))
    }

    /// Get explained variance ratio
    #[getter]
    pub fn explained_variance_ratio_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let evr = self.explained_variance_ratio.as_ref()
            .ok_or_else(|| PyValueError::new_err("PCA not fitted"))?;
        Ok(evr.clone().into_pyarray_bound(py))
    }
}
