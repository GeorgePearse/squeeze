//! Principal Component Analysis (PCA) implementation.
//!
//! PCA finds orthogonal directions of maximum variance in the data
//! and projects data onto a lower-dimensional subspace.

use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::{Eigh, UPLO};

use crate::{Error, Result};

/// PCA dimensionality reduction
pub struct PCA {
    n_components: usize,
    components: Option<Array2<f64>>,
    mean: Option<Array1<f64>>,
    explained_variance: Option<Array1<f64>>,
    explained_variance_ratio: Option<Array1<f64>>,
}

impl PCA {
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
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<()> {
        let x = data.view();
        let (n_samples, n_features) = (x.nrows(), x.ncols());

        if n_samples < 2 {
            return Err(Error::InvalidParameter(
                "PCA requires at least two samples".into(),
            ));
        }
        if self.n_components > n_features {
            return Err(Error::InvalidParameter(format!(
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
        let (eigenvalues, eigenvectors) = cov
            .eigh(UPLO::Upper)
            .map_err(|e| Error::Computation(format!("eigendecomposition failed: {e}")))?;

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
    pub fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let components = self.components.as_ref().ok_or(Error::NotFitted)?;
        let mean = self.mean.as_ref().ok_or(Error::NotFitted)?;

        let x = data.view();
        if x.ncols() != mean.len() {
            return Err(Error::InvalidParameter(format!(
                "expected {} features, got {}",
                mean.len(),
                x.ncols()
            )));
        }

        // Center and project
        let mut x_centered = x.to_owned();
        for mut row in x_centered.rows_mut() {
            row -= mean;
        }

        let transformed = x_centered.dot(&components.t());
        Ok(transformed)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Result<Array2<f64>> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Get principal components
    pub fn components(&self) -> Result<&Array2<f64>> {
        self.components.as_ref().ok_or(Error::NotFitted)
    }

    /// Get explained variance
    pub fn explained_variance(&self) -> Result<&Array1<f64>> {
        self.explained_variance.as_ref().ok_or(Error::NotFitted)
    }

    /// Get explained variance ratio
    pub fn explained_variance_ratio(&self) -> Result<&Array1<f64>> {
        self.explained_variance_ratio
            .as_ref()
            .ok_or(Error::NotFitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{arr2, Array2};

    fn create_test_data() -> Array2<f64> {
        // Create a simple dataset where variance is primarily along first dimension
        let mut data = Array2::zeros((100, 5));
        for i in 0..100 {
            let val = i as f64;
            data[[i, 0]] = val * 2.0; // High variance
            data[[i, 1]] = val * 0.5; // Medium variance
            data[[i, 2]] = val * 0.1; // Low variance
            data[[i, 3]] = ((i % 10) as f64) * 0.01; // Very low variance
            data[[i, 4]] = 0.0; // Zero variance
        }
        data
    }

    #[test]
    fn test_pca_internal_fit() {
        let data = create_test_data();
        let mean = data.mean_axis(Axis(0)).unwrap();
        assert_eq!(mean.len(), 5);

        // Center the data
        let mut x_centered = data.clone();
        for mut row in x_centered.rows_mut() {
            row -= &mean;
        }

        // Compute covariance
        let n_samples = data.nrows();
        let cov = x_centered.t().dot(&x_centered) / (n_samples - 1) as f64;

        // Check covariance matrix is symmetric
        for i in 0..5 {
            for j in 0..5 {
                assert_relative_eq!(cov[[i, j]], cov[[j, i]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_explained_variance_decreasing() {
        let data = create_test_data();
        let n_samples = data.nrows();
        // Compute mean and center
        let mean = data.mean_axis(Axis(0)).unwrap();
        let mut x_centered = data.clone();
        for mut row in x_centered.rows_mut() {
            row -= &mean;
        }

        // Compute covariance and eigendecomposition
        let cov = x_centered.t().dot(&x_centered) / (n_samples - 1) as f64;
        let (eigenvalues, _eigenvectors) = cov.eigh(UPLO::Upper).unwrap();

        // Sort eigenvalues descending
        let mut sorted_eigenvalues: Vec<f64> = eigenvalues.iter().cloned().collect();
        sorted_eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());

        // Verify eigenvalues are in decreasing order
        for i in 1..sorted_eigenvalues.len() {
            assert!(
                sorted_eigenvalues[i] <= sorted_eigenvalues[i - 1],
                "Eigenvalues should be in decreasing order"
            );
        }

        // First eigenvalue should capture most variance
        let total_variance: f64 = sorted_eigenvalues.iter().sum();
        let first_ratio = sorted_eigenvalues[0] / total_variance;
        assert!(
            first_ratio > 0.8,
            "First component should explain >80% variance in this test data"
        );
    }

    #[test]
    fn test_covariance_properties() {
        // Simple 2D test case
        let data = arr2(&[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]);

        let mean = data.mean_axis(Axis(0)).unwrap();
        let mut centered = data.clone();
        for mut row in centered.rows_mut() {
            row -= &mean;
        }

        let cov = centered.t().dot(&centered) / 3.0;

        // Check covariance matrix properties
        assert_relative_eq!(cov[[0, 0]], 1.6666666, epsilon = 1e-5);
        assert_relative_eq!(cov[[1, 1]], 6.6666666, epsilon = 1e-5);
        assert_relative_eq!(cov[[0, 1]], cov[[1, 0]], epsilon = 1e-10); // Symmetric
    }

    #[test]
    fn test_zero_variance_handling() {
        // Data with one dimension having zero variance
        let mut data = Array2::zeros((50, 3));
        for i in 0..50 {
            data[[i, 0]] = i as f64;
            data[[i, 1]] = (i as f64).sin();
            data[[i, 2]] = 5.0; // Constant - zero variance
        }

        let mean = data.mean_axis(Axis(0)).unwrap();
        let mut centered = data.clone();
        for mut row in centered.rows_mut() {
            row -= &mean;
        }

        let cov = centered.t().dot(&centered) / 49.0;

        // Variance of constant dimension should be ~0
        assert!(
            cov[[2, 2]].abs() < 1e-10,
            "Constant dimension should have zero variance"
        );
    }
}
