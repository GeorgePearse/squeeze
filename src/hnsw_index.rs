use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;

use crate::metrics;

/// Brute-force approximate nearest neighbor index
///
/// This is a simplified implementation using brute-force k-NN search.
/// It provides correct results but without the logarithmic scaling of HNSW.
/// This can be optimized to use HNSW or other algorithms later.
#[pyclass]
pub struct HnswIndex {
    /// Copy of the data for searching
    data: Vec<Vec<f32>>,
    /// Number of neighbors to return
    n_neighbors: usize,
    /// Distance metric name
    metric: String,
    /// Whether the metric is angular (cosine/correlation)
    is_angular: bool,
    /// Cached neighbor graph (for neighbor_graph property)
    neighbor_graph_cache: Option<(Vec<Vec<i64>>, Vec<Vec<f32>>)>,
}

#[pymethods]
impl HnswIndex {
    /// Create a new nearest neighbor index
    ///
    /// Parameters
    /// ----------
    /// data : ndarray of float32, shape (n_samples, n_features)
    ///     The data to index
    /// n_neighbors : int
    ///     Number of neighbors to return
    /// metric : str
    ///     Distance metric: 'euclidean', 'manhattan', 'cosine', etc.
    /// m : int
    ///     HNSW M parameter (unused in brute-force version, for API compatibility)
    /// ef_construction : int
    ///     HNSW ef_construction parameter (unused in brute-force version, for API compatibility)
    #[new]
    fn new(
        data: PyReadonlyArray2<f32>,
        n_neighbors: usize,
        metric: String,
        _m: usize,
        _ef_construction: usize,
    ) -> PyResult<Self> {
        let shape = data.shape();
        let (n_samples, n_features) = (shape[0], shape[1]);

        // Validate inputs
        if n_samples == 0 {
            return Err(PyValueError::new_err("data must have at least one sample"));
        }
        if n_neighbors == 0 {
            return Err(PyValueError::new_err("n_neighbors must be at least 1"));
        }
        if n_neighbors > n_samples {
            return Err(PyValueError::new_err(
                format!("n_neighbors ({}) cannot exceed n_samples ({})", n_neighbors, n_samples),
            ));
        }

        // Convert numpy array to Vec<Vec<f32>>
        let data_slice = data.as_slice().map_err(|e|
            PyValueError::new_err(format!("Failed to get array slice: {}", e))
        )?;

        let mut data_vec = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let row_start = i * n_features;
            let row_end = row_start + n_features;
            data_vec.push(data_slice[row_start..row_end].to_vec());
        }

        let is_angular = metric == "cosine" || metric == "correlation";

        Ok(Self {
            data: data_vec,
            n_neighbors,
            metric,
            is_angular,
            neighbor_graph_cache: None,
        })
    }

    /// Query the index for k nearest neighbors
    ///
    /// Parameters
    /// ----------
    /// queries : ndarray of float32, shape (n_queries, n_features)
    ///     Query points
    /// k : int
    ///     Number of neighbors to return
    /// ef : int
    ///     Search parameter (unused in brute-force version, for API compatibility)
    ///
    /// Returns
    /// -------
    /// indices : ndarray of int64, shape (n_queries, k)
    ///     Indices of nearest neighbors
    /// distances : ndarray of float32, shape (n_queries, k)
    ///     Distances to nearest neighbors
    fn query<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        _ef: usize,
    ) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
        let shape = queries.shape();
        let (n_queries, n_features) = (shape[0], shape[1]);

        if n_features != self.data[0].len() {
            return Err(PyValueError::new_err(
                format!("query features ({}) don't match data features ({})",
                    n_features, self.data[0].len()),
            ));
        }

        // Convert queries to Vec<Vec<f32>>
        let queries_slice = queries.as_slice().map_err(|e|
            PyValueError::new_err(format!("Failed to get queries slice: {}", e))
        )?;

        let mut queries_vec = Vec::with_capacity(n_queries);
        for i in 0..n_queries {
            let row_start = i * n_features;
            let row_end = row_start + n_features;
            queries_vec.push(queries_slice[row_start..row_end].to_vec());
        }

        // Perform queries (sequential for now, can parallelize later)
        let mut all_indices = Vec::with_capacity(n_queries);
        let mut all_distances = Vec::with_capacity(n_queries);

        for query_vec in &queries_vec {
            // Find k nearest neighbors
            let mut neighbors = Vec::with_capacity(self.data.len());

            for (i, data_vec) in self.data.iter().enumerate() {
                let dist = self.compute_distance(query_vec, data_vec);
                neighbors.push((i as i64, dist));
            }

            // Sort by distance (ascending)
            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Extract top k neighbors
            let mut indices = Vec::with_capacity(k);
            let mut distances = Vec::with_capacity(k);

            for (i, neighbor) in neighbors.iter().take(k).enumerate() {
                indices.push(neighbor.0);
                distances.push(neighbor.1);
            }

            // Pad with -1 and NaN if fewer than k neighbors
            while indices.len() < k {
                indices.push(-1);
                distances.push(f32::NAN);
            }

            all_indices.push(indices);
            all_distances.push(distances);
        }

        // Convert to numpy arrays
        let indices_array = PyArray2::from_vec2(py, &all_indices)?.to_owned();
        let distances_array = PyArray2::from_vec2(py, &all_distances)?.to_owned();

        Ok((indices_array, distances_array))
    }

    /// Get the k-nearest neighbor graph for all indexed points
    ///
    /// Returns
    /// -------
    /// indices : ndarray of int64, shape (n_samples, n_neighbors)
    ///     Indices of nearest neighbors
    /// distances : ndarray of float32, shape (n_samples, n_neighbors)
    ///     Distances to nearest neighbors
    fn neighbor_graph<'py>(&mut self, py: Python<'py>)
        -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)>
    {
        // Return cached result if available
        if let Some((indices, distances)) = &self.neighbor_graph_cache {
            let indices_py = PyArray2::from_vec2(py, indices)?.to_owned();
            let distances_py = PyArray2::from_vec2(py, distances)?.to_owned();
            return Ok((indices_py, distances_py));
        }

        // Compute neighbor graph for all points
        let mut all_indices = Vec::with_capacity(self.data.len());
        let mut all_distances = Vec::with_capacity(self.data.len());

        for query_vec in &self.data {
            // Find k nearest neighbors for this point
            let mut neighbors = Vec::with_capacity(self.data.len());

            for (i, data_vec) in self.data.iter().enumerate() {
                let dist = self.compute_distance(query_vec, data_vec);
                neighbors.push((i as i64, dist));
            }

            // Sort by distance (ascending)
            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Extract top n_neighbors neighbors
            let mut indices = Vec::with_capacity(self.n_neighbors);
            let mut distances = Vec::with_capacity(self.n_neighbors);

            for (i, neighbor) in neighbors.iter().take(self.n_neighbors).enumerate() {
                indices.push(neighbor.0);
                distances.push(neighbor.1);
            }

            // Pad if necessary
            while indices.len() < self.n_neighbors {
                indices.push(-1);
                distances.push(f32::NAN);
            }

            all_indices.push(indices);
            all_distances.push(distances);
        }

        // Cache the result
        self.neighbor_graph_cache = Some((all_indices.clone(), all_distances.clone()));

        // Convert to numpy arrays
        let indices_array = PyArray2::from_vec2(py, &all_indices)?.to_owned();
        let distances_array = PyArray2::from_vec2(py, &all_distances)?.to_owned();

        Ok((indices_array, distances_array))
    }

    /// Prepare the index for querying (no-op in brute-force version, exists for API compatibility)
    fn prepare(&mut self) -> PyResult<()> {
        Ok(())
    }

    /// Update the index with new data (insert new points)
    ///
    /// Parameters
    /// ----------
    /// new_data : ndarray of float32, shape (n_new, n_features)
    ///     New data points to add to the index
    fn update(&mut self, new_data: PyReadonlyArray2<f32>) -> PyResult<()> {
        let shape = new_data.shape();
        let (n_new, n_features) = (shape[0], shape[1]);

        if n_features != self.data[0].len() {
            return Err(PyValueError::new_err(
                format!("new_data features ({}) don't match index features ({})",
                    n_features, self.data[0].len()),
            ));
        }

        // Convert new data to Vec<Vec<f32>>
        let new_data_slice = new_data.as_slice().map_err(|e|
            PyValueError::new_err(format!("Failed to get new_data slice: {}", e))
        )?;

        let mut new_data_vec = Vec::with_capacity(n_new);
        for i in 0..n_new {
            let row_start = i * n_features;
            let row_end = row_start + n_features;
            new_data_vec.push(new_data_slice[row_start..row_end].to_vec());
        }

        // Add new data to the index
        self.data.extend(new_data_vec);

        // Clear cache
        self.neighbor_graph_cache = None;

        Ok(())
    }

    /// Get the _angular_trees property (for API compatibility)
    #[getter]
    fn _angular_trees(&self) -> bool {
        self.is_angular
    }

    /// Get the metric name
    #[getter]
    fn metric(&self) -> String {
        self.metric.clone()
    }

    /// Get the number of samples in the index
    #[getter]
    fn n_samples(&self) -> usize {
        self.data.len()
    }

    /// Get the number of features
    #[getter]
    fn n_features(&self) -> usize {
        if self.data.is_empty() {
            0
        } else {
            self.data[0].len()
        }
    }
}

impl HnswIndex {
    /// Compute distance between two vectors using the configured metric
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.metric.as_str() {
            "euclidean" | "l2" => metrics::euclidean(a, b),
            "manhattan" | "l1" | "taxicab" => metrics::manhattan(a, b),
            "cosine" => metrics::cosine(a, b),
            "chebyshev" | "linfinity" => metrics::chebyshev(a, b),
            "hamming" => metrics::hamming(a, b),
            _ => {
                // Default to euclidean for unknown metrics
                metrics::euclidean(a, b)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_distance_euclidean() {
        let index = HnswIndex {
            data: vec![],
            n_neighbors: 1,
            metric: "euclidean".to_string(),
            is_angular: false,
            neighbor_graph_cache: None,
        };

        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let dist = index.compute_distance(&a, &b);
        assert!((dist - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_distance_manhattan() {
        let index = HnswIndex {
            data: vec![],
            n_neighbors: 1,
            metric: "manhattan".to_string(),
            is_angular: false,
            neighbor_graph_cache: None,
        };

        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        let dist = index.compute_distance(&a, &b);
        assert!((dist - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_distance_cosine() {
        let index = HnswIndex {
            data: vec![],
            n_neighbors: 1,
            metric: "cosine".to_string(),
            is_angular: true,
            neighbor_graph_cache: None,
        };

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];  // Same vector
        let dist = index.compute_distance(&a, &b);
        assert!(dist < 0.01);  // Should be ~0
    }
}
