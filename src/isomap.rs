//! Isomap (Isometric Mapping) implementation.
//!
//! Isomap computes geodesic distances along the manifold by finding
//! shortest paths on a k-NN graph, then applies MDS to embed the data.

use ndarray::{Array2};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use ordered_float::OrderedFloat;

use crate::metrics_simd;
use crate::mds::{compute_distance_matrix, classical_mds};

/// Isomap dimensionality reduction
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct Isomap {
    n_components: usize,
    n_neighbors: usize,
}

#[pymethods]
impl Isomap {
    #[new]
    #[pyo3(signature = (n_components=2, n_neighbors=10))]
    pub fn new(n_components: usize, n_neighbors: usize) -> Self {
        Self {
            n_components,
            n_neighbors,
        }
    }

    /// Fit and transform data using Isomap
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

        // Build k-NN graph
        let knn_graph = self.build_knn_graph(&distances, n_samples);

        // Compute geodesic distances using Dijkstra's algorithm
        let geodesic_distances = self.compute_geodesic_distances(&knn_graph, &distances, n_samples)?;

        // Apply classical MDS to geodesic distances
        let embedding = classical_mds(&geodesic_distances, self.n_components)?;
        Ok(embedding.into_pyarray_bound(py))
    }
}

impl Isomap {
    fn build_knn_graph(&self, distances: &Array2<f64>, n_samples: usize) -> Vec<Vec<(usize, f64)>> {
        (0..n_samples).into_par_iter().map(|i| {
            let mut heap: BinaryHeap<(OrderedFloat<f64>, usize)> = BinaryHeap::new();
            
            for j in 0..n_samples {
                if i != j {
                    // Use negative distance for max-heap to act as min-heap
                    heap.push((OrderedFloat(-distances[[i, j]]), j));
                    if heap.len() > self.n_neighbors {
                        heap.pop();
                    }
                }
            }

            heap.into_iter()
                .map(|(d, j)| (j, -d.into_inner()))
                .collect()
        }).collect()
    }

    fn compute_geodesic_distances(
        &self, 
        knn_graph: &[Vec<(usize, f64)>], 
        distances: &Array2<f64>,
        n_samples: usize
    ) -> PyResult<Array2<f64>> {
        // Floyd-Warshall for shortest paths
        let mut geodesic = Array2::from_elem((n_samples, n_samples), f64::INFINITY);

        // Initialize with k-NN edges (symmetric)
        for i in 0..n_samples {
            geodesic[[i, i]] = 0.0;
            for &(j, d) in &knn_graph[i] {
                geodesic[[i, j]] = d;
                geodesic[[j, i]] = d; // Make symmetric
            }
        }

        // Floyd-Warshall
        for k in 0..n_samples {
            for i in 0..n_samples {
                for j in 0..n_samples {
                    let new_dist = geodesic[[i, k]] + geodesic[[k, j]];
                    if new_dist < geodesic[[i, j]] {
                        geodesic[[i, j]] = new_dist;
                    }
                }
            }
        }

        // Check for disconnected components
        let max_dist = geodesic.iter().cloned().fold(0.0_f64, f64::max);
        if max_dist.is_infinite() {
            return Err(PyValueError::new_err(
                "Graph is disconnected. Try increasing n_neighbors."
            ));
        }

        Ok(geodesic)
    }
}
