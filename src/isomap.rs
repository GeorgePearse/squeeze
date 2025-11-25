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
            // Use a max-heap with positive distances
            // When we pop, we remove the largest distance (farthest neighbor)
            // This keeps the k smallest distances (nearest neighbors)
            let mut heap: BinaryHeap<(OrderedFloat<f64>, usize)> = BinaryHeap::new();

            for j in 0..n_samples {
                if i != j {
                    heap.push((OrderedFloat(distances[[i, j]]), j));
                    if heap.len() > self.n_neighbors {
                        heap.pop(); // Remove the largest (farthest)
                    }
                }
            }

            heap.into_iter()
                .map(|(d, j)| (j, d.into_inner()))
                .collect()
        }).collect()
    }

    fn compute_geodesic_distances(
        &self, 
        knn_graph: &[Vec<(usize, f64)>], 
        distances: &Array2<f64>,
        n_samples: usize
    ) -> PyResult<Array2<f64>> {
        // Use parallel Dijkstra for O(n^2 log n) instead of Floyd-Warshall's O(n^3)
        let geodesic_rows: Vec<Vec<f64>> = (0..n_samples).into_par_iter().map(|source| {
            self.dijkstra_single_source(source, knn_graph, n_samples)
        }).collect();
        
        // Assemble the geodesic distance matrix
        let mut geodesic = Array2::from_elem((n_samples, n_samples), f64::INFINITY);
        for (i, row) in geodesic_rows.into_iter().enumerate() {
            for (j, dist) in row.into_iter().enumerate() {
                geodesic[[i, j]] = dist;
            }
        }
        
        // Check for disconnected components
        let max_dist = geodesic.iter()
            .filter(|&&d| !d.is_infinite())
            .cloned()
            .fold(0.0_f64, f64::max);
        
        let has_inf = geodesic.iter().any(|&d| d.is_infinite());
        if has_inf {
            return Err(PyValueError::new_err(
                "Graph is disconnected. Try increasing n_neighbors."
            ));
        }

        Ok(geodesic)
    }
    
    /// Dijkstra's algorithm for single-source shortest paths
    fn dijkstra_single_source(
        &self,
        source: usize,
        knn_graph: &[Vec<(usize, f64)>],
        n_samples: usize
    ) -> Vec<f64> {
        let mut dist = vec![f64::INFINITY; n_samples];
        dist[source] = 0.0;
        
        // Min-heap: (distance, node)
        let mut heap = BinaryHeap::new();
        heap.push((OrderedFloat(0.0), source));
        
        while let Some((OrderedFloat(d), u)) = heap.pop() {
            // Skip if we've already found a better path
            if -d > dist[u] { continue; }
            
            // Explore neighbors
            for &(v, weight) in &knn_graph[u] {
                let new_dist = dist[u] + weight;
                if new_dist < dist[v] {
                    dist[v] = new_dist;
                    heap.push((OrderedFloat(-new_dist), v));
                }
            }
            
            // Also check reverse edges to ensure symmetry
            for v in 0..n_samples {
                for &(neighbor, weight) in &knn_graph[v] {
                    if neighbor == u {
                        let new_dist = dist[u] + weight;
                        if new_dist < dist[v] {
                            dist[v] = new_dist;
                            heap.push((OrderedFloat(-new_dist), v));
                        }
                    }
                }
            }
        }
        
        dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::collections::BinaryHeap;
    use ordered_float::OrderedFloat;

    #[test]
    fn test_knn_graph_construction() {
        let isomap = Isomap::new(2, 2);
        
        // Simple distance matrix
        let distances = Array2::from_shape_vec((4, 4), vec![
            0.0, 1.0, 2.0, 3.0,
            1.0, 0.0, 1.0, 2.0,
            2.0, 1.0, 0.0, 1.0,
            3.0, 2.0, 1.0, 0.0,
        ]).unwrap();
        
        let knn_graph = isomap.build_knn_graph(&distances, 4);
        
        // Each node should have 2 neighbors
        for neighbors in &knn_graph {
            assert_eq!(neighbors.len(), 2, "Each node should have exactly k neighbors");
        }
        
        // Node 0's nearest neighbors should be 1 and 2
        let node0_neighbors: Vec<usize> = knn_graph[0].iter().map(|(idx, _)| *idx).collect();
        assert!(node0_neighbors.contains(&1));
        assert!(node0_neighbors.contains(&2));
        
        // Check that distances are correct
        for &(neighbor_idx, dist) in &knn_graph[0] {
            assert_eq!(dist, distances[[0, neighbor_idx]]);
        }
    }

    #[test]
    fn test_geodesic_distances_simple_chain() {
        let isomap = Isomap::new(2, 1);
        
        // Create a chain: 0 -- 1 -- 2 -- 3
        // Direct distances don't reflect the chain structure
        let knn_graph = vec![
            vec![(1, 1.0)],  // 0 connects to 1
            vec![(0, 1.0), (2, 1.0)],  // 1 connects to 0 and 2
            vec![(1, 1.0), (3, 1.0)],  // 2 connects to 1 and 3
            vec![(2, 1.0)],  // 3 connects to 2
        ];
        
        // Dummy distances matrix (not used in this test)
        let distances = Array2::zeros((4, 4));
        
        let geodesic = isomap.compute_geodesic_distances(&knn_graph, &distances, 4).unwrap();
        
        // Check geodesic distances along the chain
        assert_relative_eq!(geodesic[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(geodesic[[0, 1]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(geodesic[[0, 2]], 2.0, epsilon = 1e-10);  // Through node 1
        assert_relative_eq!(geodesic[[0, 3]], 3.0, epsilon = 1e-10);  // Through nodes 1 and 2
    }

    #[test]
    fn test_floyd_warshall_correctness() {
        let isomap = Isomap::new(2, 2);
        
        // Create a simple graph
        let knn_graph = vec![
            vec![(1, 2.0), (2, 5.0)],
            vec![(0, 2.0), (2, 1.0)],
            vec![(0, 5.0), (1, 1.0)],
        ];
        
        let distances = Array2::zeros((3, 3));
        let geodesic = isomap.compute_geodesic_distances(&knn_graph, &distances, 3).unwrap();
        
        // Check shortest paths
        assert_relative_eq!(geodesic[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(geodesic[[0, 1]], 2.0, epsilon = 1e-10);
        assert_relative_eq!(geodesic[[0, 2]], 3.0, epsilon = 1e-10); // 0->1->2 is shorter than 0->2
        
        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(geodesic[[i, j]], geodesic[[j, i]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_disconnected_graph_detection() {
        let isomap = Isomap::new(2, 1);

        // Create disconnected components: 0-1 and 2-3
        let knn_graph = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0)],
            vec![(3, 1.0)],
            vec![(2, 1.0)],
        ];

        let distances = Array2::zeros((4, 4));
        let result = isomap.compute_geodesic_distances(&knn_graph, &distances, 4);

        // Should return an error for disconnected graph
        // Note: We can't inspect the PyErr message without initializing Python,
        // so just verify that an error is returned
        assert!(result.is_err(), "Disconnected graph should return an error");
    }

    #[test]
    fn test_symmetric_knn_graph() {
        let isomap = Isomap::new(2, 2);
        
        // Test that the geodesic distance computation makes the graph symmetric
        let knn_graph = vec![
            vec![(1, 1.0), (2, 2.0)],  // 0 -> 1, 0 -> 2
            vec![(2, 1.5)],            // 1 -> 2 only
            vec![],                     // 2 has no outgoing edges initially
        ];
        
        let distances = Array2::zeros((3, 3));
        let geodesic = isomap.compute_geodesic_distances(&knn_graph, &distances, 3).unwrap();
        
        // Despite asymmetric k-NN graph, geodesic distances should be symmetric
        assert_relative_eq!(geodesic[[0, 1]], geodesic[[1, 0]], epsilon = 1e-10);
        assert_relative_eq!(geodesic[[0, 2]], geodesic[[2, 0]], epsilon = 1e-10);
        assert_relative_eq!(geodesic[[1, 2]], geodesic[[2, 1]], epsilon = 1e-10);
    }
}
