use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::PyErr;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::metrics;

impl From<metrics::MetricError> for PyErr {
    fn from(err: metrics::MetricError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}

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

        if !Self::is_supported_metric(metric.as_str()) {
            return Err(PyValueError::new_err(format!("unknown metric '{}'", metric)));
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
    #[pyo3(signature = (queries, k, ef, filter=None))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        ef: usize,
        filter: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
        if k == 0 {
            return Err(PyValueError::new_err("k must be at least 1"));
        }

        let mask: Option<Vec<bool>> = if let Some(filter_obj) = filter {
            let mask_array = filter_obj
                .downcast::<PyArray1<bool>>()
                .map_err(|_| {
                    PyNotImplementedError::new_err(
                        "Only boolean mask filters are supported by the Rust HNSW backend.",
                    )
                })?;

            if mask_array.ndim() != 1 {
                return Err(PyValueError::new_err(
                    "filter mask must be a 1-dimensional boolean array matching the indexed data length",
                ));
            }

            let readonly = mask_array.readonly();
            if readonly.len() != self.data.len() {
                return Err(PyValueError::new_err(format!(
                    "filter mask length ({}) must equal number of indexed samples ({})",
                    readonly.len(),
                    self.data.len()
                )));
            }

            let mask_slice = readonly.as_slice().map_err(|_| {
                PyValueError::new_err("filter mask must be contiguous in memory")
            })?;

            Some(mask_slice.to_vec())
        } else {
            None
        };

        let mask_slice = mask.as_ref().map(|m| m.as_slice());
        let _ = ef;

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

        let mut all_indices = Vec::with_capacity(n_queries);
        let mut all_distances = Vec::with_capacity(n_queries);

        for query_vec in &queries_vec {
            let mut heap = BinaryHeap::with_capacity(k);

            for (i, data_vec) in self.data.iter().enumerate() {
                if mask_slice.map_or(false, |mask_bits| !mask_bits[i]) {
                    continue;
                }

                let dist = self.compute_distance(query_vec, data_vec)?;
                push_candidate(&mut heap, HeapEntry::new(i as i64, dist), k);
            }

            let (mut indices, mut distances) = finalize_heap(heap, k);
            while indices.len() < k {
                indices.push(-1);
                distances.push(f32::NAN);
            }

            all_indices.push(indices);
            all_distances.push(distances);
        }

        #[allow(deprecated)]
        let indices_array = PyArray2::from_vec2(py, &all_indices)?.to_owned();
        #[allow(deprecated)]
        let distances_array = PyArray2::from_vec2(py, &all_distances)?.to_owned();

        Ok((indices_array, distances_array))
    }

    /// Get the k-nearest neighbor graph for all indexed points
    fn neighbor_graph<'py>(&mut self, py: Python<'py>)
        -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
        if let Some((indices, distances)) = &self.neighbor_graph_cache {
            #[allow(deprecated)]
            let indices_py = PyArray2::from_vec2(py, indices)?.to_owned();
            #[allow(deprecated)]
            let distances_py = PyArray2::from_vec2(py, distances)?.to_owned();
            return Ok((indices_py, distances_py));
        }

        let mut all_indices = Vec::with_capacity(self.data.len());
        let mut all_distances = Vec::with_capacity(self.data.len());

        for query_vec in &self.data {
            let mut heap = BinaryHeap::with_capacity(self.n_neighbors);

            for (i, data_vec) in self.data.iter().enumerate() {
                let dist = self.compute_distance(query_vec, data_vec)?;
                push_candidate(&mut heap, HeapEntry::new(i as i64, dist), self.n_neighbors);
            }

            let (mut indices, mut distances) = finalize_heap(heap, self.n_neighbors);
            while indices.len() < self.n_neighbors {
                indices.push(-1);
                distances.push(f32::NAN);
            }

            all_indices.push(indices);
            all_distances.push(distances);
        }

        #[allow(deprecated)]
        let indices_array = PyArray2::from_vec2(py, &all_indices)?.to_owned();
        #[allow(deprecated)]
        let distances_array = PyArray2::from_vec2(py, &all_distances)?.to_owned();

        self.neighbor_graph_cache = Some((all_indices.clone(), all_distances.clone()));

        Ok((indices_array, distances_array))
    }

    /// Prepare the index for querying (no-op in brute-force version, exists for API compatibility)
    fn prepare(&mut self) -> PyResult<()> {
        Ok(())
    }

    /// Update the index with new data (insert new points)
    fn update(&mut self, new_data: PyReadonlyArray2<f32>) -> PyResult<()> {
        let shape = new_data.shape();
        let (n_new, n_features) = (shape[0], shape[1]);

        if n_features != self.data[0].len() {
            return Err(PyValueError::new_err(
                format!("new_data features ({}) don't match index features ({})",
                    n_features, self.data[0].len()),
            ));
        }

        let new_data_slice = new_data.as_slice().map_err(|e|
            PyValueError::new_err(format!("Failed to get new_data slice: {}", e))
        )?;

        let mut new_data_vec = Vec::with_capacity(n_new);
        for i in 0..n_new {
            let row_start = i * n_features;
            let row_end = row_start + n_features;
            new_data_vec.push(new_data_slice[row_start..row_end].to_vec());
        }

        self.data.extend(new_data_vec);
        self.neighbor_graph_cache = None;

        Ok(())
    }

    #[getter]
    fn _angular_trees(&self) -> bool {
        self.is_angular
    }

    #[getter]
    fn metric(&self) -> String {
        self.metric.clone()
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.data.len()
    }

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
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> PyResult<f32> {
        let dist = match self.metric.as_str() {
            "euclidean" | "l2" => metrics::euclidean(a, b)?,
            "manhattan" | "l1" | "taxicab" => metrics::manhattan(a, b)?,
            "cosine" | "correlation" => metrics::cosine(a, b)?,
            "chebyshev" | "linfinity" => metrics::chebyshev(a, b)?,
            "hamming" => metrics::hamming(a, b)?,
            _ => return Err(PyValueError::new_err(format!("unknown metric '{}'", self.metric))),
        };
        Ok(dist)
    }

    fn is_supported_metric(metric: &str) -> bool {
        matches!(
            metric,
            "euclidean"
                | "l2"
                | "manhattan"
                | "l1"
                | "taxicab"
                | "cosine"
                | "correlation"
                | "chebyshev"
                | "linfinity"
                | "hamming"
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct HeapEntry {
    index: i64,
    distance: f32,
}

impl HeapEntry {
    fn new(index: i64, distance: f32) -> Self { Self { index, distance } }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.distance.to_bits() == other.distance.to_bits()
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        compare_distances(self.distance, other.distance)
            .then_with(|| self.index.cmp(&other.index))
    }
}

fn compare_distances(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.total_cmp(&b),
    }
}

fn push_candidate(heap: &mut BinaryHeap<HeapEntry>, entry: HeapEntry, k: usize) {
    if heap.len() < k {
        heap.push(entry);
    } else if let Some(worst) = heap.peek() {
        if entry < *worst {
            heap.pop();
            heap.push(entry);
        }
    }
}

fn finalize_heap(heap: BinaryHeap<HeapEntry>, k: usize) -> (Vec<i64>, Vec<f32>) {
    let mut entries = heap.into_sorted_vec();
    entries.truncate(k);

    let mut indices = Vec::with_capacity(k);
    let mut distances = Vec::with_capacity(k);
    for entry in entries {
        indices.push(entry.index);
        distances.push(entry.distance);
    }
    (indices, distances)
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
        let dist = index.compute_distance(&a, &b).unwrap();
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
        let dist = index.compute_distance(&a, &b).unwrap();
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
        let b = vec![1.0, 2.0, 3.0];
        let dist = index.compute_distance(&a, &b).unwrap();
        assert!(dist < 0.01);
    }

    #[test]
    fn test_unknown_metric_errors() {
        let index = HnswIndex {
            data: vec![],
            n_neighbors: 1,
            metric: "unknown".to_string(),
            is_angular: false,
            neighbor_graph_cache: None,
        };

        let result = index.compute_distance(&[0.0], &[0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_compare_distances_nan_ordering() {
        assert_eq!(compare_distances(f32::NAN, 1.0), Ordering::Greater);
        assert_eq!(compare_distances(1.0, f32::NAN), Ordering::Less);
        assert_eq!(compare_distances(f32::NAN, f32::NAN), Ordering::Equal);
    }

    #[test]
    fn test_finalize_heap_with_nan_and_padding() {
        let mut heap = BinaryHeap::new();
        push_candidate(&mut heap, HeapEntry::new(1, 0.5), 3);
        push_candidate(&mut heap, HeapEntry::new(2, f32::NAN), 3);
        push_candidate(&mut heap, HeapEntry::new(3, 0.3), 3);
        push_candidate(&mut heap, HeapEntry::new(4, 0.7), 3);

        let (indices, distances) = finalize_heap(heap, 3);
        assert_eq!(indices, vec![3, 1, 4]);
        assert!(distances[2] >= 0.7);
    }

    #[test]
    fn test_push_candidate_limits_size() {
        let mut heap = BinaryHeap::new();
        for i in 0..5 {
            push_candidate(&mut heap, HeapEntry::new(i, i as f32), 2);
        }
        assert_eq!(heap.len(), 2);
        let entries = heap.into_sorted_vec();
        let distances: Vec<_> = entries.iter().map(|entry| entry.distance).collect();
        assert_eq!(distances, vec![0.0, 1.0]);
    }
}
