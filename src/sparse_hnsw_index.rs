use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::PyErr;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

use crate::sparse_metrics;
use crate::hnsw_algo::Hnsw;

#[derive(Serialize, Deserialize)]
struct SparseHnswIndexState {
    indptr: Vec<i32>,
    indices: Vec<i32>,
    data: Vec<f32>,
    n_samples: usize,
    n_features: usize,
    n_neighbors: usize,
    metric: String,
    dist_p: f32,
    is_angular: bool,
    hnsw: Hnsw,
}

/// HNSW approximate nearest neighbor index for sparse data
#[pyclass(module = "umap._hnsw_backend")]
pub struct SparseHnswIndex {
    /// CSR format data
    indptr: Vec<i32>,
    indices: Vec<i32>,
    data: Vec<f32>,
    
    /// Number of samples
    n_samples: usize,
    /// Number of features (columns)
    n_features: usize,
    
    /// Number of neighbors to return
    n_neighbors: usize,
    /// Distance metric name
    metric: String,
    /// Parameter p for Minkowski distance (unused for now but kept for API parity)
    _dist_p: f32,
    /// Whether the metric is angular (cosine/correlation)
    is_angular: bool,
    
    /// Cached neighbor graph (for neighbor_graph property)
    neighbor_graph_cache: Option<(Vec<Vec<i64>>, Vec<Vec<f32>>)>,
    
    /// HNSW Graph
    hnsw: Hnsw,
}

#[pymethods]
impl SparseHnswIndex {
    /// Create a new sparse nearest neighbor index
    #[new]
    #[pyo3(signature = (data, indices, indptr, n_samples, n_features, n_neighbors, metric, m, ef_construction, dist_p=2.0, random_state=None))]
    fn new(
        data: PyReadonlyArray1<f32>,
        indices: PyReadonlyArray1<i32>,
        indptr: PyReadonlyArray1<i32>,
        n_samples: usize,
        n_features: usize,
        n_neighbors: usize,
        metric: String,
        m: usize,
        ef_construction: usize,
        dist_p: f32,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
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
            return Err(PyValueError::new_err(format!(
                "Unknown sparse metric '{}'. Supported metrics: euclidean, l2, manhattan, \
                 l1, taxicab, cosine, correlation",
                metric
            )));
        }

        // Clone data into Vecs
        let data_vec = data.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?.to_vec();
        let indices_vec = indices.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?.to_vec();
        let indptr_vec = indptr.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?.to_vec();

        // Validate indptr length
        if indptr_vec.len() != n_samples + 1 {
            return Err(PyValueError::new_err(
                format!("indptr length ({}) must be n_samples + 1 ({})", indptr_vec.len(), n_samples + 1)
            ));
        }

        let is_angular = metric == "cosine" || metric == "correlation";
        
        // Initialize HNSW with deterministic seed for reproducibility
        let seed = random_state.unwrap_or(42);
        let mut hnsw = Hnsw::new(m, ef_construction, n_samples, seed);
        
        // Build graph
        {
            let dist_func = |i: usize, j: usize| -> f32 {
                Self::compute_dist_static(
                    i, j, 
                    &indptr_vec, &indices_vec, &data_vec, 
                    &metric
                ).unwrap_or(f32::MAX)
            };
            
            for i in 0..n_samples {
                hnsw.insert(i, &dist_func);
            }
        }

        Ok(Self {
            indptr: indptr_vec,
            indices: indices_vec,
            data: data_vec,
            n_samples,
            n_features,
            n_neighbors,
            metric,
            _dist_p: dist_p,
            is_angular,
            neighbor_graph_cache: None,
            hnsw,
        })
    }

    /// Query the index for k nearest neighbors
    #[pyo3(signature = (query_data, query_indices, query_indptr, k, ef))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        query_data: PyReadonlyArray1<f32>,
        query_indices: PyReadonlyArray1<i32>,
        query_indptr: PyReadonlyArray1<i32>,
        k: usize,
        ef: usize,
    ) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
        if k == 0 {
            return Err(PyValueError::new_err("k must be at least 1"));
        }

        let q_data = query_data.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let q_indices = query_indices.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let q_indptr = query_indptr.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;

        let n_queries = q_indptr.len() - 1;

        // Parallel query execution
        let results: Result<Vec<_>, PyErr> = (0..n_queries).into_par_iter().map(|i| {
            
            // Define closure to compute distance between query i and node j
            let dist_query = |node_idx: usize| -> f32 {
                // Extract query vector i
                let q_start = q_indptr[i] as usize;
                let q_end = q_indptr[i+1] as usize;
                let q_vec_data = &q_data[q_start..q_end];
                let q_vec_indices = &q_indices[q_start..q_end];
                
                // Extract node vector j (from self)
                let c_start = self.indptr[node_idx] as usize;
                let c_end = self.indptr[node_idx+1] as usize;
                let c_vec_data = &self.data[c_start..c_end];
                let c_vec_indices = &self.indices[c_start..c_end];
                
                Self::compute_dist_vectors(
                    q_vec_indices, q_vec_data,
                    c_vec_indices, c_vec_data,
                    &self.metric
                ).unwrap_or(f32::MAX)
            };

            let found = self.hnsw.search(None, k, ef, dist_query);
            
            let mut indices = Vec::with_capacity(k);
            let mut distances = Vec::with_capacity(k);
            
            for (idx, dist) in found {
                indices.push(idx as i64);
                distances.push(dist);
                if indices.len() >= k { break; }
            }
            
            while indices.len() < k {
                indices.push(-1);
                distances.push(f32::NAN);
            }

            Ok((indices, distances))
        }).collect();

        let results = results?;
        let (all_indices, all_distances): (Vec<_>, Vec<_>) = results.into_iter().unzip();

        #[allow(deprecated)]
        let indices_array = PyArray2::from_vec2(py, &all_indices)?.to_owned();
        #[allow(deprecated)]
        let distances_array = PyArray2::from_vec2(py, &all_distances)?.to_owned();

        Ok((indices_array, distances_array))
    }

    /// Get the k-nearest neighbor graph
    fn neighbor_graph<'py>(&mut self, py: Python<'py>)
        -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
        
        if let Some((indices, distances)) = &self.neighbor_graph_cache {
            #[allow(deprecated)]
            let indices_py = PyArray2::from_vec2(py, indices)?.to_owned();
            #[allow(deprecated)]
            let distances_py = PyArray2::from_vec2(py, distances)?.to_owned();
            return Ok((indices_py, distances_py));
        }
        
        let ef = self.n_neighbors * 2;

        let results: Result<Vec<_>, PyErr> = (0..self.n_samples).into_par_iter().map(|i| {
            let dist_query = |node_idx: usize| -> f32 {
                Self::compute_dist_static(
                    i, node_idx, 
                    &self.indptr, &self.indices, &self.data, 
                    &self.metric
                ).unwrap_or(f32::MAX)
            };
            
            let mut found = self.hnsw.search(Some(i), self.n_neighbors + 1, ef, dist_query);
            
            // Filter self
            found.retain(|&(idx, _)| idx != i);
            found.truncate(self.n_neighbors);
            
            let mut indices = Vec::with_capacity(self.n_neighbors);
            let mut distances = Vec::with_capacity(self.n_neighbors);
            
            for (idx, dist) in found {
                indices.push(idx as i64);
                distances.push(dist);
            }
            
            while indices.len() < self.n_neighbors {
                indices.push(-1);
                distances.push(f32::NAN);
            }

            Ok((indices, distances))
        }).collect();

        let results = results?;
        let (all_indices, all_distances): (Vec<_>, Vec<_>) = results.into_iter().unzip();

        // Cache the results
        self.neighbor_graph_cache = Some((all_indices, all_distances));
        
        // Borrow from cache for return to avoid cloning
        let (cached_indices, cached_distances) = self.neighbor_graph_cache.as_ref().unwrap();
        #[allow(deprecated)]
        let indices_array = PyArray2::from_vec2(py, cached_indices)?.to_owned();
        #[allow(deprecated)]
        let distances_array = PyArray2::from_vec2(py, cached_distances)?.to_owned();

        Ok((indices_array, distances_array))
    }
    
    fn prepare(&mut self) -> PyResult<()> { Ok(()) }
    
    fn update(&mut self, _new_data: &Bound<'_, PyAny>) -> PyResult<()> {
        Err(PyNotImplementedError::new_err("Update not implemented for sparse index yet"))
    }
    
    // Serialization support (Pickle)
    pub fn __getstate__(&self, py: Python) -> PyResult<Py<PyBytes>> {
        let state = SparseHnswIndexState {
            indptr: self.indptr.clone(),
            indices: self.indices.clone(),
            data: self.data.clone(),
            n_samples: self.n_samples,
            n_features: self.n_features,
            n_neighbors: self.n_neighbors,
            metric: self.metric.clone(),
            dist_p: self._dist_p,
            is_angular: self.is_angular,
            hnsw: self.hnsw.clone(),
        };
        
        let encoded = bincode::serialize(&state).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new_bound(py, &encoded).into())
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        let bytes = state.as_bytes();
        let decoded: SparseHnswIndexState = bincode::deserialize(bytes).map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        self.indptr = decoded.indptr;
        self.indices = decoded.indices;
        self.data = decoded.data;
        self.n_samples = decoded.n_samples;
        self.n_features = decoded.n_features;
        self.n_neighbors = decoded.n_neighbors;
        self.metric = decoded.metric;
        self._dist_p = decoded.dist_p;
        self.is_angular = decoded.is_angular;
        self.hnsw = decoded.hnsw;
        self.neighbor_graph_cache = None;
        
        Ok(())
    }
    
    #[getter]
    fn _angular_trees(&self) -> bool { self.is_angular }
    
    #[getter]
    fn metric(&self) -> String { self.metric.clone() }
    
    #[getter]
    fn n_samples(&self) -> usize { self.n_samples }
    
    #[getter]
    fn n_features(&self) -> usize { self.n_features }
}

impl SparseHnswIndex {
    // Static helper to compute distance between two rows in the stored data
    fn compute_dist_static(
        i: usize, j: usize,
        indptr: &[i32], indices: &[i32], data: &[f32],
        metric: &str
    ) -> Result<f32, PyErr> {
        let start_i = indptr[i] as usize;
        let end_i = indptr[i+1] as usize;
        let idx_i = &indices[start_i..end_i];
        let val_i = &data[start_i..end_i];
        
        let start_j = indptr[j] as usize;
        let end_j = indptr[j+1] as usize;
        let idx_j = &indices[start_j..end_j];
        let val_j = &data[start_j..end_j];
        
        Self::compute_dist_vectors(idx_i, val_i, idx_j, val_j, metric)
    }

    // Static helper for vectors
    fn compute_dist_vectors(
        idx_a: &[i32], val_a: &[f32],
        idx_b: &[i32], val_b: &[f32],
        metric: &str
    ) -> Result<f32, PyErr> {
        match metric {
            "euclidean" | "l2" => Ok(sparse_metrics::sparse_euclidean(idx_a, val_a, idx_b, val_b)),
            "manhattan" | "l1" | "taxicab" => Ok(sparse_metrics::sparse_manhattan(idx_a, val_a, idx_b, val_b)),
            "cosine" | "correlation" => Ok(sparse_metrics::sparse_cosine(idx_a, val_a, idx_b, val_b)),
            _ => Err(PyValueError::new_err(format!("unknown sparse metric '{}'", metric))),
        }
    }

    fn is_supported_metric(metric: &str) -> bool {
        matches!(
            metric,
            "euclidean" | "l2" | "manhattan" | "l1" | "taxicab" | "cosine" | "correlation"
        )
    }
}
