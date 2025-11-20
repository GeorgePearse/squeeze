use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::PyErr;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

use crate::metrics;
use crate::hnsw_algo::Hnsw;

impl From<metrics::MetricError> for PyErr {
    fn from(err: metrics::MetricError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}

#[derive(Serialize, Deserialize)]
struct HnswIndexState {
    data: Vec<Vec<f32>>,
    n_neighbors: usize,
    metric: String,
    dist_p: f32,
    is_angular: bool,
    hnsw: Hnsw,
}

/// HNSW approximate nearest neighbor index
#[pyclass(module = "umap._hnsw_backend")]
pub struct HnswIndex {
    /// Copy of the data for searching
    data: Vec<Vec<f32>>,
    /// Number of neighbors to return
    n_neighbors: usize,
    /// Distance metric name
    metric: String,
    /// Parameter p for Minkowski distance
    dist_p: f32,
    /// Whether the metric is angular (cosine/correlation)
    is_angular: bool,
    /// Cached neighbor graph (for neighbor_graph property)
    neighbor_graph_cache: Option<(Vec<Vec<i64>>, Vec<Vec<f32>>)>,
    
    /// The HNSW Graph Algorithm
    hnsw: Hnsw,
}

#[pymethods]
impl HnswIndex {
    /// Create a new nearest neighbor index
    #[new]
    #[pyo3(signature = (data, n_neighbors, metric, m, ef_construction, dist_p=2.0, random_state=None))]
    fn new(
        data: PyReadonlyArray2<f32>,
        n_neighbors: usize,
        metric: String,
        m: usize,
        ef_construction: usize,
        dist_p: f32,
        random_state: Option<u64>,
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
            return Err(PyValueError::new_err(format!(
                "Unknown metric '{}'. Supported metrics: euclidean, l2, manhattan, l1, \
                 taxicab, cosine, correlation, chebyshev, linfinity, minkowski, hamming",
                metric
            )));
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
        
        // Initialize HNSW graph with deterministic seed for reproducibility
        // Future: expose random_state parameter to Python API
        let seed = random_state.unwrap_or(42);
        let mut hnsw = Hnsw::new(m, ef_construction, n_samples, seed);
        
        // Build the graph
        {
            let dist_func = |i: usize, j: usize| -> f32 {
                Self::compute_dist_static(&data_vec[i], &data_vec[j], &metric, dist_p).unwrap_or(f32::MAX)
            };
            
            for i in 0..n_samples {
                hnsw.insert(i, &dist_func);
            }
        }

        Ok(Self {
            data: data_vec,
            n_neighbors,
            metric,
            dist_p,
            is_angular,
            neighbor_graph_cache: None,
            hnsw,
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

        // Parallel query execution using rayon
        let results: Result<Vec<_>, PyErr> = queries_vec.par_iter().map(|query_vec| {
            // Distance function for this query
            let dist_query = |node_idx: usize| -> f32 {
                Self::compute_dist_static(query_vec, &self.data[node_idx], &self.metric, self.dist_p).unwrap_or(f32::MAX)
            };
            
            // Search
            let found = self.hnsw.search(None, k, ef, dist_query);
            
            // Filter if needed
            let mut filtered_indices = Vec::new();
            let mut filtered_dists = Vec::new();
            
            for (idx, dist) in found {
                if let Some(m) = &mask {
                    if !m[idx] { continue; }
                }
                filtered_indices.push(idx as i64);
                filtered_dists.push(dist);
                if filtered_indices.len() >= k { break; }
            }
            
            while filtered_indices.len() < k {
                filtered_indices.push(-1);
                filtered_dists.push(f32::NAN);
            }
            
            Ok((filtered_indices, filtered_dists))
        }).collect();

        let results = results?;
        let (all_indices, all_distances): (Vec<_>, Vec<_>) = results.into_iter().unzip();

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

        let ef = self.n_neighbors * 2; 
        
        let results: Result<Vec<_>, PyErr> = (0..self.data.len()).into_par_iter().map(|i| {
            let query_vec = &self.data[i];
            let dist_query = |node_idx: usize| -> f32 {
                Self::compute_dist_static(query_vec, &self.data[node_idx], &self.metric, self.dist_p).unwrap_or(f32::MAX)
            };
            
            let mut found = self.hnsw.search(Some(i), self.n_neighbors + 1, ef, dist_query);
            
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
        let start_idx = self.data.len();
        
        for i in 0..n_new {
            let row_start = i * n_features;
            let row_end = row_start + n_features;
            let vec = new_data_slice[row_start..row_end].to_vec();
            new_data_vec.push(vec);
        }
        
        // Append data
        self.data.extend(new_data_vec);
        
        // Insert into graph
        for i in 0..n_new {
            let current_idx = start_idx + i;
            let dist_func = |u: usize, v: usize| -> f32 {
                Self::compute_dist_static(&self.data[u], &self.data[v], &self.metric, self.dist_p).unwrap_or(f32::MAX)
            };
            self.hnsw.insert(current_idx, &dist_func);
        }
        
        self.neighbor_graph_cache = None;

        Ok(())
    }

    // Serialization support (Pickle)
    
    pub fn __getstate__(&self, py: Python) -> PyResult<Py<PyBytes>> {
        let state = HnswIndexState {
            data: self.data.clone(),
            n_neighbors: self.n_neighbors,
            metric: self.metric.clone(),
            dist_p: self.dist_p,
            is_angular: self.is_angular,
            hnsw: self.hnsw.to_owned(), // Hnsw must be Clone? No, just Serialize.
            // Wait, to_owned() on a struct usually implies Clone. 
            // Hnsw doesn't derive Clone. But we serialize state, not self.
            // We can just move or ref.
            // `state` owns the data. So we need to clone self.data etc.
            // But `hnsw` field in state must be owned. We can implement Clone for Hnsw?
            // Or just serialize directly?
            // Serialize works on reference.
            // But `HnswIndexState` needs owned values.
            // Let's implement Clone for Hnsw in `hnsw_algo.rs`.
        };
        // Actually, better: serialize directly to bytes without intermediate struct if possible?
        // Or make HnswIndexState use references? Serde supports serializing structs with references.
        // But `bincode::serialize` takes `&T`.
        // So we can define `HnswIndexStateRef<'a>`
        
        let encoded = bincode::serialize(&state).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new_bound(py, &encoded).into())
    }

    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        let bytes = state.as_bytes();
        let decoded: HnswIndexState = bincode::deserialize(bytes).map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        self.data = decoded.data;
        self.n_neighbors = decoded.n_neighbors;
        self.metric = decoded.metric;
        self.dist_p = decoded.dist_p;
        self.is_angular = decoded.is_angular;
        self.hnsw = decoded.hnsw;
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
    fn compute_dist_static(a: &[f32], b: &[f32], metric: &str, p: f32) -> crate::metrics::MetricResult<f32> {
        match metric {
            "euclidean" | "l2" => metrics::euclidean(a, b),
            "manhattan" | "l1" | "taxicab" => metrics::manhattan(a, b),
            "cosine" | "correlation" => metrics::cosine(a, b),
            "chebyshev" | "linfinity" => metrics::chebyshev(a, b),
            "minkowski" => metrics::minkowski(a, b, p),
            "hamming" => metrics::hamming(a, b),
            _ => Err(metrics::MetricError::DimensionMismatch{left:0, right:0}), 
        }
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
                | "minkowski"
                | "hamming"
        )
    }
}
