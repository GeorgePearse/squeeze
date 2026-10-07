use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::PyErr;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

// Use SIMD-optimized metrics for 3-4x faster distance computation
use crate::metrics_simd;
use crate::metrics::{self, MetricError, MetricResult};
use crate::hnsw_algo::{Hnsw, PruneStrategy};
use crate::compute::{Device, Metric, BRUTEFORCE_MAX_ROWS};
use ndarray::Array2;

impl From<MetricError> for PyErr {
    fn from(err: MetricError) -> Self {
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
    /// `None` when the index was built for GPU brute force and never needed the graph.
    hnsw: Option<Hnsw>,
    device: Option<String>,
    m: usize,
    ef_construction: usize,
    seed: u64,
    prune_strategy: String,
    prune_alpha: f32,
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

    /// The HNSW graph. Built eagerly on CPU; built lazily (only if a GPU call fails or a
    /// filter mask is used) when a GPU runs exact brute-force kNN instead.
    hnsw: Option<Hnsw>,
    /// The requested device string (`None` = auto) and the GPU it resolved to, if any.
    device: Option<String>,
    gpu: Option<Device>,
    /// Graph construction parameters, kept so the graph can be built lazily.
    m: usize,
    ef_construction: usize,
    seed: u64,
    prune_strategy: String,
    prune_alpha: f32,
}

#[pymethods]
impl HnswIndex {
    /// Create a new nearest neighbor index
    #[new]
    #[pyo3(signature = (data, n_neighbors, metric, m, ef_construction, dist_p=2.0, random_state=None, prune_strategy="simple", prune_alpha=1.2, device=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        data: PyReadonlyArray2<f32>,
        n_neighbors: usize,
        metric: String,
        m: usize,
        ef_construction: usize,
        dist_p: f32,
        random_state: Option<u64>,
        prune_strategy: &str,
        prune_alpha: f32,
        device: Option<String>,
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

        // Validate the pruning strategy up front (the graph may be built lazily)
        Self::parse_prune(prune_strategy, prune_alpha)?;
        let seed = random_state.unwrap_or(42);

        // Resolve the compute device. A GPU runs exact brute-force kNN for supported metrics
        // up to BRUTEFORCE_MAX_ROWS rows; otherwise the HNSW graph is built on the CPU.
        let resolved = crate::device_py::resolve(py, device.as_deref())?;
        let gpu = match resolved {
            Device::Cpu => None,
            dev if n_samples <= BRUTEFORCE_MAX_ROWS && Metric::parse(&metric).is_some() => Some(dev),
            _ => None,
        };

        let mut index = Self {
            data: data_vec,
            n_neighbors,
            metric,
            dist_p,
            is_angular,
            neighbor_graph_cache: None,
            hnsw: None,
            device,
            gpu,
            m,
            ef_construction,
            seed,
            prune_strategy: prune_strategy.to_string(),
            prune_alpha,
        };
        if index.gpu.is_none() {
            index.ensure_graph();
        }
        Ok(index)
    }

    /// Query the index for k nearest neighbors
    #[pyo3(signature = (queries, k, ef, filter=None))]
    fn query<'py>(
        &mut self,
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

        // GPU path: exact brute force (filter masks use the graph instead)
        if mask.is_none() && self.gpu.is_some() {
            let q = Array2::from_shape_vec((n_queries, n_features), queries_slice.to_vec())
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if let Some((idx, dist)) = self.brute_force(py, q, k.min(self.data.len())) {
                let indices: Vec<Vec<i64>> = idx
                    .rows()
                    .into_iter()
                    .map(|r| {
                        let mut v: Vec<i64> = r.iter().map(|&i| i as i64).collect();
                        v.resize(k, -1);
                        v
                    })
                    .collect();
                let dists: Vec<Vec<f32>> = dist
                    .rows()
                    .into_iter()
                    .map(|r| {
                        let mut v: Vec<f32> = r.to_vec();
                        v.resize(k, f32::NAN);
                        v
                    })
                    .collect();
                #[allow(deprecated)]
                let indices_array = PyArray2::from_vec2(py, &indices)?.to_owned();
                #[allow(deprecated)]
                let distances_array = PyArray2::from_vec2(py, &dists)?.to_owned();
                return Ok((indices_array, distances_array));
            }
        }
        self.ensure_graph();
        let hnsw = self.hnsw.as_ref().expect("graph built by ensure_graph");

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
            let found = hnsw.search(None, k, ef, dist_query);
            
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

        // GPU path: exact brute force over the data itself, dropping each point's own entry
        if self.gpu.is_some() {
            let n = self.data.len();
            let d = self.data[0].len();
            let flat: Vec<f32> = self.data.iter().flatten().copied().collect();
            let data = Array2::from_shape_vec((n, d), flat).map_err(|e| PyValueError::new_err(e.to_string()))?;
            let k = (self.n_neighbors + 1).min(n);
            if let Some((idx, dist)) = self.brute_force(py, data, k) {
                let mut all_indices = Vec::with_capacity(n);
                let mut all_distances = Vec::with_capacity(n);
                for i in 0..n {
                    let mut ind: Vec<i64> = Vec::with_capacity(self.n_neighbors);
                    let mut dis: Vec<f32> = Vec::with_capacity(self.n_neighbors);
                    for j in 0..k {
                        if idx[[i, j]] as usize == i || ind.len() == self.n_neighbors {
                            continue;
                        }
                        ind.push(idx[[i, j]] as i64);
                        dis.push(dist[[i, j]]);
                    }
                    while ind.len() < self.n_neighbors {
                        ind.push(-1);
                        dis.push(f32::NAN);
                    }
                    all_indices.push(ind);
                    all_distances.push(dis);
                }
                self.neighbor_graph_cache = Some((all_indices, all_distances));
                let (ci, cd) = self.neighbor_graph_cache.as_ref().unwrap();
                #[allow(deprecated)]
                let indices_array = PyArray2::from_vec2(py, ci)?.to_owned();
                #[allow(deprecated)]
                let distances_array = PyArray2::from_vec2(py, cd)?.to_owned();
                return Ok((indices_array, distances_array));
            }
        }
        self.ensure_graph();
        let hnsw = self.hnsw.as_ref().expect("graph built by ensure_graph");

        let ef = self.n_neighbors * 2; 
        
        let results: Result<Vec<_>, PyErr> = (0..self.data.len()).into_par_iter().map(|i| {
            let query_vec = &self.data[i];
            let dist_query = |node_idx: usize| -> f32 {
                Self::compute_dist_static(query_vec, &self.data[node_idx], &self.metric, self.dist_p).unwrap_or(f32::MAX)
            };
            
            let mut found = hnsw.search(Some(i), self.n_neighbors + 1, ef, dist_query);
            
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
        
        // Insert into graph (if one exists; the GPU brute-force path needs none)
        if let Some(hnsw) = self.hnsw.as_mut() {
            for i in 0..n_new {
                let current_idx = start_idx + i;
                let dist_func = |u: usize, v: usize| -> f32 {
                    Self::compute_dist_static(&self.data[u], &self.data[v], &self.metric, self.dist_p).unwrap_or(f32::MAX)
                };
                hnsw.insert(current_idx, &dist_func);
            }
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
            hnsw: self.hnsw.clone(),
            device: self.device.clone(),
            m: self.m,
            ef_construction: self.ef_construction,
            seed: self.seed,
            prune_strategy: self.prune_strategy.clone(),
            prune_alpha: self.prune_alpha,
        };

        let encoded = bincode::serialize(&state).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new_bound(py, &encoded).into())
    }

    pub fn __setstate__(&mut self, py: Python<'_>, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        let bytes = state.as_bytes();
        let decoded: HnswIndexState = bincode::deserialize(bytes).map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        self.data = decoded.data;
        self.n_neighbors = decoded.n_neighbors;
        self.metric = decoded.metric;
        self.dist_p = decoded.dist_p;
        self.is_angular = decoded.is_angular;
        self.hnsw = decoded.hnsw;
        self.device = decoded.device;
        self.m = decoded.m;
        self.ef_construction = decoded.ef_construction;
        self.seed = decoded.seed;
        self.prune_strategy = decoded.prune_strategy;
        self.prune_alpha = decoded.prune_alpha;
        self.neighbor_graph_cache = None;
        // The device is re-resolved on the loading machine; the graph is built lazily if needed.
        self.gpu = match crate::device_py::resolve(py, self.device.as_deref())? {
            Device::Cpu => None,
            dev if self.data.len() <= BRUTEFORCE_MAX_ROWS && Metric::parse(&self.metric).is_some() => Some(dev),
            _ => None,
        };
        
        Ok(())
    }

    /// The compute device this index uses for neighbour search: `"cpu"` (HNSW graph) or the
    /// GPU description (exact brute force).
    #[getter]
    fn compute_device(&self) -> String {
        match &self.gpu {
            Some(d) => d.to_string(),
            None => "cpu".to_string(),
        }
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
    fn parse_prune(prune_strategy: &str, prune_alpha: f32) -> PyResult<PruneStrategy> {
        match prune_strategy {
            "simple" => Ok(PruneStrategy::Simple),
            "robust" => Ok(PruneStrategy::RobustPrune { alpha: prune_alpha }),
            _ => Err(PyValueError::new_err(format!(
                "Unknown prune_strategy '{}'. Supported: 'simple', 'robust'",
                prune_strategy
            ))),
        }
    }

    /// Build the HNSW graph over the current data if it does not exist yet.
    fn ensure_graph(&mut self) {
        if self.hnsw.is_some() {
            return;
        }
        let prune = Self::parse_prune(&self.prune_strategy, self.prune_alpha)
            .expect("prune strategy validated in the constructor");
        let n_samples = self.data.len();
        let mut hnsw = Hnsw::with_prune_strategy(self.m, self.ef_construction, n_samples, self.seed, prune);
        {
            let data = &self.data;
            let metric = &self.metric;
            let dist_p = self.dist_p;
            let dist_func = |i: usize, j: usize| -> f32 {
                Self::compute_dist_static(&data[i], &data[j], metric, dist_p).unwrap_or(f32::MAX)
            };
            for i in 0..n_samples {
                hnsw.insert(i, &dist_func);
            }
        }
        self.hnsw = Some(hnsw);
    }

    /// Exact kNN of `queries` against the data on the GPU. On any failure, warns once,
    /// disables the GPU for this index and returns `None` so the caller uses the graph.
    fn brute_force(&mut self, py: Python<'_>, queries: Array2<f32>, k: usize) -> Option<(Array2<u32>, Array2<f32>)> {
        let device = self.gpu.clone()?;
        let metric = Metric::parse(&self.metric)?;
        let n = self.data.len();
        let d = self.data[0].len();
        let flat: Vec<f32> = self.data.iter().flatten().copied().collect();
        let data = Array2::from_shape_vec((n, d), flat).ok()?;
        let result = device
            .backend()
            .and_then(|b| b.knn_bruteforce(data.view(), queries.view(), k, metric));
        match result {
            Ok(r) => Some(r),
            Err(e) => {
                crate::device_py::warn(
                    py,
                    &format!("squeeze: kNN on {} failed, falling back to HNSW on cpu: {}", device, e),
                );
                self.gpu = None;
                None
            }
        }
    }

    fn compute_dist_static(a: &[f32], b: &[f32], metric: &str, p: f32) -> MetricResult<f32> {
        match metric {
            // Use SIMD-optimized versions for common metrics (3-4x faster!)
            "euclidean" | "l2" => metrics_simd::euclidean(a, b),
            "manhattan" | "l1" | "taxicab" => metrics_simd::manhattan(a, b),
            "cosine" | "correlation" => metrics_simd::cosine(a, b),
            // Fall back to scalar for less common metrics
            "chebyshev" | "linfinity" => metrics::chebyshev(a, b),
            "minkowski" => metrics::minkowski(a, b, p),
            "hamming" => metrics::hamming(a, b),
            _ => Err(MetricError::DimensionMismatch{left:0, right:0}),
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
