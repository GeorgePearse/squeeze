//! PLSCAN clustering (Persistent Leaves Spatial Clustering for Applications with Noise).
//!
//! This is a Rust implementation inspired by JelmerBot/fast_plscan:
//! - Builds a mutual-reachability kNN graph (HDBSCAN*-style core distances)
//! - Extracts a minimum spanning forest
//! - Builds linkage/condensed/leaf trees
//! - Selects an optimal minimum cluster size by maximizing total persistence
//! - Produces cluster labels + membership probabilities

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::metrics_simd;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersistenceMeasure {
    Size,
    Distance,
    Density,
    SizeDistance,
    SizeDensity,
}

impl PersistenceMeasure {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "size" => Some(Self::Size),
            "distance" => Some(Self::Distance),
            "density" => Some(Self::Density),
            "size-distance" => Some(Self::SizeDistance),
            "size-density" => Some(Self::SizeDensity),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlscanClusterResult {
    pub labels: Vec<i64>,
    pub probabilities: Vec<f64>,
    pub selected_min_cluster_size: Option<f64>,
    pub trace_min_size: Vec<f64>,
    pub trace_persistence: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
struct Edge {
    u: u32,
    v: u32,
    w: f64,
}

#[derive(Debug, Clone)]
struct SpanningTree {
    parent: Vec<u32>,
    child: Vec<u32>,
    distance: Vec<f64>,
}

#[derive(Debug, Clone)]
struct LinkageTree {
    left: Vec<u32>,
    right: Vec<u32>,
    child_count: Vec<u32>,
    child_size: Vec<f64>,
}

#[derive(Debug, Clone)]
struct CondensedTree {
    parent: Vec<u32>,
    child: Vec<u32>,
    distance: Vec<f64>,
    child_size: Vec<f64>,
    cluster_rows: Vec<u32>,
}

#[derive(Debug, Clone)]
struct LeafTree {
    parent: Vec<u32>,
    min_distance: Vec<f64>,
    max_distance: Vec<f64>,
    min_size: Vec<f64>,
    max_size: Vec<f64>,
}

#[derive(Debug, Clone)]
struct PersistenceTrace {
    min_size: Vec<f64>,
    persistence: Vec<f64>,
}

/// PLSCAN clustering (scikit-like interface).
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct PLSCAN {
    min_samples: usize,
    min_cluster_size: Option<f64>,
    max_cluster_size: Option<f64>,
    persistence_measure: String,

    labels: Option<Array1<i64>>,
    probabilities: Option<Array1<f64>>,
    selected_min_cluster_size: Option<f64>,
    trace_min_size: Option<Array1<f64>>,
    trace_persistence: Option<Array1<f64>>,
}

#[pymethods]
impl PLSCAN {
    #[new]
    #[pyo3(signature = (min_samples=5, min_cluster_size=None, max_cluster_size=None, persistence_measure="size"))]
    pub fn new(
        min_samples: usize,
        min_cluster_size: Option<f64>,
        max_cluster_size: Option<f64>,
        persistence_measure: &str,
    ) -> PyResult<Self> {
        if min_samples < 2 {
            return Err(PyValueError::new_err("min_samples must be >= 2"));
        }
        let _ = PersistenceMeasure::parse(persistence_measure).ok_or_else(|| {
            PyValueError::new_err(
                "persistence_measure must be one of: \"size\", \"distance\", \"density\", \"size-distance\", \"size-density\"",
            )
        })?;

        Ok(Self {
            min_samples,
            min_cluster_size,
            max_cluster_size,
            persistence_measure: persistence_measure.to_string(),
            labels: None,
            probabilities: None,
            selected_min_cluster_size: None,
            trace_min_size: None,
            trace_persistence: None,
        })
    }

    pub fn fit(&mut self, _py: Python<'_>, data: PyReadonlyArray2<f64>) -> PyResult<()> {
        let data_vec = ndarray_to_vecvec_f32(data)?;
        let measure = PersistenceMeasure::parse(&self.persistence_measure).unwrap();

        let min_cluster_size = self
            .min_cluster_size
            .unwrap_or(self.min_samples as f64);
        if min_cluster_size < self.min_samples as f64 {
            return Err(PyValueError::new_err(
                "min_cluster_size must be >= min_samples",
            ));
        }
        let max_cluster_size = self.max_cluster_size.unwrap_or(f64::INFINITY);
        if max_cluster_size <= min_cluster_size {
            return Err(PyValueError::new_err(
                "max_cluster_size must be > min_cluster_size",
            ));
        }

        let res = cluster_internal(
            &data_vec,
            self.min_samples,
            min_cluster_size,
            max_cluster_size,
            measure,
        )
        .map_err(PyValueError::new_err)?;

        self.labels = Some(Array1::from_vec(res.labels));
        self.probabilities = Some(Array1::from_vec(res.probabilities));
        self.selected_min_cluster_size = res.selected_min_cluster_size;
        self.trace_min_size = Some(Array1::from_vec(res.trace_min_size));
        self.trace_persistence = Some(Array1::from_vec(res.trace_persistence));
        Ok(())
    }

    pub fn fit_predict<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        self.fit(py, data)?;
        self.labels_(py)
    }

    #[getter]
    pub fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let labels = self
            .labels
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("PLSCAN not fitted"))?;
        Ok(labels.clone().into_pyarray_bound(py))
    }

    #[getter]
    pub fn probabilities_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let probs = self
            .probabilities
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("PLSCAN not fitted"))?;
        Ok(probs.clone().into_pyarray_bound(py))
    }

    #[getter]
    pub fn selected_min_cluster_size_(&self) -> Option<f64> {
        self.selected_min_cluster_size
    }

    #[getter]
    pub fn trace_min_size_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let v = self
            .trace_min_size
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("PLSCAN not fitted"))?;
        Ok(v.clone().into_pyarray_bound(py))
    }

    #[getter]
    pub fn trace_persistence_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let v = self
            .trace_persistence
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("PLSCAN not fitted"))?;
        Ok(v.clone().into_pyarray_bound(py))
    }
}

pub fn cluster_internal(
    data: &[Vec<f32>],
    min_samples: usize,
    min_cluster_size: f64,
    max_cluster_size: f64,
    persistence_measure: PersistenceMeasure,
) -> Result<PlscanClusterResult, String> {
    validate_data(data)?;
    if min_samples < 2 {
        return Err("min_samples must be >= 2".to_string());
    }
    if data.len() <= min_samples {
        return Err("data must have more rows than min_samples".to_string());
    }
    if min_cluster_size <= 0.0 {
        return Err("min_cluster_size must be > 0".to_string());
    }
    if max_cluster_size <= min_cluster_size {
        return Err("max_cluster_size must be > min_cluster_size".to_string());
    }

    let (knn, core_distances) = compute_knn_and_core_distances(data, min_samples)?;
    let edges = build_mutual_edges(&knn, &core_distances);
    if edges.is_empty() {
        return Err("empty mutual reachability graph".to_string());
    }

    let mst = compute_spanning_forest(data.len(), &edges);
    if mst.distance.is_empty() {
        return Err("spanning forest is empty".to_string());
    }

    let linkage = compute_linkage_tree(&mst, data.len());
    let condensed = compute_condensed_tree(&linkage, &mst, data.len(), min_cluster_size);
    let leaf_tree = if condensed.cluster_rows.is_empty() {
        leaf_tree_fallback(&mst, data.len(), min_cluster_size)
    } else {
        compute_leaf_tree(&condensed, data.len(), min_cluster_size)?
    };

    let trace = match persistence_measure {
        PersistenceMeasure::Size => compute_size_persistence(&leaf_tree),
        PersistenceMeasure::Distance => compute_distance_persistence(&leaf_tree, &condensed, data.len()),
        PersistenceMeasure::Density => compute_density_persistence(&leaf_tree, &condensed, data.len()),
        PersistenceMeasure::SizeDistance => compute_size_distance_bi_persistence(&leaf_tree, &condensed, data.len()),
        PersistenceMeasure::SizeDensity => compute_size_density_bi_persistence(&leaf_tree, &condensed, data.len()),
    };

    let best_birth = best_min_cluster_size(&trace, max_cluster_size);
    let selected_clusters = best_birth
        .map(|birth| apply_size_cut(&leaf_tree, birth))
        .unwrap_or_default();

    let (labels, probabilities) =
        compute_cluster_labels(&leaf_tree, &condensed, &selected_clusters, data.len());

    Ok(PlscanClusterResult {
        labels,
        probabilities,
        selected_min_cluster_size: best_birth,
        trace_min_size: trace.min_size,
        trace_persistence: trace.persistence,
    })
}

fn ndarray_to_vecvec_f32(data: PyReadonlyArray2<f64>) -> PyResult<Vec<Vec<f32>>> {
    let x = data.as_array();
    let (n_samples, n_features) = (x.nrows(), x.ncols());
    if n_samples == 0 {
        return Err(PyValueError::new_err("data must not be empty"));
    }
    if n_features == 0 {
        return Err(PyValueError::new_err("data must have at least 1 column"));
    }

    let mut out = Vec::with_capacity(n_samples);
    for row in x.rows() {
        let mut v = Vec::with_capacity(n_features);
        for &val in row.iter() {
            if !val.is_finite() {
                return Err(PyValueError::new_err("data must contain only finite values"));
            }
            v.push(val as f32);
        }
        out.push(v);
    }
    Ok(out)
}

fn validate_data(data: &[Vec<f32>]) -> Result<(), String> {
    if data.is_empty() {
        return Err("data must not be empty".to_string());
    }
    if data.len() < 2 {
        return Err("data must have at least 2 rows".to_string());
    }
    let dim = data[0].len();
    if dim == 0 {
        return Err("data must have at least 1 column".to_string());
    }
    for row in data {
        if row.len() != dim {
            return Err("all rows must have the same length".to_string());
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err("data must contain only finite values".to_string());
        }
    }
    Ok(())
}

fn compute_knn_and_core_distances(
    data: &[Vec<f32>],
    k: usize,
) -> Result<(Vec<Vec<(usize, f64)>>, Vec<f64>), String> {
    let n = data.len();
    if k == 0 {
        return Err("k must be >= 1".to_string());
    }
    if n <= k {
        return Err("data must have more rows than k".to_string());
    }

    let results: Vec<(Vec<(usize, f64)>, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d = metrics_simd::euclidean(&data[i], &data[j])
                        .unwrap_or(f32::INFINITY) as f64;
                    (j, d)
                })
                .collect();

            let kth = k - 1;
            dists.select_nth_unstable_by(kth, |a, b| a.1.total_cmp(&b.1));
            let mut nearest = dists[..k].to_vec();
            nearest.sort_by(|a, b| a.1.total_cmp(&b.1));
            let core = nearest[k - 1].1;
            (nearest, core)
        })
        .collect();

    let mut knn = Vec::with_capacity(n);
    let mut core = Vec::with_capacity(n);
    for (neighbors, c) in results {
        knn.push(neighbors);
        core.push(c);
    }
    Ok((knn, core))
}

fn build_mutual_edges(knn: &[Vec<(usize, f64)>], core: &[f64]) -> Vec<Edge> {
    let n = knn.len();
    let mut edges = Vec::with_capacity(n * knn[0].len());
    for i in 0..n {
        for &(j, dist) in &knn[i] {
            let u = (i as u32).min(j as u32);
            let v = (i as u32).max(j as u32);
            let w = dist.max(core[i]).max(core[j]);
            edges.push(Edge { u, v, w });
        }
    }

    // Deduplicate (keep smallest w per undirected pair).
    edges.sort_by(|a, b| {
        a.u.cmp(&b.u)
            .then_with(|| a.v.cmp(&b.v))
            .then_with(|| a.w.total_cmp(&b.w))
    });
    let mut uniq: Vec<Edge> = Vec::with_capacity(edges.len());
    for e in edges {
        if let Some(last) = uniq.last() {
            if last.u == e.u && last.v == e.v {
                continue;
            }
        }
        uniq.push(e);
    }

    uniq.sort_by(|a, b| a.w.total_cmp(&b.w));
    uniq
}

#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0u8; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut node = x;
        while self.parent[node] != root {
            let next = self.parent[node];
            self.parent[node] = root;
            node = next;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return false;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] = self.rank[ra].saturating_add(1);
        }
        true
    }
}

fn compute_spanning_forest(num_points: usize, edges: &[Edge]) -> SpanningTree {
    let mut uf = UnionFind::new(num_points);
    let mut parent = Vec::with_capacity(num_points.saturating_sub(1));
    let mut child = Vec::with_capacity(num_points.saturating_sub(1));
    let mut distance = Vec::with_capacity(num_points.saturating_sub(1));

    for e in edges {
        let u = e.u as usize;
        let v = e.v as usize;
        if uf.union(u, v) {
            parent.push(e.u);
            child.push(e.v);
            distance.push(e.w);
        }
    }

    SpanningTree {
        parent,
        child,
        distance,
    }
}

fn compute_linkage_tree(mst: &SpanningTree, num_points: usize) -> LinkageTree {
    let num_edges = mst.parent.len();
    let total_nodes = num_points + num_edges;

    let mut dsu_parent: Vec<u32> = (0..total_nodes as u32).collect();
    let mut count_by_label = vec![0u32; total_nodes];
    let mut size_by_label = vec![0.0f64; total_nodes];
    for i in 0..num_points {
        count_by_label[i] = 1;
        size_by_label[i] = 1.0;
    }

    let mut left = vec![0u32; num_edges];
    let mut right = vec![0u32; num_edges];
    let mut child_count = vec![0u32; num_edges];
    let mut child_size = vec![0.0f64; num_edges];

    for idx in 0..num_edges {
        let next = (num_points + idx) as u32;
        let a = dsu_find(&mut dsu_parent, mst.parent[idx]);
        let b = dsu_find(&mut dsu_parent, mst.child[idx]);
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        left[idx] = lo;
        right[idx] = hi;

        let a_usize = a as usize;
        let b_usize = b as usize;
        let next_usize = next as usize;

        dsu_parent[a_usize] = next;
        dsu_parent[b_usize] = next;
        dsu_parent[next_usize] = next;

        count_by_label[next_usize] = count_by_label[a_usize] + count_by_label[b_usize];
        size_by_label[next_usize] = size_by_label[a_usize] + size_by_label[b_usize];
        child_count[idx] = count_by_label[next_usize];
        child_size[idx] = size_by_label[next_usize];
    }

    LinkageTree {
        left,
        right,
        child_count,
        child_size,
    }
}

fn dsu_find(parent: &mut [u32], mut node: u32) -> u32 {
    let mut root = node as usize;
    while parent[root] != root as u32 {
        root = parent[root] as usize;
    }
    while parent[node as usize] != root as u32 {
        let next = parent[node as usize];
        parent[node as usize] = root as u32;
        node = next;
    }
    root as u32
}

#[derive(Debug, Clone)]
struct RowInfo {
    parent: u32,
    distance: f64,
    size: f64,
    left: u32,
    left_count: u32,
    left_size: f64,
    right: u32,
    right_count: u32,
    right_size: f64,
}

fn compute_condensed_tree(
    linkage_tree: &LinkageTree,
    spanning_tree: &SpanningTree,
    num_points: usize,
    min_cluster_size: f64,
) -> CondensedTree {
    let num_edges = linkage_tree.left.len();
    let buffer_size = 2 * num_edges;

    let mut parent_out = vec![0u32; buffer_size];
    let mut child_out = vec![0u32; buffer_size];
    let mut dist_out = vec![0.0f64; buffer_size];
    let mut size_out = vec![0.0f64; buffer_size];
    let mut cluster_rows = vec![0u32; num_edges];

    let mut parent_of = vec![num_points as u32; num_edges];
    let mut pending_idx = vec![0usize; num_edges];
    let mut pending_distance = vec![0.0f64; num_edges];

    let mut next_label = num_points as u32;
    let mut cluster_count = 0usize;
    let mut idx_out = 0usize;

    for rev in 0..num_edges {
        let node_idx = num_edges - 1 - rev;
        let mut row = get_row(node_idx, linkage_tree, spanning_tree, num_points, &parent_of);

        let mut out_idx = if row.size < min_cluster_size {
            let out = pending_idx[node_idx];
            row.distance = pending_distance[node_idx];
            out
        } else {
            let out = idx_out;
            idx_out += if row.left_size < min_cluster_size {
                row.left_count as usize
            } else {
                0
            };
            idx_out += if row.right_size < min_cluster_size {
                row.right_count as usize
            } else {
                0
            };
            out
        };

        store_or_delay(
            &row,
            &mut out_idx,
            num_points,
            min_cluster_size,
            &mut parent_out,
            &mut child_out,
            &mut dist_out,
            &mut size_out,
            &mut parent_of,
            &mut pending_idx,
            &mut pending_distance,
        );

        if row.left_size >= min_cluster_size && row.right_size >= min_cluster_size {
            write_merge(
                &row,
                &mut idx_out,
                &mut cluster_count,
                &mut next_label,
                num_points,
                &mut parent_out,
                &mut child_out,
                &mut dist_out,
                &mut size_out,
                &mut cluster_rows,
                &mut parent_of,
            );
        }
    }

    parent_out.truncate(idx_out);
    child_out.truncate(idx_out);
    dist_out.truncate(idx_out);
    size_out.truncate(idx_out);
    cluster_rows.truncate(cluster_count);

    CondensedTree {
        parent: parent_out,
        child: child_out,
        distance: dist_out,
        child_size: size_out,
        cluster_rows,
    }
}

fn get_row(
    node_idx: usize,
    linkage_tree: &LinkageTree,
    spanning_tree: &SpanningTree,
    num_points: usize,
    parent_of: &[u32],
) -> RowInfo {
    let left = linkage_tree.left[node_idx];
    let right = linkage_tree.right[node_idx];

    let (left_count, left_size) = if (left as usize) < num_points {
        (1u32, 1.0)
    } else {
        let idx = (left as usize) - num_points;
        (linkage_tree.child_count[idx], linkage_tree.child_size[idx])
    };
    let (right_count, right_size) = if (right as usize) < num_points {
        (1u32, 1.0)
    } else {
        let idx = (right as usize) - num_points;
        (linkage_tree.child_count[idx], linkage_tree.child_size[idx])
    };

    RowInfo {
        parent: parent_of[node_idx],
        distance: spanning_tree.distance[node_idx],
        size: linkage_tree.child_size[node_idx],
        left,
        left_count,
        left_size,
        right,
        right_count,
        right_size,
    }
}

#[allow(clippy::too_many_arguments)]
fn store_or_delay(
    row: &RowInfo,
    out_idx: &mut usize,
    num_points: usize,
    min_cluster_size: f64,
    parent_out: &mut [u32],
    child_out: &mut [u32],
    dist_out: &mut [f64],
    size_out: &mut [f64],
    parent_of: &mut [u32],
    pending_idx: &mut [usize],
    pending_distance: &mut [f64],
) {
    if (row.left as usize) < num_points {
        write_row(
            *out_idx,
            row.parent,
            row.distance,
            row.left,
            row.left_size,
            parent_out,
            child_out,
            dist_out,
            size_out,
        );
        *out_idx += 1;
    } else {
        delay_row(
            out_idx,
            row.parent,
            row.distance,
            row.left,
            row.left_count,
            row.left_size,
            num_points,
            min_cluster_size,
            parent_of,
            pending_idx,
            pending_distance,
        );
    }

    if (row.right as usize) < num_points {
        write_row(
            *out_idx,
            row.parent,
            row.distance,
            row.right,
            row.right_size,
            parent_out,
            child_out,
            dist_out,
            size_out,
        );
        *out_idx += 1;
    } else {
        delay_row(
            out_idx,
            row.parent,
            row.distance,
            row.right,
            row.right_count,
            row.right_size,
            num_points,
            min_cluster_size,
            parent_of,
            pending_idx,
            pending_distance,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn write_row(
    out_idx: usize,
    parent: u32,
    distance: f64,
    child: u32,
    child_size: f64,
    parent_out: &mut [u32],
    child_out: &mut [u32],
    dist_out: &mut [f64],
    size_out: &mut [f64],
) {
    parent_out[out_idx] = parent;
    child_out[out_idx] = child;
    dist_out[out_idx] = distance;
    size_out[out_idx] = child_size;
}

#[allow(clippy::too_many_arguments)]
fn delay_row(
    out_idx: &mut usize,
    parent: u32,
    distance: f64,
    child: u32,
    child_count: u32,
    child_size: f64,
    num_points: usize,
    min_cluster_size: f64,
    parent_of: &mut [u32],
    pending_idx: &mut [usize],
    pending_distance: &mut [f64],
) {
    let child_idx = (child as usize) - num_points;
    parent_of[child_idx] = parent;
    if child_size < min_cluster_size {
        pending_idx[child_idx] = *out_idx;
        pending_distance[child_idx] = distance;
        *out_idx += child_count as usize;
    }
}

#[allow(clippy::too_many_arguments)]
fn write_merge(
    row: &RowInfo,
    idx_out: &mut usize,
    cluster_count: &mut usize,
    next_label: &mut u32,
    num_points: usize,
    parent_out: &mut [u32],
    child_out: &mut [u32],
    dist_out: &mut [f64],
    size_out: &mut [f64],
    cluster_rows: &mut [u32],
    parent_of: &mut [u32],
) {
    let phantom_root = num_points as u32;
    let parent_label = if row.parent == phantom_root {
        *next_label += 1;
        *next_label
    } else {
        row.parent
    };

    // Left cluster row.
    parent_of[(row.left as usize) - num_points] = {
        *next_label += 1;
        *next_label
    };
    cluster_rows[*cluster_count] = *idx_out as u32;
    *cluster_count += 1;
    write_row(
        *idx_out,
        parent_label,
        row.distance,
        *next_label,
        row.left_size,
        parent_out,
        child_out,
        dist_out,
        size_out,
    );
    *idx_out += 1;

    // Right cluster row.
    parent_of[(row.right as usize) - num_points] = {
        *next_label += 1;
        *next_label
    };
    cluster_rows[*cluster_count] = *idx_out as u32;
    *cluster_count += 1;
    write_row(
        *idx_out,
        parent_label,
        row.distance,
        *next_label,
        row.right_size,
        parent_out,
        child_out,
        dist_out,
        size_out,
    );
    *idx_out += 1;
}

fn compute_leaf_tree(
    condensed_tree: &CondensedTree,
    num_points: usize,
    min_cluster_size: f64,
) -> Result<LeafTree, String> {
    if condensed_tree.cluster_rows.is_empty() {
        return Err("condensed tree has no cluster rows".to_string());
    }
    let last_cluster_row = *condensed_tree
        .cluster_rows
        .last()
        .expect("checked non-empty") as usize;
    let max_label = (condensed_tree.child[last_cluster_row] as usize)
        .checked_sub(num_points)
        .ok_or_else(|| "invalid condensed tree labels".to_string())?;
    let num_clusters = max_label + 1;

    let mut parent = vec![0u32; num_clusters];
    let mut min_distance = vec![0.0f64; num_clusters];
    let mut max_distance = vec![0.0f64; num_clusters];
    let mut min_size = vec![0.0f64; num_clusters];
    let mut max_size = vec![0.0f64; num_clusters];

    fill_min_dist(&mut min_distance, condensed_tree, num_points)?;
    fill_parent_and_max_dist(&mut parent, &mut max_distance, condensed_tree, num_points)?;
    fill_sizes(
        &mut min_size,
        &mut max_size,
        &parent,
        condensed_tree,
        num_points,
        min_cluster_size,
    )?;

    Ok(LeafTree {
        parent,
        min_distance,
        max_distance,
        min_size,
        max_size,
    })
}

fn leaf_tree_fallback(mst: &SpanningTree, num_points: usize, min_cluster_size: f64) -> LeafTree {
    let min_distance = mst.distance.first().copied().unwrap_or(0.0);
    let max_distance = mst.distance.last().copied().unwrap_or(min_distance);
    LeafTree {
        parent: vec![0u32],
        min_distance: vec![min_distance],
        max_distance: vec![max_distance],
        min_size: vec![min_cluster_size],
        max_size: vec![num_points as f64],
    }
}

fn fill_min_dist(
    min_distance: &mut [f64],
    condensed_tree: &CondensedTree,
    num_points: usize,
) -> Result<(), String> {
    for idx in 0..condensed_tree.parent.len() {
        let parent_idx = (condensed_tree.parent[idx] as usize)
            .checked_sub(num_points)
            .ok_or_else(|| "invalid condensed tree labels".to_string())?;
        if parent_idx >= min_distance.len() {
            return Err("condensed tree parent index out of bounds".to_string());
        }
        min_distance[parent_idx] = condensed_tree.distance[idx];
    }
    Ok(())
}

fn fill_parent_and_max_dist(
    parent: &mut [u32],
    max_distance: &mut [f64],
    condensed_tree: &CondensedTree,
    num_points: usize,
) -> Result<(), String> {
    parent.fill(0u32);
    let default_max = condensed_tree.distance.first().copied().unwrap_or(0.0);
    max_distance.fill(default_max);

    for &row_idx_u32 in &condensed_tree.cluster_rows {
        let row_idx = row_idx_u32 as usize;
        let child_idx = (condensed_tree.child[row_idx] as usize)
            .checked_sub(num_points)
            .ok_or_else(|| "invalid condensed tree labels".to_string())?;
        parent[child_idx] = condensed_tree.parent[row_idx]
            .checked_sub(num_points as u32)
            .ok_or_else(|| "invalid condensed tree labels".to_string())?;
        max_distance[child_idx] = condensed_tree.distance[row_idx];
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn fill_sizes(
    min_size: &mut [f64],
    max_size: &mut [f64],
    parent: &[u32],
    condensed_tree: &CondensedTree,
    num_points: usize,
    min_cluster_size: f64,
) -> Result<(), String> {
    min_size.fill(min_cluster_size);

    let num_rows = condensed_tree.cluster_rows.len();
    if num_rows % 2 != 0 {
        return Err("condensed tree cluster_rows must have even length".to_string());
    }

    let mut step = 1usize;
    while step <= num_rows {
        let row_idx = num_rows - step;
        let left_idx = condensed_tree.cluster_rows[row_idx] as usize;
        let right_idx = condensed_tree.cluster_rows[row_idx - 1] as usize;

        let size = condensed_tree.child_size[left_idx].min(condensed_tree.child_size[right_idx]);
        let out_idx = (condensed_tree.child[left_idx] as usize)
            .checked_sub(num_points)
            .ok_or_else(|| "invalid condensed tree labels".to_string())?;
        let parent_idx = (condensed_tree.parent[left_idx] as usize)
            .checked_sub(num_points)
            .ok_or_else(|| "invalid condensed tree labels".to_string())?;
        if out_idx == 0 {
            return Err("unexpected condensed tree child index 0".to_string());
        }

        max_size[out_idx] = size;
        max_size[out_idx - 1] = size;
        min_size[parent_idx] =
            min_size[parent_idx].max(size.max(min_size[out_idx - 1]).max(min_size[out_idx]));
        if parent[parent_idx] == 0 {
            min_size[0] = min_size[0].max(min_size[parent_idx]);
        }

        step += 2;
    }

    max_size[0] = num_points as f64;
    for idx in 1..max_size.len() {
        if parent[idx] == 0 {
            max_size[idx] = min_size[0];
        }
    }
    Ok(())
}

fn apply_size_cut(leaf_tree: &LeafTree, cut_size: f64) -> Vec<u32> {
    let mut selected = Vec::new();
    for idx in 0..leaf_tree.parent.len() {
        if leaf_tree.min_size[idx] <= cut_size && leaf_tree.max_size[idx] > cut_size {
            selected.push(idx as u32);
        }
    }
    selected
}

fn compute_size_persistence(leaf_tree: &LeafTree) -> PersistenceTrace {
    let (mut min_size, mut persistence) = initialize_trace(leaf_tree);
    fill_persistences(&mut min_size, &mut persistence, leaf_tree, |idx| {
        if leaf_tree.parent[idx] > 0 {
            leaf_tree.max_size[idx] - leaf_tree.min_size[idx]
        } else {
            0.0
        }
    });
    PersistenceTrace { min_size, persistence }
}

fn compute_distance_persistence(
    leaf_tree: &LeafTree,
    condensed_tree: &CondensedTree,
    num_points: usize,
) -> PersistenceTrace {
    let persistences =
        compute_persistences(leaf_tree, condensed_tree, num_points, distance_persistence);
    let (mut min_size, mut persistence) = initialize_trace(leaf_tree);
    fill_persistences(&mut min_size, &mut persistence, leaf_tree, |idx| persistences[idx]);
    PersistenceTrace { min_size, persistence }
}

fn compute_density_persistence(
    leaf_tree: &LeafTree,
    condensed_tree: &CondensedTree,
    num_points: usize,
) -> PersistenceTrace {
    let persistences = compute_persistences(leaf_tree, condensed_tree, num_points, density_persistence);
    let (mut min_size, mut persistence) = initialize_trace(leaf_tree);
    fill_persistences(&mut min_size, &mut persistence, leaf_tree, |idx| persistences[idx]);
    PersistenceTrace { min_size, persistence }
}

fn compute_size_distance_bi_persistence(
    leaf_tree: &LeafTree,
    condensed_tree: &CondensedTree,
    num_points: usize,
) -> PersistenceTrace {
    let bi = compute_bi_persistences(leaf_tree, condensed_tree, num_points, distance_persistence);
    let (mut min_size, mut persistence) = initialize_trace(leaf_tree);
    fill_persistences(&mut min_size, &mut persistence, leaf_tree, |idx| bi[idx]);
    PersistenceTrace { min_size, persistence }
}

fn compute_size_density_bi_persistence(
    leaf_tree: &LeafTree,
    condensed_tree: &CondensedTree,
    num_points: usize,
) -> PersistenceTrace {
    let bi = compute_bi_persistences(leaf_tree, condensed_tree, num_points, density_persistence);
    let (mut min_size, mut persistence) = initialize_trace(leaf_tree);
    fill_persistences(&mut min_size, &mut persistence, leaf_tree, |idx| bi[idx]);
    PersistenceTrace { min_size, persistence }
}

fn initialize_trace(leaf_tree: &LeafTree) -> (Vec<f64>, Vec<f64>) {
    let num_leaves = leaf_tree.parent.len();
    let mut thresholds = Vec::with_capacity(num_leaves.saturating_sub(1) * 2);
    for idx in 1..num_leaves {
        thresholds.push(leaf_tree.min_size[idx]);
        thresholds.push(leaf_tree.max_size[idx]);
    }
    thresholds.sort_by(|a, b| a.total_cmp(b));
    thresholds.dedup();
    let persistence = vec![0.0f64; thresholds.len()];
    (thresholds, persistence)
}

fn fill_persistences(
    thresholds: &mut [f64],
    persistence: &mut [f64],
    leaf_tree: &LeafTree,
    mut get_persistence: impl FnMut(usize) -> f64,
) {
    for idx in 1..leaf_tree.parent.len() {
        let birth = leaf_tree.min_size[idx];
        let death = leaf_tree.max_size[idx];
        if death <= birth {
            continue;
        }
        let p = get_persistence(idx);
        if p == 0.0 {
            continue;
        }

        let start = lower_bound(thresholds, birth);
        let end = start + lower_bound(&thresholds[start..], death);
        for v in &mut persistence[start..end] {
            *v += p;
        }
    }
}

fn lower_bound(sorted: &[f64], value: f64) -> usize {
    let mut left = 0usize;
    let mut right = sorted.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if sorted[mid] < value {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn upper_bound(sorted: &[f64], value: f64) -> usize {
    let mut left = 0usize;
    let mut right = sorted.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if sorted[mid] <= value {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn best_min_cluster_size(trace: &PersistenceTrace, max_cluster_size: f64) -> Option<f64> {
    let idx = upper_bound(&trace.min_size, max_cluster_size);
    if idx == 0 {
        return None;
    }
    let (best_idx, _) = trace.persistence[..idx]
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();
    Some(trace.min_size[best_idx])
}

#[inline]
fn distance_persistence(min_dist: f64, max_dist: f64) -> f64 {
    max_dist - min_dist
}

#[inline]
fn density_persistence(min_dist: f64, max_dist: f64) -> f64 {
    (-min_dist).exp() - (-max_dist).exp()
}

fn collect_leaf_children(
    leaf_tree: &LeafTree,
    condensed_tree: &CondensedTree,
    num_points: usize,
    mut pre_callback: impl FnMut(usize, f64, f64, f64),
    mut post_callback: impl FnMut(usize, f64, f64, f64),
) {
    let num_rows = condensed_tree.parent.len();
    let num_leaves = leaf_tree.parent.len();
    let mut collected = vec![0.0f64; num_leaves];

    for rev in 0..num_rows {
        let idx = num_rows - 1 - rev;
        let child = condensed_tree.child[idx] as usize;
        if child >= num_points {
            continue;
        }

        let distance = condensed_tree.distance[idx];
        let mut leaf_idx = condensed_tree.parent[idx] as usize - num_points;
        while leaf_tree.parent[leaf_idx] > 0 {
            let weight = condensed_tree.child_size[idx];
            pre_callback(leaf_idx, collected[leaf_idx], distance, weight);
            collected[leaf_idx] += weight;
            post_callback(leaf_idx, collected[leaf_idx], distance, weight);
            leaf_idx = leaf_tree.parent[leaf_idx] as usize;
        }
    }
}

fn compute_persistences(
    leaf_tree: &LeafTree,
    condensed_tree: &CondensedTree,
    num_points: usize,
    to_persistence: fn(f64, f64) -> f64,
) -> Vec<f64> {
    let num_leaves = leaf_tree.parent.len();
    let mut min_dists = vec![0.0f64; num_leaves];
    collect_leaf_children(
        leaf_tree,
        condensed_tree,
        num_points,
        |idx, size, distance, _weight| {
            if size <= leaf_tree.min_size[idx] {
                min_dists[idx] = distance;
            }
        },
        |_idx, _size, _distance, _weight| {},
    );

    let mut out = vec![0.0f64; num_leaves];
    for i in 0..num_leaves {
        out[i] = to_persistence(min_dists[i], leaf_tree.max_distance[i]);
    }
    out
}

fn compute_bi_persistences(
    leaf_tree: &LeafTree,
    condensed_tree: &CondensedTree,
    num_points: usize,
    persistence_callback: fn(f64, f64) -> f64,
) -> Vec<f64> {
    let num_leaves = leaf_tree.parent.len();
    let mut bi = vec![0.0f64; num_leaves];

    collect_leaf_children(
        leaf_tree,
        condensed_tree,
        num_points,
        |_idx, _size, _distance, _weight| {},
        |idx, size, distance, weight| {
            if size > leaf_tree.min_size[idx] && size <= leaf_tree.max_size[idx] {
                bi[idx] += weight * persistence_callback(distance, leaf_tree.max_distance[idx]);
            }
        },
    );

    bi
}

fn compute_cluster_labels(
    leaf_tree: &LeafTree,
    condensed_tree: &CondensedTree,
    selected_clusters: &[u32],
    num_points: usize,
) -> (Vec<i64>, Vec<f64>) {
    let segment_labels = compute_segment_labels(leaf_tree, selected_clusters);
    let leaf_persistence = compute_leaf_persistence(leaf_tree, selected_clusters);

    let mut labels = vec![-1i64; num_points];
    let mut probabilities = vec![0.0f64; num_points];

    for idx in 0..condensed_tree.parent.len() {
        let child = condensed_tree.child[idx] as usize;
        if child >= num_points {
            continue;
        }

        let parent_idx = condensed_tree.parent[idx] as usize - num_points;
        let label = segment_labels[parent_idx];
        labels[child] = label;

        if label >= 0 {
            let label_usize = label as usize;
            let seg_idx = selected_clusters[label_usize] as usize;
            let max_dist = leaf_tree.max_distance[seg_idx];
            let point_persistence = max_dist - condensed_tree.distance[idx];
            let denom = leaf_persistence[label_usize];
            if denom > 0.0 {
                probabilities[child] = (point_persistence / denom).min(1.0);
            }
        }
    }

    (labels, probabilities)
}

fn compute_segment_labels(leaf_tree: &LeafTree, selected: &[u32]) -> Vec<i64> {
    let num_segments = leaf_tree.parent.len();
    let mut segment_labels = vec![0i64; num_segments];
    segment_labels[0] = -1;

    let mut label = 0usize;
    for segment_idx in 1..num_segments {
        if label < selected.len() && selected[label] as usize == segment_idx {
            segment_labels[segment_idx] = label as i64;
            label += 1;
        } else {
            let p = leaf_tree.parent[segment_idx] as usize;
            segment_labels[segment_idx] = segment_labels[p];
        }
    }
    segment_labels
}

fn compute_leaf_persistence(leaf_tree: &LeafTree, selected: &[u32]) -> Vec<f64> {
    let mut persistence = vec![0.0f64; selected.len()];
    for (label, &segment_idx_u32) in selected.iter().enumerate() {
        let segment_idx = segment_idx_u32 as usize;
        persistence[label] = leaf_tree.max_distance[segment_idx] - leaf_tree.min_distance[segment_idx];
    }
    persistence
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plscan_cluster_shapes() {
        // Two well-separated blobs on a line.
        let mut x = Vec::new();
        for i in 0..20 {
            x.push(vec![i as f32 * 0.01, 0.0]);
        }
        for i in 0..20 {
            x.push(vec![100.0 + i as f32 * 0.01, 0.0]);
        }

        let res = cluster_internal(&x, 3, 3.0, f64::INFINITY, PersistenceMeasure::Size).unwrap();
        assert_eq!(res.labels.len(), 40);
        assert_eq!(res.probabilities.len(), 40);
        assert_eq!(res.trace_min_size.len(), res.trace_persistence.len());
        assert!(res.probabilities.iter().all(|p| (0.0..=1.0).contains(p)));
    }
}
