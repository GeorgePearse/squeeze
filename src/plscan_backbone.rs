//! Dimensionality reduction pipeline backed by PLSCAN clustering.
//!
//! Implements:
//! - Landmark / dataset compression via PLSCAN clusters
//! - Scale selection via PLSCAN's most persistent min cluster size
//! - Soft must-link regularization via cluster memberships

use std::collections::BTreeMap;

use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::metrics_simd;
use crate::plscan::{cluster_internal, PersistenceMeasure};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RepresentativeStrategy {
    HighProb,
    Center,
    Medoid,
}

impl RepresentativeStrategy {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "high_prob" | "high-prob" | "highprob" => Some(Self::HighProb),
            "center" | "centroid" => Some(Self::Center),
            "medoid" => Some(Self::Medoid),
            _ => None,
        }
    }
}

/// A PLSCAN-backed landmark + spectral dimensionality reducer.
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct PLSCANBackbone {
    n_components: usize,
    min_samples: usize,
    min_cluster_size: Option<f64>,
    max_cluster_size: Option<f64>,
    persistence_measure: String,

    rep_strategy: String,
    reps_per_cluster: usize,
    include_noise: bool,
    noise_samples: usize,

    n_neighbors: Option<usize>,
    neighbor_scale: f64,
    must_link_weight: f64,

    interpolation_k: usize,
    restrict_to_cluster: bool,

    embedding: Option<Array2<f64>>,
    labels: Option<Array1<i64>>,
    probabilities: Option<Array1<f64>>,
    rep_indices: Option<Array1<i64>>,
    n_neighbors_used: Option<usize>,
    selected_min_cluster_size: Option<f64>,
    trace_min_size: Option<Array1<f64>>,
    trace_persistence: Option<Array1<f64>>,
}

#[pymethods]
impl PLSCANBackbone {
    #[new]
    #[pyo3(signature = (
        n_components=2,
        min_samples=5,
        min_cluster_size=None,
        max_cluster_size=None,
        persistence_measure="size",
        rep_strategy="high_prob",
        reps_per_cluster=1,
        include_noise=false,
        noise_samples=0,
        n_neighbors=None,
        neighbor_scale=1.0,
        must_link_weight=0.0,
        interpolation_k=3,
        restrict_to_cluster=true
    ))]
    pub fn new(
        n_components: usize,
        min_samples: usize,
        min_cluster_size: Option<f64>,
        max_cluster_size: Option<f64>,
        persistence_measure: &str,
        rep_strategy: &str,
        reps_per_cluster: usize,
        include_noise: bool,
        noise_samples: usize,
        n_neighbors: Option<usize>,
        neighbor_scale: f64,
        must_link_weight: f64,
        interpolation_k: usize,
        restrict_to_cluster: bool,
    ) -> PyResult<Self> {
        if n_components == 0 {
            return Err(PyValueError::new_err("n_components must be >= 1"));
        }
        if min_samples < 2 {
            return Err(PyValueError::new_err("min_samples must be >= 2"));
        }
        if reps_per_cluster == 0 {
            return Err(PyValueError::new_err("reps_per_cluster must be >= 1"));
        }
        if interpolation_k == 0 {
            return Err(PyValueError::new_err("interpolation_k must be >= 1"));
        }
        if neighbor_scale <= 0.0 {
            return Err(PyValueError::new_err("neighbor_scale must be > 0"));
        }
        let _ = PersistenceMeasure::parse(persistence_measure).ok_or_else(|| {
            PyValueError::new_err(
                "persistence_measure must be one of: \"size\", \"distance\", \"density\", \"size-distance\", \"size-density\"",
            )
        })?;
        let _ = RepresentativeStrategy::parse(rep_strategy).ok_or_else(|| {
            PyValueError::new_err("rep_strategy must be one of: \"high_prob\", \"center\", \"medoid\"")
        })?;

        Ok(Self {
            n_components,
            min_samples,
            min_cluster_size,
            max_cluster_size,
            persistence_measure: persistence_measure.to_string(),
            rep_strategy: rep_strategy.to_string(),
            reps_per_cluster,
            include_noise,
            noise_samples,
            n_neighbors,
            neighbor_scale,
            must_link_weight,
            interpolation_k,
            restrict_to_cluster,
            embedding: None,
            labels: None,
            probabilities: None,
            rep_indices: None,
            n_neighbors_used: None,
            selected_min_cluster_size: None,
            trace_min_size: None,
            trace_persistence: None,
        })
    }

    pub fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
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

        let cluster = cluster_internal(
            &data_vec,
            self.min_samples,
            min_cluster_size,
            max_cluster_size,
            measure,
        )
        .map_err(PyValueError::new_err)?;

        let rep_strategy = RepresentativeStrategy::parse(&self.rep_strategy).unwrap();
        let (mut rep_indices, rep_cluster_labels) = select_representatives(
            &data_vec,
            &cluster.labels,
            &cluster.probabilities,
            rep_strategy,
            self.reps_per_cluster,
            self.include_noise,
            self.noise_samples,
            self.n_components,
            self.min_samples,
        );
        if rep_indices.len() < 2 {
            // Shouldn't happen after fallback, but keep a hard guard.
            rep_indices = vec![0, data_vec.len() - 1];
        }

        let rep_probs: Vec<f64> = rep_indices
            .iter()
            .map(|&idx| cluster.probabilities[idx])
            .collect();
        let rep_points: Vec<Vec<f32>> = rep_indices.iter().map(|&idx| data_vec[idx].clone()).collect();

        let n_neighbors_used = suggest_n_neighbors(
            self.n_neighbors,
            cluster.selected_min_cluster_size,
            self.min_samples,
            rep_indices.len(),
            self.neighbor_scale,
        );

        let rep_embedding = spectral_embedding(
            &rep_points,
            self.n_components,
            n_neighbors_used,
            &rep_cluster_labels,
            &rep_probs,
            self.must_link_weight,
        )
        .map_err(PyValueError::new_err)?;

        let embedding = interpolate_embedding(
            &data_vec,
            &cluster.labels,
            &rep_indices,
            &rep_cluster_labels,
            &rep_embedding,
            self.interpolation_k,
            self.restrict_to_cluster,
        )
        .map_err(PyValueError::new_err)?;

        self.embedding = Some(embedding.clone());
        self.labels = Some(Array1::from_vec(cluster.labels));
        self.probabilities = Some(Array1::from_vec(cluster.probabilities));
        self.rep_indices = Some(Array1::from_vec(rep_indices.iter().map(|&v| v as i64).collect()));
        self.n_neighbors_used = Some(n_neighbors_used);
        self.selected_min_cluster_size = cluster.selected_min_cluster_size;
        self.trace_min_size = Some(Array1::from_vec(cluster.trace_min_size));
        self.trace_persistence = Some(Array1::from_vec(cluster.trace_persistence));

        Ok(embedding.into_pyarray_bound(py))
    }

    #[getter]
    pub fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let labels = self
            .labels
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("PLSCANBackbone not fitted"))?;
        Ok(labels.clone().into_pyarray_bound(py))
    }

    #[getter]
    pub fn probabilities_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let probs = self
            .probabilities
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("PLSCANBackbone not fitted"))?;
        Ok(probs.clone().into_pyarray_bound(py))
    }

    #[getter]
    pub fn rep_indices_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let reps = self
            .rep_indices
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("PLSCANBackbone not fitted"))?;
        Ok(reps.clone().into_pyarray_bound(py))
    }

    #[getter]
    pub fn n_neighbors_used_(&self) -> Option<usize> {
        self.n_neighbors_used
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
            .ok_or_else(|| PyValueError::new_err("PLSCANBackbone not fitted"))?;
        Ok(v.clone().into_pyarray_bound(py))
    }

    #[getter]
    pub fn trace_persistence_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let v = self
            .trace_persistence
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("PLSCANBackbone not fitted"))?;
        Ok(v.clone().into_pyarray_bound(py))
    }
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

fn select_representatives(
    data: &[Vec<f32>],
    labels: &[i64],
    probabilities: &[f64],
    strategy: RepresentativeStrategy,
    reps_per_cluster: usize,
    include_noise: bool,
    noise_samples: usize,
    n_components: usize,
    min_samples: usize,
) -> (Vec<usize>, Vec<i64>) {
    let mut clusters: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    let mut noise = Vec::new();
    for (idx, &label) in labels.iter().enumerate() {
        if label >= 0 {
            clusters.entry(label).or_default().push(idx);
        } else {
            noise.push(idx);
        }
    }

    let mut reps = Vec::<usize>::new();
    let mut rep_cluster_labels = Vec::<i64>::new();
    for (&cluster_id, members) in clusters.iter() {
        let chosen = pick_cluster_reps(data, members, probabilities, strategy, reps_per_cluster);
        for idx in chosen {
            reps.push(idx);
            rep_cluster_labels.push(cluster_id);
        }
    }

    if include_noise && noise_samples > 0 && !noise.is_empty() {
        let take = noise_samples.min(noise.len());
        if take == noise.len() {
            for &idx in &noise {
                reps.push(idx);
                rep_cluster_labels.push(-1);
            }
        } else {
            let step = (noise.len() as f64) / (take as f64);
            for i in 0..take {
                let pos = ((i as f64) * step).floor() as usize;
                reps.push(noise[pos]);
                rep_cluster_labels.push(-1);
            }
        }
    }

    // Fallback if clustering produced too few representatives.
    if reps.len() < 2 {
        let target = (n_components + 1).max(2).max(min_samples);
        let take = target.min(data.len());
        let step = (data.len() as f64) / (take as f64);
        reps = (0..take)
            .map(|i| ((i as f64) * step).floor() as usize)
            .collect();
        rep_cluster_labels = reps.iter().map(|_| -1).collect();
    }

    // De-duplicate while preserving order.
    let mut seen = std::collections::BTreeSet::new();
    let mut out_reps = Vec::new();
    let mut out_labels = Vec::new();
    for (i, &idx) in reps.iter().enumerate() {
        if seen.insert(idx) {
            out_reps.push(idx);
            out_labels.push(rep_cluster_labels[i]);
        }
    }

    (out_reps, out_labels)
}

fn pick_cluster_reps(
    data: &[Vec<f32>],
    members: &[usize],
    probabilities: &[f64],
    strategy: RepresentativeStrategy,
    reps_per_cluster: usize,
) -> Vec<usize> {
    if members.is_empty() || reps_per_cluster == 0 {
        return Vec::new();
    }

    let first = match strategy {
        RepresentativeStrategy::HighProb => *members
            .iter()
            .max_by(|&&a, &&b| probabilities[a].total_cmp(&probabilities[b]))
            .unwrap(),
        RepresentativeStrategy::Center => nearest_to_centroid(data, members),
        RepresentativeStrategy::Medoid => medoid_index(data, members),
    };

    let mut chosen = vec![first];
    while chosen.len() < reps_per_cluster && chosen.len() < members.len() {
        let next = farthest_from_chosen(data, members, &chosen);
        if chosen.contains(&next) {
            break;
        }
        chosen.push(next);
    }
    chosen
}

fn nearest_to_centroid(data: &[Vec<f32>], members: &[usize]) -> usize {
    let dim = data[members[0]].len();
    let mut centroid = vec![0.0f64; dim];
    for &idx in members {
        for (j, &v) in data[idx].iter().enumerate() {
            centroid[j] += v as f64;
        }
    }
    let inv = 1.0 / (members.len() as f64);
    for v in &mut centroid {
        *v *= inv;
    }

    let mut best_idx = members[0];
    let mut best_dist = f64::INFINITY;
    for &idx in members {
        let mut sum = 0.0f64;
        for (&a, &b) in data[idx].iter().zip(&centroid) {
            let d = (a as f64) - b;
            sum += d * d;
        }
        if sum < best_dist {
            best_dist = sum;
            best_idx = idx;
        }
    }
    best_idx
}

fn medoid_index(data: &[Vec<f32>], members: &[usize]) -> usize {
    let mut best_idx = members[0];
    let mut best_sum = f64::INFINITY;

    for &i in members {
        let mut total = 0.0f64;
        for &j in members {
            if i == j {
                continue;
            }
            total += metrics_simd::euclidean(&data[i], &data[j]).unwrap_or(f32::INFINITY) as f64;
        }
        if total < best_sum {
            best_sum = total;
            best_idx = i;
        }
    }
    best_idx
}

fn farthest_from_chosen(data: &[Vec<f32>], members: &[usize], chosen: &[usize]) -> usize {
    let mut best_idx = members[0];
    let mut best_min_dist = -1.0f64;

    for &idx in members {
        if chosen.contains(&idx) {
            continue;
        }
        let mut min_dist = f64::INFINITY;
        for &c in chosen {
            min_dist = min_dist.min(
                metrics_simd::euclidean(&data[idx], &data[c]).unwrap_or(f32::INFINITY) as f64,
            );
        }
        if min_dist > best_min_dist {
            best_min_dist = min_dist;
            best_idx = idx;
        }
    }
    best_idx
}

fn suggest_n_neighbors(
    n_neighbors: Option<usize>,
    selected_min_cluster_size: Option<f64>,
    min_samples: usize,
    n_points: usize,
    neighbor_scale: f64,
) -> usize {
    if let Some(k) = n_neighbors {
        return k.max(2).min(n_points.saturating_sub(1).max(2));
    }

    let base = selected_min_cluster_size.unwrap_or(min_samples as f64);
    let scaled = (base * neighbor_scale).round() as isize;
    let mut k = usize::try_from(scaled).unwrap_or(2);
    k = k.max(2).max(min_samples);
    k.min(n_points.saturating_sub(1).max(2))
}

fn spectral_embedding(
    points: &[Vec<f32>],
    n_components: usize,
    n_neighbors: usize,
    cluster_labels: &[i64],
    probabilities: &[f64],
    must_link_weight: f64,
) -> Result<Array2<f64>, String> {
    let n = points.len();
    if n < 2 {
        return Err("need at least 2 representatives".to_string());
    }
    if n_components == 0 {
        return Err("n_components must be >= 1".to_string());
    }
    if n_components >= n {
        return Err("n_components must be < number of representatives".to_string());
    }

    let k = n_neighbors.min(n.saturating_sub(1)).max(1);
    let mut w = Array2::<f64>::zeros((n, n));
    let mut neighbor_dists = Vec::with_capacity(n * k);

    for i in 0..n {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let d = metrics_simd::euclidean(&points[i], &points[j]).unwrap_or(f32::INFINITY) as f64;
                (j, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        for &(j, d) in dists.iter().take(k) {
            neighbor_dists.push(d);
            w[[i, j]] = d;
        }
    }

    neighbor_dists.sort_by(|a, b| a.total_cmp(b));
    let sigma = if neighbor_dists.is_empty() {
        1.0
    } else {
        let mid = neighbor_dists.len() / 2;
        let s = neighbor_dists[mid];
        if s > 0.0 { s } else { 1e-6 }
    };
    let denom = 2.0 * sigma * sigma;

    for i in 0..n {
        for j in 0..n {
            let d = w[[i, j]];
            if d > 0.0 {
                w[[i, j]] = (-(d * d) / denom).exp();
            }
        }
    }

    // Symmetrize via max weight.
    for i in 0..n {
        for j in (i + 1)..n {
            let wij = w[[i, j]].max(w[[j, i]]);
            w[[i, j]] = wij;
            w[[j, i]] = wij;
        }
    }

    // Soft must-link edges within clusters, weighted by membership strengths.
    if must_link_weight > 0.0 && !cluster_labels.is_empty() {
        let mut clusters: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for (idx, &label) in cluster_labels.iter().enumerate() {
            if label >= 0 {
                clusters.entry(label).or_default().push(idx);
            }
        }
        for (_label, members) in clusters {
            for a in 0..members.len() {
                let i = members[a];
                for b in (a + 1)..members.len() {
                    let j = members[b];
                    let add = must_link_weight * probabilities[i] * probabilities[j];
                    w[[i, j]] += add;
                    w[[j, i]] += add;
                }
            }
        }
    }

    // Normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}.
    let mut degree = vec![0.0f64; n];
    for i in 0..n {
        let sum: f64 = w.row(i).iter().sum();
        degree[i] = if sum > 0.0 { sum } else { 1.0 };
    }
    let dinv_sqrt: Vec<f64> = degree.iter().map(|d| 1.0 / d.sqrt()).collect();

    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        l[[i, i]] = 1.0;
    }
    for i in 0..n {
        for j in 0..n {
            let wij = w[[i, j]];
            if wij == 0.0 {
                continue;
            }
            let val = wij * dinv_sqrt[i] * dinv_sqrt[j];
            if i == j {
                l[[i, i]] -= val;
            } else {
                l[[i, j]] = -val;
            }
        }
    }

    let (eigenvalues, eigenvectors): (Array1<f64>, Array2<f64>) = l
        .eigh(UPLO::Upper)
        .map_err(|e| format!("Eigendecomposition failed: {e}"))?;

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| eigenvalues[a].total_cmp(&eigenvalues[b]));

    let tol = 1e-10;
    let mut chosen = Vec::new();
    for &idx in &indices {
        if eigenvalues[idx] <= tol {
            continue;
        }
        chosen.push(idx);
        if chosen.len() == n_components {
            break;
        }
    }
    if chosen.len() < n_components {
        return Err("not enough non-trivial eigenvectors for requested n_components".to_string());
    }

    let mut embedding = Array2::<f64>::zeros((n, n_components));
    for (comp, &eig_idx) in chosen.iter().enumerate() {
        let col = eigenvectors.column(eig_idx);
        for i in 0..n {
            embedding[[i, comp]] = col[i];
        }
    }
    Ok(embedding)
}

#[allow(clippy::too_many_arguments)]
fn interpolate_embedding(
    data: &[Vec<f32>],
    labels: &[i64],
    rep_indices: &[usize],
    rep_cluster_labels: &[i64],
    rep_embedding: &Array2<f64>,
    interpolation_k: usize,
    restrict_to_cluster: bool,
) -> Result<Array2<f64>, String> {
    let n = data.len();
    let m = rep_indices.len();
    if m == 0 {
        return Err("no representatives selected".to_string());
    }
    if rep_cluster_labels.len() != m {
        return Err("rep_cluster_labels length mismatch".to_string());
    }
    if rep_embedding.nrows() != m {
        return Err("rep_embedding row count mismatch".to_string());
    }
    let n_components = rep_embedding.ncols();
    if n_components == 0 {
        return Err("rep_embedding must have at least 1 component".to_string());
    }

    let mut point_to_rep_pos = vec![None; n];
    for (pos, &idx) in rep_indices.iter().enumerate() {
        if idx < n {
            point_to_rep_pos[idx] = Some(pos);
        }
    }

    let mut cluster_to_rep_positions: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (pos, &label) in rep_cluster_labels.iter().enumerate() {
        cluster_to_rep_positions.entry(label).or_default().push(pos);
    }

    let all_positions: Vec<usize> = (0..m).collect();
    let eps = 1e-12_f64;

    let mut out = Array2::<f64>::zeros((n, n_components));
    for i in 0..n {
        if let Some(pos) = point_to_rep_pos[i] {
            out.row_mut(i).assign(&rep_embedding.row(pos));
            continue;
        }

        let mut candidates: &[usize] = &all_positions;
        if restrict_to_cluster {
            let label = labels[i];
            if label >= 0 {
                if let Some(v) = cluster_to_rep_positions.get(&label) {
                    if !v.is_empty() {
                        candidates = v;
                    }
                }
            }
        }

        let k = interpolation_k.min(candidates.len()).max(1);
        let mut dists: Vec<(usize, f64)> = candidates
            .iter()
            .map(|&pos| {
                let idx = rep_indices[pos];
                let d = metrics_simd::euclidean(&data[i], &data[idx]).unwrap_or(f32::INFINITY) as f64;
                (pos, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));

        if dists[0].1 == 0.0 {
            out.row_mut(i).assign(&rep_embedding.row(dists[0].0));
            continue;
        }

        let mut weight_sum = 0.0f64;
        let mut weights = Vec::with_capacity(k);
        for &(pos, dist) in dists.iter().take(k) {
            let w = 1.0 / (dist + eps);
            weights.push((pos, w));
            weight_sum += w;
        }
        if weight_sum == 0.0 {
            out.row_mut(i).assign(&rep_embedding.row(weights[0].0));
            continue;
        }

        for c in 0..n_components {
            let mut val = 0.0f64;
            for &(pos, w) in &weights {
                val += (w / weight_sum) * rep_embedding[[pos, c]];
            }
            out[[i, c]] = val;
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_embedding_shapes() {
        let mut x = Vec::new();
        for i in 0..10 {
            x.push(vec![i as f32 * 0.01, 0.0]);
        }
        for i in 0..10 {
            x.push(vec![100.0 + i as f32 * 0.01, 0.0]);
        }

        let labels = vec![0i64; x.len()];
        let probs = vec![1.0f64; x.len()];
        let emb = spectral_embedding(&x, 2, 5, &labels, &probs, 0.1).unwrap();
        assert_eq!(emb.nrows(), x.len());
        assert_eq!(emb.ncols(), 2);
    }
}

