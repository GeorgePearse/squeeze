//! UMAP (Uniform Manifold Approximation and Projection) implementation.
//!
//! This provides a Rust-backed UMAP variant intended for fast CPU execution.
//! The implementation focuses on the standard Euclidean-output UMAP objective
//! (fuzzy set cross-entropy with negative sampling).

use std::collections::BinaryHeap;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use ordered_float::OrderedFloat;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::Normal;
use rayon::prelude::*;

use crate::hnsw_algo::Hnsw;
use crate::metrics;
use crate::metrics_simd;

const GRAD_CLIP: f32 = 4.0;

#[inline]
fn clip(x: f32) -> f32 {
    x.clamp(-GRAD_CLIP, GRAD_CLIP)
}

#[derive(Clone, Copy)]
enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    Chebyshev,
    Minkowski { p: f32 },
    Hamming,
}

impl DistanceMetric {
    fn parse(name: &str, p: f32) -> Option<Self> {
        match name {
            "euclidean" | "l2" => Some(Self::Euclidean),
            "manhattan" | "l1" | "taxicab" => Some(Self::Manhattan),
            "cosine" | "correlation" => Some(Self::Cosine),
            "chebyshev" | "linfinity" => Some(Self::Chebyshev),
            "minkowski" => Some(Self::Minkowski { p }),
            "hamming" => Some(Self::Hamming),
            _ => None,
        }
    }

    #[inline]
    fn dist(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::Euclidean => metrics_simd::euclidean(a, b).unwrap_or(f32::MAX),
            Self::Manhattan => metrics_simd::manhattan(a, b).unwrap_or(f32::MAX),
            Self::Cosine => metrics_simd::cosine(a, b).unwrap_or(f32::MAX),
            Self::Chebyshev => metrics::chebyshev(a, b).unwrap_or(f32::MAX),
            Self::Minkowski { p } => metrics::minkowski(a, b, p).unwrap_or(f32::MAX),
            Self::Hamming => metrics::hamming(a, b).unwrap_or(f32::MAX),
        }
    }
}

/// Rust-backed UMAP implementation.
///
/// This is intentionally scoped to the most common UMAP configuration:
/// - kNN graph in input space
/// - Fuzzy simplicial set construction
/// - Euclidean output with SGD + negative sampling
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct UMAPRust {
    n_components: usize,
    n_neighbors: usize,
    n_epochs: usize,
    min_dist: f32,
    spread: f32,
    metric: String,
    dist_p: f32,
    initial_alpha: f32,
    negative_sample_rate: f32,
    gamma: f32,
    random_state: Option<u64>,

    // HNSW tuning (used when n_samples is large)
    m: usize,
    ef_construction: usize,
    ef_search: usize,

    embedding: Option<Array2<f32>>,
}

#[pymethods]
impl UMAPRust {
    #[new]
    #[pyo3(
        signature = (
            n_components=2,
            n_neighbors=15,
            n_epochs=0,
            min_dist=0.1,
            spread=1.0,
            metric="euclidean".to_string(),
            dist_p=2.0,
            initial_alpha=1.0,
            negative_sample_rate=5.0,
            gamma=1.0,
            random_state=None,
            m=16,
            ef_construction=200,
            ef_search=50,
        )
    )]
    pub fn new(
        n_components: usize,
        n_neighbors: usize,
        n_epochs: usize,
        min_dist: f32,
        spread: f32,
        metric: String,
        dist_p: f32,
        initial_alpha: f32,
        negative_sample_rate: f32,
        gamma: f32,
        random_state: Option<u64>,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
    ) -> Self {
        Self {
            n_components,
            n_neighbors,
            n_epochs,
            min_dist,
            spread,
            metric,
            dist_p,
            initial_alpha,
            negative_sample_rate,
            gamma,
            random_state,
            m,
            ef_construction,
            ef_search,
            embedding: None,
        }
    }

    /// Fit the model and return the embedding.
    pub fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x = data.as_array();
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples < 2 {
            return Err(PyValueError::new_err("UMAP requires at least 2 samples"));
        }
        if self.n_neighbors == 0 {
            return Err(PyValueError::new_err("n_neighbors must be at least 1"));
        }
        if self.n_neighbors >= n_samples {
            return Err(PyValueError::new_err(format!(
                "n_neighbors ({}) must be less than n_samples ({})",
                self.n_neighbors, n_samples
            )));
        }
        if self.n_components == 0 {
            return Err(PyValueError::new_err("n_components must be at least 1"));
        }
        if self.min_dist < 0.0 {
            return Err(PyValueError::new_err("min_dist must be >= 0"));
        }
        if self.spread <= 0.0 {
            return Err(PyValueError::new_err("spread must be > 0"));
        }

        let metric = DistanceMetric::parse(self.metric.as_str(), self.dist_p).ok_or_else(|| {
            PyValueError::new_err(format!(
                "Unknown metric '{}'. Supported metrics: euclidean, manhattan, cosine, correlation, chebyshev, minkowski, hamming",
                self.metric
            ))
        })?;

        // Copy input to a contiguous Vec<Vec<f32>> for fast distance computations.
        let mut data_f32 = Vec::with_capacity(n_samples);
        for row in x.rows() {
            let mut v = Vec::with_capacity(n_features);
            for &val in row.iter() {
                v.push(val as f32);
            }
            data_f32.push(v);
        }

        let (knn_indices, knn_dists) = self.compute_knn(&data_f32, metric);
        let (sigmas, rhos) = smooth_knn_dist(&knn_dists, self.n_neighbors, 1.0, 1.0);
        let (head, tail, weights) = compute_membership_strengths(
            &knn_indices,
            &knn_dists,
            &sigmas,
            &rhos,
            n_samples,
        );

        let n_epochs = if self.n_epochs == 0 {
            if n_samples <= 10_000 { 500 } else { 200 }
        } else {
            self.n_epochs
        };

        let (a, b) = find_ab_params(self.spread, self.min_dist);

        // Initialize embedding
        let mut embedding = self.initialize_embedding(n_samples);
        scale_to_10(&mut embedding, n_samples, self.n_components);

        optimize_embedding(
            &mut embedding,
            n_samples,
            self.n_components,
            &head,
            &tail,
            &weights,
            n_epochs,
            a,
            b,
            self.gamma,
            self.initial_alpha,
            self.negative_sample_rate,
            self.random_state,
        );

        let embedding_arr = Array2::from_shape_vec((n_samples, self.n_components), embedding)
            .map_err(|e| PyValueError::new_err(format!("Failed to shape embedding: {}", e)))?;

        self.embedding = Some(embedding_arr.clone());
        Ok(embedding_arr.into_pyarray_bound(py))
    }

    /// Fit the model, storing the embedding in `embedding_`.
    pub fn fit(&mut self, py: Python<'_>, data: PyReadonlyArray2<f64>) -> PyResult<()> {
        self.fit_transform(py, data)?;
        Ok(())
    }

    #[getter]
    pub fn embedding_<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let emb = self
            .embedding
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("UMAPRust not fitted"))?;
        Ok(emb.clone().into_pyarray_bound(py))
    }
}

impl UMAPRust {
    fn initialize_embedding(&self, n_samples: usize) -> Vec<f32> {
        let mut rng: StdRng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_seed(rand::random()),
        };

        let normal = Normal::new(0.0, 1e-4).unwrap();
        let mut embedding = vec![0.0f32; n_samples * self.n_components];
        for v in embedding.iter_mut() {
            *v = normal.sample(&mut rng);
        }
        embedding
    }

    fn compute_knn(
        &self,
        data: &[Vec<f32>],
        metric: DistanceMetric,
    ) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let n_samples = data.len();

        // Exact kNN is fast enough for typical datasets used in tests/benchmarks
        // (e.g. sklearn Digits), and provides better determinism and quality.
        if n_samples <= 4096 {
            return exact_knn(data, self.n_neighbors, metric);
        }

        hnsw_knn(
            data,
            self.n_neighbors,
            metric,
            self.m,
            self.ef_construction,
            self.ef_search.max(self.n_neighbors),
            self.random_state.unwrap_or(42),
        )
    }
}

fn exact_knn(
    data: &[Vec<f32>],
    k: usize,
    metric: DistanceMetric,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let n_samples = data.len();

    let results: Vec<(Vec<usize>, Vec<f32>)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let mut heap: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::with_capacity(k + 1);
            for j in 0..n_samples {
                if i == j {
                    continue;
                }
                let d = metric.dist(&data[i], &data[j]);
                if heap.len() < k {
                    heap.push((OrderedFloat(d), j));
                } else if let Some(&(OrderedFloat(max_d), _)) = heap.peek() {
                    if d < max_d {
                        heap.pop();
                        heap.push((OrderedFloat(d), j));
                    }
                }
            }

            let mut pairs: Vec<(f32, usize)> = heap
                .into_iter()
                .map(|(OrderedFloat(d), j)| (d, j))
                .collect();
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let mut indices = Vec::with_capacity(k);
            let mut dists = Vec::with_capacity(k);
            for (d, j) in pairs {
                indices.push(j);
                dists.push(d);
            }

            (indices, dists)
        })
        .collect();

    let mut indices = Vec::with_capacity(n_samples);
    let mut dists = Vec::with_capacity(n_samples);
    for (idx, dist) in results {
        indices.push(idx);
        dists.push(dist);
    }

    (indices, dists)
}

fn hnsw_knn(
    data: &[Vec<f32>],
    k: usize,
    metric: DistanceMetric,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    seed: u64,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let n_samples = data.len();

    let mut hnsw = Hnsw::new(m, ef_construction, n_samples, seed);

    // Distance function used for building the index.
    {
        let dist_fn = |i: usize, j: usize| -> f32 { metric.dist(&data[i], &data[j]) };
        for i in 0..n_samples {
            hnsw.insert(i, &dist_fn);
        }
    }

    let results: Vec<(Vec<usize>, Vec<f32>)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let query = &data[i];
            let dist_query = |node_idx: usize| -> f32 { metric.dist(query, &data[node_idx]) };
            let found = hnsw.search(None, k + 1, ef_search, dist_query);

            let mut indices = Vec::with_capacity(k);
            let mut dists = Vec::with_capacity(k);
            for (j, d) in found {
                if j == i {
                    continue;
                }
                indices.push(j);
                dists.push(d);
                if indices.len() >= k {
                    break;
                }
            }

            // Padding if needed
            while indices.len() < k {
                indices.push(i);
                dists.push(f32::INFINITY);
            }

            (indices, dists)
        })
        .collect();

    let mut indices_all = Vec::with_capacity(n_samples);
    let mut dists_all = Vec::with_capacity(n_samples);
    for (idx, dist) in results {
        indices_all.push(idx);
        dists_all.push(dist);
    }

    (indices_all, dists_all)
}

fn smooth_knn_dist(
    distances: &[Vec<f32>],
    k: usize,
    local_connectivity: f32,
    bandwidth: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n_samples = distances.len();
    let target = (k as f32).log2() * bandwidth;

    let mut sigmas = vec![1.0f32; n_samples];
    let mut rhos = vec![0.0f32; n_samples];

    for i in 0..n_samples {
        let mut non_zero: Vec<f32> = distances[i]
            .iter()
            .copied()
            .filter(|&d| d > 0.0 && d.is_finite())
            .collect();
        non_zero.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if !non_zero.is_empty() {
            if (non_zero.len() as f32) >= local_connectivity {
                let index = local_connectivity.floor() as usize;
                let interpolation = local_connectivity - index as f32;

                if index > 0 {
                    let mut rho = non_zero[index - 1];
                    if interpolation > 1e-5 && index < non_zero.len() {
                        rho += interpolation * (non_zero[index] - non_zero[index - 1]);
                    }
                    rhos[i] = rho;
                } else {
                    rhos[i] = interpolation * non_zero[0];
                }
            } else {
                rhos[i] = *non_zero.last().unwrap();
            }
        }

        let mut lo = 0.0f32;
        let mut hi = f32::INFINITY;
        let mut mid = 1.0f32;

        for _ in 0..64 {
            let mut psum = 0.0f32;
            for &d in distances[i].iter() {
                if !d.is_finite() {
                    continue;
                }
                let mut v = d - rhos[i];
                if v > 0.0 {
                    v = (-v / mid).exp();
                } else {
                    v = 1.0;
                }
                psum += v;
            }

            if (psum - target).abs() < 1e-5 {
                break;
            }

            if psum > target {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                if hi.is_infinite() {
                    mid *= 2.0;
                } else {
                    mid = (lo + hi) / 2.0;
                }
            }

            if mid < 1e-6 {
                mid = 1e-6;
                break;
            }
        }

        sigmas[i] = mid;
    }

    (sigmas, rhos)
}

fn compute_membership_strengths(
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<f32>],
    sigmas: &[f32],
    rhos: &[f32],
    n_vertices: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    let n_samples = knn_indices.len();

    // First build directed weights.
    let mut directed: Vec<(usize, usize, f32)> = Vec::with_capacity(n_samples * knn_indices[0].len());
    for i in 0..n_samples {
        for (j, &nbr) in knn_indices[i].iter().enumerate() {
            if nbr >= n_vertices || nbr == i {
                continue;
            }

            let d = knn_dists[i][j];
            if !d.is_finite() {
                continue;
            }

            let rho = rhos[i];
            let sigma = sigmas[i].max(1e-6);
            let w = if d - rho <= 0.0 {
                1.0
            } else {
                (-(d - rho) / sigma).exp()
            };

            if w > 0.0 {
                directed.push((i, nbr, w));
            }
        }
    }

    // Map directed weights for symmetric union.
    let mut weight_map = std::collections::HashMap::<u64, f32>::with_capacity(directed.len() * 2);
    for &(i, j, w) in directed.iter() {
        let key = (i as u64) * (n_vertices as u64) + (j as u64);
        weight_map.insert(key, w);
    }

    let mut head = Vec::with_capacity(directed.len());
    let mut tail = Vec::with_capacity(directed.len());
    let mut weights = Vec::with_capacity(directed.len());

    for (i, j, w_ij) in directed {
        let rev_key = (j as u64) * (n_vertices as u64) + (i as u64);
        let w_ji = weight_map.get(&rev_key).copied().unwrap_or(0.0);
        let w = w_ij + w_ji - w_ij * w_ji;
        if w > 0.0 {
            head.push(i);
            tail.push(j);
            weights.push(w);
        }
    }

    (head, tail, weights)
}

fn find_ab_params(spread: f32, min_dist: f32) -> (f32, f32) {
    // UMAP-learn typically fits these parameters with scipy curve_fit.
    // Here we do a lightweight grid search over plausible values.
    let n = 300;
    let max_x = spread * 3.0;

    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);

    for i in 0..n {
        let x = (i as f32) * max_x / ((n - 1) as f32);
        let y = if x < min_dist {
            1.0
        } else {
            (-(x - min_dist) / spread).exp()
        };
        xs.push(x);
        ys.push(y);
    }

    let mut best_a = 1.5769435;
    let mut best_b = 0.8950609;
    let mut best_err = f32::INFINITY;

    // Search b in [0.5, 2.0] and a in logspace [0.1, 10]
    for bi in 0..40 {
        let b = 0.5 + (bi as f32) * (1.5 / 39.0);
        for ai in 0..40 {
            let log_a = -1.0 + (ai as f32) * (2.0 / 39.0); // [-1, 1]
            let a = 10f32.powf(log_a);

            let mut err = 0.0f32;
            for (&x, &y) in xs.iter().zip(ys.iter()) {
                let denom = 1.0 + a * x.powf(2.0 * b);
                let f = 1.0 / denom;
                let diff = f - y;
                err += diff * diff;
            }

            if err < best_err {
                best_err = err;
                best_a = a;
                best_b = b;
            }
        }
    }

    (best_a, best_b)
}

fn scale_to_10(embedding: &mut [f32], n_samples: usize, dim: usize) {
    for d in 0..dim {
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        for i in 0..n_samples {
            let v = embedding[i * dim + d];
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }

        let range = (max_v - min_v).max(1e-6);
        for i in 0..n_samples {
            let idx = i * dim + d;
            embedding[idx] = 10.0 * (embedding[idx] - min_v) / range;
        }
    }
}

fn make_epochs_per_sample(weights: &[f32], n_epochs: usize) -> Vec<f32> {
    let mut result = vec![-1.0f32; weights.len()];
    let max_w = weights
        .iter()
        .copied()
        .fold(0.0f32, |a, b| a.max(b));

    if max_w <= 0.0 {
        return result;
    }

    for (i, &w) in weights.iter().enumerate() {
        let n_samples = (n_epochs as f32) * (w / max_w);
        if n_samples > 0.0 {
            result[i] = (n_epochs as f32) / n_samples;
        }
    }

    result
}

fn optimize_embedding(
    embedding: &mut [f32],
    n_vertices: usize,
    dim: usize,
    head: &[usize],
    tail: &[usize],
    weights: &[f32],
    n_epochs: usize,
    a: f32,
    b: f32,
    gamma: f32,
    initial_alpha: f32,
    negative_sample_rate: f32,
    random_state: Option<u64>,
) {
    let epochs_per_sample = make_epochs_per_sample(weights, n_epochs);
    let mut epoch_of_next_sample = epochs_per_sample.clone();

    let epochs_per_negative_sample: Vec<f32> = epochs_per_sample
        .iter()
        .map(|&e| if e > 0.0 { e / negative_sample_rate } else { -1.0 })
        .collect();
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();

    let mut rng: StdRng = match random_state {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_seed(rand::random()),
    };

    for n in 0..n_epochs {
        let alpha = initial_alpha * (1.0 - (n as f32) / (n_epochs as f32));

        for i in 0..epochs_per_sample.len() {
            if epochs_per_sample[i] <= 0.0 {
                continue;
            }
            if epoch_of_next_sample[i] > (n as f32) {
                continue;
            }

            let j = head[i];
            let k = tail[i];

            // Attractive update (move both points)
            let mut dist2 = 0.0f32;
            for d in 0..dim {
                let diff = embedding[j * dim + d] - embedding[k * dim + d];
                dist2 += diff * diff;
            }

            let mut grad_coeff = 0.0f32;
            if dist2 > 0.0 {
                grad_coeff = -2.0 * a * b * dist2.powf(b - 1.0);
                grad_coeff /= a * dist2.powf(b) + 1.0;
            }

            for d in 0..dim {
                let diff = embedding[j * dim + d] - embedding[k * dim + d];
                let grad_d = clip(grad_coeff * diff);
                embedding[j * dim + d] += grad_d * alpha;
                embedding[k * dim + d] -= grad_d * alpha;
            }

            epoch_of_next_sample[i] += epochs_per_sample[i];

            // Negative sampling
            let n_neg_samples = if epochs_per_negative_sample[i] > 0.0 {
                (((n as f32) - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i])
                    .floor()
                    .max(0.0) as usize
            } else {
                0
            };

            for _ in 0..n_neg_samples {
                let kk = rng.gen_range(0..n_vertices);
                if kk == j {
                    continue;
                }

                let mut dist2 = 0.0f32;
                for d in 0..dim {
                    let diff = embedding[j * dim + d] - embedding[kk * dim + d];
                    dist2 += diff * diff;
                }

                let mut grad_coeff = 0.0f32;
                if dist2 > 0.0 {
                    grad_coeff = 2.0 * gamma * b;
                    grad_coeff /= (0.001 + dist2) * (a * dist2.powf(b) + 1.0);
                }

                if grad_coeff > 0.0 {
                    for d in 0..dim {
                        let diff = embedding[j * dim + d] - embedding[kk * dim + d];
                        let grad_d = clip(grad_coeff * diff);
                        embedding[j * dim + d] += grad_d * alpha;
                    }
                }
            }

            epoch_of_next_negative_sample[i] += (n_neg_samples as f32) * epochs_per_negative_sample[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_ab_params_returns_finite() {
        let (a, b) = find_ab_params(1.0, 0.1);
        assert!(a.is_finite());
        assert!(b.is_finite());
        assert!(a > 0.0);
        assert!(b > 0.0);
    }

    #[test]
    fn test_make_epochs_per_sample_basic() {
        let weights = vec![0.0, 0.5, 1.0];
        let eps = make_epochs_per_sample(&weights, 100);
        assert_eq!(eps.len(), 3);
        assert!(eps[0] < 0.0);
        assert!(eps[1] > 0.0);
        assert!(eps[2] > 0.0);
        assert!(eps[2] <= eps[1]);
    }
}
