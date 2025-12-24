//! UMAP (Uniform Manifold Approximation and Projection) implementation.
//!
//! This is a CPU-first implementation intended to provide a real Rust backend
//! for the core UMAP algorithm:
//! 1) k-NN search (exact, O(n²) for now)
//! 2) Fuzzy simplicial set construction (smooth k-NN distances + membership)
//! 3) SGD optimization with negative sampling

use ndarray::{Array2, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use ordered_float::OrderedFloat;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::Normal;
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap};

use crate::metrics_simd;

const SMOOTH_K_TOLERANCE: f64 = 1e-5;
const MIN_K_DIST_SCALE: f64 = 1e-3;

#[inline]
fn clip(v: f64) -> f64 {
    v.clamp(-4.0, 4.0)
}

#[inline]
fn pack_u64(i: usize, j: usize) -> u64 {
    ((i as u64) << 32) | (j as u64)
}

#[inline]
fn unpack_u64(key: u64) -> (usize, usize) {
    ((key >> 32) as usize, (key as u32) as usize)
}

fn find_ab_params(spread: f64, min_dist: f64) -> (f64, f64) {
    // We fit: f(x) = 1 / (1 + a * x^(2b)) to an exponential decay curve.
    // Instead of curve_fit, use a log-linear regression on:
    //   (1/f - 1) = a * x^(2b)  => log(1/f - 1) = log(a) + 2b*log(x)
    //
    // Only use x >= min_dist where the target curve is < 1 (so logs are defined).
    if spread <= 0.0 {
        return (1.0, 1.0);
    }

    let min_x = min_dist.max(1e-6);
    let max_x = (spread * 3.0).max(min_x * 1.001);
    let n_points = 300usize;

    let mut xs = Vec::with_capacity(n_points);
    let mut ys = Vec::with_capacity(n_points);

    for t in 0..n_points {
        let frac = t as f64 / (n_points.saturating_sub(1) as f64);
        let x = min_x + frac * (max_x - min_x);
        let y = (-(x - min_dist) / spread).exp();
        // y is in (0, 1]; for x==min_dist, y==1, so skip to avoid log(0)
        if y >= 1.0 {
            continue;
        }
        xs.push(x.ln());
        ys.push((1.0 / y - 1.0).ln());
    }

    if xs.len() < 2 {
        // Fallback to common defaults
        return (1.576943460, 0.895060879);
    }

    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;

    let mut var_x = 0.0;
    let mut cov_xy = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        var_x += dx * dx;
        cov_xy += dx * dy;
    }

    if var_x <= 1e-12 {
        return (1.576943460, 0.895060879);
    }

    let slope = cov_xy / var_x;
    let intercept = mean_y - slope * mean_x;

    let a = intercept.exp().max(1e-12);
    let b = (slope / 2.0).max(1e-6);

    (a, b)
}

fn smooth_knn_dist_row(
    distances: &[f64],
    k: f64,
    mean_distances: f64,
    n_iter: usize,
    local_connectivity: f64,
    bandwidth: f64,
) -> (f64, f64) {
    // Port of umap-learn smooth_knn_dist (slightly adapted for slice inputs).
    let target = k.log2() * bandwidth;
    let mut rho = 0.0;

    let non_zero: Vec<f64> = distances.iter().copied().filter(|&d| d > 0.0).collect();
    if !non_zero.is_empty() {
        if (non_zero.len() as f64) >= local_connectivity {
            let index = local_connectivity.floor() as usize;
            let interpolation = local_connectivity - index as f64;

            if index > 0 {
                let base = non_zero[(index - 1).min(non_zero.len() - 1)];
                rho = base;
                if interpolation > SMOOTH_K_TOLERANCE && index < non_zero.len() {
                    rho += interpolation * (non_zero[index] - base);
                }
            } else {
                rho = interpolation * non_zero[0];
            }
        } else {
            rho = non_zero
                .iter()
                .copied()
                .fold(0.0_f64, |acc, v| acc.max(v));
        }
    }

    let mut lo = 0.0;
    let mut hi = f64::INFINITY;
    let mut mid = 1.0;

    for _ in 0..n_iter {
        let mut psum = 0.0;
        // Match umap-learn's behavior: start at j=1.
        for &dist in distances.iter().skip(1) {
            let d = dist - rho;
            if d > 0.0 {
                psum += (-d / mid).exp();
            } else {
                psum += 1.0;
            }
        }

        if (psum - target).abs() < SMOOTH_K_TOLERANCE {
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
    }

    let mut sigma = mid;
    if rho > 0.0 {
        let mean_ith = distances.iter().sum::<f64>() / (distances.len().max(1) as f64);
        sigma = sigma.max(MIN_K_DIST_SCALE * mean_ith);
    } else if sigma < MIN_K_DIST_SCALE * mean_distances {
        sigma = MIN_K_DIST_SCALE * mean_distances;
    }

    (sigma, rho)
}

fn compute_knn(data: &[Vec<f32>], n_neighbors: usize) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
    let n = data.len();
    let k = n_neighbors.min(n.saturating_sub(1)).max(1);

    let rows: Vec<(Vec<usize>, Vec<f64>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut heap: BinaryHeap<(OrderedFloat<f32>, usize)> = BinaryHeap::new();

            for j in 0..n {
                if i == j {
                    continue;
                }
                let d = metrics_simd::euclidean(&data[i], &data[j]).unwrap_or(f32::MAX);
                heap.push((OrderedFloat(d), j));
                if heap.len() > k {
                    heap.pop();
                }
            }

            let mut neighbors: Vec<(usize, f64)> = heap
                .into_iter()
                .map(|(d, j)| (j, d.into_inner() as f64))
                .collect();
            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let indices = neighbors.iter().map(|(j, _)| *j).collect();
            let dists = neighbors.iter().map(|(_, d)| *d).collect();

            (indices, dists)
        })
        .collect();

    let mut knn_indices = Vec::with_capacity(n);
    let mut knn_dists = Vec::with_capacity(n);
    for (idx, dist) in rows {
        knn_indices.push(idx);
        knn_dists.push(dist);
    }
    (knn_indices, knn_dists)
}

fn optimize_embedding(
    embedding: &mut Array2<f64>,
    edges: &[(usize, usize, f64)],
    n_epochs: usize,
    initial_alpha: f64,
    negative_sample_rate: usize,
    gamma: f64,
    a_param: f64,
    b_param: f64,
    seed: u64,
) {
    let n_vertices = embedding.nrows();
    let dim = embedding.ncols();

    let w_max = edges
        .iter()
        .map(|&(_, _, w)| w)
        .fold(0.0_f64, f64::max)
        .max(1e-12);

    let mut rng = StdRng::seed_from_u64(seed);

    for epoch in 0..n_epochs {
        let alpha = initial_alpha * (1.0 - epoch as f64 / n_epochs.max(1) as f64);
        if alpha <= 0.0 {
            break;
        }

        for &(i, j, w) in edges {
            let scaled_alpha = alpha * (w / w_max);

            // Attractive update
            let mut dist_sq = 0.0;
            for d in 0..dim {
                let diff = embedding[[i, d]] - embedding[[j, d]];
                dist_sq += diff * diff;
            }

            let grad_coeff = if dist_sq > 0.0 {
                let dist_pow_b = dist_sq.powf(b_param);
                -2.0 * a_param * b_param * dist_sq.powf(b_param - 1.0) / (a_param * dist_pow_b + 1.0)
            } else {
                0.0
            };

            if grad_coeff != 0.0 {
                for d in 0..dim {
                    let diff = embedding[[i, d]] - embedding[[j, d]];
                    let grad_d = clip(grad_coeff * diff);
                    embedding[[i, d]] += grad_d * scaled_alpha;
                    embedding[[j, d]] -= grad_d * scaled_alpha;
                }
            }

            // Negative sampling (repulsive)
            for _ in 0..negative_sample_rate {
                let k = rng.random_range(0..n_vertices);
                if k == i {
                    continue;
                }

                let mut dist_sq = 0.0;
                for d in 0..dim {
                    let diff = embedding[[i, d]] - embedding[[k, d]];
                    dist_sq += diff * diff;
                }

                let grad_coeff = if dist_sq > 0.0 {
                    let dist_pow_b = dist_sq.powf(b_param);
                    2.0 * gamma * b_param / ((0.001 + dist_sq) * (a_param * dist_pow_b + 1.0))
                } else {
                    0.0
                };

                if grad_coeff > 0.0 {
                    for d in 0..dim {
                        let diff = embedding[[i, d]] - embedding[[k, d]];
                        let grad_d = clip(grad_coeff * diff);
                        embedding[[i, d]] += grad_d * scaled_alpha;
                    }
                }
            }
        }
    }

    // Center embedding to prevent drift
    if let Some(mean) = embedding.mean_axis(Axis(0)) {
        for mut row in embedding.rows_mut() {
            row -= &mean;
        }
    }
}

/// UMAP dimensionality reduction (Rust backend).
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct UMAP {
    n_components: usize,
    n_neighbors: usize,
    n_epochs: usize,
    min_dist: f64,
    spread: f64,
    learning_rate: f64,
    negative_sample_rate: usize,
    gamma: f64,
    random_state: Option<u64>,
}

#[pymethods]
impl UMAP {
    #[new]
    #[pyo3(signature = (n_components=2, n_neighbors=15, n_epochs=200, min_dist=0.1, spread=1.0, learning_rate=1.0, negative_sample_rate=5, gamma=1.0, random_state=None))]
    pub fn new(
        n_components: usize,
        n_neighbors: usize,
        n_epochs: usize,
        min_dist: f64,
        spread: f64,
        learning_rate: f64,
        negative_sample_rate: usize,
        gamma: f64,
        random_state: Option<u64>,
    ) -> PyResult<Self> {
        if n_components == 0 {
            return Err(PyValueError::new_err("n_components must be >= 1"));
        }
        if n_neighbors < 2 {
            return Err(PyValueError::new_err("n_neighbors must be >= 2"));
        }
        if n_epochs == 0 {
            return Err(PyValueError::new_err("n_epochs must be >= 1"));
        }
        if min_dist < 0.0 {
            return Err(PyValueError::new_err("min_dist must be >= 0"));
        }
        if spread <= 0.0 {
            return Err(PyValueError::new_err("spread must be > 0"));
        }
        if learning_rate <= 0.0 {
            return Err(PyValueError::new_err("learning_rate must be > 0"));
        }

        Ok(Self {
            n_components,
            n_neighbors,
            n_epochs,
            min_dist,
            spread,
            learning_rate,
            negative_sample_rate,
            gamma,
            random_state,
        })
    }

    /// Fit and transform data using UMAP.
    ///
    /// Note: This is a minimal Rust implementation focusing on the core UMAP
    /// pipeline (kNN -> fuzzy graph -> SGD). It currently supports only dense
    /// inputs and uses exact kNN (O(n²)).
    pub fn fit_transform<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = data.as_array();
        let n_samples = x.nrows();
        let _n_features = x.ncols();

        if n_samples < 2 {
            return Err(PyValueError::new_err("UMAP requires at least 2 samples"));
        }

        let n_neighbors = self.n_neighbors.min(n_samples.saturating_sub(1));
        if n_neighbors < 2 {
            return Err(PyValueError::new_err(
                "n_neighbors must be < n_samples and >= 2",
            ));
        }

        // Convert to f32 for SIMD distance computation
        let data_f32: Vec<Vec<f32>> = x
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f32).collect())
            .collect();

        // 1) Exact kNN (for now)
        let (knn_indices, knn_dists) = compute_knn(&data_f32, n_neighbors);

        // Mean of kNN distances (used for sigma lower bound)
        let mean_distances = knn_dists
            .iter()
            .flat_map(|row| row.iter())
            .sum::<f64>()
            / ((n_samples * n_neighbors) as f64);

        // 2) Smooth kNN distance scaling
        let sig_rho: Vec<(f64, f64)> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                smooth_knn_dist_row(
                    &knn_dists[i],
                    n_neighbors as f64,
                    mean_distances,
                    64,
                    1.0,
                    1.0,
                )
            })
            .collect();

        let mut sigmas = vec![0.0; n_samples];
        let mut rhos = vec![0.0; n_samples];
        for (i, (sigma, rho)) in sig_rho.into_iter().enumerate() {
            sigmas[i] = sigma;
            rhos[i] = rho;
        }

        // 3) Membership strengths (directed)
        let mut directed: HashMap<u64, f64> = HashMap::with_capacity(n_samples * n_neighbors);
        for i in 0..n_samples {
            let sigma = sigmas[i];
            let rho = rhos[i];
            for (pos, &j) in knn_indices[i].iter().enumerate() {
                let d = knn_dists[i][pos];
                let val = if d - rho <= 0.0 || sigma == 0.0 {
                    1.0
                } else {
                    (-(d - rho) / sigma).exp()
                };
                if val > 0.0 {
                    directed.insert(pack_u64(i, j), val);
                }
            }
        }

        // 4) Fuzzy union to symmetrize weights
        let mut undirected: HashMap<u64, (f64, f64)> = HashMap::with_capacity(directed.len());
        for (&key, &w) in directed.iter() {
            let (i, j) = unpack_u64(key);
            let (a, b, forward) = if i < j { (i, j, true) } else { (j, i, false) };
            let entry = undirected.entry(pack_u64(a, b)).or_insert((0.0, 0.0));
            if forward {
                entry.0 = w;
            } else {
                entry.1 = w;
            }
        }

        let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(undirected.len() * 2);
        for (key, (w_ab, w_ba)) in undirected.into_iter() {
            let (a, b) = unpack_u64(key);
            let w = w_ab + w_ba - w_ab * w_ba;
            if w > 0.0 {
                edges.push((a, b, w));
                edges.push((b, a, w));
            }
        }

        if edges.is_empty() {
            return Err(PyValueError::new_err(
                "UMAP produced an empty neighbor graph; check inputs/parameters",
            ));
        }

        // 5) Initialize embedding
        let seed = self.random_state.unwrap_or(42);
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1e-4).map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mut embedding = Array2::<f64>::zeros((n_samples, self.n_components));
        for mut row in embedding.rows_mut() {
            for v in row.iter_mut() {
                *v = normal.sample(&mut rng);
            }
        }

        // 6) Optimize
        let (a_param, b_param) = find_ab_params(self.spread, self.min_dist);
        optimize_embedding(
            &mut embedding,
            &edges,
            self.n_epochs,
            self.learning_rate,
            self.negative_sample_rate,
            self.gamma,
            a_param,
            b_param,
            seed ^ 0x9E3779B97F4A7C15,
        );

        Ok(embedding.into_pyarray_bound(py))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_find_ab_params_positive() {
        let (a, b) = find_ab_params(1.0, 0.1);
        assert!(a > 0.0);
        assert!(b > 0.0);
    }

    #[test]
    fn test_smooth_knn_dist_row_outputs_positive_sigma() {
        let dists = vec![0.2, 0.25, 0.3, 0.35, 0.4];
        let (sigma, rho) = smooth_knn_dist_row(&dists, 5.0, 0.3, 64, 1.0, 1.0);
        assert!(sigma > 0.0);
        assert!(rho >= 0.0);
    }

    #[test]
    fn test_compute_knn_shapes() {
        let data = vec![
            vec![0.0_f32, 0.0],
            vec![1.0_f32, 0.0],
            vec![0.0_f32, 1.0],
            vec![1.0_f32, 1.0],
        ];
        let (idx, dist) = compute_knn(&data, 2);
        assert_eq!(idx.len(), 4);
        assert_eq!(dist.len(), 4);
        assert_eq!(idx[0].len(), 2);
        assert_eq!(dist[0].len(), 2);
    }

    #[test]
    fn test_optimize_embedding_centers() {
        let mut embedding = Array2::<f64>::zeros((10, 2));
        for i in 0..10 {
            embedding[[i, 0]] = i as f64;
            embedding[[i, 1]] = (i as f64) * 2.0;
        }
        let edges = vec![(0, 1, 1.0), (1, 0, 1.0)];
        optimize_embedding(&mut embedding, &edges, 5, 1.0, 1, 1.0, 1.0, 1.0, 42);
        let mean = embedding.mean_axis(Axis(0)).unwrap();
        assert_relative_eq!(mean[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(mean[1], 0.0, epsilon = 1e-6);
    }
}
