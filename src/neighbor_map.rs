//! Experimental CPU graph embeddings. These are additive alternatives, not replacements.
//!
//! SpectralMap uses block iteration on a sparse, locally scaled graph. NeighborMap
//! refines PCA or spectral coordinates with weighted attraction and sampled repulsion.
//! The latter is inspired by UMAP/negative-sampling layouts, not a UMAP implementation.

use ndarray::{Array2, Axis};
use ndarray_linalg::{Eigh, UPLO};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::metrics_simd;

type Graph = Vec<Vec<(usize, f64)>>;

fn validate(data: &Array2<f64>, neighbors: usize) -> Result<(), String> {
    if data.nrows() < 3 || data.ncols() == 0 {
        return Err("expected at least three samples and one feature".into());
    }
    if neighbors == 0 || neighbors >= data.nrows() {
        return Err("n_neighbors must be between 1 and n_samples - 1".into());
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err("input must contain only finite values".into());
    }
    Ok(())
}

fn normalized(data: &Array2<f64>) -> Array2<f64> {
    // A single scale preserves Euclidean neighborhoods and bounds the SIMD f32
    // distances, even for very large/small finite f64 input magnitudes.
    let scale = data.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    if scale == 0.0 {
        data.clone()
    } else {
        data / scale
    }
}

fn neighbor_graph(data: &Array2<f64>, k: usize) -> Graph {
    let rows: Vec<Vec<f32>> = data
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|&v| v as f32).collect())
        .collect();
    // Only retain k edges per sample. No dense n-by-n distance matrix.
    (0..rows.len())
        .into_par_iter()
        .map(|i| {
            let mut distances: Vec<(usize, f64)> = (0..rows.len())
                .filter(|&j| i != j)
                .map(|j| {
                    (
                        j,
                        metrics_simd::squared_euclidean(&rows[i], &rows[j]).unwrap() as f64,
                    )
                })
                .collect();
            let compare =
                |a: &(usize, f64), b: &(usize, f64)| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0));
            distances.select_nth_unstable_by(k - 1, compare);
            distances.truncate(k);
            distances.sort_unstable_by(compare);
            for (_, distance) in &mut distances {
                *distance = distance.sqrt();
            }
            let rho = distances[0].1;
            let sigma = (distances.iter().map(|(_, d)| d - rho).sum::<f64>() / k as f64).max(1e-8);
            distances
                .into_iter()
                .map(|(j, d)| (j, (-(d - rho) / sigma).exp()))
                .collect()
        })
        .collect()
}

fn center_scale(y: &mut Array2<f64>) {
    let mean = y.mean_axis(Axis(0)).unwrap();
    for mut row in y.rows_mut() {
        row -= &mean;
    }
    let scale = (y.iter().map(|v| v * v).sum::<f64>() / y.nrows() as f64)
        .sqrt()
        .max(1e-12);
    *y /= scale;
}

fn pca_init(data: &Array2<f64>) -> Result<Array2<f64>, String> {
    let mut centered = data.clone();
    let mean = data.mean_axis(Axis(0)).unwrap();
    for mut row in centered.rows_mut() {
        row -= &mean;
    }
    let (_, vectors) = centered
        .t()
        .dot(&centered)
        .eigh(UPLO::Upper)
        .map_err(|e| e.to_string())?;
    let mut y = Array2::zeros((data.nrows(), 2));
    for dim in 0..2.min(data.ncols()) {
        y.column_mut(dim)
            .assign(&centered.dot(&vectors.column(data.ncols() - dim - 1)));
    }
    center_scale(&mut y);
    Ok(y)
}

fn spectral_init(graph: &Graph, iterations: usize, seed: u64) -> Array2<f64> {
    let n = graph.len();
    let mut symmetric: Graph = vec![Vec::new(); n];
    for (i, edges) in graph.iter().enumerate() {
        for &(j, w) in edges {
            symmetric[i].push((j, 0.5 * w));
            symmetric[j].push((i, 0.5 * w));
        }
    }
    let degrees: Vec<f64> = symmetric
        .iter()
        .map(|r| r.iter().map(|(_, w)| w).sum::<f64>().sqrt())
        .collect();
    let norm = degrees.iter().map(|v| v * v).sum::<f64>().sqrt();
    let constant: Vec<f64> = degrees.iter().map(|v| v / norm).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut y = Array2::from_shape_fn((n, 2), |_| rng.random_range(-1.0..1.0));
    orthogonalize(&mut y, &constant);
    for _ in 0..iterations {
        let mut next = Array2::zeros((n, 2));
        for (i, edges) in symmetric.iter().enumerate() {
            for dim in 0..2 {
                // Shift by I so power iteration targets the largest algebraic
                // eigenvalues rather than oscillatory negative eigenvectors.
                next[[i, dim]] = 0.5 * y[[i, dim]];
                for &(j, w) in edges {
                    next[[i, dim]] += 0.5 * w * y[[j, dim]] / (degrees[i] * degrees[j]);
                }
            }
        }
        orthogonalize(&mut next, &constant);
        y = next;
    }
    center_scale(&mut y);
    y
}

fn orthogonalize(y: &mut Array2<f64>, constant: &[f64]) {
    for dim in 0..2 {
        let dot = (0..y.nrows())
            .map(|i| y[[i, dim]] * constant[i])
            .sum::<f64>();
        for i in 0..y.nrows() {
            y[[i, dim]] -= dot * constant[i];
        }
        if dim == 1 {
            let dot = (0..y.nrows()).map(|i| y[[i, 0]] * y[[i, 1]]).sum::<f64>();
            for i in 0..y.nrows() {
                y[[i, 1]] -= dot * y[[i, 0]];
            }
        }
        let norm = y
            .column(dim)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(1e-12);
        y.column_mut(dim).mapv_inplace(|v| v / norm);
    }
}

fn optimize(
    y: &mut Array2<f64>,
    graph: &Graph,
    epochs: usize,
    negatives: usize,
    rate: f64,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    // Break exact overlaps without labels or changing the neighborhood graph.
    for v in y.iter_mut() {
        *v += rng.random_range(-1e-4..1e-4);
    }
    let mut edges: Vec<(usize, usize, f64)> = graph
        .iter()
        .enumerate()
        .flat_map(|(i, row)| row.iter().map(move |&(j, w)| (i, j, w)))
        .collect();
    use rand::seq::SliceRandom;
    for epoch in 0..epochs {
        let alpha = rate * (1.0 - epoch as f64 / epochs as f64);
        edges.shuffle(&mut rng);
        for &(i, j, weight) in &edges {
            if rng.random::<f64>() > weight {
                continue;
            }
            let dx = y[[i, 0]] - y[[j, 0]];
            let dy = y[[i, 1]] - y[[j, 1]];
            let attraction = -2.0 / (1.0 + dx * dx + dy * dy);
            for (dim, delta) in [dx, dy].iter().enumerate() {
                let step = (attraction * delta).clamp(-4.0, 4.0) * alpha;
                y[[i, dim]] += step;
                y[[j, dim]] -= step;
            }
            for _ in 0..negatives {
                let other = rng.random_range(0..y.nrows());
                if other == i || other == j {
                    continue;
                }
                let dx = y[[i, 0]] - y[[other, 0]];
                let dy = y[[i, 1]] - y[[other, 1]];
                let d2 = dx * dx + dy * dy;
                let repulsion = 2.0 / ((0.01 + d2) * (1.0 + d2));
                y[[i, 0]] += (repulsion * dx).clamp(-4.0, 4.0) * alpha;
                y[[i, 1]] += (repulsion * dy).clamp(-4.0, 4.0) * alpha;
            }
        }
    }
}

/// Approximate sparse spectral embedding; experimental, Euclidean, two-dimensional.
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct SpectralMap {
    n_neighbors: usize,
    n_iter: usize,
    random_state: u64,
}

#[pymethods]
impl SpectralMap {
    #[new]
    #[pyo3(signature = (n_neighbors=15, n_iter=128, random_state=42))]
    fn new(n_neighbors: usize, n_iter: usize, random_state: u64) -> PyResult<Self> {
        if n_neighbors == 0 || n_iter == 0 {
            return Err(PyValueError::new_err(
                "n_neighbors and n_iter must be positive",
            ));
        }
        Ok(Self {
            n_neighbors,
            n_iter,
            random_state,
        })
    }

    fn fit_transform<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = data.as_array().as_standard_layout().to_owned();
        validate(&x, self.n_neighbors).map_err(PyValueError::new_err)?;
        let y = py.allow_threads(|| {
            spectral_init(
                &neighbor_graph(&normalized(&x), self.n_neighbors),
                self.n_iter,
                self.random_state,
            )
        });
        Ok(y.into_pyarray_bound(py))
    }
}

/// Sampled neighbor embedding with PCA or sparse spectral initialization.
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct NeighborMap {
    n_neighbors: usize,
    n_epochs: usize,
    negative_samples: usize,
    learning_rate: f64,
    init: String,
    random_state: u64,
}

#[pymethods]
impl NeighborMap {
    #[new]
    #[pyo3(signature = (n_neighbors=15, n_epochs=160, negative_samples=5, learning_rate=1.0, init="pca", random_state=42))]
    fn new(
        n_neighbors: usize,
        n_epochs: usize,
        negative_samples: usize,
        learning_rate: f64,
        init: &str,
        random_state: u64,
    ) -> PyResult<Self> {
        if n_neighbors == 0 || n_epochs == 0 || negative_samples == 0 {
            return Err(PyValueError::new_err(
                "neighbors, epochs and negative_samples must be positive",
            ));
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(PyValueError::new_err(
                "learning_rate must be positive and finite",
            ));
        }
        if init != "pca" && init != "spectral" {
            return Err(PyValueError::new_err("init must be 'pca' or 'spectral'"));
        }
        Ok(Self {
            n_neighbors,
            n_epochs,
            negative_samples,
            learning_rate,
            init: init.into(),
            random_state,
        })
    }

    fn fit_transform<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = data.as_array().as_standard_layout().to_owned();
        validate(&x, self.n_neighbors).map_err(PyValueError::new_err)?;
        let result: Result<Array2<f64>, String> = py.allow_threads(|| {
            let x = normalized(&x);
            let graph = neighbor_graph(&x, self.n_neighbors);
            let mut y = if self.init == "spectral" {
                spectral_init(&graph, 128, self.random_state)
            } else {
                pca_init(&x)?
            };
            optimize(
                &mut y,
                &graph,
                self.n_epochs,
                self.negative_samples,
                self.learning_rate,
                self.random_state,
            );
            if y.iter().any(|v| !v.is_finite()) {
                return Err(
                    "optimization produced non-finite coordinates; reduce learning_rate".into(),
                );
            }
            Ok(y)
        });
        Ok(result
            .map_err(PyValueError::new_err)?
            .into_pyarray_bound(py))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn graph_matches_scalar_neighbors_and_breaks_ties_by_index() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 3.0]];
        let graph = neighbor_graph(&data, 2);
        assert_eq!(
            graph[0].iter().map(|&(j, _)| j).collect::<Vec<_>>(),
            vec![1, 2]
        );
        for (i, edges) in graph.iter().enumerate() {
            assert_eq!(edges.len(), 2);
            assert!(edges.iter().all(|&(j, w)| i != j && w > 0.0 && w <= 1.0));
            let mut expected: Vec<_> = (0..data.nrows())
                .filter(|&j| i != j)
                .map(|j| (j, (&data.row(i) - &data.row(j)).mapv(|v| v * v).sum()))
                .collect();
            expected.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            assert_eq!(
                edges.iter().map(|&(j, _)| j).collect::<Vec<_>>(),
                expected[..2].iter().map(|&(j, _)| j).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn parallel_graph_is_deterministic() {
        let data = array![[0.0], [1.0], [2.0], [4.0], [8.0]];
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| neighbor_graph(&data, 3))
        };
        assert_eq!(run(1), run(4));
    }

    #[test]
    fn pca_initialization_is_centered_and_unit_radius() {
        let data = array![[0.0], [1.0], [2.0], [4.0]];
        let y = pca_init(&data).unwrap();
        assert_eq!(y.dim(), (4, 2));
        assert!(y
            .mean_axis(Axis(0))
            .unwrap()
            .iter()
            .all(|v| v.abs() < 1e-12));
        assert!((y.mapv(|v| v * v).sum() / 4.0 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn graph_normalization_handles_extreme_finite_magnitudes() {
        let data = array![[1e300], [-1e300], [0.0]];
        let scaled = normalized(&data);
        assert_eq!(scaled, array![[1.0], [-1.0], [0.0]]);
        assert!(neighbor_graph(&scaled, 2)
            .iter()
            .flatten()
            .all(|&(_, w)| w.is_finite()));
    }
}
