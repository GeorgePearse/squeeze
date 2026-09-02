//! Exact k-nearest-neighbour index for dense data.
//!
//! For the neighbour counts UMAP uses on high-dimensional embeddings (k in the hundreds,
//! d in the thousands) graph-based approximate search struggles: on 1280-d cosine data
//! with k=200 the HNSW index in this crate recovered 65% of the true neighbours and the
//! remaining recall could only be bought with an index that was slower than brute force.
//!
//! Brute force is O(n^2 d) but it is a dense matrix product, which is the one thing a CPU
//! does at full throughput. The work is split into row blocks; each rayon worker owns one
//! block, multiplies it against the whole dataset (`matrixmultiply` sgemm through
//! `ndarray::dot`) and selects the k smallest distances per row with a partial sort, so
//! the selection is parallel as well and the similarity slab never has to be materialised
//! for the full dataset at once. Memory per worker is `block * n * 4` bytes.
//!
//! Cosine distance is `1 - <a, b>` on L2-normalised rows; euclidean is
//! `sqrt(|a|^2 + |b|^2 - 2<a, b>)` with norms precomputed once. Both are exact (recall 1.0)
//! and deterministic, which the approximate backends are not.

use ndarray::{Array2, ArrayView2, Axis};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Metric {
    Cosine,
    Euclidean,
}

impl Metric {
    fn parse(name: &str) -> PyResult<Self> {
        match name {
            "cosine" | "correlation" => Ok(Metric::Cosine),
            "euclidean" | "l2" => Ok(Metric::Euclidean),
            other => Err(PyValueError::new_err(format!(
                "ExactKnnIndex supports 'cosine' and 'euclidean', got '{}'",
                other
            ))),
        }
    }
}

/// Exact brute-force k-NN index over a dense float32 matrix.
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct ExactKnnIndex {
    /// Row-normalised copy for cosine, raw copy for euclidean.
    data: Array2<f32>,
    /// Squared L2 norms of `data` rows (euclidean only; empty for cosine).
    sq_norms: Vec<f32>,
    metric: Metric,
    n_neighbors: usize,
    block: usize,
    neighbor_graph_cache: Option<(Vec<i64>, Vec<f32>)>,
}

fn prepare(rows: ArrayView2<f32>, metric: Metric) -> (Array2<f32>, Vec<f32>) {
    let d = rows.ncols();
    let mut out = rows.to_owned();
    let flat = out.as_slice_mut().expect("to_owned() yields a contiguous row-major array");
    match metric {
        Metric::Cosine => {
            flat.par_chunks_mut(d).for_each(|row| {
                let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 1e-12 {
                    row.iter_mut().for_each(|v| *v /= norm);
                }
            });
            (out, Vec::new())
        }
        Metric::Euclidean => {
            let sq: Vec<f32> = flat.par_chunks(d).map(|row| row.iter().map(|v| v * v).sum::<f32>()).collect();
            (out, sq)
        }
    }
}

/// k nearest rows of `index` for every row of `queries`, as flat row-major (n_q * k) buffers.
///
/// `self_offset = Some(o)` means query row i is index row `o + i` and must be excluded.
#[allow(clippy::too_many_arguments)]
fn knn(
    index: &Array2<f32>,
    index_sq_norms: &[f32],
    queries: ArrayView2<f32>,
    query_sq_norms: &[f32],
    metric: Metric,
    k: usize,
    block: usize,
    self_offset: Option<usize>,
) -> (Vec<i64>, Vec<f32>) {
    let n_index = index.nrows();
    let n_q = queries.nrows();
    let k = k.min(n_index);
    let index_t = index.t();

    let mut idx_out = vec![-1i64; n_q * k];
    let mut dist_out = vec![f32::NAN; n_q * k];

    let block = block.max(1);
    let out_chunks: Vec<(&mut [i64], &mut [f32])> = idx_out
        .chunks_mut(block * k)
        .zip(dist_out.chunks_mut(block * k))
        .collect();

    out_chunks
        .into_par_iter()
        .enumerate()
        .for_each(|(b, (idx_chunk, dist_chunk))| {
            let start = b * block;
            let end = (start + block).min(n_q);
            let q_block = queries.slice(ndarray::s![start..end, ..]);
            // (rows, n_index) similarity slab for this block only.
            let sims = q_block.dot(&index_t);

            let mut scratch: Vec<(f32, u32)> = Vec::with_capacity(n_index);
            for (r, sim_row) in sims.axis_iter(Axis(0)).enumerate() {
                let qi = start + r;
                scratch.clear();
                match metric {
                    Metric::Cosine => {
                        scratch.extend(sim_row.iter().enumerate().map(|(j, s)| (1.0 - s, j as u32)));
                    }
                    Metric::Euclidean => {
                        let qn = query_sq_norms[qi];
                        scratch.extend(
                            sim_row
                                .iter()
                                .enumerate()
                                .map(|(j, s)| ((qn + index_sq_norms[j] - 2.0 * s).max(0.0), j as u32)),
                        );
                    }
                }
                if let Some(o) = self_offset {
                    let me = o + qi;
                    if me < n_index {
                        scratch[me].0 = f32::INFINITY;
                    }
                }
                if k < scratch.len() {
                    scratch.select_nth_unstable_by(k, |a, b| a.0.total_cmp(&b.0));
                }
                let top = &mut scratch[..k];
                top.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let base = r * k;
                for (t, (d, j)) in top.iter().enumerate() {
                    idx_chunk[base + t] = *j as i64;
                    dist_chunk[base + t] = match metric {
                        Metric::Cosine => *d,
                        Metric::Euclidean => d.sqrt(),
                    };
                }
            }
        });

    (idx_out, dist_out)
}

fn to_py<'py>(
    py: Python<'py>,
    idx: Vec<i64>,
    dist: Vec<f32>,
    n_rows: usize,
    k: usize,
) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
    let idx_arr = Array2::from_shape_vec((n_rows, k), idx)
        .map_err(|e| PyValueError::new_err(format!("bad index shape: {}", e)))?;
    let dist_arr = Array2::from_shape_vec((n_rows, k), dist)
        .map_err(|e| PyValueError::new_err(format!("bad distance shape: {}", e)))?;
    Ok((
        PyArray2::from_owned_array_bound(py, idx_arr).unbind(),
        PyArray2::from_owned_array_bound(py, dist_arr).unbind(),
    ))
}

#[pymethods]
impl ExactKnnIndex {
    /// Build the index. `block` is the number of query rows one worker handles at a time.
    #[new]
    #[pyo3(signature = (data, n_neighbors, metric = "euclidean", block = 128))]
    fn new(data: PyReadonlyArray2<f32>, n_neighbors: usize, metric: &str, block: usize) -> PyResult<Self> {
        let metric = Metric::parse(metric)?;
        let view = data.as_array();
        if view.nrows() == 0 || view.ncols() == 0 {
            return Err(PyValueError::new_err("data must be a non-empty 2-D float32 array"));
        }
        if n_neighbors == 0 {
            return Err(PyValueError::new_err("n_neighbors must be positive"));
        }
        let (prepared, sq_norms) = prepare(view, metric);
        Ok(Self {
            data: prepared,
            sq_norms,
            metric,
            n_neighbors,
            block: block.max(1),
            neighbor_graph_cache: None,
        })
    }

    /// k-NN graph of the indexed rows themselves, each row excluded from its own list.
    fn neighbor_graph<'py>(&mut self, py: Python<'py>) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
        let n = self.data.nrows();
        let k = self.n_neighbors.min(n.saturating_sub(1)).max(1);
        if self.neighbor_graph_cache.is_none() {
            let (idx, dist) = py.allow_threads(|| {
                knn(
                    &self.data,
                    &self.sq_norms,
                    self.data.view(),
                    &self.sq_norms,
                    self.metric,
                    k,
                    self.block,
                    Some(0),
                )
            });
            self.neighbor_graph_cache = Some((idx, dist));
        }
        let (idx, dist) = self.neighbor_graph_cache.as_ref().unwrap();
        to_py(py, idx.clone(), dist.clone(), n, k)
    }

    /// k nearest indexed rows for new query rows (no self exclusion). `ef` is accepted and
    /// ignored so the call shape matches the HNSW index.
    #[pyo3(signature = (queries, k, ef = 0, filter_mask = None))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        ef: usize,
        filter_mask: Option<PyReadonlyArray2<bool>>,
    ) -> PyResult<(Py<PyArray2<i64>>, Py<PyArray2<f32>>)> {
        let _ = ef;
        if filter_mask.is_some() {
            return Err(PyValueError::new_err("filter_mask is not supported by ExactKnnIndex"));
        }
        let q = queries.as_array();
        if q.ncols() != self.data.ncols() {
            return Err(PyValueError::new_err(format!(
                "query has {} features, index has {}",
                q.ncols(),
                self.data.ncols()
            )));
        }
        let (q_prepared, q_sq) = prepare(q, self.metric);
        let k = k.min(self.data.nrows()).max(1);
        let (idx, dist) = py.allow_threads(|| {
            knn(
                &self.data,
                &self.sq_norms,
                q_prepared.view(),
                &q_sq,
                self.metric,
                k,
                self.block,
                None,
            )
        });
        to_py(py, idx, dist, q.nrows(), k)
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.data.nrows()
    }

    #[getter]
    fn n_features(&self) -> usize {
        self.data.ncols()
    }

    fn __repr__(&self) -> String {
        format!(
            "ExactKnnIndex(n_samples={}, n_features={}, n_neighbors={}, metric='{}')",
            self.data.nrows(),
            self.data.ncols(),
            self.n_neighbors,
            match self.metric {
                Metric::Cosine => "cosine",
                Metric::Euclidean => "euclidean",
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_matches_naive_cosine() {
        let n = 300;
        let d = 17;
        let mut v = Vec::with_capacity(n * d);
        let mut s = 12345u64;
        for _ in 0..n * d {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            v.push(((s >> 33) as f32 / u32::MAX as f32) - 0.5);
        }
        let x = Array2::from_shape_vec((n, d), v).unwrap();
        let (data, sq) = prepare(x.view(), Metric::Cosine);
        let k = 7;
        let (idx, _) = knn(&data, &sq, data.view(), &sq, Metric::Cosine, k, 32, Some(0));
        for i in 0..n {
            let mut naive: Vec<(f32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (1.0 - data.row(i).dot(&data.row(j)), j))
                .collect();
            naive.sort_by(|a, b| a.0.total_cmp(&b.0));
            let want: Vec<i64> = naive[..k].iter().map(|(_, j)| *j as i64).collect();
            assert_eq!(&idx[i * k..(i + 1) * k], &want[..], "row {}", i);
        }
    }
}
