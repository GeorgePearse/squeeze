//! Label-aware metrics on a 2D embedding, computed over its Delaunay graph.
//!
//! The embedding's Delaunay triangulation gives a planar neighbourhood graph whose
//! edges are weighted by Euclidean length. From it we derive, per class label:
//!
//! - a confusion matrix: how many points of each label fall inside the region a
//!   label occupies (its points plus everything one short edge away);
//! - a neighbourhood matrix: how close and how large each other label is just
//!   outside that region, scored against the empirical distribution of all pairs.
//!
//! The method follows Manz et al., "A General Framework for Comparing Embedding
//! Visualizations Across Class-Label Hierarchies" (IEEE VIS 2024).

use delaunator::{prev_halfedge, triangulate, Point, EMPTY};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Undirected weighted graph in compressed sparse row form.
struct CsrGraph {
    offsets: Vec<usize>,
    targets: Vec<usize>,
    weights: Vec<f64>,
}

impl CsrGraph {
    fn from_edges(n_nodes: usize, edges: &[(usize, usize, f64)]) -> Self {
        let mut degree = vec![0usize; n_nodes];
        for &(a, b, _) in edges {
            degree[a] += 1;
            degree[b] += 1;
        }
        let mut offsets = vec![0usize; n_nodes + 1];
        for i in 0..n_nodes {
            offsets[i + 1] = offsets[i] + degree[i];
        }
        let mut fill = offsets[..n_nodes].to_vec();
        let mut targets = vec![0usize; offsets[n_nodes]];
        let mut weights = vec![0.0f64; offsets[n_nodes]];
        for &(a, b, w) in edges {
            targets[fill[a]] = b;
            weights[fill[a]] = w;
            fill[a] += 1;
            targets[fill[b]] = a;
            weights[fill[b]] = w;
            fill[b] += 1;
        }
        Self {
            offsets,
            targets,
            weights,
        }
    }

    fn n_nodes(&self) -> usize {
        self.offsets.len() - 1
    }

    fn n_edges(&self) -> usize {
        self.targets.len() / 2
    }

    fn neighbors(&self, node: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let range = self.offsets[node]..self.offsets[node + 1];
        self.targets[range.clone()]
            .iter()
            .copied()
            .zip(self.weights[range].iter().copied())
    }

    /// Iterate each undirected edge once as `(a, b, weight)` with `a < b`.
    fn edges(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.n_nodes()).flat_map(move |a| {
            self.neighbors(a)
                .filter(move |&(b, _)| a < b)
                .map(move |(b, w)| (a, b, w))
        })
    }
}

fn euclidean(points: &[(f64, f64)], a: usize, b: usize) -> f64 {
    let dx = points[a].0 - points[b].0;
    let dy = points[a].1 - points[b].1;
    (dx * dx + dy * dy).sqrt()
}

/// Empirical cumulative distribution function over a fixed sample.
pub struct Ecdf {
    sorted: Vec<f64>,
}

impl Ecdf {
    pub fn new(mut values: Vec<f64>) -> Self {
        values.retain(|v| !v.is_nan());
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self { sorted: values }
    }

    /// Fraction of the sample that is `<= x`; 0 for an empty sample.
    pub fn eval(&self, x: f64) -> f64 {
        if self.sorted.is_empty() {
            return 0.0;
        }
        let count = self.sorted.partition_point(|&v| v <= x);
        count as f64 / self.sorted.len() as f64
    }
}

/// Mean and (population) standard deviation; `None` for an empty slice.
fn mean_std(values: &[f64]) -> Option<(f64, f64)> {
    if values.is_empty() {
        return None;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    Some((mean, var.sqrt()))
}

/// Determinant deciding whether `p` lies inside (>0), on (0) or outside (<0)
/// the circumcircle of the counter-clockwise triangle `(a, b, c)`.
fn incircle_det(a: (f64, f64), b: (f64, f64), c: (f64, f64), p: (f64, f64)) -> f64 {
    let (adx, ady) = (a.0 - p.0, a.1 - p.1);
    let (bdx, bdy) = (b.0 - p.0, b.1 - p.1);
    let (cdx, cdy) = (c.0 - p.0, c.1 - p.1);
    let (ad, bd, cd) = (
        adx * adx + ady * ady,
        bdx * bdx + bdy * bdy,
        cdx * cdx + cdy * cdy,
    );
    adx * (bdy * cd - bd * cdy) - ady * (bdx * cd - bd * cdx) + ad * (bdx * cdy - bdy * cdx)
}

/// Delaunay graph of a 2D embedding with label-aware summaries.
#[pyclass(module = "squeeze._hnsw_backend")]
pub struct LabelGraph {
    points: Vec<(f64, f64)>,
    graph: CsrGraph,
    triangles: Vec<usize>,
    halfedges: Vec<usize>,
}

impl LabelGraph {
    pub fn from_points(points: Vec<(f64, f64)>) -> Self {
        let n = points.len();
        let input: Vec<Point> = points.iter().map(|&(x, y)| Point { x, y }).collect();
        let tri = triangulate(&input);

        let mut edges: Vec<(usize, usize)> = Vec::with_capacity(tri.triangles.len());
        for t in tri.triangles.chunks_exact(3) {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                edges.push((a.min(b), a.max(b)));
            }
        }
        if tri.triangles.is_empty() {
            // Collinear (or tiny) input: delaunator returns the ordered points as the hull.
            for pair in tri.hull.windows(2) {
                edges.push((pair[0].min(pair[1]), pair[0].max(pair[1])));
            }
        }
        edges.sort_unstable();
        edges.dedup();
        let weighted: Vec<(usize, usize, f64)> = edges
            .iter()
            .map(|&(a, b)| (a, b, euclidean(&points, a, b)))
            .collect();

        Self {
            graph: CsrGraph::from_edges(n, &weighted),
            points,
            triangles: tri.triangles,
            halfedges: tri.halfedges,
        }
    }

    /// Nodes reachable from label `label`: its own points plus anything one edge
    /// away along an edge no longer than `mean + 3 * std` of the label's intra-label
    /// edge lengths. A label without intra-label edges follows no edges.
    fn confusion_set(&self, labels: &[usize], label: usize) -> Vec<bool> {
        let n = self.graph.n_nodes();
        let intra: Vec<f64> = self
            .graph
            .edges()
            .filter(|&(a, b, _)| labels[a] == label && labels[b] == label)
            .map(|(_, _, w)| w)
            .collect();
        let threshold = mean_std(&intra).map(|(m, s)| m + 3.0 * s);

        let mut reached = vec![false; n];
        for node in 0..n {
            if labels[node] != label {
                continue;
            }
            reached[node] = true;
            if let Some(t) = threshold {
                for (nb, w) in self.graph.neighbors(node) {
                    if w <= t {
                        reached[nb] = true;
                    }
                }
            }
        }
        reached
    }

    fn confusion_row(&self, labels: &[usize], n_labels: usize, label: usize) -> Vec<u64> {
        let reached = self.confusion_set(labels, label);
        let mut row = vec![0u64; n_labels];
        for (node, &hit) in reached.iter().enumerate() {
            if hit {
                row[labels[node]] += 1;
            }
        }
        row
    }

    /// Breadth-first walk up to `max_depth` hops outward from a label's confusion
    /// set, returning `(count, mean_distance)` per other label. Distance is the
    /// path length accumulated from the first node outside the set's boundary.
    fn neighborhood_row(
        &self,
        labels: &[usize],
        n_labels: usize,
        label: usize,
        max_depth: usize,
    ) -> Vec<(u64, f64)> {
        let inside = self.confusion_set(labels, label);
        let n = self.graph.n_nodes();
        let mut dist = vec![f64::INFINITY; n];
        let mut frontier: Vec<usize> = Vec::new();

        for node in 0..n {
            if !inside[node] {
                continue;
            }
            for (nb, w) in self.graph.neighbors(node) {
                if !inside[nb] && w < dist[nb] {
                    if dist[nb].is_infinite() {
                        frontier.push(nb);
                    }
                    dist[nb] = w;
                }
            }
        }

        let mut counts = vec![0u64; n_labels];
        let mut sums = vec![0.0f64; n_labels];
        let mut depth = 1;
        while !frontier.is_empty() {
            for &node in &frontier {
                counts[labels[node]] += 1;
                sums[labels[node]] += dist[node];
            }
            if depth >= max_depth {
                break;
            }
            let mut next: Vec<usize> = Vec::new();
            for &node in &frontier {
                for (nb, w) in self.graph.neighbors(node) {
                    let candidate = dist[node] + w;
                    if inside[nb] || candidate >= dist[nb] {
                        continue;
                    }
                    if dist[nb].is_infinite() {
                        next.push(nb);
                    }
                    dist[nb] = candidate;
                }
            }
            frontier = next;
            depth += 1;
        }

        counts
            .iter()
            .zip(sums.iter())
            .map(|(&c, &s)| {
                if c == 0 {
                    (0, f64::NAN)
                } else {
                    (c, s / c as f64)
                }
            })
            .collect()
    }

    pub fn confusion_matrix(&self, labels: &[usize], n_labels: usize) -> Vec<Vec<u64>> {
        (0..n_labels)
            .into_par_iter()
            .map(|l| self.confusion_row(labels, n_labels, l))
            .collect()
    }

    pub fn neighborhood_matrix(
        &self,
        labels: &[usize],
        n_labels: usize,
        max_depth: usize,
    ) -> Vec<Vec<f64>> {
        let rows: Vec<Vec<(u64, f64)>> = (0..n_labels)
            .into_par_iter()
            .map(|l| self.neighborhood_row(labels, n_labels, l, max_depth))
            .collect();

        let occupied: Vec<&(u64, f64)> = rows.iter().flatten().filter(|(c, _)| *c > 0).collect();
        let count_ecdf = Ecdf::new(occupied.iter().map(|(c, _)| *c as f64).collect());
        let dist_ecdf = Ecdf::new(occupied.iter().map(|(_, d)| *d).collect());

        rows.iter()
            .map(|row| {
                row.iter()
                    .map(|&(c, d)| {
                        if c == 0 {
                            0.0
                        } else {
                            count_ecdf.eval(c as f64) * (1.0 - dist_ecdf.eval(d))
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Number of triangles whose circumcircle passes through a fourth point, i.e.
    /// where the Delaunay triangulation is not unique.
    pub fn ambiguous_triangles(&self) -> usize {
        let scale = {
            let (mut lo, mut hi) = (
                (f64::INFINITY, f64::INFINITY),
                (f64::NEG_INFINITY, f64::NEG_INFINITY),
            );
            for &(x, y) in &self.points {
                lo = (lo.0.min(x), lo.1.min(y));
                hi = (hi.0.max(x), hi.1.max(y));
            }
            let d = ((hi.0 - lo.0).powi(2) + (hi.1 - lo.1).powi(2)).sqrt();
            if d.is_finite() && d > 0.0 {
                d
            } else {
                1.0
            }
        };
        let tol = 1e-9 * scale.powi(4);

        (0..self.triangles.len() / 3)
            .filter(|&t| {
                let (a, b, c) = (
                    self.points[self.triangles[3 * t]],
                    self.points[self.triangles[3 * t + 1]],
                    self.points[self.triangles[3 * t + 2]],
                );
                (3 * t..3 * t + 3).any(|e| {
                    let twin = self.halfedges[e];
                    twin != EMPTY
                        && incircle_det(a, b, c, self.points[self.triangles[prev_halfedge(twin)]])
                            .abs()
                            <= tol
                })
            })
            .count()
    }
}

fn to_matrix(rows: Vec<Vec<f64>>, n_labels: usize) -> ndarray::Array2<f64> {
    let mut out = ndarray::Array2::<f64>::zeros((n_labels, n_labels));
    for (i, row) in rows.into_iter().enumerate() {
        for (j, v) in row.into_iter().enumerate() {
            out[[i, j]] = v;
        }
    }
    out
}

fn check_labels(labels: &[i64], n_nodes: usize) -> PyResult<(Vec<usize>, usize)> {
    if labels.len() != n_nodes {
        return Err(PyValueError::new_err(format!(
            "labels has {} entries but the embedding has {} rows",
            labels.len(),
            n_nodes
        )));
    }
    if labels.iter().any(|&l| l < 0) {
        return Err(PyValueError::new_err(
            "labels must be non-negative integer codes",
        ));
    }
    let codes: Vec<usize> = labels.iter().map(|&l| l as usize).collect();
    let n_labels = codes.iter().max().map_or(0, |m| m + 1);
    Ok((codes, n_labels))
}

#[pymethods]
impl LabelGraph {
    /// Build the Delaunay graph of a 2D embedding of shape `(n_samples, 2)`.
    #[new]
    pub fn new(embedding: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let x = embedding.as_array();
        if x.ncols() != 2 {
            return Err(PyValueError::new_err(format!(
                "label metrics need a 2D embedding (n_components == 2), got {} columns",
                x.ncols()
            )));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(
                "embedding contains non-finite values",
            ));
        }
        let points: Vec<(f64, f64)> = x.rows().into_iter().map(|r| (r[0], r[1])).collect();
        Ok(Self::from_points(points))
    }

    #[getter]
    pub fn n_nodes(&self) -> usize {
        self.graph.n_nodes()
    }

    #[getter]
    pub fn n_edges(&self) -> usize {
        self.graph.n_edges()
    }

    #[getter]
    pub fn n_triangles(&self) -> usize {
        self.triangles.len() / 3
    }

    #[getter]
    pub fn n_ambiguous_triangles(&self) -> usize {
        self.ambiguous_triangles()
    }

    /// Label confusion counts, shape `(n_labels, n_labels)`; row `i` is the label
    /// histogram of the region claimed by label `i`.
    pub fn confusion<'py>(
        &self,
        py: Python<'py>,
        labels: PyReadonlyArray1<i64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (codes, n_labels) = check_labels(labels.as_slice()?, self.graph.n_nodes())?;
        let rows = py.allow_threads(|| self.confusion_matrix(&codes, n_labels));
        let rows = rows
            .into_iter()
            .map(|r| r.into_iter().map(|c| c as f64).collect())
            .collect();
        Ok(to_matrix(rows, n_labels).into_pyarray_bound(py))
    }

    /// Label neighbourhood scores in `[0, 1]`, shape `(n_labels, n_labels)`.
    #[pyo3(signature = (labels, max_depth=1))]
    pub fn neighborhood<'py>(
        &self,
        py: Python<'py>,
        labels: PyReadonlyArray1<i64>,
        max_depth: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        if max_depth == 0 {
            return Err(PyValueError::new_err("max_depth must be at least 1"));
        }
        let (codes, n_labels) = check_labels(labels.as_slice()?, self.graph.n_nodes())?;
        let rows = py.allow_threads(|| self.neighborhood_matrix(&codes, n_labels, max_depth));
        Ok(to_matrix(rows, n_labels).into_pyarray_bound(py))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ecdf_is_fraction_at_or_below() {
        let e = Ecdf::new(vec![3.0, 1.0, 2.0, 2.0]);
        assert_eq!(e.eval(0.5), 0.0);
        assert_eq!(e.eval(1.0), 0.25);
        assert_eq!(e.eval(2.0), 0.75);
        assert_eq!(e.eval(2.5), 0.75);
        assert_eq!(e.eval(3.0), 1.0);
        assert_eq!(e.eval(10.0), 1.0);
    }

    #[test]
    fn ecdf_ignores_nan_and_handles_empty() {
        assert_eq!(Ecdf::new(vec![]).eval(1.0), 0.0);
        let e = Ecdf::new(vec![f64::NAN, 1.0]);
        assert_eq!(e.eval(1.0), 1.0);
    }

    #[test]
    fn mean_std_uses_square_root() {
        let (m, s) = mean_std(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();
        assert!((m - 5.0).abs() < 1e-12);
        assert!((s - 2.0).abs() < 1e-12);
        assert!(mean_std(&[]).is_none());
    }

    fn two_blobs() -> (Vec<(f64, f64)>, Vec<usize>) {
        let mut pts = Vec::new();
        let mut labels = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                pts.push((i as f64, j as f64 * 0.9));
                labels.push(0);
                pts.push((i as f64 + 100.0, j as f64 * 0.9));
                labels.push(1);
            }
        }
        (pts, labels)
    }

    #[test]
    fn separated_blobs_have_diagonal_confusion() {
        let (pts, labels) = two_blobs();
        let g = LabelGraph::from_points(pts);
        let c = g.confusion_matrix(&labels, 2);
        assert_eq!(c, vec![vec![25, 0], vec![0, 25]]);
    }

    #[test]
    fn neighborhood_scores_are_symmetric_for_two_blobs() {
        let (pts, labels) = two_blobs();
        let g = LabelGraph::from_points(pts);
        let nb = g.neighborhood_matrix(&labels, 2, 1);
        assert_eq!(nb[0][0], 0.0);
        assert_eq!(nb[1][1], 0.0);
        assert!(nb[0][1] >= 0.0 && nb[0][1] <= 1.0);
        assert!((nb[0][1] - nb[1][0]).abs() < 1e-12);
    }

    #[test]
    fn collinear_and_empty_inputs_do_not_panic() {
        let g = LabelGraph::from_points(vec![]);
        assert_eq!(g.confusion_matrix(&[], 0), Vec::<Vec<u64>>::new());
        let g = LabelGraph::from_points(vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]);
        assert_eq!(g.graph.n_edges(), 2);
        let c = g.confusion_matrix(&[0, 0, 1], 2);
        assert_eq!(c[1], vec![0, 1]);
    }

    #[test]
    fn grid_triangles_are_ambiguous() {
        let pts: Vec<(f64, f64)> = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i as f64, j as f64)))
            .collect();
        let g = LabelGraph::from_points(pts);
        assert_eq!(g.triangles.len() / 3, 8);
        assert_eq!(g.ambiguous_triangles(), 8);
    }
}
