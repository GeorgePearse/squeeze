//! CPU backend: the numerical reference every GPU backend is tested against.
//!
//! kNN and pairwise kernels delegate to the SIMD metrics in `metrics_simd`; the embedding
//! sessions keep `f64` throughout and parallelise over points with rayon.

use super::{
    contiguous_f32, pairs_to_csr, triplets_to_csr, Backend, Device, GradFn, Metric, Result,
};
use crate::metrics_simd;
use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;
use std::cmp::Ordering;

/// The CPU reference backend. Stateless.
pub struct CpuBackend;

fn check_dims(a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<usize> {
    if a.ncols() != b.ncols() {
        return Err(format!(
            "feature mismatch: {} vs {} columns",
            a.ncols(),
            b.ncols()
        ));
    }
    Ok(a.ncols())
}

#[inline]
fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    match metric {
        Metric::Euclidean => metrics_simd::euclidean(a, b).unwrap_or(f32::MAX),
        Metric::Cosine => metrics_simd::cosine(a, b).unwrap_or(f32::MAX),
        Metric::Manhattan => metrics_simd::manhattan(a, b).unwrap_or(f32::MAX),
    }
}

/// Top-k by `(distance, index)` with a bounded insertion list; `k` is small relative to `n`.
fn topk_row(dists: impl Iterator<Item = (u32, f32)>, k: usize) -> Vec<(u32, f32)> {
    let mut best: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    let cmp = |a: &(u32, f32), b: &(u32, f32)| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    };
    for cand in dists {
        if best.len() == k && cmp(&cand, &best[k - 1]) != Ordering::Less {
            continue;
        }
        let pos = best.partition_point(|x| cmp(x, &cand) == Ordering::Less);
        best.insert(pos, cand);
        if best.len() > k {
            best.pop();
        }
    }
    best
}

impl Backend for CpuBackend {
    fn device(&self) -> &Device {
        &Device::Cpu
    }

    fn knn_bruteforce(
        &self,
        data: ArrayView2<f32>,
        queries: ArrayView2<f32>,
        k: usize,
        metric: Metric,
    ) -> Result<(Array2<u32>, Array2<f32>)> {
        let d = check_dims(data, queries)?;
        let n = data.nrows();
        let m = queries.nrows();
        if k == 0 || k > n {
            return Err(format!("k={} must be in 1..={}", k, n));
        }
        let data = contiguous_f32(data);
        let queries = contiguous_f32(queries);
        let rows: Vec<Vec<(u32, f32)>> = (0..m)
            .into_par_iter()
            .map(|qi| {
                let q = &queries[qi * d..(qi + 1) * d];
                topk_row(
                    (0..n).map(|i| (i as u32, distance(metric, q, &data[i * d..(i + 1) * d]))),
                    k,
                )
            })
            .collect();
        let mut idx = Array2::<u32>::zeros((m, k));
        let mut dist = Array2::<f32>::zeros((m, k));
        for (qi, row) in rows.into_iter().enumerate() {
            for (j, (i, dd)) in row.into_iter().enumerate() {
                idx[[qi, j]] = i;
                dist[[qi, j]] = dd;
            }
        }
        Ok((idx, dist))
    }

    fn pairwise_sqdist(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>> {
        let d = check_dims(a, b)?;
        let (n, m) = (a.nrows(), b.nrows());
        let a = contiguous_f32(a);
        let b = contiguous_f32(b);
        let mut out = vec![0f32; n * m];
        out.par_chunks_mut(m).enumerate().for_each(|(i, row)| {
            let ai = &a[i * d..(i + 1) * d];
            for (j, slot) in row.iter_mut().enumerate() {
                *slot =
                    metrics_simd::squared_euclidean(ai, &b[j * d..(j + 1) * d]).unwrap_or(f32::MAX);
            }
        });
        Array2::from_shape_vec((n, m), out).map_err(|e| e.to_string())
    }

    fn gram(&self, a: ArrayView2<f32>) -> Result<Array2<f32>> {
        let (n, d) = (a.nrows(), a.ncols());
        let a = contiguous_f32(a);
        let mut out = vec![0f32; n * n];
        out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let ai = &a[i * d..(i + 1) * d];
            for (j, slot) in row.iter_mut().enumerate() {
                let aj = &a[j * d..(j + 1) * d];
                let dot: f64 = ai.iter().zip(aj).map(|(&x, &y)| x as f64 * y as f64).sum();
                *slot = dot as f32;
            }
        });
        Array2::from_shape_vec((n, n), out).map_err(|e| e.to_string())
    }

    fn pacmap_session(
        &self,
        n: usize,
        dim: usize,
        near: &[(u32, u32)],
        mid_near: &[(u32, u32)],
        far: &[(u32, u32)],
        run: &mut dyn FnMut(&mut GradFn<'_>) -> Result<()>,
    ) -> Result<()> {
        let csr = pairs_to_csr(n, &[(near, 0), (mid_near, 1), (far, 2)]);
        let mut grad_fn = |y: ArrayView2<f64>, w: &[f64]| -> Result<Array2<f64>> {
            if y.nrows() != n || y.ncols() != dim || w.len() != 3 {
                return Err("pacmap: embedding shape or weights mismatch".into());
            }
            let (w_near, w_mn, w_fp) = (w[0], w[1], w[2]);
            let y = y.to_owned();
            let ys = y.as_slice().expect("owned array is contiguous");
            let mut grad = vec![0f64; n * dim];
            grad.par_chunks_mut(dim).enumerate().for_each(|(i, g)| {
                let yi = &ys[i * dim..(i + 1) * dim];
                for e in csr.offsets[i] as usize..csr.offsets[i + 1] as usize {
                    let j = csr.nbr[e] as usize;
                    let yj = &ys[j * dim..(j + 1) * dim];
                    let d2: f64 = yi.iter().zip(yj).map(|(a, b)| (a - b) * (a - b)).sum();
                    let coeff = match csr.tag[e] {
                        0 => w_near * 2.0 * 10.0 / ((10.0 + d2) * (10.0 + d2)),
                        1 => w_mn * 2.0 * 10000.0 / ((10000.0 + d2) * (10000.0 + d2)),
                        _ => -(w_fp * 2.0 / ((1.0 + d2) * (1.0 + d2))),
                    };
                    for c in 0..dim {
                        g[c] += coeff * (yi[c] - yj[c]);
                    }
                }
            });
            Array2::from_shape_vec((n, dim), grad).map_err(|e| e.to_string())
        };
        run(&mut grad_fn)
    }

    fn trimap_session(
        &self,
        n: usize,
        dim: usize,
        triplets: &[(u32, u32, u32)],
        weights: &[f64],
        run: &mut dyn FnMut(&mut GradFn<'_>) -> Result<()>,
    ) -> Result<()> {
        if weights.len() != triplets.len() {
            return Err("trimap: one weight per triplet required".into());
        }
        let csr = triplets_to_csr(n, triplets);
        let mut grad_fn = |y: ArrayView2<f64>, s: &[f64]| -> Result<Array2<f64>> {
            if y.nrows() != n || y.ncols() != dim || s.len() != 1 {
                return Err("trimap: embedding shape or scalars mismatch".into());
            }
            let scale = s[0];
            let y = y.to_owned();
            let ys = y.as_slice().expect("owned array is contiguous");
            let mut grad = vec![0f64; n * dim];
            grad.par_chunks_mut(dim).enumerate().for_each(|(p, g)| {
                for e in csr.offsets[p] as usize..csr.offsets[p + 1] as usize {
                    let t = csr.nbr[e] as usize;
                    let (i, j, k) = triplets[t];
                    let (yi, yj, yk) = (
                        &ys[i as usize * dim..(i as usize + 1) * dim],
                        &ys[j as usize * dim..(j as usize + 1) * dim],
                        &ys[k as usize * dim..(k as usize + 1) * dim],
                    );
                    let mut d_ij = 0.0;
                    let mut d_ik = 0.0;
                    for c in 0..dim {
                        d_ij += (yi[c] - yj[c]) * (yi[c] - yj[c]);
                        d_ik += (yi[c] - yk[c]) * (yi[c] - yk[c]);
                    }
                    if d_ij - d_ik + 1.0 <= 0.0 {
                        continue;
                    }
                    let sw = scale * weights[t];
                    match csr.tag[e] {
                        0 => {
                            for c in 0..dim {
                                g[c] += sw * ((yi[c] - yj[c]) - (yi[c] - yk[c]));
                            }
                        }
                        1 => {
                            for c in 0..dim {
                                g[c] -= sw * (yi[c] - yj[c]);
                            }
                        }
                        _ => {
                            for c in 0..dim {
                                g[c] += sw * (yi[c] - yk[c]);
                            }
                        }
                    }
                }
            });
            Array2::from_shape_vec((n, dim), grad).map_err(|e| e.to_string())
        };
        run(&mut grad_fn)
    }

    fn tsne_exact_session(
        &self,
        p: ArrayView2<f64>,
        dim: usize,
        run: &mut dyn FnMut(&mut GradFn<'_>) -> Result<()>,
    ) -> Result<()> {
        let n = p.nrows();
        if p.ncols() != n {
            return Err("tsne: P must be square".into());
        }
        let p = p.to_owned();
        let ps = p.as_slice().expect("owned array is contiguous");
        let mut grad_fn = |y: ArrayView2<f64>, s: &[f64]| -> Result<Array2<f64>> {
            if y.nrows() != n || y.ncols() != dim || s.len() != 1 {
                return Err("tsne: embedding shape or scalars mismatch".into());
            }
            let ex = s[0];
            let y = y.to_owned();
            let ys = y.as_slice().expect("owned array is contiguous");
            let kernel = |i: usize, j: usize| -> f64 {
                let yi = &ys[i * dim..(i + 1) * dim];
                let yj = &ys[j * dim..(j + 1) * dim];
                let d2: f64 = yi.iter().zip(yj).map(|(a, b)| (a - b) * (a - b)).sum();
                1.0 / (1.0 + d2)
            };
            // Z = Σ_{i≠j} k_ij, accumulated per row in f64 as the CPU implementation does.
            let row_sums: Vec<f64> = (0..n)
                .into_par_iter()
                .map(|i| (0..n).filter(|&j| j != i).map(|j| kernel(i, j)).sum())
                .collect();
            let z: f64 = row_sums.iter().sum();
            let inv_z = if z > 0.0 { 1.0 / z } else { 0.0 };
            let mut grad = vec![0f64; n * dim];
            grad.par_chunks_mut(dim).enumerate().for_each(|(i, g)| {
                let yi = &ys[i * dim..(i + 1) * dim];
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let kij = kernel(i, j);
                    let qij = (kij * inv_z).max(1e-12);
                    let mult = 4.0 * (ex * ps[i * n + j] - qij) * kij;
                    let yj = &ys[j * dim..(j + 1) * dim];
                    for c in 0..dim {
                        g[c] += mult * (yi[c] - yj[c]);
                    }
                }
            });
            Array2::from_shape_vec((n, dim), grad).map_err(|e| e.to_string())
        };
        run(&mut grad_fn)
    }
}

/// Centre the rows of an embedding in place (shared by the algorithms after each step).
pub fn center_rows(y: &mut Array2<f64>) {
    let mean = y.mean_axis(Axis(0)).expect("non-empty embedding");
    for mut row in y.rows_mut() {
        row -= &mean;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    fn random_matrix(rng: &mut StdRng, n: usize, d: usize) -> Array2<f32> {
        Array2::from_shape_fn((n, d), |_| rng.sample::<f32, _>(StandardNormal))
    }

    #[test]
    fn knn_matches_naive_sort() {
        let mut rng = StdRng::seed_from_u64(7);
        let data = random_matrix(&mut rng, 200, 16);
        let queries = random_matrix(&mut rng, 20, 16);
        let (idx, dist) = CpuBackend
            .knn_bruteforce(data.view(), queries.view(), 5, Metric::Euclidean)
            .unwrap();
        for q in 0..20 {
            let mut all: Vec<(f32, usize)> = (0..200)
                .map(|i| {
                    let dd: f32 = (0..16)
                        .map(|c| (data[[i, c]] - queries[[q, c]]).powi(2))
                        .sum();
                    (dd.sqrt(), i)
                })
                .collect();
            all.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for j in 0..5 {
                assert_eq!(idx[[q, j]] as usize, all[j].1);
                assert!((dist[[q, j]] - all[j].0).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn sqdist_and_gram_consistent() {
        let mut rng = StdRng::seed_from_u64(3);
        let a = random_matrix(&mut rng, 30, 8);
        let sq = CpuBackend.pairwise_sqdist(a.view(), a.view()).unwrap();
        let g = CpuBackend.gram(a.view()).unwrap();
        for i in 0..30 {
            for j in 0..30 {
                let expect = g[[i, i]] + g[[j, j]] - 2.0 * g[[i, j]];
                assert!(
                    (sq[[i, j]] - expect).abs() < 1e-3,
                    "{} vs {}",
                    sq[[i, j]],
                    expect
                );
            }
        }
    }

    #[test]
    fn pacmap_gradient_is_antisymmetric_for_one_pair() {
        let y = ndarray::arr2(&[[0.0, 0.0], [3.0, 4.0]]);
        let mut got = None;
        CpuBackend
            .pacmap_session(2, 2, &[(0, 1)], &[], &[], &mut |g| {
                got = Some(g(y.view(), &[1.0, 1.0, 1.0])?);
                Ok(())
            })
            .unwrap();
        let g = got.unwrap();
        // d² = 25, coeff = 20 / 35² ; grad_0 = coeff*(y0-y1) = -coeff*(3,4)
        let coeff = 20.0 / (35.0f64 * 35.0);
        assert!((g[[0, 0]] + coeff * 3.0).abs() < 1e-12);
        assert!((g[[0, 1]] + coeff * 4.0).abs() < 1e-12);
        assert!((g[[1, 0]] - coeff * 3.0).abs() < 1e-12);
        assert!((g[[1, 1]] - coeff * 4.0).abs() < 1e-12);
    }

    #[test]
    fn trimap_gradient_sums_to_zero() {
        let y = ndarray::arr2(&[[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]]);
        let mut got = None;
        CpuBackend
            .trimap_session(3, 2, &[(0, 1, 2)], &[2.0], &mut |g| {
                got = Some(g(y.view(), &[0.5])?);
                Ok(())
            })
            .unwrap();
        let g = got.unwrap();
        let col_sums = g.sum_axis(Axis(0));
        assert!(col_sums.iter().all(|v| v.abs() < 1e-12));
        assert!(g.iter().any(|v| v.abs() > 0.0));
    }

    #[test]
    fn tsne_gradient_matches_dense_formula() {
        let n = 6;
        let mut rng = StdRng::seed_from_u64(11);
        let mut p = Array2::from_shape_fn((n, n), |_| rng.random::<f64>());
        for i in 0..n {
            p[[i, i]] = 0.0;
        }
        let y = Array2::from_shape_fn((n, 2), |_| rng.random::<f64>());
        let mut got = None;
        CpuBackend
            .tsne_exact_session(p.view(), 2, &mut |g| {
                got = Some(g(y.view(), &[2.0])?);
                Ok(())
            })
            .unwrap();
        let g = got.unwrap();
        let mut k = Array2::<f64>::zeros((n, n));
        let mut z = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let d2 = (y[[i, 0]] - y[[j, 0]]).powi(2) + (y[[i, 1]] - y[[j, 1]]).powi(2);
                    k[[i, j]] = 1.0 / (1.0 + d2);
                    z += k[[i, j]];
                }
            }
        }
        for i in 0..n {
            let mut gi = [0.0; 2];
            for j in 0..n {
                if i != j {
                    let q = (k[[i, j]] / z).max(1e-12);
                    let mult = 4.0 * (2.0 * p[[i, j]] - q) * k[[i, j]];
                    gi[0] += mult * (y[[i, 0]] - y[[j, 0]]);
                    gi[1] += mult * (y[[i, 1]] - y[[j, 1]]);
                }
            }
            assert!((g[[i, 0]] - gi[0]).abs() < 1e-12);
            assert!((g[[i, 1]] - gi[1]).abs() < 1e-12);
        }
    }
}
