//! MLX backend for Apple Silicon (`mlx-rs`, feature `gpu-mlx`, macOS aarch64 only).
//!
//! Kernels are expressed as MLX array programs in `f32` (MLX has no `f64` on the GPU):
//! pairwise distances through the Gram expansion `|a|² + |b|² − 2 a·b`, top-k through
//! `argpartition`, and the gradient sessions through gather / scatter-add. MLX evaluates
//! lazily; every method forces evaluation before copying results to the host.

use super::{contiguous_f32, to_f32, to_f64_matrix, Backend, Device, GradFn, Metric, Result};
use mlx_rs::ops::indexing::{scatter_add_single, take_along_axis, IndexOp};
use mlx_rs::{ops, Array, Dtype};
use ndarray::{Array2, ArrayView2};

/// Floats per intermediate chunk (256 MiB of f32) used to bound temporary matrices.
const CHUNK_FLOATS: usize = 64 << 20;

fn err<E: std::fmt::Display>(e: E) -> String {
    format!("mlx: {}", e)
}

fn chip_name() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "Apple Silicon".to_string())
}

/// Try to run a tiny program on the MLX GPU device.
pub fn probe() -> Result<Device> {
    let outcome = std::panic::catch_unwind(|| -> Result<()> {
        mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let s = a.sum(None).map_err(err)?;
        let v: f32 = s.item_exact::<f32>();
        if (v - 6.0).abs() > 1e-6 {
            return Err(format!("MLX GPU self-test returned {} instead of 6", v));
        }
        Ok(())
    });
    match outcome {
        Ok(Ok(())) => Ok(Device::Mlx { name: chip_name() }),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("MLX GPU initialisation panicked".into()),
    }
}

/// The MLX backend. Stateless beyond the device description; MLX owns the GPU context.
pub struct MlxBackend {
    desc: Device,
}

impl MlxBackend {
    pub fn open(desc: &Device) -> Result<Self> {
        if !matches!(desc, Device::Mlx { .. }) {
            return Err("not an MLX device".into());
        }
        mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
        Ok(Self { desc: desc.clone() })
    }
}

fn matrix(data: &[f32], n: usize, d: usize) -> Array {
    Array::from_slice(data, &[n as i32, d as i32])
}

fn indices(idx: &[u32]) -> Array {
    Array::from_slice(idx, &[idx.len() as i32])
}

fn host_f32(a: &Array) -> Result<Vec<f32>> {
    let a = a.as_dtype(Dtype::Float32).map_err(err)?;
    a.eval().map_err(err)?;
    Ok(a.as_slice::<f32>().to_vec())
}

fn host_u32(a: &Array) -> Result<Vec<u32>> {
    let a = a.as_dtype(Dtype::Uint32).map_err(err)?;
    a.eval().map_err(err)?;
    Ok(a.as_slice::<u32>().to_vec())
}

fn rows(a: &Array, r0: usize, r1: usize) -> Array {
    a.index((r0 as i32..r1 as i32, ..))
}

/// `|q_i − x_j|²` for a block of query rows via the Gram expansion, clamped at zero.
fn sqdist_block(q: &Array, q_norm: &Array, xt: &Array, x_norm: &Array) -> Result<Array> {
    let dot = ops::matmul(q, xt).map_err(err)?;
    let two = Array::from_f32(2.0);
    let d2 = ops::subtract(
        &ops::add(q_norm, x_norm).map_err(err)?,
        &ops::multiply(&dot, &two).map_err(err)?,
    )
    .map_err(err)?;
    ops::maximum(&d2, Array::from_f32(0.0)).map_err(err)
}

fn row_norms(a: &Array, keep_as_column: bool) -> Result<Array> {
    let n = a.square().map_err(err)?.sum_axis(1, None).map_err(err)?;
    if keep_as_column {
        n.expand_dims(1).map_err(err)
    } else {
        Ok(n)
    }
}

fn normalize_rows(x: &[f32], d: usize) -> Vec<f32> {
    let mut out = x.to_vec();
    for row in out.chunks_mut(d) {
        let norm = row
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            .sqrt();
        if norm < 1e-10 {
            row.iter_mut().for_each(|v| *v = 0.0);
        } else {
            row.iter_mut().for_each(|v| *v = (*v as f64 / norm) as f32);
        }
    }
    out
}

/// Sorted top-k (ascending) of each row of `d`: `(indices [rows, k], values [rows, k])`.
fn topk_rows(d: &Array, k: usize) -> Result<(Array, Array)> {
    let part = ops::argpartition_axis(d, (k - 1) as i32, 1).map_err(err)?;
    let idx = part.index((.., ..k as i32));
    let vals = take_along_axis(d, &idx, 1).map_err(err)?;
    let order = ops::argsort_axis(&vals, 1).map_err(err)?;
    let idx_sorted = take_along_axis(&idx, &order, 1).map_err(err)?;
    let vals_sorted = take_along_axis(&vals, &order, 1).map_err(err)?;
    Ok((idx_sorted, vals_sorted))
}

/// `grad[idx[p], :] += contrib[p, :]` for `contrib [P, dim]`.
fn scatter_rows(grad: &Array, idx: &Array, contrib: &Array, dim: usize) -> Result<Array> {
    let p = contrib.shape()[0];
    let updates = contrib.reshape(&[p, 1, dim as i32]).map_err(err)?;
    scatter_add_single(grad, idx, &updates, 0).map_err(err)
}

impl Backend for MlxBackend {
    fn device(&self) -> &Device {
        &self.desc
    }

    fn knn_bruteforce(
        &self,
        data: ArrayView2<f32>,
        queries: ArrayView2<f32>,
        k: usize,
        metric: Metric,
    ) -> Result<(Array2<u32>, Array2<f32>)> {
        if data.ncols() != queries.ncols() {
            return Err(format!(
                "feature mismatch: {} vs {} columns",
                data.ncols(),
                queries.ncols()
            ));
        }
        let (n, m, d) = (data.nrows(), queries.nrows(), data.ncols());
        if k == 0 || k > n {
            return Err(format!("k={} must be in 1..={}", k, n));
        }
        let data_c = contiguous_f32(data);
        let queries_c = contiguous_f32(queries);
        let (x, q) = match metric {
            Metric::Cosine => (
                matrix(&normalize_rows(&data_c, d), n, d),
                matrix(&normalize_rows(&queries_c, d), m, d),
            ),
            _ => (matrix(&data_c, n, d), matrix(&queries_c, m, d)),
        };
        let xt = x.t();
        let x_norm = row_norms(&x, false)?;
        let mut all_idx = Vec::with_capacity(m * k);
        let mut all_dist = Vec::with_capacity(m * k);
        let chunk = match metric {
            Metric::Manhattan => (CHUNK_FLOATS / (n * d).max(1)).clamp(1, m),
            _ => (CHUNK_FLOATS / n.max(1)).clamp(1, m),
        };
        let mut r0 = 0;
        while r0 < m {
            let r1 = (r0 + chunk).min(m);
            let qb = rows(&q, r0, r1);
            let dist = match metric {
                Metric::Euclidean => {
                    let qn = row_norms(&qb, true)?;
                    sqdist_block(&qb, &qn, &xt, &x_norm)?.sqrt().map_err(err)?
                }
                Metric::Cosine => {
                    let dot = ops::matmul(&qb, &xt).map_err(err)?;
                    ops::subtract(Array::from_f32(1.0), &dot).map_err(err)?
                }
                Metric::Manhattan => {
                    // |q[:,None,:] - x[None,:,:]| summed over features
                    let qe = qb.expand_dims(1).map_err(err)?;
                    let xe = x.expand_dims(0).map_err(err)?;
                    ops::subtract(&qe, &xe)
                        .map_err(err)?
                        .abs()
                        .map_err(err)?
                        .sum_axis(2, None)
                        .map_err(err)?
                }
            };
            let (idx, vals) = topk_rows(&dist, k)?;
            all_idx.extend(host_u32(&idx)?);
            all_dist.extend(host_f32(&vals)?);
            r0 = r1;
        }
        let idx = Array2::from_shape_vec((m, k), all_idx).map_err(|e| e.to_string())?;
        let dist = Array2::from_shape_vec((m, k), all_dist).map_err(|e| e.to_string())?;
        Ok((idx, dist))
    }

    fn pairwise_sqdist(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>> {
        if a.ncols() != b.ncols() {
            return Err(format!(
                "feature mismatch: {} vs {} columns",
                a.ncols(),
                b.ncols()
            ));
        }
        let (n, m, d) = (a.nrows(), b.nrows(), a.ncols());
        let a_c = contiguous_f32(a);
        let b_c = contiguous_f32(b);
        let xa = matrix(&a_c, n, d);
        let xb = matrix(&b_c, m, d);
        let bt = xb.t();
        let b_norm = row_norms(&xb, false)?;
        let mut out = Vec::with_capacity(n * m);
        let chunk = (CHUNK_FLOATS / m.max(1)).clamp(1, n);
        let mut r0 = 0;
        while r0 < n {
            let r1 = (r0 + chunk).min(n);
            let ab = rows(&xa, r0, r1);
            let an = row_norms(&ab, true)?;
            out.extend(host_f32(&sqdist_block(&ab, &an, &bt, &b_norm)?)?);
            r0 = r1;
        }
        Array2::from_shape_vec((n, m), out).map_err(|e| e.to_string())
    }

    fn gram(&self, a: ArrayView2<f32>) -> Result<Array2<f32>> {
        let (n, d) = (a.nrows(), a.ncols());
        let a_c = contiguous_f32(a);
        let xa = matrix(&a_c, n, d);
        let at = xa.t();
        let mut out = Vec::with_capacity(n * n);
        let chunk = (CHUNK_FLOATS / n.max(1)).clamp(1, n);
        let mut r0 = 0;
        while r0 < n {
            let r1 = (r0 + chunk).min(n);
            let block = ops::matmul(rows(&xa, r0, r1), &at).map_err(err)?;
            out.extend(host_f32(&block)?);
            r0 = r1;
        }
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
        // (i indices, j indices, kind) per group; kind 0 near, 1 mid-near, 2 far
        let groups: Vec<(Array, Array, u8)> = [(near, 0u8), (mid_near, 1), (far, 2)]
            .into_iter()
            .filter(|(pairs, _)| !pairs.is_empty())
            .map(|(pairs, kind)| {
                let i: Vec<u32> = pairs.iter().map(|p| p.0).collect();
                let j: Vec<u32> = pairs.iter().map(|p| p.1).collect();
                (indices(&i), indices(&j), kind)
            })
            .collect();
        let mut grad_fn = |y: ArrayView2<f64>, w: &[f64]| -> Result<Array2<f64>> {
            if y.nrows() != n || y.ncols() != dim || w.len() != 3 {
                return Err("pacmap: embedding shape or weights mismatch".into());
            }
            let ya = matrix(&to_f32(y), n, dim);
            let mut grad = ops::zeros::<f32>(&[n as i32, dim as i32]).map_err(err)?;
            for (ii, jj, kind) in &groups {
                let yi = ya.take_axis(ii, 0).map_err(err)?;
                let yj = ya.take_axis(jj, 0).map_err(err)?;
                let diff = ops::subtract(&yi, &yj).map_err(err)?;
                let d2 = diff.square().map_err(err)?.sum_axis(1, true).map_err(err)?;
                let (weight, c, scale, sign) = match kind {
                    0 => (w[0] as f32, 10.0f32, 20.0f32, 1.0f32),
                    1 => (w[1] as f32, 10000.0, 20000.0, 1.0),
                    _ => (w[2] as f32, 1.0, 2.0, -1.0),
                };
                // coeff = sign · weight · scale / (c + d²)²
                let s = ops::add(&d2, Array::from_f32(c)).map_err(err)?;
                let denom = s.square().map_err(err)?;
                let coeff =
                    ops::divide(Array::from_f32(sign * weight * scale), &denom).map_err(err)?;
                let contrib = ops::multiply(&coeff, &diff).map_err(err)?;
                grad = scatter_rows(&grad, ii, &contrib, dim)?;
                grad = scatter_rows(&grad, jj, &contrib.negative().map_err(err)?, dim)?;
            }
            Ok(to_f64_matrix(&host_f32(&grad)?, n, dim))
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
        if triplets.is_empty() {
            return Err("trimap: no triplets".into());
        }
        let ii = indices(&triplets.iter().map(|t| t.0).collect::<Vec<_>>());
        let jj = indices(&triplets.iter().map(|t| t.1).collect::<Vec<_>>());
        let kk = indices(&triplets.iter().map(|t| t.2).collect::<Vec<_>>());
        let w32: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
        let w = Array::from_slice(&w32, &[w32.len() as i32, 1]);
        let mut grad_fn = |y: ArrayView2<f64>, s: &[f64]| -> Result<Array2<f64>> {
            if y.nrows() != n || y.ncols() != dim || s.len() != 1 {
                return Err("trimap: embedding shape or scalars mismatch".into());
            }
            let ya = matrix(&to_f32(y), n, dim);
            let yi = ya.take_axis(&ii, 0).map_err(err)?;
            let yj = ya.take_axis(&jj, 0).map_err(err)?;
            let yk = ya.take_axis(&kk, 0).map_err(err)?;
            let dij = ops::subtract(&yi, &yj).map_err(err)?;
            let dik = ops::subtract(&yi, &yk).map_err(err)?;
            let d_ij = dij.square().map_err(err)?.sum_axis(1, true).map_err(err)?;
            let d_ik = dik.square().map_err(err)?.sum_axis(1, true).map_err(err)?;
            let loss = ops::add(
                &ops::subtract(&d_ij, &d_ik).map_err(err)?,
                Array::from_f32(1.0),
            )
            .map_err(err)?;
            let active = ops::gt(&loss, Array::from_f32(0.0))
                .map_err(err)?
                .as_dtype(Dtype::Float32)
                .map_err(err)?;
            let sw = ops::multiply(
                &ops::multiply(&w, &active).map_err(err)?,
                Array::from_f32(s[0] as f32),
            )
            .map_err(err)?;
            let g_i = ops::multiply(&sw, &ops::subtract(&dij, &dik).map_err(err)?).map_err(err)?;
            let g_j = ops::multiply(&sw, &dij)
                .map_err(err)?
                .negative()
                .map_err(err)?;
            let g_k = ops::multiply(&sw, &dik).map_err(err)?;
            let mut grad = ops::zeros::<f32>(&[n as i32, dim as i32]).map_err(err)?;
            grad = scatter_rows(&grad, &ii, &g_i, dim)?;
            grad = scatter_rows(&grad, &jj, &g_j, dim)?;
            grad = scatter_rows(&grad, &kk, &g_k, dim)?;
            Ok(to_f64_matrix(&host_f32(&grad)?, n, dim))
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
        let pa = matrix(&to_f32(p), n, n);
        let eye = ops::eye::<f32>(n as i32, None, None).map_err(err)?;
        let offdiag = ops::subtract(Array::from_f32(1.0), &eye).map_err(err)?;
        let mut grad_fn = |y: ArrayView2<f64>, s: &[f64]| -> Result<Array2<f64>> {
            if y.nrows() != n || y.ncols() != dim || s.len() != 1 {
                return Err("tsne: embedding shape or scalars mismatch".into());
            }
            let ya = matrix(&to_f32(y), n, dim);
            let yn_col = row_norms(&ya, true)?;
            let yn_row = row_norms(&ya, false)?;
            let d2 = sqdist_block(&ya, &yn_col, &ya.t(), &yn_row)?;
            // k_ij = 1/(1+d²) with a zero diagonal
            let kmat = ops::multiply(
                &ops::divide(
                    Array::from_f32(1.0),
                    &ops::add(&d2, Array::from_f32(1.0)).map_err(err)?,
                )
                .map_err(err)?,
                &offdiag,
            )
            .map_err(err)?;
            let z = kmat.sum(None).map_err(err)?;
            let q = ops::maximum(
                &ops::divide(&kmat, &z).map_err(err)?,
                Array::from_f32(1e-12),
            )
            .map_err(err)?;
            let pq = ops::subtract(
                &ops::multiply(&pa, Array::from_f32(s[0] as f32)).map_err(err)?,
                &q,
            )
            .map_err(err)?;
            let mult = ops::multiply(
                &ops::multiply(&pq, &kmat).map_err(err)?,
                Array::from_f32(4.0),
            )
            .map_err(err)?;
            // grad_i = Σ_j mult_ij (y_i − y_j) = rowsum(mult)_i · y_i − (mult · y)_i
            let rowsum = mult.sum_axis(1, true).map_err(err)?;
            let grad = ops::subtract(
                &ops::multiply(&rowsum, &ya).map_err(err)?,
                &ops::matmul(&mult, &ya).map_err(err)?,
            )
            .map_err(err)?;
            Ok(to_f64_matrix(&host_f32(&grad)?, n, dim))
        };
        run(&mut grad_fn)
    }
}

#[cfg(test)]
mod tests {
    //! Run on Apple Silicon with `SQUEEZE_DEVICE=mlx`; they print device name and timings so
    //! the CI log doubles as the benchmark record for the MLX path.
    use super::super::cpu::CpuBackend;
    use super::super::Selection;
    use super::*;
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    fn backend() -> Option<MlxBackend> {
        if std::env::var(super::super::DEVICE_ENV).as_deref() != Ok("mlx") {
            eprintln!("skipping mlx tests: set SQUEEZE_DEVICE=mlx to run them");
            return None;
        }
        let sel = Selection::run(Some("mlx"));
        match &sel.chosen {
            Device::Mlx { .. } => {
                eprintln!("mlx test device: {}", sel.chosen);
                MlxBackend::open(&sel.chosen).ok()
            }
            _ => {
                eprintln!("skipping mlx tests\n{}", sel);
                None
            }
        }
    }

    fn random_matrix(rng: &mut StdRng, n: usize, d: usize) -> Array2<f32> {
        Array2::from_shape_fn((n, d), |_| rng.sample::<f32, _>(StandardNormal))
    }

    fn random_matrix_f64(rng: &mut StdRng, n: usize, d: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, d), |_| rng.sample::<f64, _>(StandardNormal))
    }

    fn max_rel_err_f32(got: &Array2<f32>, want: &Array2<f32>) -> f32 {
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        got.iter()
            .zip(want.iter())
            .map(|(g, w)| (g - w).abs() / scale)
            .fold(0f32, f32::max)
    }

    fn max_rel_err_f64(got: &Array2<f64>, want: &Array2<f64>) -> f64 {
        let scale = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-9);
        got.iter()
            .zip(want.iter())
            .map(|(g, w)| (g - w).abs() / scale)
            .fold(0f64, f64::max)
    }

    fn recall(got: &Array2<u32>, want: &Array2<u32>) -> f64 {
        let mut hit = 0usize;
        for (g, w) in got.rows().into_iter().zip(want.rows()) {
            let ws: std::collections::HashSet<u32> = w.iter().copied().collect();
            hit += g.iter().filter(|i| ws.contains(i)).count();
        }
        hit as f64 / (got.nrows() * got.ncols()) as f64
    }

    #[test]
    fn knn_matches_cpu_for_every_metric() {
        let Some(gpu) = backend() else { return };
        let mut rng = StdRng::seed_from_u64(1);
        let data = random_matrix(&mut rng, 2000, 64);
        let queries = random_matrix(&mut rng, 300, 64);
        for metric in [Metric::Euclidean, Metric::Cosine, Metric::Manhattan] {
            let t = std::time::Instant::now();
            let (gi, gd) = gpu
                .knn_bruteforce(data.view(), queries.view(), 15, metric)
                .unwrap();
            let gpu_t = t.elapsed();
            let t = std::time::Instant::now();
            let (ci, cd) = CpuBackend
                .knn_bruteforce(data.view(), queries.view(), 15, metric)
                .unwrap();
            let cpu_t = t.elapsed();
            let r = recall(&gi, &ci);
            let e = max_rel_err_f32(&gd, &cd);
            eprintln!(
                "mlx knn {} 2000x64, 300 queries: mlx {:?} cpu {:?} recall {:.4} max rel err {:.2e}",
                metric, gpu_t, cpu_t, r, e
            );
            assert!(r >= 0.999, "{} recall {}", metric, r);
            assert!(e <= 1e-4, "{} rel err {}", metric, e);
        }
    }

    #[test]
    fn sqdist_and_gram_match_cpu() {
        let Some(gpu) = backend() else { return };
        let mut rng = StdRng::seed_from_u64(3);
        let a = random_matrix(&mut rng, 1797, 64);
        let b = random_matrix(&mut rng, 300, 64);
        let t = std::time::Instant::now();
        let g_sq = gpu.pairwise_sqdist(a.view(), b.view()).unwrap();
        eprintln!("mlx sqdist 1797x300x64: {:?}", t.elapsed());
        let c_sq = CpuBackend.pairwise_sqdist(a.view(), b.view()).unwrap();
        assert!(max_rel_err_f32(&g_sq, &c_sq) <= 1e-4);
        let g_gram = gpu.gram(a.view()).unwrap();
        let c_gram = CpuBackend.gram(a.view()).unwrap();
        assert!(max_rel_err_f32(&g_gram, &c_gram) <= 1e-4);
    }

    #[test]
    fn pacmap_steps_match_cpu() {
        let Some(gpu) = backend() else { return };
        let mut rng = StdRng::seed_from_u64(4);
        let n = 500;
        let near: Vec<(u32, u32)> = (0..n as u32).map(|i| (i, (i + 1) % n as u32)).collect();
        let mid: Vec<(u32, u32)> = (0..n as u32).map(|i| (i, (i + 7) % n as u32)).collect();
        let far: Vec<(u32, u32)> = (0..n as u32).map(|i| (i, (i + 97) % n as u32)).collect();
        let y0 = random_matrix_f64(&mut rng, n, 2) * 10.0;
        let run = |b: &dyn Backend| -> Array2<f64> {
            let mut y = y0.clone();
            b.pacmap_session(n, 2, &near, &mid, &far, &mut |g| {
                for _ in 0..10 {
                    let grad = g(y.view(), &[2.0, 3.0, 1.0])?;
                    y = &y - &(grad * 0.5);
                }
                Ok(())
            })
            .unwrap();
            y
        };
        let t = std::time::Instant::now();
        let yg = run(&gpu);
        eprintln!("mlx pacmap 10 steps n=500: {:?}", t.elapsed());
        let yc = run(&CpuBackend);
        assert!(max_rel_err_f64(&yg, &yc) <= 1e-4);
    }

    #[test]
    fn trimap_steps_match_cpu() {
        let Some(gpu) = backend() else { return };
        let mut rng = StdRng::seed_from_u64(5);
        let n = 400;
        let triplets: Vec<(u32, u32, u32)> = (0..n as u32)
            .flat_map(|i| {
                [
                    (i, (i + 1) % n as u32, (i + 50) % n as u32),
                    (i, (i + 2) % n as u32, (i + 80) % n as u32),
                ]
            })
            .collect();
        let weights: Vec<f64> = triplets.iter().map(|t| 1.0 + (t.0 % 5) as f64).collect();
        let y0 = random_matrix_f64(&mut rng, n, 2);
        let run = |b: &dyn Backend| -> Array2<f64> {
            let mut y = y0.clone();
            b.trimap_session(n, 2, &triplets, &weights, &mut |g| {
                for _ in 0..10 {
                    let grad = g(y.view(), &[2.0 / triplets.len() as f64])?;
                    y = &y - &(grad * 10.0);
                }
                Ok(())
            })
            .unwrap();
            y
        };
        let t = std::time::Instant::now();
        let yg = run(&gpu);
        eprintln!("mlx trimap 10 steps n=400: {:?}", t.elapsed());
        let yc = run(&CpuBackend);
        assert!(max_rel_err_f64(&yg, &yc) <= 1e-4);
    }

    #[test]
    fn tsne_steps_match_cpu() {
        let Some(gpu) = backend() else { return };
        let mut rng = StdRng::seed_from_u64(6);
        let n = 500;
        let mut p = Array2::from_shape_fn((n, n), |_| rng.random::<f64>());
        for i in 0..n {
            p[[i, i]] = 0.0;
        }
        let total: f64 = p.sum();
        p /= total;
        let y0 = random_matrix_f64(&mut rng, n, 2) * 1e-2;
        let run = |b: &dyn Backend| -> Array2<f64> {
            let mut y = y0.clone();
            b.tsne_exact_session(p.view(), 2, &mut |g| {
                for _ in 0..10 {
                    let grad = g(y.view(), &[12.0])?;
                    y = &y - &(grad * 50.0);
                }
                Ok(())
            })
            .unwrap();
            y
        };
        let t = std::time::Instant::now();
        let yg = run(&gpu);
        eprintln!("mlx tsne 10 steps n=500: {:?}", t.elapsed());
        let yc = run(&CpuBackend);
        assert!(max_rel_err_f64(&yg, &yc) <= 1e-3);
    }
}
