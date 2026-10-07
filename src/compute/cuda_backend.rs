//! CUDA backend (feature `gpu-cuda`, `cudarc` with dynamic loading of `libcuda`).
//!
//! The kernels are the CUDA C twins of the WGSL shaders (`kernels/squeeze.cu`), compiled
//! once to PTX for `compute_75` and embedded; the driver JIT builds them for the actual GPU,
//! so the target machine needs only the NVIDIA driver, not the CUDA toolkit. The reason this
//! backend exists next to wgpu-Vulkan: sandboxed GPU runtimes (Modal's gVisor, most
//! Kubernetes GPU setups) expose CUDA but not the Vulkan/graphics driver stack, so
//! "any GPU that is available" needs a CUDA path for NVIDIA cards in containers.

use super::{
    contiguous_f32, pairs_to_csr, to_f32, to_f64_matrix, triplets_to_csr, Backend, Device, GradFn,
    Metric, Result,
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use ndarray::{Array2, ArrayView2};
use std::sync::Arc;

const PTX: &str = include_str!("kernels/squeeze.ptx");
const DMAX: usize = 16;
const BLOCK: u32 = 128;
const TILE_BUDGET_BYTES: usize = 256 << 20;
const TILE_COLS: usize = 65_536;

fn err<E: std::fmt::Display>(e: E) -> String {
    format!("cuda: {}", e)
}

/// Is a CUDA driver library present? Checked before touching `cudarc`, whose dynamic loader
/// panics when `libcuda` is missing.
fn libcuda_present() -> bool {
    const CANDIDATES: &[&str] = &[
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/lib/libcuda.so.1",
        "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
        "/usr/local/cuda/compat/libcuda.so.1",
        "C:\\Windows\\System32\\nvcuda.dll",
    ];
    if CANDIDATES.iter().any(|p| std::path::Path::new(p).exists()) {
        return true;
    }
    if let Ok(paths) = std::env::var("LD_LIBRARY_PATH") {
        for dir in paths.split(':') {
            if std::path::Path::new(dir).join("libcuda.so.1").exists() {
                return true;
            }
        }
    }
    std::process::Command::new("ldconfig")
        .arg("-p")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("libcuda.so.1"))
        .unwrap_or(false)
}

/// Find the first CUDA device.
pub fn probe() -> Result<Device> {
    if !libcuda_present() {
        return Err("no NVIDIA driver (libcuda.so.1 not found)".into());
    }
    let outcome = std::panic::catch_unwind(|| -> Result<Device> {
        let count = CudaContext::device_count().map_err(err)?;
        if count <= 0 {
            return Err("CUDA driver present but no device".into());
        }
        let ctx = CudaContext::new(0).map_err(err)?;
        let name = ctx.name().map_err(err)?;
        Ok(Device::Cuda { name, ordinal: 0 })
    });
    match outcome {
        Ok(r) => r,
        Err(_) => Err("CUDA driver initialisation panicked".into()),
    }
}

/// The CUDA backend for one device.
pub struct CudaBackend {
    desc: Device,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
}

impl CudaBackend {
    pub fn open(desc: &Device) -> Result<Self> {
        let Device::Cuda { ordinal, .. } = desc else {
            return Err("not a CUDA device".into());
        };
        let ctx = CudaContext::new(*ordinal).map_err(err)?;
        let stream = ctx.default_stream();
        let ptx = match std::env::var("SQUEEZE_CUDA_PTX") {
            Ok(path) => cudarc::nvrtc::Ptx::from_file(path),
            Err(_) => cudarc::nvrtc::Ptx::from_src(PTX),
        };
        let module = ctx.load_module(ptx).map_err(err)?;
        Ok(Self {
            desc: desc.clone(),
            ctx,
            stream,
            module,
        })
    }

    fn func(&self, name: &str) -> Result<CudaFunction> {
        self.module.load_function(name).map_err(err)
    }

    fn upload<T: cudarc::driver::DeviceRepr>(&self, data: &[T]) -> Result<CudaSlice<T>> {
        self.stream.clone_htod(data).map_err(err)
    }

    fn zeros<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>> {
        self.stream.alloc_zeros::<T>(len.max(1)).map_err(err)
    }

    fn download<T: cudarc::driver::DeviceRepr>(&self, buf: &CudaSlice<T>) -> Result<Vec<T>> {
        self.stream.synchronize().map_err(err)?;
        self.stream.clone_dtoh(buf).map_err(err)
    }

    fn write<T: cudarc::driver::DeviceRepr>(
        &self,
        dst: &mut CudaSlice<T>,
        src: &[T],
    ) -> Result<()> {
        self.stream.memcpy_htod(src, dst).map_err(err)
    }

    fn grid_1d(n: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((n as u32).div_ceil(BLOCK).max(1), 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Run the tiled pairwise kernel; `per_tile(tile, a0, a_rows, b0, b_rows)` is called after
    /// each tile completed (the stream is synchronised first).
    #[allow(clippy::too_many_arguments)]
    fn tiles(
        &self,
        a: &[f32],
        n_a: usize,
        b: &[f32],
        n_b: usize,
        d: usize,
        kernel: &str,
        mut per_tile: impl FnMut(&CudaSlice<f32>, usize, usize, usize, usize) -> Result<()>,
    ) -> Result<()> {
        if d == 0 || n_a == 0 || n_b == 0 {
            return Err("empty input".into());
        }
        let a_buf = self.upload(a)?;
        let b_buf = self.upload(b)?;
        let b_cols = n_b.min(TILE_COLS);
        let a_rows_max = n_a
            .min((TILE_BUDGET_BYTES / 4) / b_cols)
            .min(65_535 * 16)
            .max(1);
        let mut tile: CudaSlice<f32> = self.zeros(a_rows_max * b_cols)?;
        let f = self.func(kernel)?;
        let d32 = d as u32;
        let mut a0 = 0;
        while a0 < n_a {
            let a_rows = a_rows_max.min(n_a - a0);
            let mut b0 = 0;
            while b0 < n_b {
                let b_rows = b_cols.min(n_b - b0);
                let (a0u, ar, b0u, br, stride) = (
                    a0 as u32,
                    a_rows as u32,
                    b0 as u32,
                    b_rows as u32,
                    b_rows as u32,
                );
                let cfg = LaunchConfig {
                    grid_dim: (b_rows.div_ceil(16) as u32, a_rows.div_ceil(16) as u32, 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.stream
                        .launch_builder(&f)
                        .arg(&a_buf)
                        .arg(&b_buf)
                        .arg(&mut tile)
                        .arg(&a0u)
                        .arg(&ar)
                        .arg(&b0u)
                        .arg(&br)
                        .arg(&d32)
                        .arg(&stride)
                        .launch(cfg)
                        .map_err(err)?;
                }
                self.stream.synchronize().map_err(err)?;
                per_tile(&tile, a0, a_rows, b0, b_rows)?;
                b0 += b_rows;
            }
            a0 += a_rows;
        }
        Ok(())
    }

    fn pairwise(
        &self,
        a: ArrayView2<f32>,
        b: ArrayView2<f32>,
        kernel: &str,
    ) -> Result<Array2<f32>> {
        if a.ncols() != b.ncols() {
            return Err(format!(
                "feature mismatch: {} vs {} columns",
                a.ncols(),
                b.ncols()
            ));
        }
        let (n_a, n_b, d) = (a.nrows(), b.nrows(), a.ncols());
        let a = contiguous_f32(a);
        let b = contiguous_f32(b);
        let mut out = vec![0f32; n_a * n_b];
        self.tiles(
            &a,
            n_a,
            &b,
            n_b,
            d,
            kernel,
            |tile, a0, a_rows, b0, b_rows| {
                let vals = self.download(tile)?;
                for r in 0..a_rows {
                    out[(a0 + r) * n_b + b0..(a0 + r) * n_b + b0 + b_rows]
                        .copy_from_slice(&vals[r * b_rows..(r + 1) * b_rows]);
                }
                Ok(())
            },
        )?;
        Array2::from_shape_vec((n_a, n_b), out).map_err(|e| e.to_string())
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

impl Backend for CudaBackend {
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
        let (data_v, queries_v, kernel) = match metric {
            Metric::Euclidean => (data_c.to_vec(), queries_c.to_vec(), "dist_euclidean"),
            Metric::Manhattan => (data_c.to_vec(), queries_c.to_vec(), "dist_manhattan"),
            Metric::Cosine => (
                normalize_rows(&data_c, d),
                normalize_rows(&queries_c, d),
                "dist_cosine",
            ),
        };
        let mut best_idx = self.upload(&vec![u32::MAX; m * k])?;
        let mut best_dist = self.upload(&vec![f32::INFINITY; m * k])?;
        let topk = self.func("topk_merge")?;
        let k32 = k as u32;
        self.tiles(
            &queries_v,
            m,
            &data_v,
            n,
            d,
            kernel,
            |tile, q0, q_rows, b0, b_cols| {
                let (q0u, qr, b0u, bc, stride) = (
                    q0 as u32,
                    q_rows as u32,
                    b0 as u32,
                    b_cols as u32,
                    b_cols as u32,
                );
                unsafe {
                    self.stream
                        .launch_builder(&topk)
                        .arg(tile)
                        .arg(&mut best_idx)
                        .arg(&mut best_dist)
                        .arg(&q0u)
                        .arg(&qr)
                        .arg(&b0u)
                        .arg(&bc)
                        .arg(&k32)
                        .arg(&stride)
                        .launch(Self::grid_1d(q_rows))
                        .map_err(err)?;
                }
                self.stream.synchronize().map_err(err)
            },
        )?;
        let idx =
            Array2::from_shape_vec((m, k), self.download(&best_idx)?).map_err(|e| e.to_string())?;
        let dist = Array2::from_shape_vec((m, k), self.download(&best_dist)?)
            .map_err(|e| e.to_string())?;
        Ok((idx, dist))
    }

    fn pairwise_sqdist(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>> {
        self.pairwise(a, b, "dist_sqeuclidean")
    }

    fn gram(&self, a: ArrayView2<f32>) -> Result<Array2<f32>> {
        self.pairwise(a, a, "dist_dot")
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
        if dim == 0 || dim > DMAX {
            return Err(format!("embedding dimension {} not in 1..={}", dim, DMAX));
        }
        let csr = pairs_to_csr(n, &[(near, 0), (mid_near, 1), (far, 2)]);
        let nbr_off = csr.offsets.len() as u32;
        let tag_off = nbr_off + csr.nbr.len() as u32;
        let mut packed = csr.offsets;
        packed.extend_from_slice(&csr.nbr);
        packed.extend_from_slice(&csr.tag);
        let csr_buf = self.upload(&packed)?;
        let mut y_buf: CudaSlice<f32> = self.zeros(n * dim)?;
        let mut grad_buf: CudaSlice<f32> = self.zeros(n * dim)?;
        let f = self.func("pacmap_grad")?;
        let (n32, dim32) = (n as u32, dim as u32);
        let mut grad_fn = |y: ArrayView2<f64>, w: &[f64]| -> Result<Array2<f64>> {
            if y.nrows() != n || y.ncols() != dim || w.len() != 3 {
                return Err("pacmap: embedding shape or weights mismatch".into());
            }
            self.write(&mut y_buf, &to_f32(y))?;
            let (w0, w1, w2) = (w[0] as f32, w[1] as f32, w[2] as f32);
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&y_buf)
                    .arg(&csr_buf)
                    .arg(&mut grad_buf)
                    .arg(&n32)
                    .arg(&dim32)
                    .arg(&w0)
                    .arg(&w1)
                    .arg(&w2)
                    .arg(&nbr_off)
                    .arg(&tag_off)
                    .launch(Self::grid_1d(n))
                    .map_err(err)?;
            }
            Ok(to_f64_matrix(&self.download(&grad_buf)?, n, dim))
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
        if dim == 0 || dim > DMAX {
            return Err(format!("embedding dimension {} not in 1..={}", dim, DMAX));
        }
        if weights.len() != triplets.len() {
            return Err("trimap: one weight per triplet required".into());
        }
        let csr = triplets_to_csr(n, triplets);
        let nbr_off = csr.offsets.len() as u32;
        let tag_off = nbr_off + csr.nbr.len() as u32;
        let trip_off = tag_off + csr.tag.len() as u32;
        let w_off = trip_off + 3 * triplets.len() as u32;
        let mut packed = csr.offsets;
        packed.extend_from_slice(&csr.nbr);
        packed.extend_from_slice(&csr.tag);
        packed.extend(triplets.iter().flat_map(|&(i, j, k)| [i, j, k]));
        packed.extend(weights.iter().map(|&w| (w as f32).to_bits()));
        let packed_buf = self.upload(&packed)?;
        let mut y_buf: CudaSlice<f32> = self.zeros(n * dim)?;
        let mut grad_buf: CudaSlice<f32> = self.zeros(n * dim)?;
        let f = self.func("trimap_grad")?;
        let (n32, dim32) = (n as u32, dim as u32);
        let mut grad_fn = |y: ArrayView2<f64>, s: &[f64]| -> Result<Array2<f64>> {
            if y.nrows() != n || y.ncols() != dim || s.len() != 1 {
                return Err("trimap: embedding shape or scalars mismatch".into());
            }
            self.write(&mut y_buf, &to_f32(y))?;
            let scale = s[0] as f32;
            unsafe {
                self.stream
                    .launch_builder(&f)
                    .arg(&y_buf)
                    .arg(&packed_buf)
                    .arg(&mut grad_buf)
                    .arg(&n32)
                    .arg(&dim32)
                    .arg(&scale)
                    .arg(&nbr_off)
                    .arg(&tag_off)
                    .arg(&trip_off)
                    .arg(&w_off)
                    .launch(Self::grid_1d(n))
                    .map_err(err)?;
            }
            Ok(to_f64_matrix(&self.download(&grad_buf)?, n, dim))
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
        if dim == 0 || dim > DMAX {
            return Err(format!("embedding dimension {} not in 1..={}", dim, DMAX));
        }
        let p_buf = self.upload(&to_f32(p))?;
        let mut y_buf: CudaSlice<f32> = self.zeros(n * dim)?;
        // scratch = rowsum[n] | z[1] | grad[n*dim]
        let mut scratch: CudaSlice<f32> = self.zeros(n + 1 + n * dim)?;
        let rowsum = self.func("tsne_rowsum")?;
        let reduce = self.func("tsne_reduce")?;
        let grad = self.func("tsne_grad")?;
        let (n32, dim32) = (n as u32, dim as u32);
        let reduce_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut grad_fn = |y: ArrayView2<f64>, s: &[f64]| -> Result<Array2<f64>> {
            if y.nrows() != n || y.ncols() != dim || s.len() != 1 {
                return Err("tsne: embedding shape or scalars mismatch".into());
            }
            self.write(&mut y_buf, &to_f32(y))?;
            let ex = s[0] as f32;
            unsafe {
                self.stream
                    .launch_builder(&rowsum)
                    .arg(&y_buf)
                    .arg(&mut scratch)
                    .arg(&n32)
                    .arg(&dim32)
                    .launch(Self::grid_1d(n))
                    .map_err(err)?;
                self.stream
                    .launch_builder(&reduce)
                    .arg(&mut scratch)
                    .arg(&n32)
                    .launch(reduce_cfg)
                    .map_err(err)?;
                self.stream
                    .launch_builder(&grad)
                    .arg(&y_buf)
                    .arg(&p_buf)
                    .arg(&mut scratch)
                    .arg(&n32)
                    .arg(&dim32)
                    .arg(&ex)
                    .launch(Self::grid_1d(n))
                    .map_err(err)?;
            }
            let all = self.download(&scratch)?;
            Ok(to_f64_matrix(&all[n + 1..n + 1 + n * dim], n, dim))
        };
        run(&mut grad_fn)
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        let _ = self.ctx.synchronize();
    }
}

#[cfg(test)]
mod tests {
    //! Run on an NVIDIA machine with `SQUEEZE_DEVICE=cuda`; skipped elsewhere.
    use super::super::cpu::CpuBackend;
    use super::super::Selection;
    use super::*;
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    fn backend() -> Option<CudaBackend> {
        if std::env::var(super::super::DEVICE_ENV).as_deref() != Ok("cuda") {
            eprintln!("skipping cuda tests: set SQUEEZE_DEVICE=cuda to run them");
            return None;
        }
        let sel = Selection::run(Some("cuda"));
        match &sel.chosen {
            Device::Cuda { .. } => {
                eprintln!("cuda test device: {}", sel.chosen);
                CudaBackend::open(&sel.chosen).ok()
            }
            _ => {
                eprintln!("skipping cuda tests\n{}", sel);
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
            eprintln!(
                "cuda knn {} 2000x64, 300 queries: {:?}",
                metric,
                t.elapsed()
            );
            let (ci, cd) = CpuBackend
                .knn_bruteforce(data.view(), queries.view(), 15, metric)
                .unwrap();
            assert!(recall(&gi, &ci) >= 0.999);
            assert!(max_rel_err_f32(&gd, &cd) <= 1e-4);
        }
    }

    #[test]
    fn knn_tiles_over_large_inputs() {
        let Some(gpu) = backend() else { return };
        let mut rng = StdRng::seed_from_u64(2);
        let data = random_matrix(&mut rng, 70_000, 16);
        let queries = random_matrix(&mut rng, 1_100, 16);
        let (gi, gd) = gpu
            .knn_bruteforce(data.view(), queries.view(), 8, Metric::Euclidean)
            .unwrap();
        let (ci, cd) = CpuBackend
            .knn_bruteforce(data.view(), queries.view(), 8, Metric::Euclidean)
            .unwrap();
        assert!(recall(&gi, &ci) >= 0.999);
        assert!(max_rel_err_f32(&gd, &cd) <= 1e-4);
    }

    #[test]
    fn sqdist_and_gram_match_cpu() {
        let Some(gpu) = backend() else { return };
        let mut rng = StdRng::seed_from_u64(3);
        let a = random_matrix(&mut rng, 1797, 64);
        let b = random_matrix(&mut rng, 300, 64);
        assert!(
            max_rel_err_f32(
                &gpu.pairwise_sqdist(a.view(), b.view()).unwrap(),
                &CpuBackend.pairwise_sqdist(a.view(), b.view()).unwrap()
            ) <= 1e-4
        );
        assert!(
            max_rel_err_f32(
                &gpu.gram(a.view()).unwrap(),
                &CpuBackend.gram(a.view()).unwrap()
            ) <= 1e-4
        );
    }

    #[test]
    fn gradient_sessions_match_cpu() {
        let Some(gpu) = backend() else { return };
        let mut rng = StdRng::seed_from_u64(4);
        let n = 500;
        let near: Vec<(u32, u32)> = (0..n as u32).map(|i| (i, (i + 1) % n as u32)).collect();
        let mid: Vec<(u32, u32)> = (0..n as u32).map(|i| (i, (i + 7) % n as u32)).collect();
        let far: Vec<(u32, u32)> = (0..n as u32).map(|i| (i, (i + 97) % n as u32)).collect();
        let y0 = random_matrix_f64(&mut rng, n, 2) * 10.0;
        let run_pacmap = |b: &dyn Backend| -> Array2<f64> {
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
        assert!(max_rel_err_f64(&run_pacmap(&gpu), &run_pacmap(&CpuBackend)) <= 1e-4);

        let triplets: Vec<(u32, u32, u32)> = (0..n as u32)
            .map(|i| (i, (i + 1) % n as u32, (i + 50) % n as u32))
            .collect();
        let weights: Vec<f64> = triplets.iter().map(|t| 1.0 + (t.0 % 5) as f64).collect();
        let y1 = random_matrix_f64(&mut rng, n, 2);
        let run_trimap = |b: &dyn Backend| -> Array2<f64> {
            let mut y = y1.clone();
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
        assert!(max_rel_err_f64(&run_trimap(&gpu), &run_trimap(&CpuBackend)) <= 1e-4);

        let mut p = Array2::from_shape_fn((n, n), |_| rng.random::<f64>());
        for i in 0..n {
            p[[i, i]] = 0.0;
        }
        let total: f64 = p.sum();
        p /= total;
        let y2 = random_matrix_f64(&mut rng, n, 2) * 1e-2;
        let run_tsne = |b: &dyn Backend| -> Array2<f64> {
            let mut y = y2.clone();
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
        assert!(max_rel_err_f64(&run_tsne(&gpu), &run_tsne(&CpuBackend)) <= 1e-3);
    }
}
