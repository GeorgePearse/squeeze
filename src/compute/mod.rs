//! Compute backends: run the heavy kernels on whatever GPU is present, fall back to CPU.
//!
//! This module has no PyO3 types. It exposes four public types:
//!
//! - [`Device`]: what to run on (`Cpu`, `Wgpu`, `Mlx`, `Cuda`), with [`Device::auto`] runtime
//!   probing and [`Device::probe`] producing a [`Selection`] report.
//! - [`Selection`]: what was probed, what was chosen and why the other candidates were
//!   rejected. Printed by `squeeze.devices()`.
//! - [`Metric`]: the distance metrics the GPU kNN kernel supports.
//! - [`Backend`]: the kernel set every backend implements over `ndarray` views.
//!
//! GPU kernels work in `f32`; the session methods take and return `f64` so the algorithms
//! keep their `f64` state and the conversion happens once per call at the boundary.
//!
//! Device selection order for `Device::auto()`:
//! `SQUEEZE_DEVICE` env override -> MLX (macOS aarch64, feature `gpu-mlx`) -> CUDA
//! (feature `gpu-cuda`) -> wgpu discrete GPU -> wgpu integrated GPU -> CPU.
//! Software wgpu adapters (lavapipe, SwiftShader) are only used when `SQUEEZE_DEVICE=wgpu`
//! is set explicitly.

use ndarray::{Array2, ArrayView2};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

pub mod cpu;

#[cfg(feature = "gpu-wgpu")]
pub mod wgpu_backend;

#[cfg(all(feature = "gpu-mlx", target_os = "macos", target_arch = "aarch64"))]
pub mod mlx_backend;

#[cfg(feature = "gpu-cuda")]
pub mod cuda_backend;

/// Errors are plain strings: the Python layer turns them into one `warnings.warn` line and
/// falls back to the CPU, so a structured error type would add nothing.
pub type Result<T> = std::result::Result<T, String>;

/// Environment variable that forces a device: `cpu`, `gpu`, `wgpu`, `mlx`, `cuda`, `auto`.
pub const DEVICE_ENV: &str = "SQUEEZE_DEVICE";

/// Above this many indexed rows the GPU brute-force kNN is not used and HNSW on CPU is.
/// `SQUEEZE_BRUTEFORCE_MAX_ROWS` overrides it (benchmarks and experiments).
pub const BRUTEFORCE_MAX_ROWS: usize = 500_000;

/// The effective brute-force row limit: `SQUEEZE_BRUTEFORCE_MAX_ROWS` or the default.
pub fn bruteforce_max_rows() -> usize {
    std::env::var("SQUEEZE_BRUTEFORCE_MAX_ROWS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(BRUTEFORCE_MAX_ROWS)
}

/// Above this many samples exact t-SNE is not run on the GPU (Barnes-Hut on CPU instead).
pub const TSNE_EXACT_MAX_ROWS: usize = 20_000;

// ---------------------------------------------------------------------------------------------
// Metric
// ---------------------------------------------------------------------------------------------

/// Distance metrics supported by the GPU kNN and pairwise kernels. Anything else runs on CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Metric {
    Euclidean,
    Cosine,
    Manhattan,
}

impl Metric {
    /// Parse the metric names used across squeeze (`euclidean`/`l2`, `cosine`,
    /// `manhattan`/`l1`/`taxicab`). Returns `None` for metrics the GPU kernels do not cover.
    pub fn parse(name: &str) -> Option<Metric> {
        match name.to_ascii_lowercase().as_str() {
            "euclidean" | "l2" => Some(Metric::Euclidean),
            "cosine" => Some(Metric::Cosine),
            "manhattan" | "l1" | "taxicab" | "cityblock" => Some(Metric::Manhattan),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Metric::Euclidean => "euclidean",
            Metric::Cosine => "cosine",
            Metric::Manhattan => "manhattan",
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------------------------

/// A compute device. `Cpu` always exists; the others carry what the probe found.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    /// A wgpu adapter: Vulkan on Linux/Windows/Android, Metal on macOS/iOS, DX12 on Windows.
    Wgpu {
        name: String,
        /// `vulkan`, `metal`, `dx12`, `gl`.
        backend: String,
        /// `discrete`, `integrated`, `virtual`, `cpu`, `other`.
        device_type: String,
    },
    /// Apple MLX on Apple Silicon.
    Mlx {
        name: String,
    },
    /// NVIDIA CUDA through `cudarc`.
    Cuda {
        name: String,
        ordinal: usize,
    },
}

impl Device {
    /// Short kind name: `cpu`, `wgpu`, `mlx`, `cuda`.
    pub fn kind(&self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Wgpu { .. } => "wgpu",
            Device::Mlx { .. } => "mlx",
            Device::Cuda { .. } => "cuda",
        }
    }

    pub fn is_gpu(&self) -> bool {
        !matches!(self, Device::Cpu)
    }

    /// Pick the best available device (see module docs for the order). Never fails: the
    /// worst case is `Device::Cpu`. The result is computed once per process and cached.
    pub fn auto() -> Device {
        Self::probe().chosen.clone()
    }

    /// Probe every backend and report what was found. Cached per process (the probe opens
    /// GPU devices, which costs tens to hundreds of milliseconds).
    pub fn probe() -> &'static Selection {
        static PROBE: OnceLock<Selection> = OnceLock::new();
        PROBE.get_or_init(|| Selection::run(std::env::var(DEVICE_ENV).ok().as_deref()))
    }

    /// Resolve a user request. Accepts `auto`, `cpu`, `gpu` (any GPU), `wgpu`, `mlx`, `cuda`,
    /// or `None` (= `auto`). `gpu`/`wgpu`/`mlx`/`cuda` fail with a reason when that device is
    /// not available, so the caller can warn and fall back.
    pub fn resolve(request: Option<&str>) -> Result<Device> {
        let req = request.unwrap_or("auto").trim().to_ascii_lowercase();
        if req == "auto" || req.is_empty() {
            return Ok(Device::auto());
        }
        if req == "cpu" {
            return Ok(Device::Cpu);
        }
        // An explicit request should see software adapters too, hence a fresh, forced probe.
        let selection = if req == Device::probe().requested.as_deref().unwrap_or("auto") {
            Device::probe().clone()
        } else {
            Selection::run(Some(&req))
        };
        if req == "gpu" {
            return if selection.chosen.is_gpu() {
                Ok(selection.chosen)
            } else {
                Err(format!("no GPU available:\n{}", selection))
            };
        }
        if !matches!(req.as_str(), "wgpu" | "mlx" | "cuda") {
            return Err(format!(
                "unknown device '{}'; expected auto, cpu, gpu, wgpu, mlx or cuda",
                req
            ));
        }
        if selection.chosen.kind() == req {
            Ok(selection.chosen)
        } else {
            let reason = selection
                .candidates
                .iter()
                .find(|(kind, _, _)| *kind == req)
                .map(|(_, _, outcome)| outcome.clone())
                .unwrap_or_else(|| "not probed".to_string());
            Err(format!("device '{}' is not available: {}", req, reason))
        }
    }

    /// The backend for this device. Backends are created once and shared.
    pub fn backend(&self) -> Result<Arc<dyn Backend>> {
        static CACHE: OnceLock<Mutex<HashMap<Device, Arc<dyn Backend>>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(b) = cache.lock().unwrap().get(self) {
            return Ok(b.clone());
        }
        let backend: Arc<dyn Backend> = match self {
            Device::Cpu => Arc::new(cpu::CpuBackend),
            #[cfg(feature = "gpu-wgpu")]
            Device::Wgpu { .. } => Arc::new(wgpu_backend::WgpuBackend::open(self)?),
            #[cfg(not(feature = "gpu-wgpu"))]
            Device::Wgpu { .. } => return Err("built without the gpu-wgpu feature".into()),
            #[cfg(all(feature = "gpu-mlx", target_os = "macos", target_arch = "aarch64"))]
            Device::Mlx { .. } => Arc::new(mlx_backend::MlxBackend::open(self)?),
            #[cfg(not(all(feature = "gpu-mlx", target_os = "macos", target_arch = "aarch64")))]
            Device::Mlx { .. } => {
                return Err("built without the gpu-mlx feature (macOS aarch64 only)".into())
            }
            #[cfg(feature = "gpu-cuda")]
            Device::Cuda { .. } => Arc::new(cuda_backend::CudaBackend::open(self)?),
            #[cfg(not(feature = "gpu-cuda"))]
            Device::Cuda { .. } => return Err("built without the gpu-cuda feature".into()),
        };
        cache.lock().unwrap().insert(self.clone(), backend.clone());
        Ok(backend)
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Wgpu {
                name,
                backend,
                device_type,
            } => write!(f, "wgpu:{} ({}, {})", name, backend, device_type),
            Device::Mlx { name } => write!(f, "mlx:{}", name),
            Device::Cuda { name, ordinal } => write!(f, "cuda:{} (device {})", name, ordinal),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------------------------

/// The device probe report: what was requested, what was chosen, and every candidate.
#[derive(Clone, Debug)]
pub struct Selection {
    /// Value of `SQUEEZE_DEVICE` (or an explicit request) that steered the probe, if any.
    pub requested: Option<String>,
    pub chosen: Device,
    /// One entry per probed backend kind, in probe order:
    /// `(kind, device if it initialised, outcome)` where `kind` is `mlx`/`cuda`/`wgpu`/`cpu`
    /// and `outcome` is `chosen`, `available`, or the rejection reason (feature off, no
    /// driver, init error text).
    pub candidates: Vec<(String, Option<Device>, String)>,
}

impl Selection {
    /// Probe all backends. `request` is `SQUEEZE_DEVICE` or an explicit device string; when it
    /// names a backend, that backend is tried first and software adapters are allowed.
    pub fn run(request: Option<&str>) -> Selection {
        let req = request.map(|s| s.trim().to_ascii_lowercase());
        let forced = req
            .as_deref()
            .filter(|r| matches!(*r, "wgpu" | "mlx" | "cuda"));
        let mut candidates = Vec::new();
        let mut chosen: Option<Device> = None;

        if req.as_deref() == Some("cpu") {
            candidates.push((
                "cpu".into(),
                Some(Device::Cpu),
                format!("chosen ({}=cpu)", DEVICE_ENV),
            ));
            return Selection {
                requested: req,
                chosen: Device::Cpu,
                candidates,
            };
        }

        let order: Vec<&str> = match forced {
            Some(f) => std::iter::once(f)
                .chain(["mlx", "cuda", "wgpu"].into_iter().filter(|k| *k != f))
                .collect(),
            None => vec!["mlx", "cuda", "wgpu"],
        };

        for kind in order {
            let allow_software = forced == Some(kind);
            let probed: Result<Device> = match kind {
                "mlx" => probe_mlx(),
                "cuda" => probe_cuda(),
                "wgpu" => probe_wgpu(allow_software),
                _ => unreachable!(),
            };
            match probed {
                Ok(dev) => {
                    let outcome = if chosen.is_none() {
                        chosen = Some(dev.clone());
                        "chosen".to_string()
                    } else {
                        "available".to_string()
                    };
                    candidates.push((kind.into(), Some(dev), outcome));
                }
                Err(reason) => candidates.push((kind.into(), None, reason)),
            }
        }

        let cpu_outcome = if chosen.is_none() {
            "chosen (fallback)"
        } else {
            "available"
        };
        candidates.push(("cpu".into(), Some(Device::Cpu), cpu_outcome.into()));

        Selection {
            requested: req,
            chosen: chosen.unwrap_or(Device::Cpu),
            candidates,
        }
    }
}

impl fmt::Display for Selection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "squeeze compute devices")?;
        match &self.requested {
            Some(r) => writeln!(f, "  {}={}", DEVICE_ENV, r)?,
            None => writeln!(f, "  {} unset (auto)", DEVICE_ENV)?,
        }
        writeln!(f, "  chosen: {}", self.chosen)?;
        for (kind, device, outcome) in &self.candidates {
            match device {
                Some(d) => writeln!(f, "  {:<5} {:<10} {}", kind, outcome, d)?,
                None => writeln!(f, "  {:<5} {:<10} {}", kind, "rejected", outcome)?,
            }
        }
        Ok(())
    }
}

fn probe_mlx() -> Result<Device> {
    #[cfg(all(feature = "gpu-mlx", target_os = "macos", target_arch = "aarch64"))]
    {
        mlx_backend::probe()
    }
    #[cfg(not(all(feature = "gpu-mlx", target_os = "macos", target_arch = "aarch64")))]
    {
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            Err("built without the gpu-mlx feature".into())
        } else {
            Err("MLX needs macOS on Apple Silicon".into())
        }
    }
}

fn probe_cuda() -> Result<Device> {
    #[cfg(feature = "gpu-cuda")]
    {
        cuda_backend::probe()
    }
    #[cfg(not(feature = "gpu-cuda"))]
    {
        Err("built without the gpu-cuda feature".into())
    }
}

#[allow(unused_variables)]
fn probe_wgpu(allow_software: bool) -> Result<Device> {
    #[cfg(feature = "gpu-wgpu")]
    {
        wgpu_backend::probe(allow_software)
    }
    #[cfg(not(feature = "gpu-wgpu"))]
    {
        Err("built without the gpu-wgpu feature".into())
    }
}

// ---------------------------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------------------------

/// A gradient evaluator handed to the caller of a session: `(embedding [n, dim], scalars)
/// -> gradient [n, dim]`. The meaning of `scalars` depends on the session:
/// PaCMAP `[w_near, w_mid_near, w_far]`, TriMap `[scale]`, t-SNE `[exaggeration]`.
pub type GradFn<'a> = dyn FnMut(ArrayView2<f64>, &[f64]) -> Result<Array2<f64>> + 'a;

/// The kernel set. Shapes: `data [n, d]`, `queries [m, d]`; all row-major contiguous.
///
/// Every method may fail (device lost, out of memory); callers fall back to the CPU backend.
pub trait Backend: Send + Sync {
    fn device(&self) -> &Device;

    /// Exact k-nearest neighbours of `queries` among `data`, `k <= n`. Returns
    /// `(indices [m, k], distances [m, k])` sorted by ascending distance. Distances are the
    /// metric's own (euclidean: root distance; cosine: `1 - cos`; manhattan: L1).
    fn knn_bruteforce(
        &self,
        data: ArrayView2<f32>,
        queries: ArrayView2<f32>,
        k: usize,
        metric: Metric,
    ) -> Result<(Array2<u32>, Array2<f32>)>;

    /// Squared euclidean distances `[n, m]` between the rows of `a [n, d]` and `b [m, d]`.
    fn pairwise_sqdist(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>>;

    /// Gram matrix `a · aᵀ` as `[n, n]`.
    fn gram(&self, a: ArrayView2<f32>) -> Result<Array2<f32>>;

    /// PaCMAP gradient session. `near`, `mid_near` and `far` are `(i, j)` pairs; the
    /// gradient for `scalars = [w_near, w_mn, w_fp]` is
    /// `near/mid: +coeff · (y_i - y_j)` on `i` and the opposite on `j`, `far: -coeff · (y_i - y_j)`,
    /// with `coeff_near = w·20/(10+d²)²`, `coeff_mid = w·20000/(10000+d²)²`,
    /// `coeff_far = w·2/(1+d²)²`. `run` receives the gradient evaluator and drives the loop.
    fn pacmap_session(
        &self,
        n: usize,
        dim: usize,
        near: &[(u32, u32)],
        mid_near: &[(u32, u32)],
        far: &[(u32, u32)],
        run: &mut dyn FnMut(&mut GradFn<'_>) -> Result<()>,
    ) -> Result<()>;

    /// TriMap gradient session over `(anchor, positive, negative)` triplets with per-triplet
    /// `weights`. For `scalars = [scale]` and `loss = d²_ij - d²_ik + 1 > 0`:
    /// `g_i += s·w·((y_i-y_j) - (y_i-y_k))`, `g_j -= s·w·(y_i-y_j)`, `g_k += s·w·(y_i-y_k)`.
    fn trimap_session(
        &self,
        n: usize,
        dim: usize,
        triplets: &[(u32, u32, u32)],
        weights: &[f64],
        run: &mut dyn FnMut(&mut GradFn<'_>) -> Result<()>,
    ) -> Result<()>;

    /// Exact t-SNE gradient session for the joint probabilities `p [n, n]`. For
    /// `scalars = [exaggeration]`: `q_ij = k_ij / Σ k`, `k_ij = 1/(1+d²_ij)`,
    /// `g_i = 4 Σ_j (ex·p_ij - q_ij) k_ij (y_i - y_j)`, with `q` floored at `1e-12` after
    /// normalisation as in the CPU implementation.
    fn tsne_exact_session(
        &self,
        p: ArrayView2<f64>,
        dim: usize,
        run: &mut dyn FnMut(&mut GradFn<'_>) -> Result<()>,
    ) -> Result<()>;
}

// ---------------------------------------------------------------------------------------------
// Shared helpers for backends (not public API)
// ---------------------------------------------------------------------------------------------

/// Per-point adjacency in CSR form for pair kernels: entry `e` of point `i` is
/// `(nbr[e], tag[e])` for `e in offsets[i]..offsets[i+1]`.
pub(crate) struct Csr {
    pub offsets: Vec<u32>,
    pub nbr: Vec<u32>,
    pub tag: Vec<u32>,
}

/// Build the symmetric CSR adjacency of tagged pairs. Each pair `(i, j)` appears in both
/// `i`'s and `j`'s list with the same tag.
pub(crate) fn pairs_to_csr(n: usize, groups: &[(&[(u32, u32)], u32)]) -> Csr {
    let mut counts = vec![0u32; n + 1];
    for (pairs, _) in groups {
        for &(i, j) in *pairs {
            counts[i as usize + 1] += 1;
            counts[j as usize + 1] += 1;
        }
    }
    for i in 0..n {
        counts[i + 1] += counts[i];
    }
    let nnz = counts[n] as usize;
    let mut nbr = vec![0u32; nnz];
    let mut tag = vec![0u32; nnz];
    let mut fill = counts.clone();
    for (pairs, t) in groups {
        for &(i, j) in *pairs {
            let a = fill[i as usize] as usize;
            nbr[a] = j;
            tag[a] = *t;
            fill[i as usize] += 1;
            let b = fill[j as usize] as usize;
            nbr[b] = i;
            tag[b] = *t;
            fill[j as usize] += 1;
        }
    }
    Csr {
        offsets: counts,
        nbr,
        tag,
    }
}

/// Per-point roles in triplets: entry `e` of point `p` is `(triplet[e], role[e])` with role
/// `0` anchor, `1` positive, `2` negative.
pub(crate) fn triplets_to_csr(n: usize, triplets: &[(u32, u32, u32)]) -> Csr {
    let mut counts = vec![0u32; n + 1];
    for &(i, j, k) in triplets {
        counts[i as usize + 1] += 1;
        counts[j as usize + 1] += 1;
        counts[k as usize + 1] += 1;
    }
    for i in 0..n {
        counts[i + 1] += counts[i];
    }
    let nnz = counts[n] as usize;
    let mut nbr = vec![0u32; nnz];
    let mut tag = vec![0u32; nnz];
    let mut fill = counts.clone();
    for (t, &(i, j, k)) in triplets.iter().enumerate() {
        for (p, role) in [(i, 0u32), (j, 1u32), (k, 2u32)] {
            let a = fill[p as usize] as usize;
            nbr[a] = t as u32;
            tag[a] = role;
            fill[p as usize] += 1;
        }
    }
    Csr {
        offsets: counts,
        nbr,
        tag,
    }
}

/// Convert an `f64` matrix to a contiguous `f32` vector (row-major).
#[allow(dead_code)]
pub(crate) fn to_f32(a: ArrayView2<f64>) -> Vec<f32> {
    a.iter().map(|&v| v as f32).collect()
}

/// Wrap a contiguous `f32` buffer as an `f64` matrix.
#[allow(dead_code)]
pub(crate) fn to_f64_matrix(v: &[f32], n: usize, dim: usize) -> Array2<f64> {
    Array2::from_shape_vec((n, dim), v.iter().map(|&x| x as f64).collect())
        .expect("shape matches buffer length")
}

/// Row-major contiguous copy of an `f32` view (no copy when already standard layout).
pub(crate) fn contiguous_f32<'a>(a: ArrayView2<'a, f32>) -> std::borrow::Cow<'a, [f32]> {
    match a.to_slice() {
        Some(s) => std::borrow::Cow::Borrowed(s),
        None => std::borrow::Cow::Owned(a.iter().copied().collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_parse_covers_aliases() {
        assert_eq!(Metric::parse("L2"), Some(Metric::Euclidean));
        assert_eq!(Metric::parse("taxicab"), Some(Metric::Manhattan));
        assert_eq!(Metric::parse("cosine"), Some(Metric::Cosine));
        assert_eq!(Metric::parse("hamming"), None);
    }

    #[test]
    fn resolve_cpu_and_auto_never_fail() {
        assert_eq!(Device::resolve(Some("cpu")).unwrap(), Device::Cpu);
        let _ = Device::resolve(None).unwrap();
        let _ = Device::resolve(Some("auto")).unwrap();
        assert!(Device::resolve(Some("tpu")).is_err());
    }

    #[test]
    fn selection_report_lists_every_kind() {
        let sel = Selection::run(None);
        let kinds: Vec<_> = sel.candidates.iter().map(|c| c.0.as_str()).collect();
        assert_eq!(kinds, ["mlx", "cuda", "wgpu", "cpu"]);
        let text = sel.to_string();
        assert!(text.contains("chosen:"));
    }

    #[test]
    fn csr_is_symmetric_and_complete() {
        let near = [(0u32, 1u32), (1, 2)];
        let far = [(0u32, 2u32)];
        let csr = pairs_to_csr(3, &[(&near, 0), (&far, 2)]);
        assert_eq!(csr.offsets, vec![0, 2, 4, 6]);
        // point 1 sees 0 and 2 as near
        let s = &csr.nbr[2..4];
        assert!(s.contains(&0) && s.contains(&2));
        assert!(csr.tag[2..4].iter().all(|&t| t == 0));
    }

    #[test]
    fn triplet_csr_roles() {
        let csr = triplets_to_csr(3, &[(0, 1, 2)]);
        assert_eq!(csr.offsets, vec![0, 1, 2, 3]);
        assert_eq!(csr.tag, vec![0, 1, 2]);
        assert_eq!(csr.nbr, vec![0, 0, 0]);
    }
}
