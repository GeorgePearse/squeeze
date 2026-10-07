//! PyO3 glue between the Python API and `compute`: device-string parsing, the one-line
//! warning on fallback, and the `squeeze.devices()` report. Nothing in `compute` depends on
//! this module.

use crate::compute::{self, cpu::CpuBackend, Backend, Device};
use ndarray::{Array2, ArrayView2};
use pyo3::exceptions::{PyRuntimeWarning, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

/// Emit one `RuntimeWarning` line (never raises).
pub fn warn(py: Python<'_>, message: &str) {
    let category = py.get_type_bound::<PyRuntimeWarning>();
    let _ = PyErr::warn_bound(py, &category, message, 1);
}

/// Resolve the `device=` argument. `None`/`"auto"` never fail. An explicit device that is
/// not available warns and falls back to the CPU; an unknown name raises `ValueError`.
pub fn resolve(py: Python<'_>, device: Option<&str>) -> PyResult<Device> {
    match Device::resolve(device) {
        Ok(d) => Ok(d),
        Err(msg) if msg.starts_with("unknown device") => Err(PyValueError::new_err(msg)),
        Err(msg) => {
            warn(py, &format!("squeeze: falling back to cpu: {}", msg));
            Ok(Device::Cpu)
        }
    }
}

/// Kinds of work with different GPU break-even sizes (measured on a T4, see docs/gpu.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Work {
    /// Pairwise distance matrix (MDS, Isomap, LLE, PHATE, TriMap, PaCMAP, t-SNE affinities).
    Pairwise,
    /// Exact brute-force k-NN instead of the HNSW graph.
    Knn,
    /// Exact t-SNE gradient instead of Barnes-Hut.
    TsneExact,
    /// Per-iteration PaCMAP / TriMap gradient steps (launch-latency bound at small n).
    Step,
}

/// Below this many rows `device="auto"` keeps the work on the CPU even when a GPU exists.
/// Environment overrides: `SQUEEZE_GPU_MIN_ROWS` (pairwise and k-NN), `SQUEEZE_GPU_MIN_ROWS_TSNE`,
/// `SQUEEZE_GPU_MIN_ROWS_STEP`.
pub fn gpu_min_rows(work: Work) -> usize {
    let env = |key: &str| {
        std::env::var(key)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
    };
    match work {
        Work::Pairwise | Work::Knn => env("SQUEEZE_GPU_MIN_ROWS").unwrap_or(4_096),
        Work::TsneExact => env("SQUEEZE_GPU_MIN_ROWS_TSNE").unwrap_or(1_000),
        Work::Step => env("SQUEEZE_GPU_MIN_ROWS_STEP").unwrap_or(100_000),
    }
}

/// Was a device named explicitly, either as the `device=` argument or through `SQUEEZE_DEVICE`?
fn explicit(request: Option<&str>) -> bool {
    let named = |r: &str| {
        let r = r.trim().to_ascii_lowercase();
        !r.is_empty() && r != "auto"
    };
    request.is_some_and(named) || Device::probe().requested.as_deref().is_some_and(named)
}

/// Resolve `device=` for a piece of work on `n` rows. `auto` only picks a GPU when the work
/// is large enough to pay for the transfers (see [`gpu_min_rows`]); an explicit device is
/// always honoured.
pub fn resolve_for(py: Python<'_>, device: Option<&str>, n: usize, work: Work) -> PyResult<Device> {
    let resolved = resolve(py, device)?;
    if resolved.is_gpu() && !explicit(device) && n < gpu_min_rows(work) {
        return Ok(Device::Cpu);
    }
    Ok(resolved)
}

/// The backend for `device`, or the CPU backend with a warning when it cannot be opened.
pub fn backend(py: Python<'_>, device: &Device) -> Arc<dyn Backend> {
    match device.backend() {
        Ok(b) => b,
        Err(e) => {
            warn(
                py,
                &format!(
                    "squeeze: {} unavailable, falling back to cpu: {}",
                    device, e
                ),
            );
            Arc::new(CpuBackend)
        }
    }
}

/// Run `f` on the device's backend; on any error warn once and rerun it on the CPU.
pub fn with_fallback<T>(
    py: Python<'_>,
    device: &Device,
    what: &str,
    f: impl Fn(&dyn Backend) -> compute::Result<T>,
) -> PyResult<T> {
    let primary = backend(py, device);
    match f(primary.as_ref()) {
        Ok(v) => Ok(v),
        Err(e) if primary.device().is_gpu() => {
            warn(
                py,
                &format!(
                    "squeeze: {} failed on {}, falling back to cpu: {}",
                    what,
                    primary.device(),
                    e
                ),
            );
            f(&CpuBackend).map_err(PyValueError::new_err)
        }
        Err(e) => Err(PyValueError::new_err(e)),
    }
}

/// Euclidean distance matrix `[n, n]` in `f64` from `f64` data, computed on `device` in `f32`
/// with a zero diagonal. Used by MDS, Isomap, LLE, PHATE, TriMap and PaCMAP.
pub fn distance_matrix(
    py: Python<'_>,
    device: &Device,
    x: ArrayView2<f64>,
) -> PyResult<Array2<f64>> {
    let x32 = x.mapv(|v| v as f32);
    let sq = with_fallback(py, device, "pairwise distances", |b| {
        b.pairwise_sqdist(x32.view(), x32.view())
    })?;
    let n = sq.nrows();
    let mut out = sq.mapv(|v| (v.max(0.0) as f64).sqrt());
    for i in 0..n {
        out[[i, i]] = 0.0;
    }
    Ok(out)
}

/// `squeeze.devices()`: the device probe report as text.
#[pyfunction]
pub fn devices() -> String {
    Device::probe().to_string()
}

/// `squeeze.default_device()`: the device `device="auto"` resolves to, e.g. `"cpu"` or
/// `"wgpu:NVIDIA T4 (vulkan, discrete)"`.
#[pyfunction]
pub fn default_device() -> String {
    Device::auto().to_string()
}

/// `squeeze.resolve_device(name)`: what a `device=` argument resolves to, as text.
#[pyfunction]
#[pyo3(signature = (device=None))]
pub fn resolve_device(py: Python<'_>, device: Option<&str>) -> PyResult<String> {
    Ok(resolve(py, device)?.to_string())
}
