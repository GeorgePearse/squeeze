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

/// The backend for `device`, or the CPU backend with a warning when it cannot be opened.
pub fn backend(py: Python<'_>, device: &Device) -> Arc<dyn Backend> {
    match device.backend() {
        Ok(b) => b,
        Err(e) => {
            warn(
                py,
                &format!("squeeze: {} unavailable, falling back to cpu: {}", device, e),
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
pub fn distance_matrix(py: Python<'_>, device: &Device, x: ArrayView2<f64>) -> PyResult<Array2<f64>> {
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
