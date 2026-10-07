//! CUDA backend placeholder (feature `gpu-cuda`).
//!
//! NVIDIA GPUs already run through the wgpu backend on Vulkan. A native CUDA path
//! (`cudarc`, same kernels in CUDA C) is only worth its build complexity if the
//! wgpu-Vulkan numbers on a real NVIDIA card leave a large factor on the table; until that
//! measurement says so, this module reports CUDA as unavailable so `Device::auto()` falls
//! through to wgpu. See `docs/gpu.md` for the T4 comparison that informs this decision.

use super::{Device, Result};

/// CUDA is never selected: the backend is not implemented in this build.
pub fn probe() -> Result<Device> {
    Err("CUDA backend not implemented; NVIDIA GPUs use wgpu on Vulkan".into())
}

/// Placeholder so `Device::backend()` has a target for `Device::Cuda`.
pub struct CudaBackend;

impl CudaBackend {
    pub fn open(_desc: &Device) -> Result<super::cpu::CpuBackend> {
        Err("CUDA backend not implemented; NVIDIA GPUs use wgpu on Vulkan".into())
    }
}
