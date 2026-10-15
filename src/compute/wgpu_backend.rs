//! wgpu backend: portable compute shaders (WGSL) on Vulkan, Metal and DX12.
//!
//! All kernels work in `f32`. Pairwise work is tiled through a reusable `[a_rows, b_cols]`
//! scratch buffer sized from the adapter limits, so inputs far larger than one binding are
//! handled; a single input matrix (data or queries) must still fit in one buffer, otherwise
//! the call fails and the caller falls back to the CPU.

use super::{
    contiguous_f32, pairs_to_csr, to_f32, to_f64_matrix, triplets_to_csr, Backend, Device, GradFn,
    Metric, Result,
};
use ndarray::{Array2, ArrayView2};
use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};
use wgpu::util::DeviceExt;

const DIST_TILE_SRC: &str = include_str!("shaders/dist_tile.wgsl");
const TOPK_MERGE_SRC: &str = include_str!("shaders/topk_merge.wgsl");
const PACMAP_SRC: &str = include_str!("shaders/pacmap_grad.wgsl");
const TRIMAP_SRC: &str = include_str!("shaders/trimap_grad.wgsl");
const TSNE_SRC: &str = include_str!("shaders/tsne_exact.wgsl");

/// Largest embedding dimension the gradient shaders support (private arrays).
const DMAX: usize = 16;
/// Workgroup size of the 1-D kernels.
const WG: u32 = 64;
/// Upper bound on the pairwise scratch tile, in bytes.
const TILE_BUDGET_BYTES: u64 = 256 << 20;
/// Columns (data rows) per pairwise tile; also the scan length of the top-k merge.
const TILE_COLS: usize = 65_536;

/// A pairwise reduction variant: the inner accumulation and the final map.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Variant {
    SqEuclidean,
    Euclidean,
    Manhattan,
    Dot,
    Cosine,
}

impl Variant {
    fn key(self) -> &'static str {
        match self {
            Variant::SqEuclidean => "dist_sqeuclidean",
            Variant::Euclidean => "dist_euclidean",
            Variant::Manhattan => "dist_manhattan",
            Variant::Dot => "dist_dot",
            Variant::Cosine => "dist_cosine",
        }
    }

    fn source(self) -> String {
        let (acc, fin) = match self {
            Variant::SqEuclidean => ("let v = x - y; acc = acc + v * v;", "let result = acc;"),
            Variant::Euclidean => (
                "let v = x - y; acc = acc + v * v;",
                "let result = sqrt(acc);",
            ),
            Variant::Manhattan => ("acc = acc + abs(x - y);", "let result = acc;"),
            Variant::Dot => ("acc = acc + x * y;", "let result = acc;"),
            Variant::Cosine => ("acc = acc + x * y;", "let result = 1.0 - acc;"),
        };
        DIST_TILE_SRC
            .replace("//ACC//", acc)
            .replace("//FINAL//", fin)
    }
}

/// The wgpu backend for one adapter.
pub struct WgpuBackend {
    desc: Device,
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: wgpu::Limits,
    pipelines: Mutex<HashMap<String, wgpu::ComputePipeline>>,
    /// Last error reported through the uncaptured-error handler, if any.
    last_error: Arc<Mutex<Option<String>>>,
}

fn new_instance() -> wgpu::Instance {
    // Honours WGPU_BACKEND (e.g. `vulkan`, `metal`, `dx12`) and the other wgpu env knobs.
    wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env())
}

fn device_type_name(t: wgpu::DeviceType) -> &'static str {
    match t {
        wgpu::DeviceType::DiscreteGpu => "discrete",
        wgpu::DeviceType::IntegratedGpu => "integrated",
        wgpu::DeviceType::VirtualGpu => "virtual",
        wgpu::DeviceType::Cpu => "cpu",
        wgpu::DeviceType::Other => "other",
    }
}

fn is_software(info: &wgpu::AdapterInfo) -> bool {
    let name = info.name.to_ascii_lowercase();
    info.device_type == wgpu::DeviceType::Cpu
        || name.contains("llvmpipe")
        || name.contains("lavapipe")
        || name.contains("swiftshader")
}

/// Lower is better: hardware class first, then API (Vulkan/Metal/DX12 before GL).
fn rank(info: &wgpu::AdapterInfo) -> Option<(u8, u8)> {
    let class = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    };
    let api = match info.backend {
        wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::Dx12 => 0,
        wgpu::Backend::Gl => 1,
        _ => return None,
    };
    Some((class, api))
}

fn describe(info: &wgpu::AdapterInfo) -> Device {
    Device::Wgpu {
        name: info.name.clone(),
        backend: info.backend.to_str().to_string(),
        device_type: device_type_name(info.device_type).to_string(),
    }
}

/// Enumerate adapters and pick the best one. Software adapters (lavapipe, SwiftShader,
/// anything of type `Cpu`) and virtual/other adapters are only eligible when
/// `allow_software` is set, i.e. when the user forced `SQUEEZE_DEVICE=wgpu`.
pub fn probe(allow_software: bool) -> Result<Device> {
    let instance = new_instance();
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    if adapters.is_empty() {
        return Err("no wgpu adapter found (no Vulkan, Metal or DX12 driver)".into());
    }
    let mut best: Option<((u8, u8), wgpu::AdapterInfo)> = None;
    let mut skipped = Vec::new();
    for adapter in &adapters {
        let info = adapter.get_info();
        let Some(r) = rank(&info) else {
            continue;
        };
        let hardware = r.0 <= 1 && !is_software(&info);
        if !hardware && !allow_software {
            skipped.push(format!(
                "{} ({}, {})",
                info.name,
                info.backend.to_str(),
                device_type_name(info.device_type)
            ));
            continue;
        }
        if best.as_ref().map_or(true, |(br, _)| r < *br) {
            best = Some((r, info));
        }
    }
    match best {
        Some((_, info)) => Ok(describe(&info)),
        None if skipped.is_empty() => Err("no usable wgpu adapter".into()),
        None => Err(format!(
            "only software/virtual adapters: {}; set {}=wgpu to use them",
            skipped.join(", "),
            super::DEVICE_ENV
        )),
    }
}

impl WgpuBackend {
    /// Open the adapter described by `desc` (as produced by [`probe`]).
    pub fn open(desc: &Device) -> Result<Self> {
        let Device::Wgpu { name, backend, .. } = desc else {
            return Err("not a wgpu device".into());
        };
        let instance = new_instance();
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let adapter = adapters
            .into_iter()
            .find(|a| {
                let info = a.get_info();
                &info.name == name && info.backend.to_str() == backend
            })
            .ok_or_else(|| format!("wgpu adapter '{}' ({}) is no longer present", name, backend))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("squeeze"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .map_err(|e| format!("wgpu request_device failed on '{}': {}", name, e))?;
        let last_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let slot = last_error.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            *slot.lock().unwrap() = Some(e.to_string());
        }));
        let limits = device.limits();
        Ok(Self {
            desc: desc.clone(),
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            limits,
            pipelines: Mutex::new(HashMap::new()),
            last_error,
        })
    }

    /// Adapter limits relevant to sizing (binding size, buffer size), for reports and tests.
    pub fn limits(&self) -> (u64, u64) {
        (
            self.limits.max_storage_buffer_binding_size,
            self.limits.max_buffer_size,
        )
    }

    fn take_error(&self, what: &str) -> Result<()> {
        match self.last_error.lock().unwrap().take() {
            Some(e) => Err(format!(
                "wgpu error during {} on {}: {}",
                what, self.desc, e
            )),
            None => Ok(()),
        }
    }

    fn pipeline(
        &self,
        key: &str,
        source: impl FnOnce() -> String,
        entry: &str,
    ) -> Result<wgpu::ComputePipeline> {
        self.pipeline_with_layout(key, source, entry, None)
    }

    /// Explicit layout for kernels whose entry points do not all touch every binding (an
    /// implicit layout would then reject a shared bind group): bindings 0 and 1 read-only
    /// storage, 2 read-write storage, 3 uniform.
    fn shared_layout(&self) -> wgpu::PipelineLayout {
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("squeeze-shared"),
                entries: &[
                    storage(0, true),
                    storage(1, true),
                    storage(2, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        self.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("squeeze-shared"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            })
    }

    fn pipeline_with_layout(
        &self,
        key: &str,
        source: impl FnOnce() -> String,
        entry: &str,
        layout: Option<&wgpu::PipelineLayout>,
    ) -> Result<wgpu::ComputePipeline> {
        if let Some(p) = self.pipelines.lock().unwrap().get(key) {
            return Ok(p.clone());
        }
        let src = source();
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(key),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(key),
                layout,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!("shader '{}' failed to compile: {}", key, e));
        }
        self.pipelines
            .lock()
            .unwrap()
            .insert(key.to_string(), pipeline.clone());
        Ok(pipeline)
    }

    fn check_buffer(&self, label: &str, bytes: u64) -> Result<()> {
        let max = self
            .limits
            .max_storage_buffer_binding_size
            .min(self.limits.max_buffer_size);
        if bytes > max {
            return Err(format!(
                "{} needs {} MiB but {} allows {} MiB per buffer",
                label,
                bytes >> 20,
                self.desc,
                max >> 20
            ));
        }
        Ok(())
    }

    fn storage_from<T: bytemuck::Pod>(&self, label: &str, data: &[T]) -> Result<wgpu::Buffer> {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        self.check_buffer(label, bytes.len().max(4) as u64)?;
        // Zero-length bindings are invalid; pad to one element.
        let buf = if bytes.is_empty() {
            self.storage_zeroed(label, 4)
        } else {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                })
        };
        Ok(buf)
    }

    fn storage_zeroed(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn uniform(&self, label: &str, words: &[u32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(words),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn bind(&self, pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    fn encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("squeeze"),
            })
    }

    fn dispatch(
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind: &wgpu::BindGroup,
        groups: (u32, u32, u32),
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
    }

    fn submit(&self, encoder: wgpu::CommandEncoder, what: &str) -> Result<()> {
        self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("wgpu poll failed during {}: {}", what, e))?;
        self.take_error(what)
    }

    /// Copy `bytes` from `src` back to the host.
    fn readback(&self, src: &wgpu::Buffer, bytes: u64, what: &str) -> Result<Vec<u8>> {
        self.readback_at(src, 0, bytes, what)
    }

    /// Copy `bytes` starting at `offset` (a multiple of 4) from `src` back to the host.
    fn readback_at(
        &self,
        src: &wgpu::Buffer,
        offset: u64,
        bytes: u64,
        what: &str,
    ) -> Result<Vec<u8>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.encoder();
        encoder.copy_buffer_to_buffer(src, offset, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let (tx, rx) = mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("wgpu poll failed during {}: {}", what, e))?;
        rx.recv()
            .map_err(|_| format!("wgpu map callback dropped during {}", what))?
            .map_err(|e| format!("wgpu buffer map failed during {}: {}", what, e))?;
        let out = {
            let view = staging
                .slice(..)
                .get_mapped_range()
                .map_err(|e| format!("wgpu mapped range failed during {}: {}", what, e))?;
            view.to_vec()
        };
        staging.unmap();
        self.take_error(what)?;
        Ok(out)
    }

    fn groups_1d(&self, n: usize) -> Result<u32> {
        let groups = (n as u64).div_ceil(WG as u64);
        if groups > self.limits.max_compute_workgroups_per_dimension as u64 {
            return Err(format!(
                "{} rows exceed the {} workgroups per dispatch on {}",
                n, self.limits.max_compute_workgroups_per_dimension, self.desc
            ));
        }
        Ok(groups as u32)
    }

    /// Run the tiled pairwise kernel over all `a × b` tiles; `per_tile` is called after each
    /// tile has been computed into `tile` (row stride `b_rows`) and its submission completed.
    #[allow(clippy::too_many_arguments)]
    fn tiles(
        &self,
        a: &[f32],
        n_a: usize,
        b: &[f32],
        n_b: usize,
        d: usize,
        variant: Variant,
        mut per_tile: impl FnMut(&wgpu::Buffer, usize, usize, usize, usize) -> Result<()>,
    ) -> Result<()> {
        if d == 0 || n_a == 0 || n_b == 0 {
            return Err("empty input".into());
        }
        let a_buf = self.storage_from("a", a)?;
        let b_buf = self.storage_from("b", b)?;
        let max_rows = (self.limits.max_compute_workgroups_per_dimension as usize) * 16;
        let b_cols = n_b.min(TILE_COLS).min(max_rows);
        let budget_floats =
            (TILE_BUDGET_BYTES.min(self.limits.max_storage_buffer_binding_size) / 4) as usize;
        let a_rows_max = n_a.min(budget_floats / b_cols).min(max_rows).max(1);
        let tile = self.storage_zeroed("tile", (a_rows_max * b_cols * 4) as u64);
        let params = self.uniform("dist-params", &[0; 8]);
        let pipeline = self.pipeline(variant.key(), || variant.source(), "main")?;
        let bind = self.bind(&pipeline, &[&a_buf, &b_buf, &tile, &params]);

        let mut a0 = 0;
        while a0 < n_a {
            let a_rows = a_rows_max.min(n_a - a0);
            let mut b0 = 0;
            while b0 < n_b {
                let b_rows = b_cols.min(n_b - b0);
                self.queue.write_buffer(
                    &params,
                    0,
                    bytemuck::cast_slice(&[
                        a0 as u32,
                        a_rows as u32,
                        b0 as u32,
                        b_rows as u32,
                        d as u32,
                        b_rows as u32,
                        0,
                        0,
                    ]),
                );
                let mut enc = self.encoder();
                Self::dispatch(
                    &mut enc,
                    &pipeline,
                    &bind,
                    (b_rows.div_ceil(16) as u32, a_rows.div_ceil(16) as u32, 1),
                );
                self.submit(enc, variant.key())?;
                per_tile(&tile, a0, a_rows, b0, b_rows)?;
                b0 += b_rows;
            }
            a0 += a_rows;
        }
        Ok(())
    }

    /// Pairwise matrix `[n_a, n_b]` for a variant, assembled on the host tile by tile.
    fn pairwise(
        &self,
        a: ArrayView2<f32>,
        b: ArrayView2<f32>,
        variant: Variant,
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
            variant,
            |tile, a0, a_rows, b0, b_rows| {
                let bytes = self.readback(tile, (a_rows * b_rows * 4) as u64, variant.key())?;
                let vals: &[f32] = bytemuck::cast_slice(&bytes);
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

/// L2-normalise rows for the cosine kernel; rows with (near) zero norm become zero so their
/// cosine distance is 1, matching `metrics_simd::cosine`.
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

impl Backend for WgpuBackend {
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
        let (data_v, queries_v, variant) = match metric {
            Metric::Euclidean => (data_c.to_vec(), queries_c.to_vec(), Variant::Euclidean),
            Metric::Manhattan => (data_c.to_vec(), queries_c.to_vec(), Variant::Manhattan),
            Metric::Cosine => (
                normalize_rows(&data_c, d),
                normalize_rows(&queries_c, d),
                Variant::Cosine,
            ),
        };
        let best_idx = self.storage_from("best_idx", &vec![u32::MAX; m * k])?;
        let best_dist = self.storage_from("best_dist", &vec![f32::INFINITY; m * k])?;
        let params = self.uniform("topk-params", &[0; 8]);
        let topk = self.pipeline("topk_merge", || TOPK_MERGE_SRC.to_string(), "main")?;
        // Queries are the `a` side (rows of the tile), data the `b` side (columns).
        self.tiles(
            &queries_v,
            m,
            &data_v,
            n,
            d,
            variant,
            |tile, q0, q_rows, b0, b_cols| {
                self.queue.write_buffer(
                    &params,
                    0,
                    bytemuck::cast_slice(&[
                        q0 as u32,
                        q_rows as u32,
                        b0 as u32,
                        b_cols as u32,
                        k as u32,
                        b_cols as u32,
                        0,
                        0,
                    ]),
                );
                let bind = self.bind(&topk, &[tile, &best_idx, &best_dist, &params]);
                let mut enc = self.encoder();
                Self::dispatch(&mut enc, &topk, &bind, (self.groups_1d(q_rows)?, 1, 1));
                self.submit(enc, "topk_merge")
            },
        )?;
        let idx_bytes = self.readback(&best_idx, (m * k * 4) as u64, "knn readback")?;
        let dist_bytes = self.readback(&best_dist, (m * k * 4) as u64, "knn readback")?;
        let idx =
            Array2::from_shape_vec((m, k), bytemuck::cast_slice::<u8, u32>(&idx_bytes).to_vec())
                .map_err(|e| e.to_string())?;
        let dist = Array2::from_shape_vec(
            (m, k),
            bytemuck::cast_slice::<u8, f32>(&dist_bytes).to_vec(),
        )
        .map_err(|e| e.to_string())?;
        Ok((idx, dist))
    }

    fn pairwise_sqdist(&self, a: ArrayView2<f32>, b: ArrayView2<f32>) -> Result<Array2<f32>> {
        self.pairwise(a, b, Variant::SqEuclidean)
    }

    fn gram(&self, a: ArrayView2<f32>) -> Result<Array2<f32>> {
        self.pairwise(a, a, Variant::Dot)
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
        let groups = self.groups_1d(n)?;
        let csr = pairs_to_csr(n, &[(near, 0), (mid_near, 1), (far, 2)]);
        let nbr_off = csr.offsets.len() as u32;
        let tag_off = nbr_off + csr.nbr.len() as u32;
        let mut packed = csr.offsets;
        packed.extend_from_slice(&csr.nbr);
        packed.extend_from_slice(&csr.tag);
        let csr_buf = self.storage_from("csr", &packed)?;
        let y_buf = self.storage_zeroed("y", (n * dim * 4) as u64);
        let grad_buf = self.storage_zeroed("grad", (n * dim * 4) as u64);
        let params = self.uniform("pacmap-params", &[0; 8]);
        let pipeline = self.pipeline("pacmap_grad", || PACMAP_SRC.to_string(), "main")?;
        let bind = self.bind(&pipeline, &[&y_buf, &csr_buf, &grad_buf, &params]);
        let mut grad_fn = |y: ArrayView2<f64>, w: &[f64]| -> Result<Array2<f64>> {
            if y.nrows() != n || y.ncols() != dim || w.len() != 3 {
                return Err("pacmap: embedding shape or weights mismatch".into());
            }
            self.queue
                .write_buffer(&y_buf, 0, bytemuck::cast_slice(&to_f32(y)));
            self.queue.write_buffer(
                &params,
                0,
                bytemuck::cast_slice(&[
                    n as u32,
                    dim as u32,
                    (w[0] as f32).to_bits(),
                    (w[1] as f32).to_bits(),
                    (w[2] as f32).to_bits(),
                    nbr_off,
                    tag_off,
                    0,
                ]),
            );
            let mut enc = self.encoder();
            Self::dispatch(&mut enc, &pipeline, &bind, (groups, 1, 1));
            self.submit(enc, "pacmap_grad")?;
            let bytes = self.readback(&grad_buf, (n * dim * 4) as u64, "pacmap_grad")?;
            Ok(to_f64_matrix(bytemuck::cast_slice(&bytes), n, dim))
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
        let groups = self.groups_1d(n)?;
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
        let packed_buf = self.storage_from("trimap-packed", &packed)?;
        let y_buf = self.storage_zeroed("y", (n * dim * 4) as u64);
        let grad_buf = self.storage_zeroed("grad", (n * dim * 4) as u64);
        let params = self.uniform("trimap-params", &[0; 8]);
        let pipeline = self.pipeline("trimap_grad", || TRIMAP_SRC.to_string(), "main")?;
        let bind = self.bind(&pipeline, &[&y_buf, &packed_buf, &grad_buf, &params]);
        let mut grad_fn = |y: ArrayView2<f64>, s: &[f64]| -> Result<Array2<f64>> {
            if y.nrows() != n || y.ncols() != dim || s.len() != 1 {
                return Err("trimap: embedding shape or scalars mismatch".into());
            }
            self.queue
                .write_buffer(&y_buf, 0, bytemuck::cast_slice(&to_f32(y)));
            self.queue.write_buffer(
                &params,
                0,
                bytemuck::cast_slice(&[
                    n as u32,
                    dim as u32,
                    (s[0] as f32).to_bits(),
                    nbr_off,
                    tag_off,
                    trip_off,
                    w_off,
                    0,
                ]),
            );
            let mut enc = self.encoder();
            Self::dispatch(&mut enc, &pipeline, &bind, (groups, 1, 1));
            self.submit(enc, "trimap_grad")?;
            let bytes = self.readback(&grad_buf, (n * dim * 4) as u64, "trimap_grad")?;
            Ok(to_f64_matrix(bytemuck::cast_slice(&bytes), n, dim))
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
        let groups = self.groups_1d(n)?;
        let p_buf = self.storage_from("p", &to_f32(p))?;
        let y_buf = self.storage_zeroed("y", (n * dim * 4) as u64);
        // scratch = rowsum[n] | z[1] | grad[n*dim]
        let grad_offset = ((n + 1) * 4) as u64;
        let scratch = self.storage_zeroed("scratch", grad_offset + (n * dim * 4) as u64);
        let params = self.uniform("tsne-params", &[0; 4]);
        let layout = self.shared_layout();
        let rowsum_p = self.pipeline_with_layout(
            "tsne_rowsum",
            || TSNE_SRC.to_string(),
            "rowsum_main",
            Some(&layout),
        )?;
        let reduce_p = self.pipeline_with_layout(
            "tsne_reduce",
            || TSNE_SRC.to_string(),
            "reduce_main",
            Some(&layout),
        )?;
        let grad_p = self.pipeline_with_layout(
            "tsne_grad",
            || TSNE_SRC.to_string(),
            "grad_main",
            Some(&layout),
        )?;
        let bufs = [&y_buf, &p_buf, &scratch, &params];
        let bind_rowsum = self.bind(&rowsum_p, &bufs);
        let bind_reduce = self.bind(&reduce_p, &bufs);
        let bind_grad = self.bind(&grad_p, &bufs);
        let mut grad_fn = |y: ArrayView2<f64>, s: &[f64]| -> Result<Array2<f64>> {
            if y.nrows() != n || y.ncols() != dim || s.len() != 1 {
                return Err("tsne: embedding shape or scalars mismatch".into());
            }
            self.queue
                .write_buffer(&y_buf, 0, bytemuck::cast_slice(&to_f32(y)));
            self.queue.write_buffer(
                &params,
                0,
                bytemuck::cast_slice(&[n as u32, dim as u32, (s[0] as f32).to_bits(), 0]),
            );
            let mut enc = self.encoder();
            Self::dispatch(&mut enc, &rowsum_p, &bind_rowsum, (groups, 1, 1));
            Self::dispatch(&mut enc, &reduce_p, &bind_reduce, (1, 1, 1));
            Self::dispatch(&mut enc, &grad_p, &bind_grad, (groups, 1, 1));
            self.submit(enc, "tsne_exact")?;
            let bytes =
                self.readback_at(&scratch, grad_offset, (n * dim * 4) as u64, "tsne_exact")?;
            Ok(to_f64_matrix(bytemuck::cast_slice(&bytes), n, dim))
        };
        run(&mut grad_fn)
    }
}

#[cfg(test)]
mod tests {
    //! These tests need a wgpu adapter. They run under lavapipe (software Vulkan) locally
    //! and in CI with `SQUEEZE_DEVICE=wgpu`; without an adapter they print a skip note.
    use super::super::cpu::CpuBackend;
    use super::super::Selection;
    use super::*;
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    fn backend() -> Option<WgpuBackend> {
        // Only probe when asked: forcing "wgpu" would otherwise pick up whatever software
        // adapter the machine has (old Mesa lavapipe builds crash on compute shaders).
        if std::env::var(super::super::DEVICE_ENV).as_deref() != Ok("wgpu") {
            eprintln!("skipping wgpu tests: set SQUEEZE_DEVICE=wgpu to run them");
            return None;
        }
        let sel = Selection::run(Some("wgpu"));
        match &sel.chosen {
            Device::Wgpu { .. } => match WgpuBackend::open(&sel.chosen) {
                Ok(b) => {
                    eprintln!("wgpu test device: {} limits {:?}", sel.chosen, b.limits());
                    Some(b)
                }
                Err(e) => {
                    eprintln!("skipping wgpu tests: {}", e);
                    None
                }
            },
            _ => {
                eprintln!("skipping wgpu tests: no adapter\n{}", sel);
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

    fn assert_close(name: &str, got: &Array2<f32>, want: &Array2<f32>, rel: f32) {
        assert_eq!(got.shape(), want.shape(), "{} shape", name);
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!(
                (g - w).abs() <= rel * scale.max(w.abs()),
                "{}: {} vs {} (scale {})",
                name,
                g,
                w,
                scale
            );
        }
    }

    fn assert_close_f64(name: &str, got: &Array2<f64>, want: &Array2<f64>, rel: f64) {
        assert_eq!(got.shape(), want.shape(), "{} shape", name);
        let scale = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-9);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!(
                (g - w).abs() <= rel * scale,
                "{}: {} vs {} (scale {})",
                name,
                g,
                w,
                scale
            );
        }
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
        let data = random_matrix(&mut rng, 600, 37);
        let queries = random_matrix(&mut rng, 50, 37);
        for metric in [Metric::Euclidean, Metric::Cosine, Metric::Manhattan] {
            let t = std::time::Instant::now();
            let (gi, gd) = gpu
                .knn_bruteforce(data.view(), queries.view(), 12, metric)
                .unwrap();
            eprintln!("wgpu knn {} 600x37, 50 queries: {:?}", metric, t.elapsed());
            let (ci, cd) = CpuBackend
                .knn_bruteforce(data.view(), queries.view(), 12, metric)
                .unwrap();
            let r = recall(&gi, &ci);
            assert!(r >= 0.999, "{} recall {}", metric, r);
            assert_close(&format!("knn {}", metric), &gd, &cd, 1e-4);
        }
    }

    #[test]
    fn knn_tiles_over_large_inputs() {
        let Some(gpu) = backend() else { return };
        // 70k rows forces two column tiles (65 536 + 4 464); 1 100 queries forces two row
        // tiles under the 256 MiB scratch budget.
        let mut rng = StdRng::seed_from_u64(2);
        let data = random_matrix(&mut rng, 70_000, 16);
        let queries = random_matrix(&mut rng, 1_100, 16);
        let t = std::time::Instant::now();
        let (gi, gd) = gpu
            .knn_bruteforce(data.view(), queries.view(), 8, Metric::Euclidean)
            .unwrap();
        eprintln!("wgpu knn 70000x16, 1100 queries: {:?}", t.elapsed());
        let (ci, cd) = CpuBackend
            .knn_bruteforce(data.view(), queries.view(), 8, Metric::Euclidean)
            .unwrap();
        assert!(recall(&gi, &ci) >= 0.999);
        assert_close("tiled knn", &gd, &cd, 1e-4);
    }

    #[test]
    fn sqdist_and_gram_match_cpu() {
        let Some(gpu) = backend() else { return };
        let mut rng = StdRng::seed_from_u64(3);
        let a = random_matrix(&mut rng, 300, 64);
        let b = random_matrix(&mut rng, 70, 64);
        let g_sq = gpu.pairwise_sqdist(a.view(), b.view()).unwrap();
        let c_sq = CpuBackend.pairwise_sqdist(a.view(), b.view()).unwrap();
        assert_close("sqdist", &g_sq, &c_sq, 1e-4);
        let g_gram = gpu.gram(a.view()).unwrap();
        let c_gram = CpuBackend.gram(a.view()).unwrap();
        assert_close("gram", &g_gram, &c_gram, 1e-4);
    }

    #[test]
    fn pacmap_steps_match_cpu() {
        let Some(gpu) = backend() else { return };
        let mut rng = StdRng::seed_from_u64(4);
        let n = 200;
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
        let yg = run(&gpu);
        let yc = run(&CpuBackend);
        assert_close_f64("pacmap 10 steps", &yg, &yc, 1e-4);
    }

    #[test]
    fn trimap_steps_match_cpu() {
        let Some(gpu) = backend() else { return };
        let mut rng = StdRng::seed_from_u64(5);
        let n = 150;
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
        let yg = run(&gpu);
        let yc = run(&CpuBackend);
        assert_close_f64("trimap 10 steps", &yg, &yc, 1e-4);
    }

    #[test]
    fn tsne_steps_match_cpu() {
        let Some(gpu) = backend() else { return };
        let mut rng = StdRng::seed_from_u64(6);
        let n = 120;
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
        let yg = run(&gpu);
        let yc = run(&CpuBackend);
        assert_close_f64("tsne 10 steps", &yg, &yc, 1e-3);
    }
}
