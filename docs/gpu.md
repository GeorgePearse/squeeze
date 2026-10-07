# GPU compute

Squeeze runs on the CPU everywhere. When the machine has a GPU, the heavy kernels run on
it without any change to your code: pairwise distances, exact k-nearest neighbours, and
the embedding gradients of PaCMAP, TriMap and exact t-SNE. The CPU path stays the
numerical reference and the fallback.

```python
import squeeze

print(squeeze.devices())          # what was probed, what was chosen, why the rest was not
print(squeeze.default_device())   # what device="auto" resolves to

emb = squeeze.PaCMAP(n_components=2, random_state=42).fit_transform(X)            # auto
emb = squeeze.TSNE(n_components=2, random_state=42, device="cpu").fit_transform(X)  # force CPU
emb = squeeze.UMAP(n_neighbors=15, device="gpu").fit_transform(X)                  # any GPU, else warn + CPU
```

## Backends

| Backend | Where | How | Feature flag |
|---|---|---|---|
| `cpu` | everywhere | SIMD (AVX2/NEON) distances, rayon-parallel kernels, `f64` state | always on |
| `wgpu` | Linux, Windows, macOS, Android | WGSL compute shaders through [wgpu](https://wgpu.rs): Vulkan, Metal or DX12 | `gpu-wgpu` (default on) |
| `mlx` | macOS on Apple Silicon | [MLX](https://github.com/ml-explore/mlx) array programs through `mlx-rs` | `gpu-mlx` |
| `cuda` | NVIDIA (Linux, Windows) | CUDA C twins of the WGSL kernels, compiled to PTX and JIT-built by the driver through [cudarc](https://github.com/coreylowman/cudarc); needs only the NVIDIA driver, no CUDA toolkit | `gpu-cuda` |

GPU kernels work in `f32`. The algorithms keep their `f64` state and convert at the
boundary once per call; reductions on the CPU keep `f64` accumulation. Everything
compiles and all tests pass with `--no-default-features` (CPU only).

## Device selection

`device="auto"` (the default) resolves once per process, in this order:

1. `SQUEEZE_DEVICE` environment variable, if set (`cpu`, `gpu`, `wgpu`, `mlx`, `cuda`, `auto`).
2. MLX, on macOS aarch64 when built with `gpu-mlx`.
3. CUDA, when built with `gpu-cuda` and `libcuda.so.1` is present (so on a machine with both a
   Vulkan ICD and CUDA the CUDA backend wins; sandboxed GPU containers such as Modal's expose
   CUDA but no working Vulkan driver, which is why this backend exists).
4. A wgpu adapter of type *discrete GPU*, then *integrated GPU*. Vulkan, Metal and DX12
   adapters are preferred over OpenGL.
5. CPU.

Software and virtual Vulkan adapters (Mesa lavapipe, SwiftShader, VM adapters) are never
picked by `auto`; set `SQUEEZE_DEVICE=wgpu` to use them. That is how the GPU code paths
run on GPU-less Linux CI. `WGPU_BACKEND=vulkan|metal|dx12|gl` restricts the APIs wgpu
enumerates.

`device=` accepts the same strings on every Rust reducer except PCA (whose work is a
d x d covariance), on `HnswIndexWrapper`, and on `squeeze.UMAP`. An explicit device that is
not available warns and falls back to the CPU; an unknown name raises `ValueError`.

`auto` is also size-aware, because a GPU only pays for the transfers above a certain
input size (measured on a T4, see the benchmarks below):

| work | GPU from | override |
|---|---:|---|
| pairwise distance matrix, brute-force k-NN | 4 096 rows | `SQUEEZE_GPU_MIN_ROWS` |
| exact t-SNE gradient | 1 000 rows | `SQUEEZE_GPU_MIN_ROWS_TSNE` |
| PaCMAP / TriMap gradient steps | 100 000 rows | `SQUEEZE_GPU_MIN_ROWS_STEP` |

Below the threshold `auto` keeps that work on the CPU even when a GPU is present. An
explicit `device="gpu"` (or `SQUEEZE_DEVICE`) always uses the device.

`squeeze.devices()` prints the probe report. On a Linux box with only the Mesa software
driver it reads:

```
squeeze compute devices
  SQUEEZE_DEVICE unset (auto)
  chosen: cpu
  mlx   rejected   MLX needs macOS on Apple Silicon
  cuda  rejected   built without the gpu-cuda feature
  wgpu  rejected   only software/virtual adapters: llvmpipe (LLVM 11.0.1, 256 bits) (vulkan, cpu); set SQUEEZE_DEVICE=wgpu to use them
  cpu   chosen (fallback) cpu
```

## What runs where

| Work | CPU | GPU |
|---|---|---|
| k-NN for UMAP and `HnswIndexWrapper` | HNSW graph (approximate) | exact brute force for euclidean, cosine and manhattan, up to 500 000 rows (`SQUEEZE_BRUTEFORCE_MAX_ROWS` overrides); other metrics, sparse input, filter masks and larger inputs use the CPU graph |
| Pairwise distance matrix (MDS, Isomap, LLE, PHATE, TriMap, PaCMAP, t-SNE) | SIMD, rayon | tiled kernel through a scratch buffer sized from the adapter limits |
| PaCMAP and TriMap optimisation | per-point gather over the pair/triplet adjacency | same gather kernel; the pair structure is uploaded once per fit |
| t-SNE gradient | Barnes-Hut above 1 000 samples in 2-D, exact below | exact `O(n²)` gradient up to 20 000 samples (Barnes-Hut stays on the CPU above, or when `use_barnes_hut=True`) |
| PCA, spectral solvers, UMAP layout | CPU | CPU |

Any GPU initialisation or kernel failure (device lost, out of memory, buffer over the
adapter's binding limit) produces one `RuntimeWarning` naming the device and the error
and the call completes on the CPU. The fallback path never raises.

## Correctness

Every kernel is tested against the CPU reference on random data: k-NN indices identical
up to distance ties (recall >= 0.999) with distances within 1e-4 relative, squared
distances and Gram matrices within 1e-4, and ten optimisation steps of PaCMAP, TriMap and
exact t-SNE from the same start within 1e-4 (1e-3 for t-SNE). End to end on sklearn
Digits, trustworthiness at k = 15 of every reducer on a GPU is within 0.01 of the CPU
result with the same seeds (`squeeze/tests/test_devices.py`).

The Linux CI job runs the Rust and Python GPU tests under Mesa lavapipe (software
Vulkan); the macOS 15 job builds with `gpu-mlx` and runs them on MLX and on Metal through
wgpu. Lavapipe older than Mesa 23 is not supported (Debian 11's Mesa 20.3 crashes on
compute shaders); `auto` never selects it anyway.

## Benchmarks

All runs use the same seeds on every device; trustworthiness (T) is scikit-learn's at
k = 15; time is the whole `fit_transform`. The CPU and GPU runs of one table share the
machine, so the speedup is like for like.

### NVIDIA Tesla T4 (Modal container, 8 vCPU, CUDA backend, driver 610.57.04)

Digits (1 797 x 64):

| algorithm | CPU (s) | CUDA (s) | speedup | T CPU | T CUDA |
|---|---:|---:|---:|---:|---:|
| UMAP | 21.44 | 20.83 | 1.03x | 0.9871 | 0.9871 |
| t-SNE (exact on GPU, Barnes-Hut on CPU) | 8.97 | 6.02 | 1.49x | 0.9897 | 0.9892 |
| MDS | 15.38 | 16.36 | 0.94x | 0.8962 | 0.8962 |
| Isomap | 7.32 | 8.22 | 0.89x | 0.8347 | 0.8347 |
| LLE | 18.55 | 36.54 | 0.51x | 0.9141 | 0.9141 |
| PHATE | 10.31 | 11.17 | 0.92x | 0.8279 | 0.8279 |
| TriMap | 0.83 | 2.56 | 0.33x | 0.5027 | 0.5027 |
| PaCMAP | 0.41 | 0.90 | 0.45x | 0.9804 | 0.9804 |

Fashion-MNIST, 2 000-image stratified sample of the official test split (2 000 x 784):

| algorithm | CPU (s) | CUDA (s) | speedup | T CPU | T CUDA |
|---|---:|---:|---:|---:|---:|
| UMAP | 24.17 | 23.80 | 1.02x | 0.9747 | 0.9747 |
| t-SNE | 11.03 | 3.40 | 3.24x | 0.9805 | 0.9800 |
| MDS | 15.85 | 16.37 | 0.97x | 0.9253 | 0.9253 |
| Isomap | 10.64 | 11.53 | 0.92x | 0.9224 | 0.9224 |
| LLE | 22.28 | 25.20 | 0.88x | 0.9117 | 0.9117 |
| PHATE | 13.79 | 15.38 | 0.90x | 0.9056 | 0.9056 |
| TriMap | 0.90 | 4.40 | 0.20x | 0.5021 | 0.5021 |
| PaCMAP | 0.49 | 1.37 | 0.35x | 0.9677 | 0.9679 |

Larger inputs (UMAP on the 10 000-image Fashion-MNIST sample; k-NN alone, k = 15, on
synthetic Gaussian data; "recall" is against exact search on a 200-row sample):

| workload | CPU (s) | CUDA (s) | speedup | quality CPU | quality CUDA |
|---|---:|---:|---:|---:|---:|
| UMAP, Fashion-MNIST 10 000 x 784 | 76.37 | 72.00 | 1.06x | T 0.9780 | T 0.9780 |
| k-NN, 100 000 x 128 (HNSW on CPU, brute force on GPU) | 282.25 | 16.62 | 17.0x | recall 0.59 | recall 1.00 |
| k-NN, 1 000 000 x 128 (brute force on GPU) | KNN1M_CPU | KNN1M_GPU | KNN1M_SPEED | KNN1M_QCPU | KNN1M_QGPU |

The pattern across both datasets: the exact t-SNE gradient is the one per-iteration kernel
that wins at these sizes (n² work per step); the pairwise distance matrix is a small part of
the quadratic algorithms, whose time goes to SMACOF, eigensolvers, Dijkstra and power
iterations on the CPU, so moving it is neutral; and the PaCMAP / TriMap steps are
launch-latency bound (450-800 round trips over ~50k pairs each), so they lose. Brute-force
k-NN is both much faster and exact once the input has tens of thousands of rows. These
measurements set the `auto` thresholds above. The LLE slowdown on Digits (0.51x) was not
reproduced on Fashion-MNIST (0.88x) and is noted, not explained.

### Apple Silicon (GitHub `macos-15` runner): MLX versus Metal through wgpu

MACOS_PLACEHOLDER

### Software Vulkan (Mesa lavapipe)

Correctness only: a software rasteriser, so no speed claims. All kernel tests and the
end-to-end device tests pass under lavapipe on this box (Mesa 26.2.4) and on the Ubuntu CI
runner (Mesa 24). It is how the GPU code paths are exercised without a GPU.

### Cost

MODAL_COST_PLACEHOLDER

Benchmark driver: `scripts/benchmark_gpu.py` (per-job subprocesses, JSON lines, Markdown
table and plot); `scripts/modal_gpu_benchmark.py` runs it on a Modal GPU.

## Building

```bash
# default: CPU + wgpu
uv run --no-sync maturin develop --release --features extension-module
# Apple Silicon with MLX as well
uv run --no-sync maturin develop --release --features extension-module,gpu-mlx
# NVIDIA with the CUDA backend as well (the embedded PTX targets compute_75 and newer)
uv run --no-sync maturin develop --release --features extension-module,gpu-cuda
# CPU only
uv run --no-sync maturin develop --release --features extension-module --no-default-features
```

Runtime requirements: a Vulkan loader and driver on Linux (`libvulkan1` plus the vendor
ICD; `mesa-vulkan-drivers` for lavapipe), nothing extra on macOS (Metal) or Windows
(DX12). Python-level GPU libraries (torch, cupy, jax, mlx-python) are not used.

## Design notes

The compute layer is `src/compute/` and has no PyO3 types, so it survives a Rust-only
refactor. It exposes four public types:

- `Device`: `Cpu | Wgpu { name, backend, device_type } | Mlx { name } | Cuda { name, ordinal }`
  with `Device::auto()`, `Device::probe()`, `Device::resolve(request)` and `Device::backend()`.
- `Selection`: the probe report (`requested`, `chosen`, `candidates`), printed by `squeeze.devices()`.
- `Metric`: the metrics the GPU k-NN supports (`Euclidean`, `Cosine`, `Manhattan`).
- `Backend`: the kernel trait every backend implements over `ndarray` views:
  `knn_bruteforce`, `pairwise_sqdist`, `gram`, and the gradient *sessions*
  `pacmap_session`, `trimap_session`, `tsne_exact_session`. A session uploads the constant
  structure (pairs, triplets, the P matrix) once and hands the algorithm a gradient
  closure, so the optimisation loop stays in the algorithm and nothing is re-uploaded per
  iteration.

The wgpu kernels use at most three storage bindings each so minimal Vulkan
implementations work, and tile pairwise work through a scratch buffer bounded by the
adapter's `max_storage_buffer_binding_size` (256 MiB cap). The CUDA kernels
(`src/compute/kernels/squeeze.cu`) mirror them one to one and are regenerated with
`nvcc -arch=compute_75 -ptx -O3` into `squeeze.ptx` (`SQUEEZE_CUDA_PTX=<file>` loads
another build at runtime).
