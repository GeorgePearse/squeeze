"""Run scripts/benchmark_gpu.py on a Modal GPU (wgpu over Vulkan on NVIDIA).

Build the wheel first (any CPython >= 3.9, abi3):

    uvx uv@0.12.23 run --no-sync maturin build --release --features extension-module \
        -o /var/tmp/squeeze-gpu/wheels

Then, with the Modal CLI authenticated:

    modal run --detach --env main scripts/modal_gpu_benchmark.py \
        --args "--devices cpu,wgpu --datasets digits,fashion-mnist --fashion-samples 2000"
    modal volume get squeeze-gpu-benchmark <run-id> /var/tmp/squeeze-gpu/bench-t4

Environment knobs: ``SQUEEZE_MODAL_GPU`` (default ``T4``; ``L4`` also works),
``SQUEEZE_WHEEL_DIR`` (default ``/var/tmp/squeeze-gpu/wheels``). One container at a time.
"""

# ruff: noqa: PLC0415, T201, S603, S607, E501, D103, ANN001, ANN201, S108

from __future__ import annotations

import os
import pathlib
import time

import modal

GPU = os.environ.get("SQUEEZE_MODAL_GPU", "T4")
WHEEL_DIR = pathlib.Path(os.environ.get("SQUEEZE_WHEEL_DIR", "/var/tmp/squeeze-gpu/wheels"))
SCRIPTS = pathlib.Path(__file__).resolve().parent

wheels = sorted(WHEEL_DIR.glob("squeeze-*.whl"))
if not wheels:
    message = f"no squeeze wheel in {WHEEL_DIR}; build one with maturin first"
    raise SystemExit(message)
WHEEL = wheels[-1]

# The NVIDIA Vulkan ICD points at the driver's GLX library, which the container runtime
# mounts when the driver capabilities include "graphics" (hence NVIDIA_DRIVER_CAPABILITIES=all).
NVIDIA_ICD = '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0"}}'

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libvulkan1", "vulkan-tools", "libgomp1")
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all", "PYTHONUNBUFFERED": "1"})
    .run_commands(
        "mkdir -p /usr/share/vulkan/icd.d",
        f"printf '%s' '{NVIDIA_ICD}' > /usr/share/vulkan/icd.d/nvidia_icd.json",
    )
    .pip_install(
        "numpy>2",
        "scipy>=1.3.1",
        "scikit-learn>=1.6",
        "numba>=0.51.2",
        "pynndescent>=0.5",
        "tqdm",
        "matplotlib",
        "pandas",
    )
    .add_local_file(WHEEL, f"/wheels/{WHEEL.name}", copy=True)
    .run_commands(f"pip install /wheels/{WHEEL.name}")
    .add_local_file(SCRIPTS / "benchmark_gpu.py", "/root/scripts/benchmark_gpu.py")
    .add_local_file(SCRIPTS / "benchmark_datasets.py", "/root/scripts/benchmark_datasets.py")
)

app = modal.App("squeeze-gpu-benchmark")
results = modal.Volume.from_name("squeeze-gpu-benchmark", create_if_missing=True)


@app.function(
    gpu=GPU,
    cpu=8.0,
    memory=32768,
    image=image,
    volumes={"/results": results},
    timeout=4 * 3600,
    max_containers=1,
)
def run(args: str, run_id: str) -> str:
    import subprocess
    import sys

    out = pathlib.Path("/results") / run_id
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "run.log").open("w")

    def sh(cmd: list[str], **kw) -> None:
        print("$", " ".join(cmd), flush=True)
        log.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, **kw)
        print(proc.stdout[-6000:], proc.stderr[-3000:], flush=True)
        log.write(proc.stdout + proc.stderr + "\n")
        log.flush()

    sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"])
    sh(["vulkaninfo", "--summary"])
    sh([sys.executable, "-c", "import squeeze; print(squeeze.devices()); print(squeeze.default_device())"])
    sh([sys.executable, "/root/scripts/benchmark_gpu.py", *args.split(), "--out", str(out)])
    log.close()
    results.commit()
    return (out / "results.md").read_text() if (out / "results.md").exists() else "no results.md"


@app.local_entrypoint()
def main(args: str = "--devices cpu,wgpu --datasets digits", run_id: str = "") -> None:
    run_id = run_id or f"{GPU.lower()}-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"run id: {run_id}  gpu: {GPU}  wheel: {WHEEL.name}")
    print(run.remote(args, run_id))
    print(f"fetch with: modal volume get squeeze-gpu-benchmark {run_id} /var/tmp/squeeze-gpu/bench-{run_id}")
