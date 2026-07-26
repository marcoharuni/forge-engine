"""Run the optional CuTe DSL SwiGLU lab on one H100."""

from __future__ import annotations

import re
import statistics
import subprocess
from collections.abc import Callable
from pathlib import Path

import modal

APP_NAME = "forge-engine-cute-h100-validation"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPOSITORY = Path("/repo")
HF_HOME = Path("/cache/huggingface")
CUTE_ARTIFACT_DIRECTORY = Path("/tmp/forge-cute-h100-artifacts")
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 50
REPOSITORY_IGNORES = [
    ".git",
    ".git/**",
    ".venv",
    ".venv/**",
    ".agents",
    ".agents/**",
    ".codex",
    ".codex/**",
    "**/__pycache__/**",
    "**/.pytest_cache/**",
    "**/target/**",
    "*.safetensors",
    "models/**",
]

cute_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.3-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("binutils", "ninja-build")
    .add_local_dir(
        REPOSITORY_ROOT,
        remote_path=str(REMOTE_REPOSITORY),
        copy=True,
        ignore=REPOSITORY_IGNORES,
    )
    .run_commands('python -m pip install "/repo[dev,cute]"')
    .env(
        {
            "HF_HOME": str(HF_HOME),
            "MAX_JOBS": "2",
            "TORCH_CUDA_ARCH_LIST": "9.0+PTX",
            "CUTE_DSL_DUMP_DIR": str(CUTE_ARTIFACT_DIRECTORY),
            "CUTE_DSL_KEEP": "ptx,cubin",
        }
    )
    .workdir(REMOTE_REPOSITORY)
)

app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name(
    "forge-engine-hf-cache",
    create_if_missing=True,
)


def _benchmark_cuda(
    torch: object,
    operation: Callable[[], object],
    *,
    work_items: int,
    throughput_unit: str,
) -> dict[str, object]:
    """Measure one CUDA operation with events and synchronization."""
    with torch.inference_mode():
        for _ in range(WARMUP_ITERATIONS):
            operation()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = int(torch.cuda.memory_allocated())
        starts = [
            torch.cuda.Event(enable_timing=True) for _ in range(BENCHMARK_ITERATIONS)
        ]
        ends = [
            torch.cuda.Event(enable_timing=True) for _ in range(BENCHMARK_ITERATIONS)
        ]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            operation()
            end.record()
        torch.cuda.synchronize()
    elapsed_ms = [
        float(start.elapsed_time(end)) for start, end in zip(starts, ends, strict=True)
    ]
    ordered = sorted(elapsed_ms)
    median_ms = float(statistics.median(ordered))
    peak_memory = int(torch.cuda.max_memory_allocated())
    return {
        "warmup": WARMUP_ITERATIONS,
        "iterations": BENCHMARK_ITERATIONS,
        "synchronized": True,
        "latency_ms": {
            "minimum": ordered[0],
            "p10": ordered[int(0.10 * (len(ordered) - 1))],
            "median": median_ms,
            "p90": ordered[int(0.90 * (len(ordered) - 1))],
            "maximum": ordered[-1],
        },
        "throughput": work_items / (median_ms / 1_000.0),
        "throughput_unit": throughput_unit,
        "allocated_before_bytes": allocated_before,
        "peak_memory_bytes": peak_memory,
        "peak_increment_bytes": peak_memory - allocated_before,
    }


def _artifact_counts() -> dict[str, object]:
    """Read the generated CuTe PTX, cubin, and SASS evidence."""
    artifacts = {
        suffix: sorted(CUTE_ARTIFACT_DIRECTORY.rglob(f"*{suffix}"))
        for suffix in (".ptx", ".cubin")
    }
    missing = [suffix for suffix, paths in artifacts.items() if not paths]
    if missing:
        raise RuntimeError(f"CuTe did not retain artifacts for {missing}")
    ptx = "\n".join(path.read_text(errors="replace") for path in artifacts[".ptx"])
    sass_parts: list[str] = []
    sass_paths: list[Path] = []
    for cubin_path in artifacts[".cubin"]:
        completed = subprocess.run(
            ["nvdisasm", str(cubin_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        sass_parts.append(completed.stdout)
        sass_path = cubin_path.with_suffix(".sass")
        sass_path.write_text(completed.stdout)
        sass_paths.append(sass_path)
    sass = "\n".join(sass_parts)
    if ".entry" not in ptx:
        raise RuntimeError("CuTe PTX contains no kernel entry")
    if len(sass) < 100:
        raise RuntimeError("CuTe SASS output is empty")
    return {
        "ptx_files": tuple(str(path) for path in artifacts[".ptx"]),
        "cubin_files": tuple(str(path) for path in artifacts[".cubin"]),
        "sass_files": tuple(str(path) for path in sass_paths),
        "ptx_entries": ptx.count(".entry"),
        "ptx_global_loads": ptx.count("ld.global"),
        "ptx_global_stores": ptx.count("st.global"),
        "sass_instructions": len(
            re.findall(r"/\*[0-9a-fA-F]+\*/\s+([A-Z][A-Z0-9.]*)", sass)
        ),
        "sass_global_loads": sass.count("LDG"),
        "sass_global_stores": sass.count("STG"),
    }


@app.function(
    image=cute_image,
    gpu="H100",
    timeout=30 * 60,
    volumes={"/cache": hf_cache_volume},
)
def validate_cute_h100() -> dict[str, object]:
    """Validate the exact Qwen-sized fused CuTe lab and its artifacts."""
    import cutlass
    import torch

    from forge_engine.kernels.cute_swiglu import (
        cute_gate_up_swiglu_lab,
        gate_up_swiglu_reference,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one GPU, found {torch.cuda.device_count()}"
        )
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name:
        raise RuntimeError(f"expected an H100 GPU, found {device_name!r}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CuTe lab requires BF16 support")
    nvcc = subprocess.run(
        ["nvcc", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[-1]
    software = {
        "gpu": device_name,
        "compute_capability": ".".join(
            str(part) for part in torch.cuda.get_device_capability(0)
        ),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutlass": getattr(cutlass, "__version__", "unknown"),
        "nvcc": nvcc,
        "precision": "bfloat16",
    }
    print(f"CUTE_H100_SOFTWARE={software}", flush=True)
    CUTE_ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(731)
    inputs = torch.randn((1, 2_560), device=device, dtype=torch.bfloat16) * 0.02
    gate_weight = (
        torch.randn(
            (9_728, 2_560),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    up_weight = (
        torch.randn(
            (9_728, 2_560),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    with torch.inference_mode():
        expected = gate_up_swiglu_reference(
            inputs,
            gate_weight,
            up_weight,
        )
        actual = cute_gate_up_swiglu_lab(
            inputs,
            gate_weight,
            up_weight,
            require_cute=True,
        )
        torch.cuda.synchronize()
    maximum_absolute_error = float(
        (actual.float() - expected.float()).abs().max().item()
    )
    if not torch.allclose(actual, expected, atol=0.03, rtol=0.10):
        raise AssertionError(
            "CuTe/reference mismatch: "
            f"maximum_absolute_error={maximum_absolute_error:.8f}"
        )

    benchmark = {
        "workload": (
            "tokens=1, hidden=2560, intermediate=9728, BF16, "
            "fused gate/up plus SiLU product"
        ),
        "custom": _benchmark_cuda(
            torch,
            lambda: cute_gate_up_swiglu_lab(
                inputs,
                gate_weight,
                up_weight,
                require_cute=True,
            ),
            work_items=1,
            throughput_unit="tokens/s",
        ),
        "reference": _benchmark_cuda(
            torch,
            lambda: gate_up_swiglu_reference(
                inputs,
                gate_weight,
                up_weight,
            ),
            work_items=1,
            throughput_unit="tokens/s",
        ),
        "maximum_absolute_error": maximum_absolute_error,
    }
    artifacts = _artifact_counts()
    print(f"CUTE_H100_BENCHMARK={benchmark}", flush=True)
    print(f"CUTE_H100_PTX_SASS={artifacts}", flush=True)
    return {
        "software": software,
        "benchmark": benchmark,
        "artifacts": artifacts,
    }


@app.local_entrypoint()
def main() -> None:
    """Invoke the optional H100 validator and print retained evidence."""
    result = validate_cute_h100.remote()
    print("CUTE_H100_ACCEPTANCE=PASS")
    for name, value in result.items():
        print(f"{name}={value}")
