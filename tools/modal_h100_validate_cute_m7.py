"""Run the optional M7 CuTe DSL SwiGLU lab on one Modal H100."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import modal

from tools.modal_l4_validate import hf_cache_volume
from tools.modal_l4_validate_m7 import (
    ARTIFACT_DIRECTORY,
    _benchmark_cuda,
    m7_image,
)

APP_NAME = "forge-engine-m7-cute-h100-validation"
CUTE_ARTIFACT_DIRECTORY = ARTIFACT_DIRECTORY / "cute"

cute_image = (
    m7_image.run_commands('python -m pip install "/repo[cute]"')
    .env(
        {
            "CUTE_DSL_DUMP_DIR": str(CUTE_ARTIFACT_DIRECTORY),
            "CUTE_DSL_KEEP": "ptx,cubin",
        }
    )
)

app = modal.App(APP_NAME)


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
def validate_cute_m7() -> dict[str, object]:
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
    print(f"M7_CUTE_SOFTWARE={software}", flush=True)
    CUTE_ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(731)
    inputs = (
        torch.randn((1, 2_560), device=device, dtype=torch.bfloat16)
        * 0.02
    )
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
    print(f"M7_CUTE_BENCHMARK={benchmark}", flush=True)
    print(f"M7_CUTE_PTX_SASS={artifacts}", flush=True)
    return {
        "software": software,
        "benchmark": benchmark,
        "artifacts": artifacts,
    }


@app.local_entrypoint()
def main() -> None:
    """Invoke the optional H100 validator and print retained evidence."""
    result = validate_cute_m7.remote()
    print("M7_CUTE_H100_ACCEPTANCE=PASS")
    for name, value in result.items():
        print(f"{name}={value}")
