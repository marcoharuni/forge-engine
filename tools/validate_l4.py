"""Run consolidated correctness, kernel, and serving checks on one L4."""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import modal

from forge_engine.config import DEFAULT_MODEL_ID, SUPPORTED_MODEL_REVISION

APP_NAME = "forge-engine-l4-validation"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPOSITORY = Path("/repo")
HF_HOME = Path("/cache/huggingface")
PROMPT = "Write the first ten positive integers separated by spaces."
EXPECTED_PREFIX = "1 2 3 4 "
DECODE_STEPS = 8
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 50
PROBABILITY_TV_LIMIT = 0.05
ARTIFACT_DIRECTORY = Path("/tmp/forge-l4-artifacts")
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
RUST_CLIENT = (
    REMOTE_REPOSITORY / "rust" / "streamer" / "target" / "release" / "forge-streamer"
)
SERVING_PORT = 8765
SERVING_REQUESTS = 8
SERVING_CONCURRENCY = 4
SERVING_CANCEL_EVERY = 4

l4_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.3-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install(
        "binutils",
        "build-essential",
        "ca-certificates",
        "curl",
        "ninja-build",
    )
    .add_local_dir(
        REPOSITORY_ROOT,
        remote_path=str(REMOTE_REPOSITORY),
        copy=True,
        ignore=REPOSITORY_IGNORES,
    )
    .run_commands('python -m pip install "/repo[dev]"')
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf "
        "https://sh.rustup.rs -o /tmp/rustup-init.sh",
        "sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain stable",
        "/root/.cargo/bin/cargo build --release --locked "
        "--manifest-path /repo/rust/streamer/Cargo.toml",
    )
    .env(
        {
            "HF_HOME": str(HF_HOME),
            "MAX_JOBS": "2",
            "TORCH_CUDA_ARCH_LIST": "8.9+PTX",
            "TRITON_DUMP_DIR": str(ARTIFACT_DIRECTORY / "triton"),
            "TRITON_KERNEL_DUMP": "1",
        }
    )
    .workdir(REMOTE_REPOSITORY)
)

app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name(
    "forge-engine-hf-cache",
    create_if_missing=True,
)


def _record(
    condition: bool,
    message: str,
    failures: list[str],
) -> None:
    """Collect one failure so independent L4 phases can still run."""
    if not condition:
        failures.append(message)
        print(f"L4_CHECK_FAILURE={message}", flush=True)


def _parse_metrics(text: str) -> dict[str, float]:
    """Parse numeric Prometheus samples."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.rsplit(" ", 1)
        values[name] = float(value)
    return values


def _tool_version(command: list[str]) -> str:
    """Return the final non-empty version line from a system tool."""
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [
        line.strip()
        for line in (completed.stdout + completed.stderr).splitlines()
        if line.strip()
    ]
    return lines[-1]


def _benchmark_cuda(
    torch: object,
    operation: Callable[[], object],
    *,
    work_items: int,
    throughput_unit: str,
) -> dict[str, object]:
    """Measure one CUDA operation with events and explicit synchronization."""
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


def _cuobjdump(path: Path, mode: str) -> str:
    """Read PTX or SASS from one CUDA binary using the toolkit."""
    completed = subprocess.run(
        ["cuobjdump", mode, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout + completed.stderr


def _assembly_counts(ptx: str, sass: str) -> dict[str, int]:
    """Count stable, reviewable features without asserting optimization."""
    return {
        "ptx_entries": ptx.count(".entry"),
        "ptx_global_loads": ptx.count("ld.global"),
        "ptx_global_stores": ptx.count("st.global"),
        "ptx_barriers": ptx.count("bar.sync"),
        "sass_instructions": len(
            re.findall(r"/\*[0-9a-fA-F]+\*/\s+([A-Z][A-Z0-9.]*)", sass)
        ),
        "sass_global_loads": sass.count("LDG"),
        "sass_global_stores": sass.count("STG"),
        "sass_barriers": sass.count("BAR"),
    }


def _inspect_artifacts(failures: list[str]) -> dict[str, object]:
    """Inspect retained Triton artifacts and the compiled CUDA extension."""
    from forge_engine._triton_kernels import triton_kernel_artifacts
    from forge_engine.kernels.cuda_paged import cuda_extension_path

    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    observations: dict[str, object] = {}
    triton_artifacts = triton_kernel_artifacts()
    required = {
        "residual_add_rms_norm",
        "restricted_causal_prefill",
    }
    _record(
        required <= triton_artifacts.keys(),
        f"missing Triton artifacts: {sorted(required - triton_artifacts.keys())}",
        failures,
    )
    for name in sorted(required & triton_artifacts.keys()):
        artifact = triton_artifacts[name]
        ptx_value = artifact.get("ptx", "")
        ptx = (
            ptx_value.decode(errors="replace")
            if isinstance(ptx_value, bytes)
            else str(ptx_value)
        )
        cubin_value = artifact.get("cubin")
        _record(
            ".entry" in ptx,
            f"{name} generated PTX has no entry",
            failures,
        )
        _record(
            isinstance(cubin_value, bytes) and len(cubin_value) > 0,
            f"{name} generated cubin is missing",
            failures,
        )
        sass = ""
        if isinstance(cubin_value, bytes) and cubin_value:
            cubin_path = ARTIFACT_DIRECTORY / f"{name}.cubin"
            cubin_path.write_bytes(cubin_value)
            sass = _cuobjdump(cubin_path, "--dump-sass")
            _record(
                len(sass) > 100,
                f"{name} cuobjdump SASS was empty",
                failures,
            )
        observations[name] = _assembly_counts(ptx, sass)

    extension = cuda_extension_path()
    _record(
        extension is not None,
        "CUDA paged extension path was not retained",
        failures,
    )
    if extension is not None:
        extension_path = Path(extension)
        ptx = _cuobjdump(extension_path, "--dump-ptx")
        sass = _cuobjdump(extension_path, "--dump-sass")
        _record(
            ".entry" in ptx and "paged_gqa_kernel" in ptx,
            "CUDA paged extension PTX entry was not found",
            failures,
        )
        _record(
            len(sass) > 100 and "paged_gqa_kernel" in sass,
            "CUDA paged extension SASS function was not found",
            failures,
        )
        observations["cuda_paged_gqa"] = _assembly_counts(ptx, sass)
    return observations


def _probability_tv(torch: object, left: object, right: object) -> float:
    """Return maximum total variation between token distributions."""
    left_probabilities = torch.softmax(left.float(), dim=-1)
    right_probabilities = torch.softmax(right.float(), dim=-1)
    return float(
        (0.5 * (left_probabilities - right_probabilities).abs().sum(dim=-1))
        .max()
        .item()
    )


def _validate_serving(failures: list[str]) -> dict[str, object]:
    """Validate the final HTTP path with the external Rust client."""
    import httpx

    server_command = [
        "forge-engine",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVING_PORT),
        "--max-new-tokens",
        "32",
        "--max-requests",
        "8",
        "--max-batch-size",
        "4",
        "--token-budget",
        "192",
        "--block-capacity",
        "128",
    ]
    print(f"L4_SERVER_COMMAND={' '.join(server_command)}", flush=True)
    server = subprocess.Popen(
        server_command,
        cwd=REMOTE_REPOSITORY,
        stderr=subprocess.STDOUT,
    )
    base_url = f"http://127.0.0.1:{SERVING_PORT}"
    warmup_text = ""
    report: dict[str, object] = {}
    benchmark: dict[str, object] = {}
    try:
        deadline = time.monotonic() + 180.0
        with httpx.Client(
            base_url=base_url,
            timeout=2.0,
            trust_env=False,
        ) as client:
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(
                        f"server exited during startup with {server.returncode}"
                    )
                try:
                    response = client.get("/health")
                    if response.status_code == 200:
                        health = response.json()
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.25)
            else:
                raise RuntimeError("server did not become healthy in 180s")

        _record(
            health.get("model") == DEFAULT_MODEL_ID
            and health.get("revision") == SUPPORTED_MODEL_REVISION,
            f"unexpected health identity: {health!r}",
            failures,
        )

        warmup = subprocess.run(
            [
                str(RUST_CLIENT),
                "chat",
                "--base-url",
                base_url,
                "--prompt",
                PROMPT,
                "--max-tokens",
                "8",
            ],
            cwd=REMOTE_REPOSITORY,
            text=True,
            capture_output=True,
            timeout=180.0,
            check=False,
        )
        print(warmup.stderr, end="", flush=True)
        # The deterministic output intentionally ends with a space. Remove only
        # the line ending written by the CLI, not semantic model whitespace.
        warmup_text = warmup.stdout.rstrip("\r\n")
        _record(
            warmup.returncode == 0 and warmup_text.startswith(EXPECTED_PREFIX),
            (
                f"warm-up chat exit={warmup.returncode}, "
                f"output={warmup_text!r}, stderr={warmup.stderr!r}"
            ),
            failures,
        )

        with httpx.Client(
            base_url=base_url,
            timeout=10.0,
            trust_env=False,
        ) as client:
            before = _parse_metrics(client.get("/metrics").text)

        load = subprocess.Popen(
            [
                str(RUST_CLIENT),
                "load",
                "--base-url",
                base_url,
                "--prompt",
                ("Write the first one hundred positive integers separated by spaces."),
                "--max-tokens",
                "32",
                "--requests",
                str(SERVING_REQUESTS),
                "--concurrency",
                str(SERVING_CONCURRENCY),
                "--cancel-every",
                str(SERVING_CANCEL_EVERY),
                "--cancel-after-events",
                "0",
                "--cancel-max-tokens",
                "512",
                "--metrics-timeout-seconds",
                "30",
            ],
            cwd=REMOTE_REPOSITORY,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        max_active = 0
        poll_failures = 0
        poll_started = time.monotonic()
        with httpx.Client(
            base_url=base_url,
            timeout=2.0,
            trust_env=False,
        ) as client:
            while load.poll() is None:
                try:
                    current = _parse_metrics(client.get("/metrics").text)
                    max_active = max(
                        max_active,
                        int(current.get("forge_requests_active", 0.0)),
                    )
                except httpx.HTTPError:
                    poll_failures += 1
                if time.monotonic() - poll_started > 300.0:
                    load.kill()
                    raise RuntimeError("Rust load exceeded 300 seconds")
                time.sleep(0.02)
        stdout, stderr = load.communicate(timeout=10.0)
        print(stderr, end="", flush=True)
        _record(
            load.returncode == 0,
            f"Rust load exited {load.returncode}: {stderr!r}",
            failures,
        )
        try:
            report = json.loads(stdout)
        except json.JSONDecodeError as error:
            failures.append(f"Rust load emitted invalid JSON: {error}")
            report = {}

        with httpx.Client(
            base_url=base_url,
            timeout=10.0,
            trust_env=False,
        ) as client:
            after = _parse_metrics(client.get("/metrics").text)
            final_health = client.get("/health").json()

        client_report = report.get("client", {})
        server_delta = report.get("server_delta", {})
        agreement = report.get("agreement", {})
        results = report.get("results", [])
        expected_cancelled = SERVING_REQUESTS // SERVING_CANCEL_EVERY
        expected_finished = SERVING_REQUESTS - expected_cancelled
        finished_outputs = [
            result.get("text", "")
            for result in results
            if result.get("status") == "finished"
        ]
        _record(
            client_report.get("finished") == expected_finished
            and client_report.get("cancelled") == expected_cancelled
            and client_report.get("failed") == 0,
            f"unexpected client result counts: {client_report!r}",
            failures,
        )
        _record(
            agreement.get("passed") is True,
            f"client/server metrics disagreed: {agreement!r}",
            failures,
        )
        _record(
            len(finished_outputs) == expected_finished
            and all(output.startswith(EXPECTED_PREFIX) for output in finished_outputs),
            f"finished outputs failed deterministic prefix: {finished_outputs!r}",
            failures,
        )
        _record(
            max_active >= 2,
            f"maximum observed active requests was {max_active}",
            failures,
        )
        scheduler = final_health["scheduler"]
        _record(
            scheduler["waiting"] == 0
            and scheduler["running"] == 0
            and scheduler["reserved_blocks"] == 0
            and scheduler["allocated_blocks"] == 0,
            f"final scheduler state was {scheduler!r}",
            failures,
        )
        generated_tokens = int(
            after["forge_generated_tokens_total"]
            - before["forge_generated_tokens_total"]
        )
        wall_seconds = float(client_report.get("wall_seconds", 0.0))
        _record(
            generated_tokens > 0 and wall_seconds > 0.0,
            (
                f"invalid throughput inputs: tokens={generated_tokens}, "
                f"wall={wall_seconds}"
            ),
            failures,
        )
        benchmark = {
            "warmup_requests": 1,
            "measured_requests": SERVING_REQUESTS,
            "concurrency_limit": SERVING_CONCURRENCY,
            "generated_tokens": generated_tokens,
            "wall_seconds": wall_seconds,
            "throughput_tokens_per_second": (
                generated_tokens / wall_seconds if wall_seconds > 0.0 else None
            ),
            "max_observed_active_requests": max_active,
            "peak_cuda_allocated_bytes": int(
                after['forge_cuda_memory_bytes{state="peak_allocated"}']
            ),
            "metrics_poll_failures": poll_failures,
            "client_latency_seconds": {
                "mean_ttft": client_report.get("mean_ttft_seconds"),
                "mean_itl": client_report.get("mean_itl_seconds"),
                "p50_duration": client_report.get("p50_duration_seconds"),
                "p95_duration": client_report.get("p95_duration_seconds"),
            },
            "server_latency_seconds": {
                "mean_ttft": server_delta.get("mean_ttft_seconds"),
                "mean_itl": server_delta.get("mean_itl_seconds"),
                "mean_duration": server_delta.get("mean_duration_seconds"),
            },
        }
        print(
            f"L4_SERVING_BENCHMARK={json.dumps(benchmark, sort_keys=True)}",
            flush=True,
        )
    except BaseException as error:
        failures.append(f"serving validation raised {error!r}")
        print(f"L4_SERVING_EXCEPTION={error!r}", flush=True)
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10.0)
        print(f"L4_SERVER_EXIT={server.returncode}", flush=True)
    return {
        "warmup_chat": warmup_text,
        "client": report.get("client", {}),
        "server_delta": report.get("server_delta", {}),
        "agreement": report.get("agreement", {}),
        "benchmark": benchmark,
    }


@app.function(
    image=l4_image,
    gpu="L4",
    timeout=30 * 60,
    volumes={"/cache": hf_cache_volume},
)
def validate_l4() -> dict[str, object]:
    """Validate L4 kernels, integration, measurements, and artifacts."""
    import torch
    import transformers
    import triton

    from forge_engine.cache import PagedKVBlockPool
    from forge_engine.engine import GenerationCore, _normalize_tokenizer_output
    from forge_engine.kernels import (
        causal_prefill_attention_reference,
        residual_add_rms_norm,
        residual_add_rms_norm_reference,
        restricted_causal_prefill_attention,
    )
    from forge_engine.kernels.cuda_paged import (
        paged_gqa_decode,
        paged_gqa_decode_reference,
    )
    from forge_engine.kernels.cute_swiglu import (
        cute_gate_up_swiglu_lab,
        gate_up_swiglu_reference,
    )
    from forge_engine.weights import (
        download_supported_snapshot,
        load_staged_model,
        load_supported_tokenizer,
    )

    failures: list[str] = []
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one GPU, found {torch.cuda.device_count()}"
        )
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name:
        raise RuntimeError(f"expected an L4 GPU, found {device_name!r}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("L4 acceptance requires BF16 support")
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    software = {
        "gpu": device_name,
        "compute_capability": ".".join(
            str(part) for part in torch.cuda.get_device_capability(0)
        ),
        "driver": driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "triton": triton.__version__,
        "nvcc": _tool_version(["nvcc", "--version"]),
        "cuobjdump": _tool_version(["cuobjdump", "--version"]),
        "precision": "bfloat16",
        "model_revision": SUPPORTED_MODEL_REVISION,
    }
    print(f"L4_SOFTWARE={software}", flush=True)
    (ARTIFACT_DIRECTORY / "triton").mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REMOTE_REPOSITORY,
        check=True,
    )
    subprocess.run(
        [
            "/root/.cargo/bin/cargo",
            "test",
            "--release",
            "--locked",
            "--manifest-path",
            str(REMOTE_REPOSITORY / "rust" / "streamer" / "Cargo.toml"),
        ],
        cwd=REMOTE_REPOSITORY,
        check=True,
    )

    benchmarks: dict[str, object] = {}
    residual = torch.randn(
        (512, 2_560),
        dtype=torch.bfloat16,
        device=device,
    )
    update = torch.randn_like(residual)
    weight = torch.randn(
        (2_560,),
        dtype=torch.bfloat16,
        device=device,
    )
    try:
        expected_sum, expected_norm = residual_add_rms_norm_reference(
            residual,
            update,
            weight,
            1e-6,
        )
        actual_sum, actual_norm = residual_add_rms_norm(
            residual,
            update,
            weight,
            1e-6,
            require_triton=True,
        )
        _record(
            torch.allclose(actual_sum, expected_sum, atol=0.02, rtol=0.01),
            "Triton residual sum exceeded tolerance",
            failures,
        )
        _record(
            torch.allclose(actual_norm, expected_norm, atol=0.04, rtol=0.02),
            "Triton RMSNorm exceeded tolerance",
            failures,
        )
        benchmarks["triton_residual_rms"] = {
            "workload": "[512, 2560] BF16; epsilon=1e-6",
            "custom": _benchmark_cuda(
                torch,
                lambda: residual_add_rms_norm(
                    residual,
                    update,
                    weight,
                    1e-6,
                    require_triton=True,
                ),
                work_items=residual.numel(),
                throughput_unit="elements/s",
            ),
            "reference": _benchmark_cuda(
                torch,
                lambda: residual_add_rms_norm_reference(
                    residual,
                    update,
                    weight,
                    1e-6,
                ),
                work_items=residual.numel(),
                throughput_unit="elements/s",
            ),
        }
        print("L4_TRITON_RMS=PASS", flush=True)
    except BaseException as error:
        failures.append(f"Triton residual RMS phase raised {error!r}")
        print(f"L4_TRITON_RMS_EXCEPTION={error!r}", flush=True)

    query = torch.randn(
        (1, 8, 128, 128),
        dtype=torch.bfloat16,
        device=device,
    )
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    try:
        expected_prefill = causal_prefill_attention_reference(
            query,
            key,
            value,
        )
        actual_prefill = restricted_causal_prefill_attention(
            query,
            key,
            value,
            require_triton=True,
        )
        prefill_error = float(
            (actual_prefill.float() - expected_prefill.float()).abs().max()
        )
        _record(
            torch.allclose(
                actual_prefill,
                expected_prefill,
                atol=0.05,
                rtol=0.03,
            ),
            f"restricted Triton prefill max_abs={prefill_error:.6f}",
            failures,
        )
        benchmarks["triton_restricted_prefill"] = {
            "workload": "[1, 8, 128, 128] causal BF16; head_dim=128",
            "custom": _benchmark_cuda(
                torch,
                lambda: restricted_causal_prefill_attention(
                    query,
                    key,
                    value,
                    require_triton=True,
                ),
                work_items=query.shape[1] * query.shape[2],
                throughput_unit="query-heads/s",
            ),
            "reference": _benchmark_cuda(
                torch,
                lambda: causal_prefill_attention_reference(
                    query,
                    key,
                    value,
                ),
                work_items=query.shape[1] * query.shape[2],
                throughput_unit="query-heads/s",
            ),
            "maximum_absolute_error": prefill_error,
        }
        print("L4_TRITON_PREFILL_LAB=PASS", flush=True)
    except BaseException as error:
        failures.append(f"restricted Triton prefill phase raised {error!r}")
        print(f"L4_TRITON_PREFILL_EXCEPTION={error!r}", flush=True)

    past_length = 127
    block_size = 16
    page_count = (past_length + block_size - 1) // block_size
    paged_query = torch.randn(
        (1, 32, 1, 128),
        dtype=torch.bfloat16,
        device=device,
    )
    key_pages = tuple(
        torch.randn(
            (1, 8, block_size, 128),
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(page_count)
    )
    value_pages = tuple(torch.randn_like(page) for page in key_pages)
    current_key = torch.randn(
        (1, 8, 1, 128),
        dtype=torch.bfloat16,
        device=device,
    )
    current_value = torch.randn_like(current_key)
    try:
        expected_decode = paged_gqa_decode_reference(
            paged_query,
            key_pages,
            value_pages,
            current_key,
            current_value,
            past_length,
        )
        actual_decode = paged_gqa_decode(
            paged_query,
            key_pages,
            value_pages,
            current_key,
            current_value,
            past_length,
            require_cuda_kernel=True,
        )
        decode_error = float(
            (actual_decode.float() - expected_decode.float()).abs().max()
        )
        _record(
            torch.allclose(
                actual_decode,
                expected_decode,
                atol=0.04,
                rtol=0.03,
            ),
            f"CUDA paged GQA max_abs={decode_error:.6f}",
            failures,
        )
        benchmarks["cuda_paged_gqa_decode"] = {
            "workload": ("past=127, block=16, Q=32, KV=8, head_dim=128, BF16"),
            "custom": _benchmark_cuda(
                torch,
                lambda: paged_gqa_decode(
                    paged_query,
                    key_pages,
                    value_pages,
                    current_key,
                    current_value,
                    past_length,
                    require_cuda_kernel=True,
                ),
                work_items=paged_query.shape[1],
                throughput_unit="query-heads/s",
            ),
            "reference": _benchmark_cuda(
                torch,
                lambda: paged_gqa_decode_reference(
                    paged_query,
                    key_pages,
                    value_pages,
                    current_key,
                    current_value,
                    past_length,
                ),
                work_items=paged_query.shape[1],
                throughput_unit="query-heads/s",
            ),
            "maximum_absolute_error": decode_error,
        }
        print("L4_CUDA_PAGED_GQA=PASS", flush=True)
    except BaseException as error:
        failures.append(f"CUDA paged GQA phase raised {error!r}")
        print(f"L4_CUDA_PAGED_GQA_EXCEPTION={error!r}", flush=True)

    cute_evidence: dict[str, object] = {}
    try:
        cute_input = torch.randn(
            (2, 16),
            dtype=torch.bfloat16,
            device=device,
        )
        cute_gate = torch.randn(
            (8, 16),
            dtype=torch.bfloat16,
            device=device,
        )
        cute_up = torch.randn_like(cute_gate)
        fallback = cute_gate_up_swiglu_lab(
            cute_input,
            cute_gate,
            cute_up,
        )
        reference = gate_up_swiglu_reference(
            cute_input,
            cute_gate,
            cute_up,
        )
        _record(
            torch.equal(fallback, reference),
            "CuTe L4 automatic fallback changed values",
            failures,
        )
        strict_rejected = False
        try:
            cute_gate_up_swiglu_lab(
                cute_input,
                cute_gate,
                cute_up,
                require_cute=True,
            )
        except RuntimeError as error:
            strict_rejected = "H100 or B200" in str(error)
        _record(
            strict_rejected,
            "CuTe strict path did not reject unsupported L4 hardware",
            failures,
        )
        cute_evidence = {
            "status": "SKIP_L4_HARDWARE",
            "fallback_verified": True,
            "strict_guard_verified": strict_rejected,
            "required_hardware": "H100 or B200",
        }
        print("L4_CUTE_LAB=SKIP_L4_HARDWARE", flush=True)
    except BaseException as error:
        failures.append(f"CuTe fallback phase raised {error!r}")
        print(f"L4_CUTE_FALLBACK_EXCEPTION={error!r}", flush=True)

    model_evidence: dict[str, object] = {}
    try:
        import functools

        import forge_engine.qwen3 as qwen3_module

        snapshot = download_supported_snapshot()
        _record(
            snapshot.name == SUPPORTED_MODEL_REVISION,
            f"snapshot revision is {snapshot.name}",
            failures,
        )
        tokenizer, _ = load_supported_tokenizer(snapshot)
        model, _ = load_staged_model(
            snapshot,
            device=device,
            dtype=torch.bfloat16,
        )
        pool = PagedKVBlockPool(block_size=16, capacity=64)
        core = GenerationCore(model, pool)
        encoding = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids, tokenizer_mask = _normalize_tokenizer_output(encoding)
        complete_ids = input_ids.to(device)
        attention_mask = (
            torch.ones_like(complete_ids)
            if tokenizer_mask is None
            else tokenizer_mask.to(device)
        )
        original_residual = qwen3_module.residual_add_rms_norm
        original_paged = qwen3_module.paged_gqa_decode
        qwen3_module.residual_add_rms_norm = functools.partial(
            original_residual,
            require_triton=True,
        )
        qwen3_module.paged_gqa_decode = functools.partial(
            original_paged,
            require_cuda_kernel=True,
        )
        state = None
        generated: list[int] = []
        probability_distances: list[float] = []
        try:
            with torch.inference_mode():
                state = core.prefill(complete_ids, attention_mask)
                for step in range(DECODE_STEPS):
                    uncached = core.uncached(complete_ids, attention_mask)
                    custom_logits = state.logits[:, -1, :]
                    reference_logits = uncached.logits[:, -1, :]
                    custom_next = custom_logits.argmax(dim=-1)
                    reference_next = reference_logits.argmax(dim=-1)
                    _record(
                        torch.equal(custom_next, reference_next),
                        f"integrated greedy mismatch at step {step}",
                        failures,
                    )
                    probability_distances.append(
                        _probability_tv(
                            torch,
                            custom_logits,
                            reference_logits,
                        )
                    )
                    generated.append(int(custom_next.item()))
                    complete_ids = torch.cat(
                        (complete_ids, custom_next.unsqueeze(1)),
                        dim=1,
                    )
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            torch.ones(
                                (1, 1),
                                dtype=attention_mask.dtype,
                                device=device,
                            ),
                        ),
                        dim=1,
                    )
                    if step + 1 < DECODE_STEPS:
                        state = core.decode(
                            custom_next.unsqueeze(1),
                            state,
                        )
        finally:
            qwen3_module.residual_add_rms_norm = original_residual
            qwen3_module.paged_gqa_decode = original_paged
            if state is not None:
                state.cache.clear()
        maximum_tv = max(probability_distances)
        decoded = tokenizer.decode(
            generated,
            skip_special_tokens=False,
        )
        _record(
            maximum_tv <= PROBABILITY_TV_LIMIT,
            f"integrated probability_tv={maximum_tv:.8f}",
            failures,
        )
        _record(
            decoded.startswith(EXPECTED_PREFIX),
            f"integrated greedy output was {decoded!r}",
            failures,
        )
        _record(
            pool.allocated_block_count == 0,
            "integrated model phase leaked KV blocks",
            failures,
        )
        model_evidence = {
            "revision": snapshot.name,
            "greedy_token_ids": generated,
            "greedy_text": decoded,
            "maximum_probability_tv": maximum_tv,
            "strict_triton_and_cuda_integration": True,
        }
        print(f"L4_MODEL_INTEGRATION={model_evidence}", flush=True)
    except BaseException as error:
        failures.append(f"strict model integration phase raised {error!r}")
        print(f"L4_MODEL_INTEGRATION_EXCEPTION={error!r}", flush=True)

    artifacts: dict[str, object] = {}
    try:
        artifacts = _inspect_artifacts(failures)
        print(f"L4_PTX_SASS={artifacts}", flush=True)
    except BaseException as error:
        failures.append(f"PTX/SASS inspection phase raised {error!r}")
        print(f"L4_ARTIFACT_EXCEPTION={error!r}", flush=True)

    serving_evidence = _validate_serving(failures)
    print(f"L4_KERNEL_BENCHMARKS={benchmarks}", flush=True)
    hf_cache_volume.commit()
    if failures:
        raise AssertionError("L4 acceptance failed:\n- " + "\n- ".join(failures))
    return {
        "software": software,
        "benchmarks": benchmarks,
        "artifacts": artifacts,
        "model": model_evidence,
        "cute": cute_evidence,
        "serving": serving_evidence,
    }


@app.local_entrypoint()
def main() -> None:
    """Invoke the consolidated L4 validator and print retained evidence."""
    result = validate_l4.remote()
    print("L4_ACCEPTANCE=PASS")
    for name, value in result.items():
        print(f"{name}={value}")
