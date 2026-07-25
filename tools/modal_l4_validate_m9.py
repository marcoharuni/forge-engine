"""Collect consolidated Milestone 9 release evidence on one Modal L4."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import modal

from forge_engine.config import DEFAULT_MODEL_ID, SUPPORTED_MODEL_REVISION
from tools.modal_l4_validate import REMOTE_REPOSITORY, hf_cache_volume
from tools.modal_l4_validate_m8 import CLIENT, EXPECTED_PREFIX, PROMPT, m8_image

APP_NAME = "forge-engine-m9-validation"
PORT = 8765
REQUESTS = 8
CONCURRENCY = 4
CANCEL_EVERY = 4

app = modal.App(APP_NAME)


def _parse_metrics(text: str) -> dict[str, float]:
    """Parse the numeric subset of Prometheus text exposition."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.rsplit(" ", 1)
        values[name] = float(value)
    return values


def _record(
    condition: bool,
    message: str,
    failures: list[str],
) -> None:
    """Collect an independent acceptance failure."""
    if not condition:
        failures.append(message)
        print(f"M9_CHECK_FAILURE={message}", flush=True)


@app.function(
    image=m8_image,
    gpu="L4",
    timeout=30 * 60,
    volumes={"/cache": hf_cache_volume},
)
def validate_m9() -> dict[str, object]:
    """Run tests and measure the complete serving path under concurrency."""
    import httpx
    import torch
    import transformers

    failures: list[str] = []
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one GPU, found {torch.cuda.device_count()}"
        )
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name:
        raise RuntimeError(f"expected an L4 GPU, found {device_name!r}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("M9 acceptance requires L4 BF16 support")
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
            str(value) for value in torch.cuda.get_device_capability(0)
        ),
        "driver": driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "precision": "bfloat16",
        "model": DEFAULT_MODEL_ID,
        "model_revision": SUPPORTED_MODEL_REVISION,
    }
    print(f"M9_SOFTWARE={json.dumps(software, sort_keys=True)}", flush=True)

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

    server_command = [
        "forge-engine",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
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
    print(f"M9_SERVER_COMMAND={' '.join(server_command)}", flush=True)
    server = subprocess.Popen(
        server_command,
        cwd=REMOTE_REPOSITORY,
        stderr=subprocess.STDOUT,
    )
    base_url = f"http://127.0.0.1:{PORT}"
    chat_output = ""
    load_report: dict[str, object] = {}
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
                    health_response = client.get("/health")
                    if health_response.status_code == 200:
                        health = health_response.json()
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
                str(CLIENT),
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
        chat_output = warmup.stdout.strip()
        _record(
            warmup.returncode == 0
            and chat_output.startswith(EXPECTED_PREFIX),
            (
                f"warm-up chat exit={warmup.returncode}, "
                f"output={chat_output!r}, stderr={warmup.stderr!r}"
            ),
            failures,
        )

        with httpx.Client(
            base_url=base_url,
            timeout=10.0,
            trust_env=False,
        ) as client:
            before = _parse_metrics(client.get("/metrics").text)

        load_command = [
            str(CLIENT),
            "load",
            "--base-url",
            base_url,
            "--prompt",
            PROMPT,
            "--max-tokens",
            "32",
            "--requests",
            str(REQUESTS),
            "--concurrency",
            str(CONCURRENCY),
            "--cancel-every",
            str(CANCEL_EVERY),
            "--cancel-after-events",
            "0",
            "--cancel-max-tokens",
            "512",
            "--metrics-timeout-seconds",
            "30",
        ]
        load = subprocess.Popen(
            load_command,
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
            load_report = json.loads(stdout)
        except json.JSONDecodeError as error:
            failures.append(f"Rust load emitted invalid JSON: {error}")
            print(f"M9_LOAD_STDOUT={stdout!r}", flush=True)
            load_report = {}

        with httpx.Client(
            base_url=base_url,
            timeout=10.0,
            trust_env=False,
        ) as client:
            after = _parse_metrics(client.get("/metrics").text)
            final_health = client.get("/health").json()

        client_report = load_report.get("client", {})
        server_delta = load_report.get("server_delta", {})
        agreement = load_report.get("agreement", {})
        results = load_report.get("results", [])
        expected_cancelled = REQUESTS // CANCEL_EVERY
        expected_finished = REQUESTS - expected_cancelled
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
            and all(
                output.startswith(EXPECTED_PREFIX)
                for output in finished_outputs
            ),
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
                f"invalid token throughput inputs: tokens={generated_tokens}, "
                f"wall={wall_seconds}"
            ),
            failures,
        )
        benchmark = {
            "workload": {
                "prompt": PROMPT,
                "requests": REQUESTS,
                "concurrency_limit": CONCURRENCY,
                "finished_max_tokens": 32,
                "cancelled_requests": expected_cancelled,
                "cancel_after_content_events": 0,
            },
            "warmup_requests": 1,
            "measured_requests": REQUESTS,
            "synchronization": (
                "Each terminal SSE event follows the synchronous scheduler/CUDA "
                "step; metrics and health were fetched after every request "
                "reached a terminal state."
            ),
            "client_latency_seconds": {
                "mean_ttft": client_report.get("mean_ttft_seconds"),
                "mean_itl": client_report.get("mean_itl_seconds"),
                "p50_duration": client_report.get(
                    "p50_duration_seconds"
                ),
                "p95_duration": client_report.get(
                    "p95_duration_seconds"
                ),
            },
            "server_latency_seconds": {
                "mean_ttft": server_delta.get("mean_ttft_seconds"),
                "mean_itl": server_delta.get("mean_itl_seconds"),
                "mean_duration": server_delta.get(
                    "mean_duration_seconds"
                ),
            },
            "generated_tokens": generated_tokens,
            "wall_seconds": wall_seconds,
            "throughput_tokens_per_second": (
                generated_tokens / wall_seconds
                if wall_seconds > 0.0
                else None
            ),
            "peak_cuda_allocated_bytes": int(
                after[
                    'forge_cuda_memory_bytes{state="peak_allocated"}'
                ]
            ),
            "max_observed_active_requests": max_active,
            "metrics_poll_failures": poll_failures,
        }
        print(
            f"M9_BENCHMARK={json.dumps(benchmark, sort_keys=True)}",
            flush=True,
        )
        print(
            f"M9_CLIENT={json.dumps(client_report, sort_keys=True)}",
            flush=True,
        )
        print(
            f"M9_SERVER={json.dumps(server_delta, sort_keys=True)}",
            flush=True,
        )
    except BaseException as error:
        failures.append(f"M9 validation raised {error!r}")
        print(f"M9_EXCEPTION={error!r}", flush=True)
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10.0)
        print(f"M9_SERVER_EXIT={server.returncode}", flush=True)

    torch.cuda.synchronize()
    hf_cache_volume.commit()
    if failures:
        print(f"M9_FAILURES={failures!r}", flush=True)
        raise RuntimeError(
            f"M9 acceptance found {len(failures)} failure(s); "
            "see M9_FAILURES above"
        )
    return {
        "software": software,
        "warmup_chat": chat_output,
        "client": load_report["client"],
        "server_delta": load_report["server_delta"],
        "agreement": load_report["agreement"],
        "benchmark": benchmark,
    }


@app.local_entrypoint()
def main() -> None:
    """Invoke M9 validation and print the retained release evidence."""
    result = validate_m9.remote()
    print("M9_ACCEPTANCE=PASS")
    for name, value in result.items():
        print(f"{name}={value}")
