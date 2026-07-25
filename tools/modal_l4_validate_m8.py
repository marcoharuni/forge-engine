"""Run Milestone 8 Rust client/load acceptance on one Modal L4."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import modal

from forge_engine.config import DEFAULT_MODEL_ID, SUPPORTED_MODEL_REVISION
from tools.modal_l4_validate import (
    REMOTE_REPOSITORY,
    hf_cache_volume,
    image,
)

APP_NAME = "forge-engine-m8-validation"
PORT = 8765
CLIENT = (
    REMOTE_REPOSITORY
    / "rust"
    / "streamer"
    / "target"
    / "release"
    / "forge-streamer"
)
PROMPT = "Write the first one hundred positive integers separated by spaces."
EXPECTED_PREFIX = "1 2 3 4"
REQUESTS = 6
CANCEL_EVERY = 3

m8_image = (
    image.apt_install("build-essential", "ca-certificates", "curl")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf "
        "https://sh.rustup.rs -o /tmp/rustup-init.sh",
        "sh /tmp/rustup-init.sh -y --profile minimal "
        "--default-toolchain stable",
        "/root/.cargo/bin/cargo build --release --locked "
        "--manifest-path /repo/rust/streamer/Cargo.toml",
    )
)
app = modal.App(APP_NAME)


def _record(
    condition: bool,
    message: str,
    failures: list[str],
) -> None:
    """Collect a failed condition while allowing later checks to run."""
    if not condition:
        failures.append(message)
        print(f"M8_CHECK_FAILURE={message}", flush=True)


@app.function(
    image=m8_image,
    gpu="L4",
    timeout=30 * 60,
    volumes={"/cache": hf_cache_volume},
)
def validate_m8() -> dict[str, object]:
    """Exercise Rust SSE chat, load, cancellation, and metric agreement."""
    import httpx
    import torch

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
        raise RuntimeError("M8 acceptance requires L4 BF16 support")
    print(f"GPU={device_name}; torch={torch.__version__}", flush=True)

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
        "6",
        "--max-batch-size",
        "3",
        "--token-budget",
        "128",
        "--block-capacity",
        "96",
    ]
    print(f"M8_SERVER_COMMAND={' '.join(server_command)}", flush=True)
    process = subprocess.Popen(
        server_command,
        cwd=REMOTE_REPOSITORY,
        stderr=subprocess.STDOUT,
    )
    base_url = f"http://127.0.0.1:{PORT}"
    chat_output = ""
    load_report: dict[str, object] = {}
    try:
        deadline = time.monotonic() + 180.0
        with httpx.Client(
            base_url=base_url,
            timeout=2.0,
            trust_env=False,
        ) as client:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"server exited during startup with {process.returncode}"
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
            health.get("model") == DEFAULT_MODEL_ID,
            f"health model was {health.get('model')!r}",
            failures,
        )
        _record(
            health.get("revision") == SUPPORTED_MODEL_REVISION,
            f"health revision was {health.get('revision')!r}",
            failures,
        )
        print("M8_STARTUP_PHASE=COMPLETE", flush=True)

        chat = subprocess.run(
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
        print(chat.stderr, end="", flush=True)
        chat_output = chat.stdout.strip()
        _record(
            chat.returncode == 0,
            f"Rust chat exited {chat.returncode}: {chat.stderr!r}",
            failures,
        )
        _record(
            chat_output.startswith(EXPECTED_PREFIX),
            f"Rust chat output was {chat_output!r}",
            failures,
        )
        print(f"M8_RUST_CHAT={chat_output!r}", flush=True)

        load = subprocess.run(
            [
                str(CLIENT),
                "load",
                "--base-url",
                base_url,
                "--prompt",
                PROMPT,
                "--max-tokens",
                "24",
                "--requests",
                str(REQUESTS),
                "--concurrency",
                "3",
                "--cancel-every",
                str(CANCEL_EVERY),
                "--cancel-after-events",
                "0",
                "--cancel-max-tokens",
                "512",
                "--metrics-timeout-seconds",
                "30",
            ],
            cwd=REMOTE_REPOSITORY,
            text=True,
            capture_output=True,
            timeout=300.0,
            check=False,
        )
        print(load.stderr, end="", flush=True)
        _record(
            load.returncode == 0,
            f"Rust load exited {load.returncode}: {load.stderr!r}",
            failures,
        )
        try:
            load_report = json.loads(load.stdout)
        except json.JSONDecodeError as error:
            failures.append(f"Rust load emitted invalid JSON: {error}")
            print(f"M8_LOAD_STDOUT={load.stdout!r}", flush=True)
            load_report = {}

        client_report = load_report.get("client", {})
        server_delta = load_report.get("server_delta", {})
        agreement = load_report.get("agreement", {})
        expected_cancelled = REQUESTS // CANCEL_EVERY
        expected_finished = REQUESTS - expected_cancelled
        _record(
            client_report.get("requests") == REQUESTS,
            f"client request count was {client_report.get('requests')!r}",
            failures,
        )
        _record(
            client_report.get("finished") == expected_finished,
            f"client finished count was {client_report.get('finished')!r}",
            failures,
        )
        _record(
            client_report.get("cancelled") == expected_cancelled,
            f"client cancelled count was {client_report.get('cancelled')!r}",
            failures,
        )
        _record(
            client_report.get("failed") == 0,
            f"client failed count was {client_report.get('failed')!r}",
            failures,
        )
        _record(
            client_report.get("mean_ttft_seconds") is not None
            and client_report.get("mean_itl_seconds") is not None
            and client_report.get("p50_duration_seconds") is not None
            and client_report.get("p95_duration_seconds") is not None,
            "client latency report was incomplete",
            failures,
        )
        _record(
            agreement.get("passed") is True,
            f"client/server agreement was {agreement!r}",
            failures,
        )
        _record(
            server_delta.get("requests") == REQUESTS
            and server_delta.get("finished") == expected_finished
            and server_delta.get("cancelled") == expected_cancelled
            and server_delta.get("failed") == 0
            and expected_finished
            <= server_delta.get("ttft_count", -1)
            <= REQUESTS
            and server_delta.get("duration_count") == REQUESTS,
            f"server metric deltas were {server_delta!r}",
            failures,
        )
        _record(
            server_delta.get("mean_ttft_seconds") is not None
            and server_delta.get("mean_itl_seconds") is not None
            and server_delta.get("mean_duration_seconds") is not None,
            "server latency report was incomplete",
            failures,
        )
        print(
            f"M8_LOAD_CLIENT={json.dumps(client_report, sort_keys=True)}",
            flush=True,
        )
        print(
            f"M8_LOAD_SERVER={json.dumps(server_delta, sort_keys=True)}",
            flush=True,
        )
        print("M8_RUST_LOAD_PHASE=COMPLETE", flush=True)

        with httpx.Client(
            base_url=base_url,
            timeout=10.0,
            trust_env=False,
        ) as client:
            final_health = client.get("/health").json()
        scheduler = final_health["scheduler"]
        _record(
            scheduler["waiting"] == 0
            and scheduler["running"] == 0
            and scheduler["reserved_blocks"] == 0
            and scheduler["allocated_blocks"] == 0,
            f"final scheduler state was {scheduler!r}",
            failures,
        )
        print("M8_CLEANUP_PHASE=COMPLETE", flush=True)
    except BaseException as error:
        failures.append(f"M8 validation raised {error!r}")
        print(f"M8_EXCEPTION={error!r}", flush=True)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
        print(f"M8_SERVER_EXIT={process.returncode}", flush=True)

    torch.cuda.synchronize()
    hf_cache_volume.commit()
    if failures:
        print(f"M8_FAILURES={failures!r}", flush=True)
        raise RuntimeError(
            f"M8 acceptance found {len(failures)} failure(s); "
            "see M8_FAILURES above"
        )
    return {
        "gpu": device_name,
        "revision": SUPPORTED_MODEL_REVISION,
        "chat": chat_output,
        "client": load_report["client"],
        "server_delta": load_report["server_delta"],
        "agreement": load_report["agreement"],
    }


@app.local_entrypoint()
def main() -> None:
    """Invoke M8 validation and print retained evidence."""
    result = validate_m8.remote()
    print("M8_ACCEPTANCE=PASS")
    for name, value in result.items():
        print(f"{name}={value}")
