"""Build the release Dockerfile and smoke-test it on one Modal L4."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

APP_NAME = "forge-engine-m9-docker-validation"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PORT = 8765
EXPECTED_PREFIX = "1 2 3 4"
DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SUPPORTED_MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"

docker_image = modal.Image.from_dockerfile(
    REPOSITORY_ROOT / "Dockerfile",
    context_dir=REPOSITORY_ROOT,
    add_python="3.11",
).entrypoint([])
app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name(
    "forge-engine-hf-cache",
    create_if_missing=True,
)


@app.function(
    image=docker_image,
    gpu="L4",
    timeout=30 * 60,
    volumes={"/cache": hf_cache_volume},
)
def validate_docker_m9() -> dict[str, object]:
    """Start the packaged service and validate health plus streaming chat."""
    python = "/opt/forge-venv/bin/python"
    environment = json.loads(
        subprocess.run(
            [
                python,
                "-c",
                (
                    "import json, torch, transformers; "
                    "print(json.dumps({'cuda': torch.cuda.is_available(), "
                    "'gpu': torch.cuda.get_device_name(0), "
                    "'torch': torch.__version__, "
                    "'transformers': transformers.__version__}))"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    if environment["cuda"] is not True or "L4" not in environment["gpu"]:
        raise RuntimeError(f"unexpected Docker GPU environment: {environment}")

    command = [
        "/opt/forge-venv/bin/forge-engine",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--max-new-tokens",
        "16",
        "--max-requests",
        "2",
        "--max-batch-size",
        "2",
        "--block-capacity",
        "64",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{PORT}"
    output = ""
    try:
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Docker service exited during startup: {process.returncode}"
                )
            try:
                with urllib.request.urlopen(
                    f"{base_url}/health",
                    timeout=2.0,
                ) as response:
                    if response.status == 200:
                        health = json.loads(response.read())
                        break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.25)
        else:
            raise RuntimeError("Docker service did not become healthy in 180s")
        if (
            health.get("model") != DEFAULT_MODEL_ID
            or health.get("revision") != SUPPORTED_MODEL_REVISION
        ):
            raise RuntimeError(f"unexpected Docker health: {health}")

        body = json.dumps(
            {
                "model": DEFAULT_MODEL_ID,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Write the first ten positive integers "
                            "separated by spaces."
                        ),
                    }
                ],
                "stream": True,
                "max_tokens": 8,
                "temperature": 0.0,
            }
        ).encode()
        request = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        saw_done = False
        with urllib.request.urlopen(request, timeout=180.0) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Docker chat returned HTTP {response.status}"
                )
            if not response.headers.get("X-Forge-Request-ID"):
                raise RuntimeError("Docker chat omitted request ID header")
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    saw_done = True
                    continue
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(f"Docker stream failed: {chunk}")
                output += chunk["choices"][0]["delta"].get("content", "")
        if not saw_done:
            raise RuntimeError("Docker stream omitted data: [DONE]")
        if not output.startswith(EXPECTED_PREFIX):
            raise RuntimeError(f"unexpected Docker chat output: {output!r}")

        with urllib.request.urlopen(
            f"{base_url}/health",
            timeout=10.0,
        ) as response:
            final_health = json.loads(response.read())
        scheduler = final_health["scheduler"]
        if any(
            scheduler[name] != 0
            for name in (
                "waiting",
                "running",
                "reserved_blocks",
                "allocated_blocks",
            )
        ):
            raise RuntimeError(f"Docker scheduler leaked state: {scheduler}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
        server_log = (
            process.stdout.read()
            if process.stdout is not None
            else ""
        )
        print(server_log, end="", flush=True)
        print(f"M9_DOCKER_SERVER_EXIT={process.returncode}", flush=True)

    if (
        "fatal error:" in server_log
        or "Traceback (most recent call last)" in server_log
    ):
        raise RuntimeError("Docker service log contains a compilation/runtime error")
    hf_cache_volume.commit()
    return {
        "environment": environment,
        "model_revision": SUPPORTED_MODEL_REVISION,
        "chat": output,
    }


@app.local_entrypoint()
def main() -> None:
    """Build and invoke the actual Docker release image."""
    result = validate_docker_m9.remote()
    print("M9_DOCKER_ACCEPTANCE=PASS")
    for name, value in result.items():
        print(f"{name}={value}")
