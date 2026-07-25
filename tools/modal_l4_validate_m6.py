"""Run Milestone 6 serving acceptance on one Modal L4."""

from __future__ import annotations

import subprocess
import sys

import modal

from forge_engine.config import DEFAULT_MODEL_ID, SUPPORTED_MODEL_REVISION
from tools.modal_l4_validate import (
    REMOTE_REPOSITORY,
    hf_cache_volume,
    image,
)

APP_NAME = "forge-engine-m6-validation"
PORT = 8765
PROMPT = "Write the first one hundred positive integers separated by spaces."
EXPECTED_PREFIX = "1 2 3 4 "
MAX_NEW_TOKENS = 32

app = modal.App(APP_NAME)


def _record(
    condition: bool,
    message: str,
    failures: list[str],
) -> None:
    """Collect a failed condition while allowing later checks to run."""
    if not condition:
        failures.append(message)
        print(f"M6_CHECK_FAILURE={message}", flush=True)


def _metric_value(metrics: str, sample: str) -> float | None:
    """Read one exact Prometheus sample without another dependency."""
    prefix = f"{sample} "
    for line in metrics.splitlines():
        if line.startswith(prefix):
            try:
                return float(line[len(prefix) :])
            except ValueError:
                return None
    return None


@app.function(
    image=image,
    gpu="L4",
    timeout=30 * 60,
    volumes={"/cache": hf_cache_volume},
)
def validate_m6() -> dict[str, object]:
    """Start the real service and validate concurrent streaming clients."""
    import asyncio
    import time

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
        raise RuntimeError("M6 acceptance requires L4 BF16 support")
    print(f"GPU={device_name}; torch={torch.__version__}", flush=True)

    subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REMOTE_REPOSITORY,
        check=True,
    )

    command = [
        "forge-engine",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--max-requests",
        "4",
        "--max-batch-size",
        "3",
        "--token-budget",
        "128",
        "--block-capacity",
        "64",
    ]
    print(f"M6_SERVER_COMMAND={' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=REMOTE_REPOSITORY,
        stderr=subprocess.STDOUT,
    )
    base_url = f"http://127.0.0.1:{PORT}"
    startup_health: dict[str, object] = {}
    client_evidence: dict[str, object] = {}
    metrics_text = ""
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
                        startup_health = response.json()
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.25)
            else:
                raise RuntimeError("server did not become healthy in 180s")

            _record(
                startup_health.get("status") == "ok",
                f"health status was {startup_health.get('status')!r}",
                failures,
            )
            _record(
                startup_health.get("model") == DEFAULT_MODEL_ID,
                f"health model was {startup_health.get('model')!r}",
                failures,
            )
            _record(
                startup_health.get("revision") == SUPPORTED_MODEL_REVISION,
                f"health revision was {startup_health.get('revision')!r}",
                failures,
            )
            browser = client.get("/")
            _record(
                browser.status_code == 200
                and "ForgeEngine Chat" in browser.text
                and "/v1/chat/completions" in browser.text,
                "browser chat page was missing or incomplete",
                failures,
            )
        print("M6_STARTUP_PHASE=COMPLETE", flush=True)

        async def exercise_clients() -> tuple[
            dict[str, dict[str, object]],
            int,
        ]:
            async with httpx.AsyncClient(
                base_url=base_url,
                timeout=180.0,
                trust_env=False,
            ) as client:
                async def stream_one(label: str) -> dict[str, object]:
                    output = ""
                    role_seen = False
                    done_seen = False
                    finish_reason: str | None = None
                    model_values: set[str] = set()
                    async with client.stream(
                        "POST",
                        "/v1/chat/completions",
                        json={
                            "model": DEFAULT_MODEL_ID,
                            "messages": [
                                {"role": "user", "content": PROMPT}
                            ],
                            "stream": True,
                            "max_tokens": MAX_NEW_TOKENS,
                        },
                    ) as response:
                        if response.status_code != 200:
                            body = await response.aread()
                            raise RuntimeError(
                                f"{label} HTTP {response.status_code}: "
                                f"{body.decode(errors='replace')}"
                            )
                        async for line in response.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = line[6:]
                            if data == "[DONE]":
                                done_seen = True
                                continue
                            event = __import__("json").loads(data)
                            if "error" in event:
                                raise RuntimeError(
                                    f"{label} stream error: {event['error']}"
                                )
                            model_values.add(event["model"])
                            choice = event["choices"][0]
                            delta = choice["delta"]
                            role_seen = role_seen or (
                                delta.get("role") == "assistant"
                            )
                            output += delta.get("content", "")
                            if choice["finish_reason"] is not None:
                                finish_reason = choice["finish_reason"]
                    return {
                        "output": output,
                        "role_seen": role_seen,
                        "done_seen": done_seen,
                        "finish_reason": finish_reason,
                        "models": tuple(sorted(model_values)),
                    }

                tasks = {
                    label: asyncio.create_task(stream_one(label))
                    for label in ("alpha", "beta", "gamma")
                }
                max_active = 0
                while not all(task.done() for task in tasks.values()):
                    try:
                        health = (await client.get("/health")).json()
                        scheduler = health["scheduler"]
                        active = (
                            int(scheduler["waiting"])
                            + int(scheduler["running"])
                        )
                        max_active = max(max_active, active)
                    except (httpx.HTTPError, KeyError, TypeError, ValueError):
                        pass
                    await asyncio.sleep(0.01)
                return (
                    {
                        label: await task
                        for label, task in tasks.items()
                    },
                    max_active,
                )

        try:
            results, max_active = asyncio.run(exercise_clients())
            outputs = {
                label: str(result["output"])
                for label, result in results.items()
            }
            for label, result in results.items():
                _record(
                    str(result["output"]).startswith(EXPECTED_PREFIX),
                    f"{label} output was {result['output']!r}",
                    failures,
                )
                _record(
                    result["role_seen"] is True,
                    f"{label} did not receive assistant role",
                    failures,
                )
                _record(
                    result["done_seen"] is True,
                    f"{label} did not receive [DONE]",
                    failures,
                )
                _record(
                    result["finish_reason"] in {"stop", "length"},
                    f"{label} finish reason was "
                    f"{result['finish_reason']!r}",
                    failures,
                )
                _record(
                    result["models"] == (DEFAULT_MODEL_ID,),
                    f"{label} model fields were {result['models']!r}",
                    failures,
                )
            _record(
                len(set(outputs.values())) == 1,
                f"deterministic client outputs differed: {outputs!r}",
                failures,
            )
            _record(
                max_active >= 2,
                f"maximum observed active requests was {max_active}",
                failures,
            )
            client_evidence = {
                "outputs": outputs,
                "max_active": max_active,
                "finish_reasons": {
                    label: result["finish_reason"]
                    for label, result in results.items()
                },
            }
            for label, output in outputs.items():
                print(
                    f"M6_CLIENT_{label.upper()}={output!r}",
                    flush=True,
                )
            print(f"M6_MAX_ACTIVE={max_active}", flush=True)
            print("M6_CONCURRENT_CLIENT_PHASE=COMPLETE", flush=True)
        except BaseException as error:
            failures.append(f"concurrent client phase raised {error!r}")
            print(f"M6_CONCURRENT_CLIENT_EXCEPTION={error!r}", flush=True)

        try:
            with httpx.Client(
                base_url=base_url,
                timeout=10.0,
                trust_env=False,
            ) as client:
                metrics_response = client.get("/metrics")
                metrics_text = metrics_response.text
                final_health = client.get("/health").json()
            _record(
                metrics_response.status_code == 200,
                f"metrics HTTP status was {metrics_response.status_code}",
                failures,
            )
            _record(
                _metric_value(metrics_text, "forge_requests_total") == 3,
                "admitted request metric was not 3",
                failures,
            )
            _record(
                _metric_value(
                    metrics_text,
                    'forge_requests_terminal_total{status="finished"}',
                )
                == 3,
                "finished request metric was not 3",
                failures,
            )
            _record(
                _metric_value(
                    metrics_text,
                    "forge_time_to_first_text_seconds_count",
                )
                == 3,
                "TTFT metric count was not 3",
                failures,
            )
            _record(
                _metric_value(
                    metrics_text,
                    "forge_request_duration_seconds_count",
                )
                == 3,
                "duration metric count was not 3",
                failures,
            )
            for metric_name in (
                "forge_scheduler_requests",
                "forge_inter_text_latency_seconds",
                "forge_generated_tokens_total",
                "forge_cuda_memory_bytes",
                "forge_kv_blocks",
            ):
                _record(
                    metric_name in metrics_text,
                    f"metrics omitted {metric_name}",
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
            print("M6_OBSERVABILITY_PHASE=COMPLETE", flush=True)
        except BaseException as error:
            failures.append(f"observability phase raised {error!r}")
            print(f"M6_OBSERVABILITY_EXCEPTION={error!r}", flush=True)
    except BaseException as error:
        failures.append(f"startup phase raised {error!r}")
        print(f"M6_STARTUP_EXCEPTION={error!r}", flush=True)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
        print(f"M6_SERVER_EXIT={process.returncode}", flush=True)

    torch.cuda.synchronize()
    hf_cache_volume.commit()
    if failures:
        print(f"M6_FAILURES={failures!r}", flush=True)
        raise RuntimeError(
            f"M6 acceptance found {len(failures)} failure(s); "
            "see M6_FAILURES above"
        )
    return {
        "gpu": device_name,
        "revision": SUPPORTED_MODEL_REVISION,
        "health": startup_health,
        "clients": client_evidence,
        "metrics_requests": _metric_value(
            metrics_text,
            "forge_requests_total",
        ),
    }


@app.local_entrypoint()
def main() -> None:
    """Invoke M6 validation and print retained evidence."""
    result = validate_m6.remote()
    print("M6_ACCEPTANCE=PASS")
    for name, value in result.items():
        print(f"{name}={value}")
