"""Run the Milestone 1 acceptance checks on one Modal L4 GPU."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "forge-engine-m1-validation"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_CODE = "COBALT-731"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REMOTE_REPOSITORY = Path("/repo")
HF_HOME = Path("/cache/huggingface")
PROMPTS = (
    "Remember this code exactly: COBALT-731. Reply only with STORED.\n"
    "What code did I ask you to remember?\n"
)
TRANSCRIPT_INPUT = (
    "=== CHAT INPUT ===\n"
    "1. Remember this code exactly: COBALT-731. Reply only with STORED.\n"
    "2. What code did I ask you to remember?\n"
    "=== CHAT OUTPUT ===\n"
)
REPOSITORY_IGNORES = [
    ".git",
    ".git/**",
    ".venv",
    ".venv/**",
    ".agents",
    ".agents/**",
    ".codex",
    ".codex/**",
    ".cache",
    ".cache/**",
    "**/.cache",
    "**/.cache/**",
    "**/__pycache__",
    "**/__pycache__/**",
    ".pytest_cache",
    ".pytest_cache/**",
    "**/.pytest_cache",
    "**/.pytest_cache/**",
    ".mypy_cache",
    ".mypy_cache/**",
    "**/.mypy_cache",
    "**/.mypy_cache/**",
    ".ruff_cache",
    ".ruff_cache/**",
    "**/.ruff_cache",
    "**/.ruff_cache/**",
    ".uv_cache",
    ".uv_cache/**",
    "**/.uv_cache",
    "**/.uv_cache/**",
    ".tox",
    ".tox/**",
    "**/.tox",
    "**/.tox/**",
    ".nox",
    ".nox/**",
    "**/.nox",
    "**/.nox/**",
    "build",
    "build/**",
    "**/build",
    "**/build/**",
    "dist",
    "dist/**",
    "**/dist",
    "**/dist/**",
    "*.egg-info",
    "*.egg-info/**",
    "**/*.egg-info",
    "**/*.egg-info/**",
    "logs",
    "logs/**",
    "**/logs",
    "**/logs/**",
    "*.log",
    "**/*.log",
    "models",
    "models/**",
    "**/models",
    "**/models/**",
    "*.bin",
    "**/*.bin",
    "*.ckpt",
    "**/*.ckpt",
    "*.pt",
    "**/*.pt",
    "*.pth",
    "**/*.pth",
    "*.safetensors",
    "**/*.safetensors",
    "*.gguf",
    "**/*.gguf",
    "*.onnx",
    "**/*.onnx",
    "*.whl",
    "**/*.whl",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_dir(
        REPOSITORY_ROOT,
        remote_path=str(REMOTE_REPOSITORY),
        copy=True,
        ignore=REPOSITORY_IGNORES,
    )
    .run_commands('python -m pip install "/repo[dev]"')
    .env({"HF_HOME": str(HF_HOME)})
    .workdir(REMOTE_REPOSITORY)
)
app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name(
    "forge-engine-hf-cache",
    create_if_missing=True,
)


def _stream_chat() -> str:
    """Run terminal chat with fixed stdin while streaming and retaining output."""
    print(TRANSCRIPT_INPUT, end="", flush=True)
    process = subprocess.Popen(
        ["forge-engine", "chat", "--max-new-tokens", "64"],
        cwd=REMOTE_REPOSITORY,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdin is None or process.stdout is None:
        process.kill()
        raise RuntimeError("failed to open chat subprocess pipes")

    process.stdin.write(PROMPTS)
    process.stdin.close()

    transcript_parts = [TRANSCRIPT_INPUT]
    while chunk := process.stdout.read(1):
        print(chunk, end="", flush=True)
        transcript_parts.append(chunk)

    return_code = process.wait()
    transcript = "".join(transcript_parts)
    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            process.args,
            output=transcript,
        )
    return transcript


def _model_commit_sha() -> str:
    """Read the exact commit selected by Hugging Face from its cache reference."""
    repository_cache = HF_HOME / "hub" / f"models--{MODEL_ID.replace('/', '--')}"
    commit_sha = (repository_cache / "refs" / "main").read_text().strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None:
        raise RuntimeError(f"invalid Hugging Face commit SHA: {commit_sha!r}")
    if not (repository_cache / "snapshots" / commit_sha).is_dir():
        raise RuntimeError(
            f"Hugging Face snapshot for commit {commit_sha} is missing"
        )
    return commit_sha


@app.function(
    image=image,
    gpu="L4",
    timeout=30 * 60,
    volumes={"/cache": hf_cache_volume},
)
def validate_m1() -> tuple[str, str]:
    """Run CPU-safe tests and the real-L4 two-turn terminal conversation."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one GPU, found {torch.cuda.device_count()}"
        )
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name:
        raise RuntimeError(f"expected an L4 GPU, found {device_name!r}")
    print(f"GPU={device_name}", flush=True)

    subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REMOTE_REPOSITORY,
        check=True,
    )
    transcript = _stream_chat()

    hf_cache_volume.commit()
    print("Committed forge-engine-hf-cache", flush=True)

    commit_sha = _model_commit_sha()
    print(f"HUGGING_FACE_MODEL_COMMIT_SHA={commit_sha}", flush=True)

    answers = transcript.split("ForgeEngine: ")[1:]
    if len(answers) < 2:
        raise RuntimeError("chat transcript does not contain two answers")
    second_answer = answers[1].split("\nYou:", maxsplit=1)[0]
    if MODEL_CODE not in second_answer:
        raise RuntimeError(
            f"second answer did not contain {MODEL_CODE}: {second_answer!r}"
        )

    return transcript, commit_sha


@app.local_entrypoint()
def main() -> None:
    """Invoke the remote validator and print its retained transcript."""
    transcript, commit_sha = validate_m1.remote()
    print("\n=== RETAINED TRANSCRIPT ===")
    print(transcript, end="" if transcript.endswith("\n") else "\n")
    print(f"HUGGING_FACE_MODEL_COMMIT_SHA={commit_sha}")
