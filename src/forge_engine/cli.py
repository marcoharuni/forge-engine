"""ForgeEngine command-line entry point."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from forge_engine import __version__
from forge_engine.config import DEFAULT_MODEL_ID, SUPPORTED_MODEL_REVISION, EngineConfig
from forge_engine.engine import GenerationEngine
from forge_engine.model import ChatMessage, ForgeEngineError
from forge_engine.sampling import normalize_stop_strings
from forge_engine.scheduler import SchedulerConfig
from forge_engine.server import run_server


def build_parser() -> argparse.ArgumentParser:
    """Build the ForgeEngine argument parser."""
    parser = argparse.ArgumentParser(prog="forge-engine")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="start an interactive CUDA chat")
    _add_generation_arguments(chat)
    serve = commands.add_parser(
        "serve",
        help="start the streaming HTTP and browser chat service",
    )
    _add_generation_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--max-requests", type=int, default=16)
    serve.add_argument("--max-batch-size", type=int, default=8)
    serve.add_argument("--token-budget", type=int, default=256)
    serve.add_argument("--block-size", type=int, default=16)
    serve.add_argument("--block-capacity", type=int, default=1_024)
    commands.add_parser(
        "doctor",
        help="print the local ForgeEngine and CUDA environment",
    )
    return parser


def _add_generation_arguments(
    command: argparse.ArgumentParser,
) -> None:
    """Add sampling defaults shared by terminal and HTTP serving."""
    command.add_argument("--max-new-tokens", type=int, default=256)
    command.add_argument("--temperature", type=float, default=0.0)
    command.add_argument("--top-k", type=int)
    command.add_argument("--top-p", type=float, default=1.0)
    command.add_argument("--min-p", type=float, default=0.0)
    command.add_argument(
        "--stop",
        action="append",
        help="stop string; may be supplied more than once",
    )
    command.add_argument("--seed", type=int)


def run_chat(config: EngineConfig) -> int:
    """Run a single-user terminal chat until end of input."""
    engine = GenerationEngine.from_config(config)
    messages: list[ChatMessage] = []

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not prompt:
            continue

        messages.append({"role": "user", "content": prompt})
        print("ForgeEngine: ", end="", flush=True)
        response = ""
        for fragment in engine.stream(messages):
            print(fragment, end="", flush=True)
            response += fragment
        print()
        messages.append({"role": "assistant", "content": response})


def run_doctor() -> int:
    """Print concise diagnostics without loading model weights."""
    import torch

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "unavailable"
    compute_capability = (
        ".".join(str(part) for part in torch.cuda.get_device_capability(0))
        if cuda_available
        else "unavailable"
    )
    bf16_supported = bool(torch.cuda.is_bf16_supported()) if cuda_available else False
    hf_home = os.environ.get(
        "HF_HOME",
        str(Path.home() / ".cache" / "huggingface"),
    )
    fields = (
        ("forge_engine_version", __version__),
        ("python_version", platform.python_version()),
        ("pytorch_version", torch.__version__),
        ("cuda_available", cuda_available),
        ("gpu_name", gpu_name),
        ("compute_capability", compute_capability),
        ("bf16_supported", bf16_supported),
        ("cuda_toolkit_available", shutil.which("nvcc") is not None),
        ("g++_available", shutil.which("g++") is not None),
        ("ninja_available", shutil.which("ninja") is not None),
        ("hf_home", hf_home),
        ("model", DEFAULT_MODEL_ID),
        ("model_revision", SUPPORTED_MODEL_REVISION),
    )
    for name, value in fields:
        print(f"{name}={value}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested ForgeEngine command."""
    args = build_parser().parse_args(argv)

    try:
        if args.command == "doctor":
            return run_doctor()
        engine_config = EngineConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            stop_strings=normalize_stop_strings(args.stop),
            seed=args.seed,
        )
        if args.command == "chat":
            return run_chat(engine_config)
        if args.command == "serve":
            if not 1 <= args.port <= 65_535:
                raise ValueError("port must be in [1, 65535]")
            run_server(
                engine_config,
                SchedulerConfig(
                    max_requests=args.max_requests,
                    max_batch_size=args.max_batch_size,
                    token_budget=args.token_budget,
                    block_size=args.block_size,
                    block_capacity=args.block_capacity,
                ),
                host=args.host,
                port=args.port,
            )
            return 0
    except (ForgeEngineError, ValueError) as error:
        print(f"forge-engine: {error}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
