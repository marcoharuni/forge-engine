"""ForgeEngine command-line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from forge_engine.config import EngineConfig
from forge_engine.engine import GreedyEngine
from forge_engine.model import ChatMessage, ForgeEngineError


def build_parser() -> argparse.ArgumentParser:
    """Build the ForgeEngine argument parser."""
    parser = argparse.ArgumentParser(prog="forge-engine")
    commands = parser.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="start an interactive CUDA chat")
    chat.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def run_chat(config: EngineConfig) -> int:
    """Run a single-user terminal chat until end of input."""
    engine = GreedyEngine.from_config(config)
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested ForgeEngine command."""
    args = build_parser().parse_args(argv)

    try:
        if args.command == "chat":
            return run_chat(
                EngineConfig(
                    max_new_tokens=args.max_new_tokens,
                )
            )
    except (ForgeEngineError, ValueError) as error:
        print(f"forge-engine: {error}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
