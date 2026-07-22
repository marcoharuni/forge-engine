"""Console entry-point placeholders for ForgeEngine developer and runtime commands."""

from collections.abc import Callable
from typing import NoReturn

# TODO: Replace command placeholders after their underlying runtime interfaces exist.


def _unimplemented(command: str) -> NoReturn:
    """Report that a scaffolded command does not have runtime behavior yet."""
    raise NotImplementedError(f"{command} is a scaffold; runtime behavior is not implemented")


def engine_main() -> NoReturn:
    """Start the future offline engine command."""
    _unimplemented("forge-engine")


def serve_main() -> NoReturn:
    """Start the future serving command."""
    _unimplemented("forge-serve")


def inspect_main() -> NoReturn:
    """Start the future model inspection command."""
    _unimplemented("forge-inspect")


def memory_main() -> NoReturn:
    """Start the future memory estimation command."""
    _unimplemented("forge-memory")


def bench_main() -> NoReturn:
    """Start the future benchmark command."""
    _unimplemented("forge-bench")


COMMANDS: dict[str, Callable[[], NoReturn]] = {
    "engine": engine_main,
    "serve": serve_main,
    "inspect": inspect_main,
    "memory": memory_main,
    "bench": bench_main,
}
