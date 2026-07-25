"""Tests for the interactive command."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest import TestCase
from unittest.mock import patch

from forge_engine.cli import build_parser, main
from forge_engine.config import EngineConfig
from forge_engine.scheduler import SchedulerConfig


class FakeEngine:
    """Small streaming engine fake."""

    def __init__(self) -> None:
        """Record each complete conversation passed by the CLI."""
        self.calls: list[list[dict[str, str]]] = []

    def stream(self, messages: object) -> object:
        """Return a distinct response for each terminal turn."""
        conversation = [dict(message) for message in messages]
        self.calls.append(conversation)
        if len(self.calls) == 1:
            return iter(("First", " response"))
        return iter(("Second", " response"))


class CLITests(TestCase):
    """Command-line behavior."""

    def test_chat_rejects_arbitrary_model_selection(self) -> None:
        """The CLI exposes no option for loading an unsupported model."""
        with (
            redirect_stderr(StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            build_parser().parse_args(["chat", "--model", "other/model"])

        self.assertEqual(raised.exception.code, 2)

    def test_chat_streams_two_turns_with_history(self) -> None:
        """The second turn includes the first user and assistant messages."""
        output = StringIO()
        engine = FakeEngine()

        with (
            patch(
                "forge_engine.cli.GenerationEngine.from_config",
                return_value=engine,
            ),
            patch(
                "builtins.input",
                side_effect=("First question", "Second question", EOFError),
            ),
            redirect_stdout(output),
        ):
            result = main(["chat"])

        self.assertEqual(result, 0)
        self.assertEqual(
            engine.calls,
            [
                [{"role": "user", "content": "First question"}],
                [
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First response"},
                    {"role": "user", "content": "Second question"},
                ],
            ],
        )
        self.assertIn("ForgeEngine: First response", output.getvalue())
        self.assertIn("ForgeEngine: Second response", output.getvalue())

    def test_serve_builds_engine_and_scheduler_configuration(self) -> None:
        """One command passes bounded serving configuration to the server."""
        with patch("forge_engine.cli.run_server") as run_server:
            result = main(
                [
                    "serve",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "9000",
                    "--max-new-tokens",
                    "12",
                    "--max-requests",
                    "3",
                    "--max-batch-size",
                    "2",
                    "--token-budget",
                    "24",
                    "--block-capacity",
                    "128",
                ]
            )

        self.assertEqual(result, 0)
        run_server.assert_called_once_with(
            EngineConfig(max_new_tokens=12),
            SchedulerConfig(
                max_requests=3,
                max_batch_size=2,
                token_budget=24,
                block_size=16,
                block_capacity=128,
            ),
            host="0.0.0.0",
            port=9000,
        )
