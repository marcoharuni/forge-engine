"""Tests for bounded iteration-level concurrent scheduling."""

from __future__ import annotations

from contextlib import contextmanager
from unittest import TestCase
from unittest.mock import patch

import torch

from forge_engine.qwen3 import Qwen3Config, Qwen3ForCausalLM
from forge_engine.sampling import SamplingParams
from forge_engine.scheduler import (
    ConcurrentGenerationEngine,
    OverloadedError,
    RequestStatus,
    SchedulerConfig,
)


def tiny_model() -> Qwen3ForCausalLM:
    """Build a deterministic small CPU model for scheduler state tests."""
    torch.manual_seed(11)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        eos_token_id=31,
        tie_word_embeddings=True,
        attention_bias=False,
    )
    return Qwen3ForCausalLM(config).eval()


class SchedulerTokenizer:
    """Return stable prompt tensors and append-only token text."""

    eos_token_id = None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
    ) -> torch.Tensor:
        """Create one three-token prompt derived from its final message."""
        assert tokenize and add_generation_prompt and return_tensors == "pt"
        marker = sum(ord(char) for char in conversation[-1]["content"]) % 20
        return torch.tensor([[1, 2, marker + 3]], dtype=torch.long)

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        """Render token IDs in an append-only representation."""
        assert skip_special_tokens
        return "".join(f"<{token_id}>" for token_id in token_ids)


@contextmanager
def cpu_scheduler_checks() -> object:
    """Disable only CUDA-specific outer-engine guards for tiny CPU tests."""
    with (
        patch("forge_engine.scheduler._validate_input_ids"),
        patch("forge_engine.scheduler._validate_attention_mask"),
        patch("forge_engine.scheduler._validate_step"),
    ):
        yield


def scheduler(
    *,
    model: object | None = None,
    max_requests: int = 4,
    max_batch_size: int = 3,
    token_budget: int = 8,
    block_capacity: int = 12,
    max_new_tokens: int = 3,
) -> ConcurrentGenerationEngine:
    """Build a small scheduler with explicit resource bounds."""
    return ConcurrentGenerationEngine(
        SchedulerTokenizer(),
        tiny_model() if model is None else model,
        SamplingParams(max_new_tokens=max_new_tokens),
        SchedulerConfig(
            max_requests=max_requests,
            max_batch_size=max_batch_size,
            token_budget=token_budget,
            block_size=2,
            block_capacity=block_capacity,
        ),
    )


class SchedulerTests(TestCase):
    """Admission, fairness, continuous work, and cleanup."""

    def test_invalid_scheduler_limits_are_rejected(self) -> None:
        """Every scheduler resource bound must be positive."""
        names = (
            "max_requests",
            "max_batch_size",
            "token_budget",
            "block_size",
            "block_capacity",
        )
        for name in names:
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, name),
            ):
                SchedulerConfig(**{name: 0})

    def test_block_reservation_rejects_overload_before_prefill(self) -> None:
        """Worst-case KV capacity is reserved before any model allocation."""
        engine = scheduler(block_capacity=4)
        first = engine.submit(
            [{"role": "user", "content": "first"}],
            request_id="first",
        )

        with self.assertRaisesRegex(OverloadedError, "unreserved"):
            engine.submit(
                [{"role": "user", "content": "second"}],
                request_id="second",
            )

        self.assertEqual(first, "first")
        self.assertEqual(engine.cache_pool.allocated_block_count, 0)
        self.assertEqual(engine.snapshot().reserved_blocks, 3)
        self.assertTrue(engine.cancel(first))
        self.assertEqual(engine.snapshot().reserved_blocks, 0)

    def test_request_count_rejects_overload_before_tokenization(self) -> None:
        """The active-request bound rejects additional queued work."""
        engine = scheduler(max_requests=1)
        engine.submit(
            [{"role": "user", "content": "first"}],
            request_id="first",
        )

        with self.assertRaisesRegex(OverloadedError, "request limit"):
            engine.submit(
                [{"role": "user", "content": "second"}],
                request_id="second",
            )

    def test_new_request_joins_running_batch_and_all_streams_finish(self) -> None:
        """A later request joins active decode work without draining first."""
        engine = scheduler()
        engine.submit(
            [{"role": "user", "content": "alpha"}],
            request_id="alpha",
        )
        engine.submit(
            [{"role": "user", "content": "beta"}],
            request_id="beta",
        )

        with cpu_scheduler_checks():
            first_events = engine.step()
            self.assertEqual(engine.snapshot().last_batch, ("alpha", "beta"))
            engine.submit(
                [{"role": "user", "content": "gamma"}],
                request_id="gamma",
            )
            second_events = engine.step()
            self.assertEqual(
                engine.snapshot().last_batch,
                ("alpha", "beta", "gamma"),
            )
            remaining_events = engine.run_until_idle()

        all_events = first_events + second_events + remaining_events
        outputs = {
            request_id: "".join(
                event.text for event in all_events if event.request_id == request_id
            )
            for request_id in ("alpha", "beta", "gamma")
        }
        self.assertTrue(all(outputs.values()))
        for request_id in outputs:
            view = engine.request(request_id)
            self.assertIs(view.status, RequestStatus.FINISHED)
            self.assertEqual(view.sampled_tokens, 3)
            self.assertEqual(view.reserved_blocks, 0)
        self.assertEqual(engine.cache_pool.allocated_block_count, 0)
        self.assertEqual(engine.snapshot().reserved_blocks, 0)

    def test_compatible_requests_share_tensor_model_calls(self) -> None:
        """One scheduler work group becomes one actual model batch."""

        class CountingModel:
            """Record model input batch sizes around the tiny Qwen model."""

            def __init__(self) -> None:
                self.inner = tiny_model()
                self.device = self.inner.device
                self.batch_sizes: list[int] = []

            def forward(self, **kwargs: object) -> object:
                input_ids = kwargs["input_ids"]
                self.batch_sizes.append(input_ids.shape[0])
                return self.inner.forward(**kwargs)

        model = CountingModel()
        engine = scheduler(model=model, max_batch_size=2)
        for request_id in ("a", "b"):
            engine.submit(
                [{"role": "user", "content": request_id}],
                request_id=request_id,
            )

        with cpu_scheduler_checks():
            engine.step()
            first_model_batches = engine.snapshot().model_batches
            engine.step()
            second_model_batches = engine.snapshot().model_batches

        self.assertEqual(first_model_batches, (("a", "b"),))
        self.assertEqual(second_model_batches, (("a", "b"),))
        self.assertEqual(model.batch_sizes, [2, 2])

    def test_round_robin_respects_one_request_batch_limit(self) -> None:
        """A small work batch rotates fairly through queued requests."""
        engine = scheduler(max_batch_size=1, token_budget=1)
        for request_id in ("a", "b", "c"):
            engine.submit(
                [{"role": "user", "content": request_id}],
                request_id=request_id,
            )

        batches = []
        with cpu_scheduler_checks():
            for _ in range(4):
                engine.step()
                batches.append(engine.snapshot().last_batch)

        self.assertEqual(batches, [("a",), ("b",), ("c",), ("a",)])

    def test_cancellation_reclaims_blocks_and_reservation(self) -> None:
        """Cancelling running work immediately frees all capacity."""
        engine = scheduler(max_batch_size=1)
        request_id = engine.submit(
            [{"role": "user", "content": "cancel me"}],
            request_id="cancel",
        )

        with cpu_scheduler_checks():
            engine.step()
        self.assertGreater(engine.cache_pool.allocated_block_count, 0)
        self.assertTrue(engine.cancel(request_id))

        self.assertIs(
            engine.request(request_id).status,
            RequestStatus.CANCELLED,
        )
        self.assertEqual(engine.cache_pool.allocated_block_count, 0)
        self.assertEqual(engine.snapshot().reserved_blocks, 0)
        self.assertFalse(engine.cancel(request_id))

    def test_decode_failure_does_not_leak_capacity(self) -> None:
        """Per-request model errors become terminal and release resources."""

        class FailingModel:
            """Delegate prefill, then fail the first decode call."""

            def __init__(self) -> None:
                self.inner = tiny_model()
                self.device = self.inner.device
                self.calls = 0

            def forward(self, **kwargs: object) -> object:
                self.calls += 1
                if self.calls == 2:
                    raise ValueError("planned decode failure")
                return self.inner.forward(**kwargs)

        engine = scheduler(model=FailingModel(), max_batch_size=1)
        request_id = engine.submit(
            [{"role": "user", "content": "fail"}],
            request_id="failure",
        )

        with cpu_scheduler_checks():
            engine.step()
            events = engine.step()

        view = engine.request(request_id)
        self.assertIs(view.status, RequestStatus.FAILED)
        self.assertIn("planned decode failure", view.error or "")
        self.assertEqual(events[-1].status, RequestStatus.FAILED)
        self.assertEqual(engine.cache_pool.allocated_block_count, 0)
        self.assertEqual(engine.snapshot().reserved_blocks, 0)

    def test_duplicate_and_unknown_request_ids_fail_clearly(self) -> None:
        """Request identity errors never alter admission counters."""
        engine = scheduler()
        engine.submit(
            [{"role": "user", "content": "known"}],
            request_id="known",
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            engine.submit(
                [{"role": "user", "content": "duplicate"}],
                request_id="known",
            )
        with self.assertRaisesRegex(KeyError, "unknown request_id"):
            engine.request("missing")

    def test_only_terminal_request_history_can_be_forgotten(self) -> None:
        """Serving may discard finished metadata but never active work."""
        engine = scheduler()
        request_id = engine.submit(
            [{"role": "user", "content": "known"}],
            request_id="known",
        )

        with self.assertRaisesRegex(ValueError, "active"):
            engine.forget(request_id)
        engine.cancel(request_id)
        engine.forget(request_id)

        with self.assertRaisesRegex(KeyError, "unknown request_id"):
            engine.request(request_id)
        replacement = engine.submit(
            [{"role": "user", "content": "replacement"}],
            request_id="replacement",
        )
        with cpu_scheduler_checks():
            engine.run_until_idle()
        self.assertIs(
            engine.request(replacement).status,
            RequestStatus.FINISHED,
        )
