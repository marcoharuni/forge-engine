"""Tests for explicit prefill/decode state and incremental text."""

from __future__ import annotations

from unittest import TestCase

import torch

from forge_engine.cache import PagedKVBlockPool
from forge_engine.engine import GenerationCore, IncrementalDetokenizer
from forge_engine.qwen3 import Qwen3Config, Qwen3ForCausalLM


def tiny_model() -> Qwen3ForCausalLM:
    """Build a deterministic Qwen3-shaped CPU model for cache comparison."""
    torch.manual_seed(7)
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


class MappingTokenizer:
    """Decode complete token prefixes from a fixed mapping."""

    eos_token_id = 0

    def __init__(self, decoded: dict[tuple[int, ...], str]) -> None:
        self._decoded = decoded

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        """Return the configured full-prefix decode."""
        if not skip_special_tokens:
            raise AssertionError("special tokens must be skipped")
        return self._decoded[tuple(token_ids)]


class GenerationCoreTests(TestCase):
    """Cached/uncached agreement and detokenization boundaries."""

    def test_cached_decode_matches_uncached_forward(self) -> None:
        """Explicit one-token decode agrees with a complete recomputation."""
        model = tiny_model()
        pool = PagedKVBlockPool(block_size=2, capacity=4)
        core = GenerationCore(model, pool)
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        prompt_mask = torch.ones_like(prompt)
        complete = prompt

        with torch.inference_mode():
            state = core.prefill(prompt, prompt_mask)
            for token_id in (4, 5, 6, 7):
                next_token = torch.tensor([[token_id]], dtype=torch.long)
                state = core.decode(next_token, state)
                complete = torch.cat((complete, next_token), dim=1)
                uncached = core.uncached(
                    complete,
                    torch.ones_like(complete),
                )
                torch.testing.assert_close(
                    state.logits[:, -1, :],
                    uncached.logits[:, -1, :],
                    atol=1e-5,
                    rtol=1e-5,
                )

        self.assertEqual(state.cache.sequence_length, 7)
        self.assertEqual(state.cache.block_table, (0, 1, 2, 3))
        state.cache.clear()
        self.assertEqual(pool.allocated_block_count, 0)

    def test_tensor_batched_prefill_and_decode_match_uncached(self) -> None:
        """Compatible requests share model calls without changing logits."""
        model = tiny_model()
        pool = PagedKVBlockPool(block_size=2, capacity=8)
        core = GenerationCore(model, pool)
        prompts = (
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            torch.tensor([[5, 6, 7]], dtype=torch.long),
        )
        masks = tuple(torch.ones_like(prompt) for prompt in prompts)
        next_tokens = (
            torch.tensor([[4]], dtype=torch.long),
            torch.tensor([[8]], dtype=torch.long),
        )

        with torch.inference_mode():
            states = core.prefill_batch(tuple(zip(prompts, masks, strict=True)))
            for prompt, mask, state in zip(
                prompts,
                masks,
                states,
                strict=True,
            ):
                uncached = core.uncached(prompt, mask)
                torch.testing.assert_close(
                    state.logits,
                    uncached.logits,
                    atol=1e-5,
                    rtol=1e-5,
                )
            states = core.decode_batch(next_tokens, states)
            for prompt, next_token, state in zip(
                prompts,
                next_tokens,
                states,
                strict=True,
            ):
                complete = torch.cat((prompt, next_token), dim=1)
                uncached = core.uncached(complete, torch.ones_like(complete))
                torch.testing.assert_close(
                    state.logits[:, -1, :],
                    uncached.logits[:, -1, :],
                    atol=1e-5,
                    rtol=1e-5,
                )

        for state in states:
            state.cache.clear()
        self.assertEqual(pool.allocated_block_count, 0)

    def test_detokenizer_withholds_stop_across_tokens(self) -> None:
        """Neither a partial nor complete cross-token stop leaks."""
        tokenizer = MappingTokenizer(
            {
                (1,): "answer<",
                (1, 2): "answer<END>",
            }
        )
        detokenizer = IncrementalDetokenizer(tokenizer, ("<END>",))

        self.assertEqual(detokenizer.push(1), "answer")
        self.assertEqual(detokenizer.push(2), "")
        self.assertTrue(detokenizer.stopped)
        self.assertEqual(detokenizer.finish(), "")

    def test_detokenizer_waits_for_split_unicode(self) -> None:
        """A trailing replacement marker is revised before it is emitted."""
        tokenizer = MappingTokenizer(
            {
                (1,): "caf\ufffd",
                (1, 2): "café",
            }
        )
        detokenizer = IncrementalDetokenizer(tokenizer)

        self.assertEqual(detokenizer.push(1), "caf")
        self.assertEqual(detokenizer.push(2), "é")
        self.assertEqual(detokenizer.finish(), "")

    def test_finish_flushes_partial_stop_prefix(self) -> None:
        """A token-limit finish emits an incomplete stop prefix."""
        tokenizer = MappingTokenizer({(1,): "answer<"})
        detokenizer = IncrementalDetokenizer(tokenizer, ("<END>",))

        self.assertEqual(detokenizer.push(1), "answer")
        self.assertEqual(detokenizer.finish(), "<")
