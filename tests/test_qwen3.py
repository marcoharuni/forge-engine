"""CPU-safe numerical tests for the minimal Qwen3 runner."""

from __future__ import annotations

from unittest import TestCase

import torch

from forge_engine.qwen3 import Qwen3Config, Qwen3ForCausalLM, RMSNorm


def tiny_config() -> Qwen3Config:
    """Return a small architecture with Qwen3-compatible relationships."""
    return Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        eos_token_id=2,
        tie_word_embeddings=True,
        attention_bias=False,
    )


class Qwen3Tests(TestCase):
    """Reference math, shapes, guards, and contiguous cache behavior."""

    def test_rms_norm_matches_direct_reference(self) -> None:
        """RMSNorm accumulates variance in float32."""
        layer = RMSNorm(4, 1e-6)
        layer.weight.data.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        inputs = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
        expected = inputs * torch.rsqrt(
            inputs.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        expected = expected * layer.weight

        torch.testing.assert_close(layer(inputs), expected)

    def test_cached_last_token_matches_full_forward(self) -> None:
        """A prompt split into prefill and decode matches full attention."""
        torch.manual_seed(7)
        model = Qwen3ForCausalLM(tiny_config()).eval()
        complete_ids = torch.tensor([[4, 5, 6]], dtype=torch.long)

        with torch.inference_mode():
            complete = model(input_ids=complete_ids)
            prefill = model(input_ids=complete_ids[:, :2])
            decode = model(
                input_ids=complete_ids[:, 2:],
                attention_mask=torch.ones((1, 3), dtype=torch.long),
                past_key_values=prefill.past_key_values,
            )

        torch.testing.assert_close(
            decode.logits[:, -1], complete.logits[:, -1], atol=1e-5, rtol=1e-5
        )
        self.assertEqual(decode.past_key_values[0][0].shape, (1, 1, 3, 4))
        self.assertEqual(len(complete.layer_hidden_states), 2)

    def test_rejects_attention_mask_on_another_device(self) -> None:
        """Input metadata is checked before model math."""
        model = Qwen3ForCausalLM(tiny_config()).to("meta")
        input_ids = torch.empty((1, 1), dtype=torch.long, device="meta")
        mask = torch.ones((1, 1), dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "attention_mask"):
            model(input_ids=input_ids, attention_mask=mask)
