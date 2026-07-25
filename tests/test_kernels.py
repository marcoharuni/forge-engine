"""CPU-safe references, guards, and fallback tests for M7 kernels."""

from __future__ import annotations

from unittest import TestCase

import torch

from forge_engine.kernels import (
    causal_prefill_attention_reference,
    residual_add_rms_norm,
    residual_add_rms_norm_reference,
    restricted_causal_prefill_attention,
)


class ResidualRMSNormTests(TestCase):
    """Reference accuracy and guarded fallback behavior."""

    def test_reference_matches_direct_float32_accumulation(self) -> None:
        residual = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
        update = torch.tensor([[0.5, 1.0, -1.5, 2.0]])
        weight = torch.tensor([1.0, 2.0, 3.0, 4.0])
        summed, normalized = residual_add_rms_norm_reference(
            residual,
            update,
            weight,
            1e-6,
        )
        expected_sum = residual + update
        expected_norm = expected_sum * torch.rsqrt(
            expected_sum.square().mean(dim=-1, keepdim=True) + 1e-6
        )

        torch.testing.assert_close(summed, expected_sum)
        torch.testing.assert_close(normalized, expected_norm * weight)

    def test_cpu_automatically_uses_reference_fallback(self) -> None:
        residual = torch.randn(2, 3, 8)
        update = torch.randn_like(residual)
        weight = torch.randn(8)

        actual = residual_add_rms_norm(
            residual,
            update,
            weight,
            1e-6,
        )
        expected = residual_add_rms_norm_reference(
            residual,
            update,
            weight,
            1e-6,
        )

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_strict_triton_rejects_cpu_before_import(self) -> None:
        inputs = torch.randn(2, 8)
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            residual_add_rms_norm(
                inputs,
                torch.zeros_like(inputs),
                torch.ones(8),
                1e-6,
                require_triton=True,
            )

    def test_rejects_shape_and_metadata_mismatches(self) -> None:
        residual = torch.randn(2, 8)
        with self.assertRaisesRegex(ValueError, "shapes"):
            residual_add_rms_norm_reference(
                residual,
                torch.randn(1, 8),
                torch.ones(8),
                1e-6,
            )
        with self.assertRaisesRegex(ValueError, "weight"):
            residual_add_rms_norm_reference(
                residual,
                torch.randn_like(residual),
                torch.ones(7),
                1e-6,
            )


class RestrictedPrefillTests(TestCase):
    """Restricted causal attention reference and fallback."""

    def test_reference_matches_scaled_dot_product_attention(self) -> None:
        torch.manual_seed(7)
        query = torch.randn(1, 2, 5, 4)
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        actual = causal_prefill_attention_reference(query, key, value)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
        )

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_cpu_automatically_uses_reference_fallback(self) -> None:
        query = torch.randn(1, 2, 4, 8)
        key = torch.randn_like(query)
        value = torch.randn_like(query)

        actual = restricted_causal_prefill_attention(query, key, value)
        expected = causal_prefill_attention_reference(query, key, value)

        torch.testing.assert_close(actual, expected)

    def test_strict_lab_reports_unsupported_cpu(self) -> None:
        query = torch.randn(1, 2, 4, 128)
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            restricted_causal_prefill_attention(
                query,
                query,
                query,
                require_triton=True,
            )

    def test_rejects_mismatched_qkv(self) -> None:
        query = torch.randn(1, 2, 4, 8)
        with self.assertRaisesRegex(ValueError, "shapes"):
            causal_prefill_attention_reference(
                query,
                torch.randn(1, 2, 3, 8),
                query,
            )
