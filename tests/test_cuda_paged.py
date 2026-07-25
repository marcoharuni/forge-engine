"""CPU-safe reference and guard tests for CUDA paged GQA decode."""

from __future__ import annotations

from unittest import TestCase

import torch

from forge_engine.kernels.cuda_paged import (
    paged_gqa_decode,
    paged_gqa_decode_reference,
)


def paged_inputs(
    *,
    head_dim: int = 8,
    past_length: int = 5,
    block_size: int = 4,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    torch.Tensor,
    torch.Tensor,
]:
    """Build a fragmented logical sequence with two KV heads."""
    torch.manual_seed(17)
    query = torch.randn(1, 4, 1, head_dim)
    page_count = (past_length + block_size - 1) // block_size
    keys = tuple(
        torch.randn(1, 2, block_size, head_dim)
        for _ in range(page_count)
    )
    values = tuple(torch.randn_like(key) for key in keys)
    current_key = torch.randn(1, 2, 1, head_dim)
    current_value = torch.randn_like(current_key)
    return query, keys, values, current_key, current_value


class PagedGQADecodeTests(TestCase):
    """Numerics, GQA mapping, strict mode, and metadata guards."""

    def test_reference_matches_explicit_contiguous_gqa(self) -> None:
        query, keys, values, current_key, current_value = paged_inputs()
        actual = paged_gqa_decode_reference(
            query,
            keys,
            values,
            current_key,
            current_value,
            5,
        )
        contiguous_key = torch.cat(
            (keys[0], keys[1][:, :, :1], current_key),
            dim=2,
        ).repeat_interleave(2, dim=1)
        contiguous_value = torch.cat(
            (values[0], values[1][:, :, :1], current_value),
            dim=2,
        ).repeat_interleave(2, dim=1)
        probabilities = torch.softmax(
            torch.matmul(query, contiguous_key.transpose(-2, -1))
            * query.shape[-1] ** -0.5,
            dim=-1,
        )
        expected = torch.matmul(probabilities, contiguous_value)

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_cpu_automatically_uses_reference_fallback(self) -> None:
        query, keys, values, current_key, current_value = paged_inputs()
        actual = paged_gqa_decode(
            query,
            keys,
            values,
            current_key,
            current_value,
            5,
        )
        expected = paged_gqa_decode_reference(
            query,
            keys,
            values,
            current_key,
            current_value,
            5,
        )
        torch.testing.assert_close(actual, expected)

    def test_strict_cuda_mode_rejects_cpu(self) -> None:
        query, keys, values, current_key, current_value = paged_inputs(
            head_dim=128
        )
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            paged_gqa_decode(
                query,
                keys,
                values,
                current_key,
                current_value,
                5,
                require_cuda_kernel=True,
            )

    def test_rejects_incomplete_page_table(self) -> None:
        query, keys, values, current_key, current_value = paged_inputs()
        with self.assertRaisesRegex(ValueError, "page count"):
            paged_gqa_decode_reference(
                query,
                keys[:1],
                values[:1],
                current_key,
                current_value,
                5,
            )

    def test_rejects_short_physical_page(self) -> None:
        query, keys, values, current_key, current_value = paged_inputs()
        short_key = keys[1][:, :, :1, :]
        with self.assertRaisesRegex(ValueError, "shape changed"):
            paged_gqa_decode_reference(
                query,
                (keys[0], short_key),
                values,
                current_key,
                current_value,
                5,
            )
