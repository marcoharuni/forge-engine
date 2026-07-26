"""Tests for the contiguous correctness-baseline KV cache."""

from __future__ import annotations

from unittest import TestCase

import torch

from forge_engine.cache import (
    ContiguousKVCache,
    KVCacheCapacityError,
    PagedKVBlockPool,
    PagedKVCache,
)


def cache_layers(
    layer_count: int = 2,
    sequence_length: int = 3,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Build small non-contiguous key/value tensors."""
    layers = []
    for _ in range(layer_count):
        key = torch.randn(1, 2, 4, sequence_length).transpose(2, 3)
        value = torch.randn(1, 2, 4, sequence_length).transpose(2, 3)
        layers.append((key, value))
    return tuple(layers)


def numbered_cache_layers(
    layer_count: int = 2,
    sequence_length: int = 5,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Build deterministic cache tensors for exact paging checks."""
    layers = []
    for layer_index in range(layer_count):
        positions = torch.arange(sequence_length).view(1, 1, -1, 1) * 10
        heads = torch.arange(2).view(1, 2, 1, 1) * 100
        dimensions = torch.arange(4).view(1, 1, 1, 4)
        values = (positions + heads + dimensions).float()
        key = values + layer_index * 1_000
        value = values + layer_index * 1_000 + 500
        layers.append((key, value))
    return tuple(layers)


class ContiguousKVCacheTests(TestCase):
    """Cache ownership, validation, replacement, and cleanup."""

    def test_replace_makes_every_tensor_contiguous(self) -> None:
        """Model cache outputs are normalized into contiguous storage."""
        cache = ContiguousKVCache(layer_count=2)

        cache.replace(cache_layers())

        self.assertEqual(cache.layer_count, 2)
        self.assertEqual(cache.sequence_length, 3)
        self.assertTrue(
            all(
                key.is_contiguous() and value.is_contiguous()
                for key, value in cache.layers
            )
        )

    def test_replace_extends_and_clear_releases_layers(self) -> None:
        """A later model result replaces state and clear drops references."""
        cache = ContiguousKVCache()
        cache.replace(cache_layers(sequence_length=3))
        cache.replace(cache_layers(sequence_length=4))

        self.assertEqual(cache.sequence_length, 4)
        cache.clear()

        self.assertTrue(cache.empty)
        self.assertEqual(cache.sequence_length, 0)
        with self.assertRaisesRegex(ValueError, "empty cache"):
            cache.as_model_input()

    def test_rejects_layer_count_and_sequence_mismatches(self) -> None:
        """All layers must preserve count and cached sequence length."""
        cache = ContiguousKVCache(layer_count=2)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            cache.replace(cache_layers(layer_count=1))

        layers = list(cache_layers())
        layers[1] = cache_layers(layer_count=1, sequence_length=4)[0]
        with self.assertRaisesRegex(ValueError, "share a sequence length"):
            cache.replace(tuple(layers))

    def test_rejects_mismatched_key_value_metadata(self) -> None:
        """Each key/value pair must share shape, dtype, and device."""
        key = torch.randn(1, 2, 3, 4)
        value = torch.randn(1, 2, 4, 4)
        cache = ContiguousKVCache()

        with self.assertRaisesRegex(ValueError, "shapes must match"):
            cache.replace(((key, value),))


class PagedKVCacheTests(TestCase):
    """Physical allocation, reuse, fragmentation, and cleanup."""

    def test_prefill_materializes_exact_original_tensors(self) -> None:
        """A multi-block prefill gathers back without changing values."""
        original = numbered_cache_layers(sequence_length=5)
        pool = PagedKVBlockPool(block_size=2, capacity=4)
        cache = PagedKVCache(pool)

        cache.replace(original)
        materialized = cache.as_model_input()

        self.assertEqual(cache.block_table, (0, 1, 2))
        self.assertEqual(cache.sequence_length, 5)
        self.assertEqual(pool.allocated_block_count, 3)
        for actual_layer, expected_layer in zip(
            materialized,
            original,
            strict=True,
        ):
            torch.testing.assert_close(actual_layer[0], expected_layer[0])
            torch.testing.assert_close(actual_layer[1], expected_layer[1])

    def test_append_uses_partial_block_then_allocates_next_block(self) -> None:
        """Single-token decode fills a tail before reserving another block."""
        pool = PagedKVBlockPool(block_size=2, capacity=4)
        cache = PagedKVCache(pool)
        cache.replace(numbered_cache_layers(sequence_length=3))

        cache.append(numbered_cache_layers(sequence_length=4))
        self.assertEqual(cache.block_table, (0, 1))
        cache.append(numbered_cache_layers(sequence_length=5))

        self.assertEqual(cache.block_table, (0, 1, 2))
        self.assertEqual(cache.sequence_length, 5)
        for actual, expected in zip(
            cache.as_model_input(),
            numbered_cache_layers(sequence_length=5),
            strict=True,
        ):
            torch.testing.assert_close(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1])

    def test_direct_decode_appends_one_token_and_exposes_layer_pages(self) -> None:
        """The CUDA path reads pages directly and writes only new KV."""
        pool = PagedKVBlockPool(block_size=2, capacity=4)
        cache = PagedKVCache(pool)
        original = numbered_cache_layers(sequence_length=3)
        cache.replace(original)
        new_token = tuple(
            (key[:, :, -1:, :] + 100, value[:, :, -1:, :] + 100)
            for key, value in original
        )

        cache.append_token(new_token)
        view = cache.layer_view(1)

        self.assertEqual(cache.sequence_length, 4)
        self.assertEqual(view.sequence_length, 4)
        self.assertEqual(view.block_size, 2)
        self.assertEqual(len(view.keys), 2)
        expected_key = torch.cat(
            (original[1][0], new_token[1][0]),
            dim=2,
        )
        torch.testing.assert_close(
            cache.as_model_input()[1][0],
            expected_key,
        )

    def test_freed_holes_are_reused_without_moving_live_blocks(self) -> None:
        """A fragmented free ID can extend another live sequence."""
        pool = PagedKVBlockPool(block_size=2, capacity=5)
        first = PagedKVCache(pool)
        second = PagedKVCache(pool)
        first.replace(numbered_cache_layers(sequence_length=3))
        second.replace(numbered_cache_layers(sequence_length=3))
        self.assertEqual(first.block_table, (0, 1))
        self.assertEqual(second.block_table, (2, 3))

        first.clear()
        second.append(numbered_cache_layers(sequence_length=4))
        second.append(numbered_cache_layers(sequence_length=5))

        self.assertEqual(second.block_table, (2, 3, 0))
        self.assertEqual(pool.allocated_block_count, 3)
        for actual, expected in zip(
            second.as_model_input(),
            numbered_cache_layers(sequence_length=5),
            strict=True,
        ):
            torch.testing.assert_close(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1])

    def test_clear_reclaims_and_reuses_materialized_blocks(self) -> None:
        """Cleanup returns IDs while retaining their tensor allocations."""
        pool = PagedKVBlockPool(block_size=2, capacity=3)
        cache = PagedKVCache(pool)
        cache.replace(numbered_cache_layers(sequence_length=3))
        self.assertEqual(pool.materialized_block_count, 2)

        cache.clear()
        replacement = PagedKVCache(pool)
        replacement.replace(numbered_cache_layers(sequence_length=2))

        self.assertEqual(replacement.block_table, (0,))
        self.assertEqual(pool.allocated_block_count, 1)
        self.assertEqual(pool.materialized_block_count, 2)

    def test_capacity_failure_is_transactional(self) -> None:
        """An oversized prefill leaks neither block IDs nor tensors."""
        pool = PagedKVBlockPool(block_size=2, capacity=1)
        cache = PagedKVCache(pool)

        with self.assertRaisesRegex(
            KVCacheCapacityError,
            "only 1 are free",
        ):
            cache.replace(numbered_cache_layers(sequence_length=3))

        self.assertTrue(cache.empty)
        self.assertEqual(pool.allocated_block_count, 0)
        self.assertEqual(pool.materialized_block_count, 0)

    def test_decode_capacity_failure_preserves_owned_state(self) -> None:
        """A failed extension leaves the existing block table releasable."""
        pool = PagedKVBlockPool(block_size=2, capacity=1)
        cache = PagedKVCache(pool)
        cache.replace(numbered_cache_layers(sequence_length=2))

        with self.assertRaises(KVCacheCapacityError):
            cache.append(numbered_cache_layers(sequence_length=3))

        self.assertEqual(cache.sequence_length, 2)
        self.assertEqual(cache.block_table, (0,))
        self.assertEqual(pool.allocated_block_count, 1)
        cache.clear()
        self.assertEqual(pool.allocated_block_count, 0)

    def test_pool_rejects_double_release(self) -> None:
        """Invalid ownership changes fail before corrupting the free list."""
        pool = PagedKVBlockPool(block_size=2, capacity=2)
        cache = PagedKVCache(pool)
        cache.replace(numbered_cache_layers(sequence_length=1))
        block_ids = cache.block_table
        cache.clear()

        with self.assertRaisesRegex(ValueError, "unallocated"):
            pool.release(block_ids)
