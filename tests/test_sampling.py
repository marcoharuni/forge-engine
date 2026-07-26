"""Tests for deterministic and filtered token selection."""

from __future__ import annotations

from unittest import TestCase

import torch

from forge_engine.sampling import (
    SamplingParams,
    filter_logits,
    normalize_stop_strings,
    sample_token,
)


class SamplingTests(TestCase):
    """Sampling guards, filters, and reproducibility."""

    def test_invalid_parameters_are_rejected(self) -> None:
        """Unsafe sampling ranges fail before model loading."""
        invalid = (
            {"temperature": -0.1},
            {"temperature": float("inf")},
            {"temperature": float("nan")},
            {"top_k": 0},
            {"top_p": 0.0},
            {"top_p": 1.1},
            {"min_p": -0.1},
            {"min_p": 1.1},
            {"max_new_tokens": 0},
            {"stop_strings": ("",)},
            {"stop_strings": ("done", "done")},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                SamplingParams(**values)

    def test_greedy_returns_argmax(self) -> None:
        """Zero temperature selects the maximum logit in every row."""
        logits = torch.tensor([[1.0, 3.0, 2.0], [7.0, -1.0, 4.0]])

        actual = sample_token(logits, SamplingParams())

        torch.testing.assert_close(actual, torch.tensor([1, 0]))

    def test_top_k_keeps_only_largest_logits(self) -> None:
        """Top-k filtering retains exactly k finite candidates."""
        logits = torch.tensor([[0.0, 3.0, 2.0, 1.0]])
        params = SamplingParams(temperature=1.0, top_k=2)

        filtered = filter_logits(logits, params)

        self.assertEqual(
            torch.isfinite(filtered).nonzero()[:, 1].tolist(),
            [1, 2],
        )

    def test_top_p_keeps_smallest_probability_prefix(self) -> None:
        """Nucleus filtering always retains the leading candidate."""
        logits = torch.tensor([[5.0, 4.0, 0.0]])
        params = SamplingParams(temperature=1.0, top_p=0.7)

        filtered = filter_logits(logits, params)

        self.assertEqual(
            torch.isfinite(filtered).nonzero()[:, 1].tolist(),
            [0],
        )

    def test_min_p_uses_probability_relative_to_best_token(self) -> None:
        """Min-p removes tokens below a fraction of the best probability."""
        logits = torch.tensor([[0.0, -1.0, -3.0]])
        params = SamplingParams(temperature=1.0, min_p=0.2)

        filtered = filter_logits(logits, params)

        self.assertEqual(
            torch.isfinite(filtered).nonzero()[:, 1].tolist(),
            [0, 1],
        )

    def test_seeded_sampling_is_reproducible(self) -> None:
        """Identical device generators produce the same sample sequence."""
        logits = torch.zeros((1, 8))
        params = SamplingParams(temperature=1.0)
        left = torch.Generator().manual_seed(731)
        right = torch.Generator().manual_seed(731)

        left_tokens = [
            sample_token(logits, params, generator=left).item() for _ in range(16)
        ]
        right_tokens = [
            sample_token(logits, params, generator=right).item() for _ in range(16)
        ]

        self.assertEqual(left_tokens, right_tokens)

    def test_nonfinite_logits_are_rejected(self) -> None:
        """NaN and rows without any finite candidate fail clearly."""
        with self.assertRaisesRegex(ValueError, "NaN"):
            sample_token(torch.tensor([[0.0, float("nan")]]), SamplingParams())
        with self.assertRaisesRegex(ValueError, "finite"):
            sample_token(
                torch.tensor([[float("-inf"), float("-inf")]]),
                SamplingParams(),
            )
        with self.assertRaisesRegex(ValueError, "positive infinity"):
            sample_token(
                torch.tensor([[0.0, float("inf")]]),
                SamplingParams(),
            )

    def test_stop_values_are_normalized(self) -> None:
        """CLI list values become immutable runtime values."""
        self.assertEqual(normalize_stop_strings(None), ())
        self.assertEqual(normalize_stop_strings(["a", "b"]), ("a", "b"))
