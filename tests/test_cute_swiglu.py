"""CPU-safe reference and hardware guards for the CuTe SwiGLU lab."""

from __future__ import annotations

from unittest import TestCase

import torch

from forge_engine.kernels.cute_swiglu import (
    cute_gate_up_swiglu_lab,
    gate_up_swiglu_reference,
)


class CuteSwiGLUTests(TestCase):
    """Reference numerics, automatic fallback, and strict hardware gating."""

    def test_reference_matches_direct_pytorch(self) -> None:
        torch.manual_seed(23)
        inputs = torch.randn(3, 8)
        gate = torch.randn(12, 8)
        up = torch.randn(12, 8)
        expected = (
            torch.nn.functional.silu(inputs @ gate.T) * (inputs @ up.T)
        )

        actual = gate_up_swiglu_reference(inputs, gate, up)

        torch.testing.assert_close(actual, expected)

    def test_cpu_automatically_uses_reference_fallback(self) -> None:
        inputs = torch.randn(2, 8)
        gate = torch.randn(12, 8)
        up = torch.randn(12, 8)

        actual = cute_gate_up_swiglu_lab(inputs, gate, up)
        expected = gate_up_swiglu_reference(inputs, gate, up)

        torch.testing.assert_close(actual, expected)

    def test_strict_lab_rejects_unprepared_hardware(self) -> None:
        inputs = torch.randn(2, 8)
        weights = torch.randn(12, 8)
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            cute_gate_up_swiglu_lab(
                inputs,
                weights,
                weights,
                require_cute=True,
            )

    def test_rejects_projection_shape_mismatch(self) -> None:
        inputs = torch.randn(2, 8)
        with self.assertRaisesRegex(ValueError, "matching"):
            gate_up_swiglu_reference(
                inputs,
                torch.randn(12, 8),
                torch.randn(11, 8),
            )
