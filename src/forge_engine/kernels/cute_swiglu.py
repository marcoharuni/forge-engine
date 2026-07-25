"""H100/B200 CuTe DSL fused gate/up SwiGLU experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def gate_up_swiglu_reference(
    inputs: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
) -> Tensor:
    """Apply both projections and the SiLU product in readable PyTorch."""
    _validate_gate_up(inputs, gate_weight, up_weight)
    from torch.nn import functional as F

    return F.silu(F.linear(inputs, gate_weight)) * F.linear(
        inputs,
        up_weight,
    )


def cute_gate_up_swiglu_lab(
    inputs: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    *,
    require_cute: bool = False,
) -> Tensor:
    """Run the deliberately simple CuTe fused projection lab or fallback."""
    _validate_gate_up(inputs, gate_weight, up_weight)
    reason = _cute_unsupported_reason(inputs, gate_weight, up_weight)
    if reason is None:
        try:
            from forge_engine._cute_swiglu import run_cute_gate_up_swiglu

            return run_cute_gate_up_swiglu(
                inputs,
                gate_weight,
                up_weight,
            )
        except Exception:
            if require_cute:
                raise
    elif require_cute:
        raise RuntimeError(f"CuTe SwiGLU lab unavailable: {reason}")
    return gate_up_swiglu_reference(inputs, gate_weight, up_weight)


def _validate_gate_up(
    inputs: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
) -> None:
    if inputs.ndim != 2 or inputs.shape[0] < 1 or inputs.shape[1] < 1:
        raise ValueError("inputs must have shape [tokens, hidden]")
    if gate_weight.ndim != 2 or gate_weight.shape != up_weight.shape:
        raise ValueError("gate and up weights must have matching rank-2 shapes")
    if gate_weight.shape[1] != inputs.shape[1]:
        raise ValueError("projection input dimension must match inputs")
    if not inputs.is_floating_point():
        raise ValueError("inputs and weights must be floating point")
    common = (inputs.dtype, inputs.device)
    if (gate_weight.dtype, gate_weight.device) != common or (
        up_weight.dtype,
        up_weight.device,
    ) != common:
        raise ValueError("inputs and weights must share dtype and device")


def _cute_unsupported_reason(
    inputs: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
) -> str | None:
    import torch

    if inputs.device.type != "cuda":
        return "CUDA is required"
    device_name = torch.cuda.get_device_name(inputs.device)
    if "H100" not in device_name and "B200" not in device_name:
        return "only prepared H100 or B200 hardware is allowed"
    if inputs.dtype != torch.bfloat16:
        return "only BF16 is supported"
    if tuple(inputs.shape[1:]) != (2_560,):
        return "hidden dimension must be 2560"
    if tuple(gate_weight.shape) != (9_728, 2_560):
        return "Qwen3 gate/up weights must have shape [9728, 2560]"
    if inputs.shape[0] > 128:
        return "the lab supports at most 128 tokens"
    if not (
        inputs.is_contiguous()
        and gate_weight.is_contiguous()
        and up_weight.is_contiguous()
    ):
        return "all tensors must be contiguous"
    return None
