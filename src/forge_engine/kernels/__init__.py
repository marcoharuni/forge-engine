"""Guarded low-level kernels with readable PyTorch fallbacks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def residual_add_rms_norm_reference(
    residual: Tensor,
    update: Tensor,
    weight: Tensor,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    """Add one residual update and RMS-normalize in readable PyTorch."""
    _validate_residual_norm(residual, update, weight, epsilon)
    import torch

    summed = residual + update
    normalized = summed.float()
    variance = normalized.square().mean(dim=-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + epsilon)
    return summed, weight * normalized.to(summed.dtype)


def residual_add_rms_norm(
    residual: Tensor,
    update: Tensor,
    weight: Tensor,
    epsilon: float,
    *,
    require_triton: bool = False,
) -> tuple[Tensor, Tensor]:
    """Use fused Triton inference when supported, otherwise use PyTorch."""
    _validate_residual_norm(residual, update, weight, epsilon)
    reason = _triton_residual_norm_unsupported_reason(
        residual,
        update,
        weight,
    )
    if reason is None:
        try:
            from forge_engine._triton_kernels import (
                triton_residual_add_rms_norm,
            )

            return triton_residual_add_rms_norm(
                residual,
                update,
                weight,
                epsilon,
            )
        except Exception:
            if require_triton:
                raise
    elif require_triton:
        raise RuntimeError(f"Triton residual RMSNorm unavailable: {reason}")
    return residual_add_rms_norm_reference(
        residual,
        update,
        weight,
        epsilon,
    )


def causal_prefill_attention_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Compute causal attention directly for lab correctness."""
    _validate_prefill_attention(query, key, value, scale)
    import torch
    from torch.nn import functional as F

    effective_scale = (
        query.shape[-1] ** -0.5 if scale is None else scale
    )
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
    scores = scores * effective_scale
    sequence_length = query.shape[-2]
    allowed = torch.ones(
        (sequence_length, sequence_length),
        dtype=torch.bool,
        device=query.device,
    ).tril()
    scores = scores.masked_fill(~allowed, float("-inf"))
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32)
    return torch.matmul(probabilities, value.float()).to(query.dtype)


def restricted_causal_prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    scale: float | None = None,
    require_triton: bool = False,
) -> Tensor:
    """Run the restricted Triton online-softmax lab or its reference."""
    _validate_prefill_attention(query, key, value, scale)
    reason = _triton_prefill_unsupported_reason(query, key, value)
    if reason is None:
        try:
            from forge_engine._triton_kernels import (
                triton_restricted_causal_prefill,
            )

            return triton_restricted_causal_prefill(
                query,
                key,
                value,
                query.shape[-1] ** -0.5 if scale is None else scale,
            )
        except Exception:
            if require_triton:
                raise
    elif require_triton:
        raise RuntimeError(f"Triton prefill lab unavailable: {reason}")
    return causal_prefill_attention_reference(
        query,
        key,
        value,
        scale=scale,
    )


def _validate_residual_norm(
    residual: Tensor,
    update: Tensor,
    weight: Tensor,
    epsilon: float,
) -> None:
    """Guard shape, dtype, device, and numerical metadata."""
    if getattr(residual, "ndim", None) is None or residual.ndim < 1:
        raise ValueError("residual must have at least one dimension")
    if residual.shape != update.shape:
        raise ValueError("residual and update shapes must match")
    if residual.dtype != update.dtype or residual.device != update.device:
        raise ValueError("residual and update metadata must match")
    if not residual.is_floating_point():
        raise ValueError("residual and update must be floating point")
    if weight.ndim != 1 or weight.shape[0] != residual.shape[-1]:
        raise ValueError("weight must match the final hidden dimension")
    if weight.dtype != residual.dtype or weight.device != residual.device:
        raise ValueError("weight metadata must match residual")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")


def _validate_prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float | None,
) -> None:
    """Guard common reference and lab attention metadata."""
    if query.ndim != 4:
        raise ValueError("query must have shape [batch, heads, sequence, head]")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value shapes must match")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value dtypes must match")
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value devices must match")
    if not query.is_floating_point():
        raise ValueError("query, key, and value must be floating point")
    if query.shape[0] < 1 or query.shape[1] < 1:
        raise ValueError("attention batch and head counts must be positive")
    if query.shape[2] < 1 or query.shape[3] < 1:
        raise ValueError("attention sequence and head dimensions must be positive")
    if scale is not None and (not math.isfinite(scale) or scale <= 0.0):
        raise ValueError("attention scale must be finite and positive")


def _triton_residual_norm_unsupported_reason(
    residual: Tensor,
    update: Tensor,
    weight: Tensor,
) -> str | None:
    if residual.device.type != "cuda":
        return "CUDA is required"
    if residual.dtype not in _supported_half_dtypes():
        return "only FP16 and BF16 are supported"
    if not (
        residual.is_contiguous()
        and update.is_contiguous()
        and weight.is_contiguous()
    ):
        return "all tensors must be contiguous"
    feature_size = residual.shape[-1]
    block_size = 1 << (feature_size - 1).bit_length()
    if block_size > 65_536 // residual.element_size():
        return "one row exceeds the 64 KiB fused limit"
    return None


def _triton_prefill_unsupported_reason(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> str | None:
    if query.device.type != "cuda":
        return "CUDA is required"
    if query.dtype not in _supported_half_dtypes():
        return "only FP16 and BF16 are supported"
    if not (
        query.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
    ):
        return "all tensors must be contiguous"
    if query.shape[0] != 1:
        return "batch size must be exactly 1"
    if query.shape[-1] != 128:
        return "head dimension must be exactly 128"
    if query.shape[-2] > 2_048:
        return "sequence length must not exceed 2048"
    return None


def _supported_half_dtypes() -> tuple[object, object]:
    import torch

    return torch.float16, torch.bfloat16
