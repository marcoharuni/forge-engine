"""Triton implementations imported only on CUDA-capable installations."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

_LAST_ARTIFACTS: dict[str, dict[str, str | bytes]] = {}


def _capture_artifacts(name: str, compiled: object) -> None:
    """Retain generated PTX/cubin for explicit M7 inspection."""
    assembly = getattr(compiled, "asm", {})
    if not isinstance(assembly, dict):
        return
    selected = {
        kind: artifact
        for kind in ("ptx", "cubin")
        if isinstance((artifact := assembly.get(kind)), (str, bytes))
    }
    if selected:
        _LAST_ARTIFACTS[name] = selected


def triton_kernel_artifacts() -> dict[str, dict[str, str | bytes]]:
    """Return generated artifacts captured by successful kernel launches."""
    return {name: dict(artifacts) for name, artifacts in _LAST_ARTIFACTS.items()}


@triton.jit
def _residual_rms_kernel(
    residual_pointer,
    update_pointer,
    weight_pointer,
    summed_pointer,
    normalized_pointer,
    row_size: tl.constexpr,
    epsilon: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < row_size
    offset = row * row_size + columns
    residual = tl.load(residual_pointer + offset, mask=mask, other=0.0)
    update = tl.load(update_pointer + offset, mask=mask, other=0.0)
    summed = residual.to(tl.float32) + update.to(tl.float32)
    variance = tl.sum(summed * summed, axis=0) / row_size
    inverse_rms = tl.rsqrt(variance + epsilon)
    weight = tl.load(weight_pointer + columns, mask=mask, other=0.0)
    tl.store(summed_pointer + offset, summed, mask=mask)
    tl.store(
        normalized_pointer + offset,
        summed * inverse_rms * weight,
        mask=mask,
    )


def triton_residual_add_rms_norm(
    residual: Tensor,
    update: Tensor,
    weight: Tensor,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    """Launch one program per flattened input row."""
    summed = torch.empty_like(residual)
    normalized = torch.empty_like(residual)
    row_size = residual.shape[-1]
    rows = residual.numel() // row_size
    block_size = triton.next_power_of_2(row_size)
    warps = 8 if block_size >= 8_192 else 4
    compiled = _residual_rms_kernel[(rows,)](
        residual,
        update,
        weight,
        summed,
        normalized,
        row_size,
        epsilon,
        BLOCK_SIZE=block_size,
        num_warps=warps,
    )
    _capture_artifacts("residual_add_rms_norm", compiled)
    return summed, normalized


@triton.jit
def _restricted_causal_prefill_kernel(
    query_pointer,
    key_pointer,
    value_pointer,
    output_pointer,
    query_head_stride: tl.constexpr,
    query_sequence_stride: tl.constexpr,
    key_head_stride: tl.constexpr,
    key_sequence_stride: tl.constexpr,
    value_head_stride: tl.constexpr,
    value_sequence_stride: tl.constexpr,
    output_head_stride: tl.constexpr,
    output_sequence_stride: tl.constexpr,
    sequence_length: tl.constexpr,
    scale: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_index = tl.program_id(0)
    head_index = tl.program_id(1)
    dimensions = tl.arange(0, HEAD_DIM)
    query = tl.load(
        query_pointer
        + head_index * query_head_stride
        + query_index * query_sequence_stride
        + dimensions
    ).to(tl.float32)
    maximum = -float("inf")
    denominator = 0.0
    accumulator = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    key_offsets = tl.arange(0, BLOCK_N)
    for start in range(0, sequence_length, BLOCK_N):
        positions = start + key_offsets
        allowed = (positions < sequence_length) & (positions <= query_index)
        keys = tl.load(
            key_pointer
            + head_index * key_head_stride
            + positions[:, None] * key_sequence_stride
            + dimensions[None, :],
            mask=allowed[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(allowed, scores, -float("inf"))
        block_maximum = tl.max(scores, axis=0)
        next_maximum = tl.maximum(maximum, block_maximum)
        correction = tl.exp(maximum - next_maximum)
        probabilities = tl.exp(scores - next_maximum)
        probabilities = tl.where(allowed, probabilities, 0.0)
        values = tl.load(
            value_pointer
            + head_index * value_head_stride
            + positions[:, None] * value_sequence_stride
            + dimensions[None, :],
            mask=allowed[:, None],
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * correction + tl.sum(
            probabilities[:, None] * values, axis=0
        )
        denominator = denominator * correction + tl.sum(probabilities, axis=0)
        maximum = next_maximum
    tl.store(
        output_pointer
        + head_index * output_head_stride
        + query_index * output_sequence_stride
        + dimensions,
        accumulator / denominator,
    )


def triton_restricted_causal_prefill(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
) -> Tensor:
    """Launch the batch-one, head-dimension-128 causal prefill lab."""
    output = torch.empty_like(query)
    sequence_length = query.shape[2]
    compiled = _restricted_causal_prefill_kernel[(sequence_length, query.shape[1])](
        query,
        key,
        value,
        output,
        query.stride(1),
        query.stride(2),
        key.stride(1),
        key.stride(2),
        value.stride(1),
        value.stride(2),
        output.stride(1),
        output.stride(2),
        sequence_length,
        scale,
        HEAD_DIM=128,
        BLOCK_N=32,
        num_warps=4,
        num_stages=2,
    )
    _capture_artifacts("restricted_causal_prefill", compiled)
    return output
