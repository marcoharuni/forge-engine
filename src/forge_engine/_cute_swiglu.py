"""CuTe DSL source imported only on prepared H100/B200 environments."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from torch import Tensor


@cute.kernel
def _gate_up_swiglu_kernel(
    inputs: cute.Tensor,
    gate_weight: cute.Tensor,
    up_weight: cute.Tensor,
    output: cute.Tensor,
):
    """Compute one fused output element per CUDA thread."""
    threads = 256
    block_index, _, _ = cute.arch.block_idx()
    thread_index, _, _ = cute.arch.thread_idx()
    linear_index = block_index * threads + thread_index
    token_count = inputs.shape[0]
    output_size = gate_weight.shape[0]
    hidden_size = inputs.shape[1]
    if linear_index < token_count * output_size:
        token_index = linear_index // output_size
        output_index = linear_index % output_size
        gate = cutlass.Float32(0.0)
        up = cutlass.Float32(0.0)
        for hidden_index in range(hidden_size):
            input_value = inputs[token_index, hidden_index]
            gate += input_value * gate_weight[output_index, hidden_index]
            up += input_value * up_weight[output_index, hidden_index]
        sigmoid = cutlass.Float32(1.0) / (
            cutlass.Float32(1.0)
            + cute.exp2(-gate * cutlass.Float32(1.4426950408889634))
        )
        output[token_index, output_index] = gate * sigmoid * up


@cute.jit
def _launch_gate_up_swiglu(
    inputs: cute.Tensor,
    gate_weight: cute.Tensor,
    up_weight: cute.Tensor,
    output: cute.Tensor,
):
    threads = 256
    elements = inputs.shape[0] * gate_weight.shape[0]
    blocks = (elements + threads - 1) // threads
    _gate_up_swiglu_kernel(
        inputs,
        gate_weight,
        up_weight,
        output,
    ).launch(
        grid=(blocks, 1, 1),
        block=(threads, 1, 1),
    )


def run_cute_gate_up_swiglu(
    inputs: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
) -> Tensor:
    """Convert PyTorch tensors through DLPack and invoke the CuTe lab."""
    output = torch.empty(
        (inputs.shape[0], gate_weight.shape[0]),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    _launch_gate_up_swiglu(
        cute.runtime.from_dlpack(inputs),
        cute.runtime.from_dlpack(gate_weight),
        cute.runtime.from_dlpack(up_weight),
        cute.runtime.from_dlpack(output),
    )
    return output
