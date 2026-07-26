"""CUDA C++ single-token paged GQA decode with PyTorch fallback."""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

_EXTENSION: object | None = None
_EXTENSION_ERROR: BaseException | None = None
_EXTENSION_LOCK = threading.Lock()


def paged_gqa_decode_reference(
    query: Tensor,
    key_pages: Sequence[Tensor],
    value_pages: Sequence[Tensor],
    current_key: Tensor,
    current_value: Tensor,
    past_length: int,
    *,
    scale: float | None = None,
) -> Tensor:
    """Gather one layer's logical pages and compute readable GQA attention."""
    block_size, groups = _validate_paged_decode(
        query,
        key_pages,
        value_pages,
        current_key,
        current_value,
        past_length,
        scale,
    )
    import torch

    remaining = past_length
    keys = []
    values = []
    for key_page, value_page in zip(
        key_pages,
        value_pages,
        strict=True,
    ):
        count = min(block_size, remaining)
        keys.append(key_page[:, :, :count, :])
        values.append(value_page[:, :, :count, :])
        remaining -= count
    keys.append(current_key)
    values.append(current_value)
    key = torch.cat(tuple(keys), dim=2)
    value = torch.cat(tuple(values), dim=2)
    repeated_key = key.repeat_interleave(groups, dim=1)
    repeated_value = value.repeat_interleave(groups, dim=1)
    effective_scale = query.shape[-1] ** -0.5 if scale is None else scale
    scores = torch.matmul(
        query.float(),
        repeated_key.float().transpose(-2, -1),
    )
    probabilities = torch.softmax(
        scores * effective_scale,
        dim=-1,
        dtype=torch.float32,
    )
    return torch.matmul(probabilities, repeated_value.float()).to(query.dtype)


def paged_gqa_decode(
    query: Tensor,
    key_pages: Sequence[Tensor],
    value_pages: Sequence[Tensor],
    current_key: Tensor,
    current_value: Tensor,
    past_length: int,
    *,
    scale: float | None = None,
    require_cuda_kernel: bool = False,
) -> Tensor:
    """Run the CUDA C++ kernel when guarded metadata is supported."""
    _validate_paged_decode(
        query,
        key_pages,
        value_pages,
        current_key,
        current_value,
        past_length,
        scale,
    )
    reason = _cuda_unsupported_reason(
        query,
        key_pages,
        value_pages,
        current_key,
        current_value,
    )
    if reason is None:
        try:
            extension = _load_extension()
            effective_scale = query.shape[-1] ** -0.5 if scale is None else scale
            return extension.paged_gqa_decode_cuda(
                query,
                list(key_pages),
                list(value_pages),
                current_key,
                current_value,
                past_length,
                effective_scale,
            )
        except Exception:
            if require_cuda_kernel:
                raise
    elif require_cuda_kernel:
        raise RuntimeError(f"CUDA paged GQA unavailable: {reason}")
    return paged_gqa_decode_reference(
        query,
        key_pages,
        value_pages,
        current_key,
        current_value,
        past_length,
        scale=scale,
    )


def cuda_extension_path() -> str | None:
    """Return the compiled extension path after a successful strict launch."""
    return getattr(_EXTENSION, "__file__", None)


def _load_extension() -> object:
    """JIT compile once; retain the first failure for cheap future fallback."""
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        raise RuntimeError("CUDA paged extension build previously failed") from (
            _EXTENSION_ERROR
        )
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if _EXTENSION_ERROR is not None:
            raise RuntimeError(
                "CUDA paged extension build previously failed"
            ) from _EXTENSION_ERROR
        try:
            from torch.utils.cpp_extension import load_inline

            _EXTENSION = load_inline(
                name="forge_paged_gqa_decode_v2",
                cpp_sources=_CPP_SOURCE,
                cuda_sources=_CUDA_SOURCE,
                functions=["paged_gqa_decode_cuda"],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "-lineinfo"],
                with_cuda=True,
                verbose=False,
                keep_intermediates=True,
            )
        except BaseException as error:
            _EXTENSION_ERROR = error
            raise
    return _EXTENSION


def _validate_paged_decode(
    query: Tensor,
    key_pages: Sequence[Tensor],
    value_pages: Sequence[Tensor],
    current_key: Tensor,
    current_value: Tensor,
    past_length: int,
    scale: float | None,
) -> tuple[int, int]:
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[2] != 1:
        raise ValueError("query must have shape [1, query_heads, 1, head_dim]")
    if not query.is_floating_point():
        raise ValueError("query must be floating point")
    if not key_pages or len(key_pages) != len(value_pages):
        raise ValueError("key and value pages must be non-empty and aligned")
    first_key = key_pages[0]
    if first_key.ndim != 4 or first_key.shape[0] != 1:
        raise ValueError("pages must have shape [1, kv_heads, block, head_dim]")
    block_size = first_key.shape[2]
    kv_heads = first_key.shape[1]
    head_dim = first_key.shape[3]
    if block_size < 1 or kv_heads < 1 or head_dim < 1:
        raise ValueError("page dimensions must be positive")
    if query.shape[1] % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if query.shape[3] != head_dim:
        raise ValueError("query and page head dimensions must match")
    expected_current = (1, kv_heads, 1, head_dim)
    if tuple(current_key.shape) != expected_current:
        raise ValueError("current key shape does not match paged layout")
    if current_key.shape != current_value.shape:
        raise ValueError("current key and value shapes must match")
    common = (query.dtype, query.device)
    expected_page = (1, kv_heads, block_size, head_dim)
    for page in tuple(key_pages) + tuple(value_pages):
        if page.ndim != 4 or tuple(page.shape) != expected_page:
            raise ValueError("paged key/value tensor shape changed")
        if (page.dtype, page.device) != common:
            raise ValueError("all paged decode metadata must match query")
    for current in (current_key, current_value):
        if (current.dtype, current.device) != common:
            raise ValueError("all paged decode metadata must match query")
    if (
        not isinstance(past_length, int)
        or isinstance(past_length, bool)
        or past_length < 1
    ):
        raise ValueError("past_length must be positive")
    required_pages = (past_length + block_size - 1) // block_size
    if len(key_pages) != required_pages:
        raise ValueError("page count does not cover past_length exactly")
    if scale is not None and (not math.isfinite(scale) or scale <= 0.0):
        raise ValueError("attention scale must be finite and positive")
    return block_size, query.shape[1] // kv_heads


def _cuda_unsupported_reason(
    query: Tensor,
    key_pages: Sequence[Tensor],
    value_pages: Sequence[Tensor],
    current_key: Tensor,
    current_value: Tensor,
) -> str | None:
    import torch

    if query.device.type != "cuda":
        return "CUDA is required"
    if query.dtype not in (torch.float16, torch.bfloat16):
        return "only FP16 and BF16 are supported"
    if query.shape[-1] != 128:
        return "head dimension must be exactly 128"
    if not all(
        tensor.is_contiguous()
        for tensor in (
            query,
            *key_pages,
            *value_pages,
            current_key,
            current_value,
        )
    ):
        return "all tensors must be contiguous"
    return None


_CPP_SOURCE = r"""
#include <vector>
torch::Tensor paged_gqa_decode_cuda(
    torch::Tensor query,
    std::vector<torch::Tensor> key_pages,
    std::vector<torch::Tensor> value_pages,
    torch::Tensor current_key,
    torch::Tensor current_value,
    int64_t past_length,
    double scale);
"""


_CUDA_SOURCE = r"""
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void paged_gqa_kernel(
    const scalar_t* query,
    const scalar_t* const* key_pages,
    const scalar_t* const* value_pages,
    const scalar_t* current_key,
    const scalar_t* current_value,
    scalar_t* output,
    int query_heads,
    int kv_heads,
    int block_size,
    int past_length,
    float scale) {
  constexpr int HEAD_DIM = 128;
  const int query_head = blockIdx.x;
  const int dimension = threadIdx.x;
  const int groups = query_heads / kv_heads;
  const int kv_head = query_head / groups;
  const float query_value = static_cast<float>(
      query[query_head * HEAD_DIM + dimension]);
  float accumulator = 0.0f;
  __shared__ float reduction[HEAD_DIM];
  __shared__ float probability;
  __shared__ float correction;
  __shared__ float maximum;
  __shared__ float denominator;
  if (dimension == 0) {
    maximum = -INFINITY;
    denominator = 0.0f;
  }
  __syncthreads();

  for (int position = 0; position <= past_length; ++position) {
    const scalar_t* key;
    const scalar_t* value;
    int offset;
    if (position == past_length) {
      key = current_key;
      value = current_value;
      offset = kv_head * HEAD_DIM;
    } else {
      const int logical_block = position / block_size;
      const int block_offset = position % block_size;
      key = key_pages[logical_block];
      value = value_pages[logical_block];
      offset = (kv_head * block_size + block_offset) * HEAD_DIM;
    }
    reduction[dimension] =
        query_value * static_cast<float>(key[offset + dimension]);
    __syncthreads();
    for (int stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
      if (dimension < stride) {
        reduction[dimension] += reduction[dimension + stride];
      }
      __syncthreads();
    }
    if (dimension == 0) {
      const float score = reduction[0] * scale;
      const float next_maximum = fmaxf(maximum, score);
      correction = expf(maximum - next_maximum);
      probability = expf(score - next_maximum);
      denominator = denominator * correction + probability;
      maximum = next_maximum;
    }
    __syncthreads();
    accumulator = accumulator * correction
        + probability * static_cast<float>(value[offset + dimension]);
    __syncthreads();
  }
  output[query_head * HEAD_DIM + dimension] =
      static_cast<scalar_t>(accumulator / denominator);
}

torch::Tensor paged_gqa_decode_cuda(
    torch::Tensor query,
    std::vector<torch::Tensor> key_pages,
    std::vector<torch::Tensor> value_pages,
    torch::Tensor current_key,
    torch::Tensor current_value,
    int64_t past_length,
    double scale) {
  auto output = torch::empty_like(query);
  const int64_t page_count = static_cast<int64_t>(key_pages.size());
  std::vector<int64_t> host_key_pointers(page_count);
  std::vector<int64_t> host_value_pointers(page_count);
  for (int64_t index = 0; index < page_count; ++index) {
    host_key_pointers[index] =
        reinterpret_cast<int64_t>(key_pages[index].data_ptr());
    host_value_pointers[index] =
        reinterpret_cast<int64_t>(value_pages[index].data_ptr());
  }
  auto pointer_options = torch::TensorOptions()
      .dtype(torch::kInt64)
      .device(query.device());
  auto key_pointer_tensor = torch::empty({page_count}, pointer_options);
  auto value_pointer_tensor = torch::empty({page_count}, pointer_options);
  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(query.get_device()).stream();
  C10_CUDA_CHECK(cudaMemcpyAsync(
      key_pointer_tensor.data_ptr<int64_t>(),
      host_key_pointers.data(),
      page_count * sizeof(int64_t),
      cudaMemcpyHostToDevice,
      stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      value_pointer_tensor.data_ptr<int64_t>(),
      host_value_pointers.data(),
      page_count * sizeof(int64_t),
      cudaMemcpyHostToDevice,
      stream));
  const int query_heads = static_cast<int>(query.size(1));
  const int kv_heads = static_cast<int>(current_key.size(1));
  const int block_size = static_cast<int>(key_pages[0].size(2));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "forge_paged_gqa_decode",
      [&] {
        paged_gqa_kernel<scalar_t><<<query_heads, 128, 0, stream>>>(
            query.data_ptr<scalar_t>(),
            reinterpret_cast<const scalar_t* const*>(
                key_pointer_tensor.data_ptr<int64_t>()),
            reinterpret_cast<const scalar_t* const*>(
                value_pointer_tensor.data_ptr<int64_t>()),
            current_key.data_ptr<scalar_t>(),
            current_value.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            query_heads,
            kv_heads,
            block_size,
            static_cast<int>(past_length),
            static_cast<float>(scale));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""
