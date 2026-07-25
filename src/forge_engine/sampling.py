"""Guarded token sampling for the explicit generation core."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Generator, Tensor


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Parameters controlling token sampling."""

    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0
    min_p: float = 0.0
    max_new_tokens: int = 256
    stop_strings: tuple[str, ...] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous or unsafe sampling settings."""
        if not math.isfinite(self.temperature) or self.temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 0.0 <= self.min_p <= 1.0:
            raise ValueError("min_p must be in [0, 1]")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if any(not stop for stop in self.stop_strings):
            raise ValueError("stop strings must be non-empty")
        if len(set(self.stop_strings)) != len(self.stop_strings):
            raise ValueError("stop strings must be unique")

    @property
    def greedy(self) -> bool:
        """Return whether deterministic argmax selection is configured."""
        return self.temperature == 0.0


def sample_token(
    logits: Tensor,
    params: SamplingParams,
    *,
    generator: Generator | None = None,
) -> Tensor:
    """Select one token per batch row after guarded probability filtering."""
    import torch

    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 1:
        raise ValueError("logits must have shape [batch, vocabulary]")
    if not torch.is_floating_point(logits):
        raise ValueError("logits must use a floating-point dtype")
    isnan = getattr(torch, "isnan", None)
    if callable(isnan) and isnan(logits).any():
        raise ValueError("logits must not contain NaN")
    isposinf = getattr(torch, "isposinf", None)
    if callable(isposinf) and isposinf(logits).any():
        raise ValueError("logits must not contain positive infinity")
    isfinite = getattr(torch, "isfinite", None)
    if callable(isfinite) and not isfinite(logits).any(dim=-1).all():
        raise ValueError("each logits row must contain a finite value")
    if params.greedy:
        return logits.argmax(dim=-1)

    filtered = filter_logits(logits, params)
    probabilities = torch.softmax(filtered, dim=-1, dtype=torch.float32)
    return torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    ).squeeze(-1)


def filter_logits(logits: Tensor, params: SamplingParams) -> Tensor:
    """Apply temperature, top-k, top-p, and min-p in that order."""
    import torch

    if params.greedy:
        raise ValueError("greedy sampling does not filter logits")
    filtered = logits.float()
    filtered = (
        filtered - filtered.max(dim=-1, keepdim=True).values
    ) / params.temperature
    negative_infinity = float("-inf")

    if params.top_k is not None and params.top_k < filtered.shape[-1]:
        top_values, top_indices = torch.topk(
            filtered,
            k=params.top_k,
            dim=-1,
        )
        kept = torch.full_like(filtered, negative_infinity)
        filtered = kept.scatter(-1, top_indices, top_values)

    if params.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            filtered,
            dim=-1,
            descending=True,
        )
        sorted_probabilities = torch.softmax(
            sorted_logits,
            dim=-1,
            dtype=torch.float32,
        )
        remove = sorted_probabilities.cumsum(dim=-1) > params.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(
            remove,
            negative_infinity,
        )
        filtered = torch.full_like(filtered, negative_infinity).scatter(
            -1,
            sorted_indices,
            sorted_logits,
        )

    if params.min_p > 0.0:
        probabilities = torch.softmax(
            filtered,
            dim=-1,
            dtype=torch.float32,
        )
        threshold = probabilities.max(dim=-1, keepdim=True).values
        remove = probabilities < threshold * params.min_p
        filtered = filtered.masked_fill(remove, negative_infinity)

    if not torch.isfinite(filtered).any(dim=-1).all():
        raise RuntimeError("sampling filters removed every token")
    return filtered


def normalize_stop_strings(values: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize optional CLI stop strings into an immutable tuple."""
    return () if values is None else tuple(values)
