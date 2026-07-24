"""Sampling parameter types."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Parameters controlling token sampling."""

    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 256
