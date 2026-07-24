"""Explicit greedy decoding with a transformer key-value cache."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Protocol

from forge_engine.config import EngineConfig
from forge_engine.model import (
    CUDAOutOfMemoryError,
    ChatMessage,
    LanguageModel,
    LoadedModel,
    Tokenizer,
    load_model,
)


class InferenceEngine(Protocol):
    """Streaming interface consumed by front ends."""

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        """Stream decoded fragments for a chat conversation."""
        ...


class GreedyEngine:
    """A small single-GPU greedy inference engine."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        model: LanguageModel,
        max_new_tokens: int,
    ) -> None:
        """Store the loaded components and generation limit."""
        self._tokenizer = tokenizer
        self._model = model
        self._max_new_tokens = max_new_tokens

    @classmethod
    def from_config(cls, config: EngineConfig) -> GreedyEngine:
        """Load a CUDA model and construct an engine."""
        loaded: LoadedModel = load_model(config)
        return cls(loaded.tokenizer, loaded.model, config.max_new_tokens)

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        """Stream greedy decoded text for a chat conversation."""
        import torch

        eos_ids = _eos_ids(self._tokenizer.eos_token_id)
        generated_ids: list[int] = []
        decoded = ""
        cached_sequence_length = 0

        try:
            tokenizer_output = self._tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            input_ids, attention_mask = _normalize_tokenizer_output(
                tokenizer_output
            )
            input_ids = input_ids.to(self._model.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._model.device)
            _validate_input_ids(input_ids, torch)
            _validate_attention_mask(attention_mask, input_ids)
            past_key_values: object | None = None

            with torch.inference_mode():
                for _ in range(self._max_new_tokens):
                    input_length = input_ids.shape[1]
                    output = self._model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    _validate_logits(output.logits, input_ids, torch)
                    cached_sequence_length += input_length
                    _validate_cache(
                        output.past_key_values,
                        batch_size=input_ids.shape[0],
                        sequence_length=cached_sequence_length,
                    )
                    next_token = output.logits[:, -1, :].argmax(dim=-1)
                    token_id = int(next_token.item())
                    past_key_values = output.past_key_values

                    if token_id in eos_ids:
                        break

                    generated_ids.append(token_id)
                    text = self._tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                    )
                    fragment = text[len(decoded) :]
                    decoded = text
                    if fragment:
                        yield fragment

                    input_ids = next_token.unsqueeze(-1)
                    if attention_mask is not None:
                        attention_mask = torch.cat(
                            (
                                attention_mask,
                                torch.ones(
                                    (attention_mask.shape[0], 1),
                                    dtype=attention_mask.dtype,
                                    device=self._model.device,
                                ),
                            ),
                            dim=1,
                        )
        except torch.cuda.OutOfMemoryError as error:
            raise CUDAOutOfMemoryError(
                "CUDA out of memory during inference. "
                "Start a new process after freeing GPU memory."
            ) from error


def _normalize_tokenizer_output(
    tokenizer_output: object,
) -> tuple[object, object | None]:
    """Extract and normalize token tensors without removing the batch axis."""
    if isinstance(tokenizer_output, Mapping):
        if "input_ids" not in tokenizer_output:
            raise ValueError("tokenizer output must contain input_ids")
        input_ids = tokenizer_output["input_ids"]
        attention_mask = tokenizer_output.get("attention_mask")
    else:
        input_ids = tokenizer_output
        attention_mask = None

    input_ids = _normalize_token_tensor(input_ids, "input_ids")
    if attention_mask is not None:
        attention_mask = _normalize_token_tensor(
            attention_mask,
            "attention_mask",
        )
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask shape must match input_ids shape: "
                f"{tuple(attention_mask.shape)} != {tuple(input_ids.shape)}"
            )
    return input_ids, attention_mask


def _normalize_token_tensor(tensor: object, name: str) -> object:
    """Add a batch axis to rank-one tokens and reject unsupported ranks."""
    ndim = getattr(tensor, "ndim", None)
    if ndim == 1:
        return tensor.unsqueeze(0)
    if ndim == 2:
        return tensor
    shape = getattr(tensor, "shape", None)
    raise ValueError(
        f"tokenizer {name} must have shape [sequence] or [batch, sequence], "
        f"got {shape!r}"
    )


def _validate_input_ids(input_ids: object, torch: object) -> None:
    """Require a non-empty rank-two CUDA integer token tensor."""
    if getattr(input_ids, "ndim", None) != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("input_ids must contain one non-empty prompt")
    if getattr(input_ids, "dtype", None) != torch.long:
        raise ValueError("input_ids must use torch.long")
    if getattr(getattr(input_ids, "device", None), "type", None) != "cuda":
        raise ValueError("input_ids must be on a CUDA device")


def _validate_attention_mask(
    attention_mask: object | None,
    input_ids: object,
) -> None:
    """Require the tokenizer mask to match prompt shape and CUDA device."""
    if attention_mask is None:
        return
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask shape must match input_ids shape")
    if attention_mask.device != input_ids.device:
        raise ValueError("attention_mask must be on the input_ids device")
    if getattr(attention_mask.device, "type", None) != "cuda":
        raise ValueError("attention_mask must be on a CUDA device")


def _validate_logits(logits: object, input_ids: object, torch: object) -> None:
    """Require floating CUDA logits aligned with the current input."""
    if getattr(logits, "ndim", None) != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if (
        logits.shape[0] != input_ids.shape[0]
        or logits.shape[1] != input_ids.shape[1]
        or logits.shape[2] < 1
    ):
        raise ValueError("logits shape must match input batch and sequence")
    if not torch.is_floating_point(logits):
        raise ValueError("logits must use a floating-point dtype")
    if getattr(getattr(logits, "device", None), "type", None) != "cuda":
        raise ValueError("logits must be on a CUDA device")


def _validate_cache(
    cache: object,
    *,
    batch_size: int,
    sequence_length: int,
) -> None:
    """Require non-empty rank-four floating CUDA key/value cache tensors."""
    layers = _cache_layers(cache)
    if not layers:
        raise ValueError("past_key_values must contain at least one layer")

    for layer_index, layer in enumerate(layers):
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            raise ValueError(
                f"cache layer {layer_index} must contain key and value tensors"
            )
        key, value = layer[0], layer[1]
        for name, tensor in (("key", key), ("value", value)):
            if getattr(tensor, "ndim", None) != 4:
                raise ValueError(
                    f"cache layer {layer_index} {name} must be rank four"
                )
            if (
                tensor.shape[0] != batch_size
                or tensor.shape[-2] != sequence_length
            ):
                raise ValueError(
                    f"cache layer {layer_index} {name} shape is inconsistent"
                )
            is_floating_point = getattr(tensor, "is_floating_point", None)
            if not callable(is_floating_point) or not is_floating_point():
                raise ValueError(
                    f"cache layer {layer_index} {name} must be floating point"
                )
            if getattr(getattr(tensor, "device", None), "type", None) != "cuda":
                raise ValueError(
                    f"cache layer {layer_index} {name} must be on CUDA"
                )
        if key.shape != value.shape or key.dtype != value.dtype:
            raise ValueError(
                f"cache layer {layer_index} key/value metadata must match"
            )


def _cache_layers(cache: object) -> object:
    """Expose cache layers from Transformers cache or legacy tuple formats."""
    to_legacy_cache = getattr(cache, "to_legacy_cache", None)
    if callable(to_legacy_cache):
        return to_legacy_cache()
    return cache


def _eos_ids(value: int | list[int] | None) -> set[int]:
    """Normalize tokenizer end-of-sequence IDs."""
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return set(value)
