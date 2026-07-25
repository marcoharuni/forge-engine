"""Explicit prefill, decode, sampling, stopping, and text streaming."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from forge_engine.cache import PagedKVBlockPool, PagedKVCache
from forge_engine.config import EngineConfig
from forge_engine.model import (
    CUDAOutOfMemoryError,
    ChatMessage,
    LanguageModel,
    LoadedModel,
    Tokenizer,
    load_model,
)
from forge_engine.sampling import SamplingParams, sample_token


class InferenceEngine(Protocol):
    """Streaming interface consumed by front ends."""

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        """Stream decoded fragments for a chat conversation."""
        ...


@dataclass(slots=True)
class GenerationState:
    """Mutable state passed from prefill through single-token decode steps."""

    logits: object
    attention_mask: object | None
    cache: PagedKVCache


class GenerationCore:
    """Prefill/decode over paged state with reference cache materialization."""

    def __init__(
        self,
        model: LanguageModel,
        cache_pool: PagedKVBlockPool | None = None,
    ) -> None:
        self._model = model
        self._cache_pool = (
            PagedKVBlockPool() if cache_pool is None else cache_pool
        )

    def prefill(
        self,
        input_ids: object,
        attention_mask: object | None,
    ) -> GenerationState:
        """Process a complete prompt and initialize its contiguous KV cache."""
        output = self._model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )
        cache = PagedKVCache(self._cache_pool)
        try:
            cache.replace(output.past_key_values)
            if cache.sequence_length != input_ids.shape[1]:
                raise ValueError("prefill cache length must match the prompt")
        except BaseException:
            cache.clear()
            raise
        return GenerationState(
            logits=output.logits,
            attention_mask=attention_mask,
            cache=cache,
        )

    def prefill_batch(
        self,
        inputs: Sequence[tuple[object, object | None]],
    ) -> list[GenerationState]:
        """Prefill compatible requests in one tensor-batched model call."""
        import torch

        if not inputs:
            raise ValueError("prefill batch must not be empty")
        if len(inputs) == 1:
            input_ids, attention_mask = inputs[0]
            return [self.prefill(input_ids, attention_mask)]
        shape = inputs[0][0].shape
        mask_present = inputs[0][1] is not None
        for input_ids, attention_mask in inputs:
            if input_ids.shape != shape or input_ids.shape[0] != 1:
                raise ValueError("prefill batch input shapes must match")
            if (attention_mask is not None) != mask_present:
                raise ValueError("prefill batch mask presence must match")
            if attention_mask is not None and attention_mask.shape != shape:
                raise ValueError("prefill batch mask shapes must match inputs")
        batched_ids = torch.cat(
            tuple(input_ids for input_ids, _mask in inputs),
            dim=0,
        )
        batched_mask = (
            torch.cat(
                tuple(
                    attention_mask
                    for _input_ids, attention_mask in inputs
                    if attention_mask is not None
                ),
                dim=0,
            )
            if mask_present
            else None
        )
        output = self._model.forward(
            input_ids=batched_ids,
            attention_mask=batched_mask,
            past_key_values=None,
            use_cache=True,
        )
        split_caches = _split_cache_batch(
            output.past_key_values,
            len(inputs),
        )
        states: list[GenerationState] = []
        try:
            for index, ((input_ids, attention_mask), layers) in enumerate(
                zip(inputs, split_caches, strict=True)
            ):
                cache = PagedKVCache(self._cache_pool)
                cache.replace(layers)
                if cache.sequence_length != input_ids.shape[1]:
                    cache.clear()
                    raise ValueError(
                        "prefill cache length must match the prompt"
                    )
                states.append(
                    GenerationState(
                        logits=output.logits[index : index + 1],
                        attention_mask=attention_mask,
                        cache=cache,
                    )
                )
        except BaseException:
            for state in states:
                state.cache.clear()
            raise
        return states

    def decode(
        self,
        input_ids: object,
        state: GenerationState,
    ) -> GenerationState:
        """Process exactly one token using and extending existing cache state."""
        import torch

        if getattr(input_ids, "ndim", None) != 2 or input_ids.shape[1] != 1:
            raise ValueError("decode input_ids must have shape [batch, 1]")
        previous_length = state.cache.sequence_length
        if previous_length < 1:
            raise ValueError("decode requires a populated cache")
        attention_mask = state.attention_mask
        if attention_mask is not None:
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ),
                dim=1,
            )
        direct_paged = bool(
            getattr(self._model, "supports_paged_decode", False)
            and (
                attention_mask is None
                or bool(attention_mask.bool().all().item())
            )
        )
        output = self._model.forward(
            input_ids=input_ids,
            attention_mask=None if direct_paged else attention_mask,
            past_key_values=(
                state.cache
                if direct_paged
                else state.cache.as_model_input()
            ),
            use_cache=True,
        )
        if direct_paged:
            state.cache.append_token(output.past_key_values)
        else:
            state.cache.append(output.past_key_values)
        if state.cache.sequence_length != previous_length + 1:
            raise ValueError("decode must extend cache by exactly one token")
        state.logits = output.logits
        state.attention_mask = attention_mask
        return state

    def decode_batch(
        self,
        inputs: Sequence[object],
        states: Sequence[GenerationState],
    ) -> list[GenerationState]:
        """Decode compatible requests in one tensor-batched model call."""
        import torch

        if not inputs or len(inputs) != len(states):
            raise ValueError("decode batch inputs and states must align")
        if len(inputs) == 1:
            return [self.decode(inputs[0], states[0])]
        sequence_length = states[0].cache.sequence_length
        mask_present = states[0].attention_mask is not None
        extended_masks: list[object | None] = []
        materialized = []
        for input_ids, state in zip(inputs, states, strict=True):
            if getattr(input_ids, "ndim", None) != 2 or input_ids.shape != (
                1,
                1,
            ):
                raise ValueError("decode batch inputs must have shape [1, 1]")
            if state.cache.sequence_length != sequence_length:
                raise ValueError("decode batch cache lengths must match")
            if (state.attention_mask is not None) != mask_present:
                raise ValueError("decode batch mask presence must match")
            attention_mask = state.attention_mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (1, 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
            extended_masks.append(attention_mask)
            materialized.append(state.cache.as_model_input())
        layer_count = len(materialized[0])
        batched_cache = tuple(
            (
                torch.cat(
                    tuple(cache[layer_index][0] for cache in materialized),
                    dim=0,
                ),
                torch.cat(
                    tuple(cache[layer_index][1] for cache in materialized),
                    dim=0,
                ),
            )
            for layer_index in range(layer_count)
        )
        batched_mask = (
            torch.cat(
                tuple(
                    attention_mask
                    for attention_mask in extended_masks
                    if attention_mask is not None
                ),
                dim=0,
            )
            if mask_present
            else None
        )
        output = self._model.forward(
            input_ids=torch.cat(tuple(inputs), dim=0),
            attention_mask=batched_mask,
            past_key_values=batched_cache,
            use_cache=True,
        )
        split_caches = _split_cache_batch(
            output.past_key_values,
            len(states),
        )
        for index, (state, layers, attention_mask) in enumerate(
            zip(states, split_caches, extended_masks, strict=True)
        ):
            state.cache.append(layers)
            if state.cache.sequence_length != sequence_length + 1:
                raise ValueError("decode must extend cache by exactly one token")
            state.logits = output.logits[index : index + 1]
            state.attention_mask = attention_mask
        return list(states)

    def uncached(
        self,
        input_ids: object,
        attention_mask: object | None,
    ) -> object:
        """Run the full sequence without cache for correctness comparisons."""
        return self._model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
        )


class IncrementalDetokenizer:
    """Emit stable decoded text while withholding possible stop prefixes."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        stop_strings: Sequence[str] = (),
    ) -> None:
        if any(not stop for stop in stop_strings):
            raise ValueError("stop strings must be non-empty")
        self._tokenizer = tokenizer
        self._stop_strings = tuple(stop_strings)
        self._token_ids: list[int] = []
        self._emitted = ""
        self._decoded = ""
        self._stopped = False

    @property
    def stopped(self) -> bool:
        """Return whether a complete configured stop string was observed."""
        return self._stopped

    @property
    def token_ids(self) -> tuple[int, ...]:
        """Return all non-EOS token IDs consumed so far."""
        return tuple(self._token_ids)

    def push(self, token_id: int) -> str:
        """Decode one additional token and return newly stable text."""
        if self._stopped:
            raise RuntimeError("cannot push tokens after a stop string")
        self._token_ids.append(token_id)
        self._decoded = self._tokenizer.decode(
            self._token_ids,
            skip_special_tokens=True,
        )
        return self._advance(final=False)

    def finish(self) -> str:
        """Flush an incomplete text or stop prefix when generation ends."""
        if self._stopped:
            return ""
        return self._advance(final=True)

    def _advance(self, *, final: bool) -> str:
        if not self._decoded.startswith(self._emitted):
            raise ValueError("tokenizer revised text that was already emitted")
        stop_at = _first_stop(self._decoded, self._stop_strings)
        if stop_at is not None:
            stable_end = stop_at
            self._stopped = True
        elif final:
            stable_end = len(self._decoded)
        else:
            stop_holdback = _stop_prefix_length(
                self._decoded,
                self._stop_strings,
            )
            unicode_holdback = _replacement_suffix_length(self._decoded)
            stable_end = len(self._decoded) - max(
                stop_holdback,
                unicode_holdback,
            )
        if stable_end < len(self._emitted):
            raise ValueError("tokenizer changed a previously stable text prefix")
        fragment = self._decoded[len(self._emitted) : stable_end]
        self._emitted = self._decoded[:stable_end]
        return fragment


class GenerationEngine:
    """Single-request generation with explicit state transitions."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        model: LanguageModel,
        sampling_params: SamplingParams,
        cache_pool: PagedKVBlockPool | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._sampling_params = sampling_params
        self._core = GenerationCore(model, cache_pool)

    @classmethod
    def from_config(cls, config: EngineConfig) -> GenerationEngine:
        """Load a CUDA model and construct the configured engine."""
        loaded: LoadedModel = load_model(config)
        return cls(loaded.tokenizer, loaded.model, config.sampling_params())

    def stream(self, messages: Sequence[ChatMessage]) -> Iterator[str]:
        """Stream sampled text until EOS, stop string, or token limit."""
        import torch

        params = self._sampling_params
        eos_ids = _eos_ids(self._tokenizer.eos_token_id)
        detokenizer = IncrementalDetokenizer(
            self._tokenizer,
            params.stop_strings,
        )
        state: GenerationState | None = None
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
            generator = _make_generator(
                torch,
                self._model.device,
                params.seed,
            )

            with torch.inference_mode():
                state = self._core.prefill(input_ids, attention_mask)
                _validate_step(state, input_ids, torch)
                for step_index in range(params.max_new_tokens):
                    next_token = sample_token(
                        state.logits[:, -1, :],
                        params,
                        generator=generator,
                    )
                    token_id = int(next_token.item())
                    if token_id in eos_ids:
                        break

                    fragment = detokenizer.push(token_id)
                    if fragment:
                        yield fragment
                    if detokenizer.stopped:
                        break
                    if step_index + 1 == params.max_new_tokens:
                        break

                    decode_ids = next_token.unsqueeze(-1)
                    state = self._core.decode(decode_ids, state)
                    _validate_step(state, decode_ids, torch)

            final_fragment = detokenizer.finish()
            if final_fragment:
                yield final_fragment
        except torch.cuda.OutOfMemoryError as error:
            raise CUDAOutOfMemoryError(
                "CUDA out of memory during inference. "
                "Start a new process after freeing GPU memory."
            ) from error
        finally:
            if state is not None:
                state.cache.clear()


GreedyEngine = GenerationEngine


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


def _validate_step(
    state: GenerationState,
    input_ids: object,
    torch: object,
) -> None:
    """Validate logits and all cache layers after one model call."""
    _validate_logits(state.logits, input_ids, torch)
    _validate_cache(
        state.cache.layers,
        batch_size=input_ids.shape[0],
        sequence_length=state.cache.sequence_length,
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
    if not cache:
        raise ValueError("past_key_values must contain at least one layer")

    for layer_index, layer in enumerate(cache):
        if not isinstance(layer, (tuple, list)) or len(layer) != 2:
            raise ValueError(
                f"cache layer {layer_index} must contain key and value tensors"
            )
        key, value = layer
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


def _make_generator(
    torch: object,
    device: object,
    seed: int | None,
) -> object | None:
    """Create an optional device-local random generator."""
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _first_stop(text: str, stop_strings: Sequence[str]) -> int | None:
    """Return the earliest complete stop-string offset."""
    positions = [
        position
        for stop in stop_strings
        if (position := text.find(stop)) >= 0
    ]
    return min(positions) if positions else None


def _stop_prefix_length(text: str, stop_strings: Sequence[str]) -> int:
    """Return the longest suffix that may grow into a stop string."""
    longest = 0
    for stop in stop_strings:
        maximum = min(len(text), len(stop) - 1)
        for size in range(1, maximum + 1):
            if text.endswith(stop[:size]):
                longest = max(longest, size)
    return longest


def _replacement_suffix_length(text: str) -> int:
    """Hold trailing Unicode replacement characters for later byte tokens."""
    return len(text) - len(text.rstrip("\ufffd"))


def _eos_ids(value: int | list[int] | None) -> set[int]:
    """Normalize tokenizer end-of-sequence IDs."""
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return set(value)


def _split_cache_batch(
    past_key_values: object,
    batch_size: int,
) -> list[tuple[tuple[object, object], ...]]:
    """Split a legacy layer cache into independent batch-one tuples."""
    if not isinstance(past_key_values, (tuple, list)) or not past_key_values:
        raise ValueError("past_key_values must contain cache layers")
    requests: list[list[tuple[object, object]]] = [
        [] for _ in range(batch_size)
    ]
    for layer_index, layer in enumerate(past_key_values):
        if not isinstance(layer, (tuple, list)) or len(layer) != 2:
            raise ValueError(
                f"cache layer {layer_index} must contain key and value"
            )
        key, value = layer
        if key.shape[0] != batch_size or value.shape[0] != batch_size:
            raise ValueError("model cache batch does not match request batch")
        for batch_index in range(batch_size):
            requests[batch_index].append(
                (
                    key[batch_index : batch_index + 1],
                    value[batch_index : batch_index + 1],
                )
            )
    return [tuple(layers) for layers in requests]
