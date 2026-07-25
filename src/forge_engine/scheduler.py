"""Bounded iteration-level scheduling for concurrent generation requests."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from forge_engine.cache import PagedKVBlockPool
from forge_engine.config import EngineConfig
from forge_engine.engine import (
    GenerationCore,
    GenerationState,
    IncrementalDetokenizer,
    _eos_ids,
    _make_generator,
    _normalize_tokenizer_output,
    _validate_attention_mask,
    _validate_input_ids,
    _validate_step,
)
from forge_engine.model import (
    ChatMessage,
    LanguageModel,
    LoadedModel,
    Tokenizer,
    load_model,
)
from forge_engine.sampling import SamplingParams, sample_token


class RequestStatus(str, Enum):
    """Lifecycle state for one admitted request."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        """Return whether no more scheduler work is allowed."""
        return self in {
            RequestStatus.FINISHED,
            RequestStatus.CANCELLED,
            RequestStatus.FAILED,
        }


class OverloadedError(RuntimeError):
    """Raised when bounded admission cannot safely reserve a request."""


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Limits controlling admission and each scheduling iteration."""

    max_requests: int = 16
    max_batch_size: int = 8
    token_budget: int = 256
    block_size: int = 16
    block_capacity: int = 1_024

    def __post_init__(self) -> None:
        """Reject limits that could deadlock or disable scheduling."""
        for name in (
            "max_requests",
            "max_batch_size",
            "token_budget",
            "block_size",
            "block_capacity",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")


@dataclass(frozen=True, slots=True)
class RequestView:
    """Public immutable request state without model tensor references."""

    request_id: str
    status: RequestStatus
    prompt_tokens: int
    sampled_tokens: int
    reserved_blocks: int
    finish_reason: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One text or lifecycle event emitted by a scheduling iteration."""

    request_id: str
    text: str
    status: RequestStatus


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    """Small observable scheduler state for tests and future metrics."""

    iteration: int
    waiting: int
    running: int
    terminal: int
    reserved_blocks: int
    allocated_blocks: int
    last_batch: tuple[str, ...]
    model_batches: tuple[tuple[str, ...], ...]


@dataclass(slots=True)
class _Request:
    """Internal tensors and decoding state for one request."""

    request_id: str
    input_ids: object
    attention_mask: object | None
    prompt_tokens: int
    params: SamplingParams
    reserved_blocks: int
    detokenizer: IncrementalDetokenizer
    eos_ids: set[int]
    status: RequestStatus = RequestStatus.WAITING
    sampled_tokens: int = 0
    state: GenerationState | None = None
    pending_token: object | None = None
    generator: object | None = None
    finish_reason: str | None = None
    error: str | None = None
    reservation_released: bool = False


class ConcurrentGenerationEngine:
    """Run bounded requests in round-robin token-budgeted iterations."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        model: LanguageModel,
        default_sampling_params: SamplingParams,
        scheduler_config: SchedulerConfig | None = None,
        cache_pool: PagedKVBlockPool | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._default_sampling_params = default_sampling_params
        self._config = (
            SchedulerConfig() if scheduler_config is None else scheduler_config
        )
        if cache_pool is None:
            cache_pool = PagedKVBlockPool(
                block_size=self._config.block_size,
                capacity=self._config.block_capacity,
            )
        elif (
            cache_pool.block_size != self._config.block_size
            or cache_pool.capacity != self._config.block_capacity
        ):
            raise ValueError("cache pool does not match scheduler configuration")
        self._pool = cache_pool
        self._core = GenerationCore(model, cache_pool)
        self._requests: dict[str, _Request] = {}
        self._order: deque[str] = deque()
        self._reserved_blocks = 0
        self._next_request_number = 1
        self._iteration = 0
        self._last_batch: tuple[str, ...] = ()
        self._last_model_batches: tuple[tuple[str, ...], ...] = ()

    @classmethod
    def from_config(
        cls,
        engine_config: EngineConfig,
        scheduler_config: SchedulerConfig | None = None,
    ) -> ConcurrentGenerationEngine:
        """Load the supported CUDA model and build a concurrent engine."""
        loaded: LoadedModel = load_model(engine_config)
        return cls(
            loaded.tokenizer,
            loaded.model,
            engine_config.sampling_params(),
            scheduler_config,
        )

    @property
    def idle(self) -> bool:
        """Return whether no admitted request still needs work."""
        return all(
            request.status.terminal for request in self._requests.values()
        )

    @property
    def cache_pool(self) -> PagedKVBlockPool:
        """Expose pool counters and capacity for admission evidence."""
        return self._pool

    def submit(
        self,
        messages: Sequence[ChatMessage],
        *,
        request_id: str | None = None,
        sampling_params: SamplingParams | None = None,
    ) -> str:
        """Tokenize and reserve worst-case blocks before accepting work."""
        active_count = sum(
            not request.status.terminal
            for request in self._requests.values()
        )
        if active_count >= self._config.max_requests:
            raise OverloadedError(
                f"request limit {self._config.max_requests} is full"
            )
        assigned_id = (
            self._new_request_id() if request_id is None else request_id
        )
        if assigned_id in self._requests:
            raise ValueError(f"duplicate request_id: {assigned_id}")
        params = (
            self._default_sampling_params
            if sampling_params is None
            else sampling_params
        )
        tokenizer_output = self._tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids, attention_mask = _normalize_tokenizer_output(
            tokenizer_output
        )
        if input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("each request must contain one non-empty prompt")
        prompt_tokens = int(input_ids.shape[1])
        reserved_blocks = _ceil_div(
            prompt_tokens + params.max_new_tokens,
            self._pool.block_size,
        )
        if reserved_blocks > self._pool.capacity:
            raise OverloadedError(
                f"request needs {reserved_blocks} KV blocks but pool "
                f"capacity is {self._pool.capacity}"
            )
        available = self._pool.capacity - self._reserved_blocks
        if reserved_blocks > available:
            raise OverloadedError(
                f"request needs {reserved_blocks} KV blocks but only "
                f"{available} are unreserved"
            )

        request = _Request(
            request_id=assigned_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_tokens=prompt_tokens,
            params=params,
            reserved_blocks=reserved_blocks,
            detokenizer=IncrementalDetokenizer(
                self._tokenizer,
                params.stop_strings,
            ),
            eos_ids=_eos_ids(self._tokenizer.eos_token_id),
        )
        self._requests[assigned_id] = request
        self._order.append(assigned_id)
        self._reserved_blocks += reserved_blocks
        return assigned_id

    def cancel(self, request_id: str) -> bool:
        """Cancel non-terminal work and reclaim all owned resources."""
        request = self._request(request_id)
        if request.status.terminal:
            return False
        request.status = RequestStatus.CANCELLED
        request.finish_reason = "cancelled"
        self._cleanup(request)
        return True

    def step(self) -> list[StreamEvent]:
        """Advance a fresh request batch once under the token budget."""
        selected = self._select_batch()
        self._last_batch = tuple(request.request_id for request in selected)
        groups = self._group_selected(selected)
        self._last_model_batches = tuple(
            tuple(request.request_id for request in group)
            for group in groups
        )
        self._iteration += 1
        events: list[StreamEvent] = []
        import torch

        with torch.inference_mode():
            for group in groups:
                try:
                    status = group[0].status
                    if status is RequestStatus.WAITING:
                        events.extend(self._prefill_group(group, torch))
                    elif status is RequestStatus.RUNNING:
                        events.extend(self._decode_group(group, torch))
                    else:
                        raise RuntimeError(
                            f"unsupported request state: {status}"
                        )
                except Exception as error:
                    for request in group:
                        request.status = RequestStatus.FAILED
                        request.finish_reason = "error"
                        request.error = f"{type(error).__name__}: {error}"
                        self._cleanup(request)
                        events.append(
                            StreamEvent(
                                request_id=request.request_id,
                                text="",
                                status=request.status,
                            )
                        )
        self._discard_terminal_order_entries()
        return events

    def run_until_idle(self) -> list[StreamEvent]:
        """Run iterations until every admitted request is terminal."""
        events: list[StreamEvent] = []
        while not self.idle:
            batch_events = self.step()
            if not self._last_batch:
                raise RuntimeError("scheduler made no progress")
            events.extend(batch_events)
        return events

    def request(self, request_id: str) -> RequestView:
        """Return an immutable status view for one known request."""
        request = self._request(request_id)
        return RequestView(
            request_id=request.request_id,
            status=request.status,
            prompt_tokens=request.prompt_tokens,
            sampled_tokens=request.sampled_tokens,
            reserved_blocks=(
                0 if request.reservation_released else request.reserved_blocks
            ),
            finish_reason=request.finish_reason,
            error=request.error,
        )

    def forget(self, request_id: str) -> None:
        """Discard terminal request history after its consumer is done."""
        request = self._request(request_id)
        if not request.status.terminal:
            raise ValueError("cannot forget an active request")
        self._order = deque(
            queued_id for queued_id in self._order if queued_id != request_id
        )
        del self._requests[request_id]

    def snapshot(self) -> SchedulerSnapshot:
        """Return current counts without exposing request tensors."""
        requests = tuple(self._requests.values())
        return SchedulerSnapshot(
            iteration=self._iteration,
            waiting=sum(
                request.status is RequestStatus.WAITING
                for request in requests
            ),
            running=sum(
                request.status is RequestStatus.RUNNING
                for request in requests
            ),
            terminal=sum(request.status.terminal for request in requests),
            reserved_blocks=self._reserved_blocks,
            allocated_blocks=self._pool.allocated_block_count,
            last_batch=self._last_batch,
            model_batches=self._last_model_batches,
        )

    def _select_batch(self) -> list[_Request]:
        """Round-robin select at most one work item per active request."""
        selected: list[_Request] = []
        remaining_budget = self._config.token_budget
        candidates = len(self._order)
        for _ in range(candidates):
            request_id = self._order.popleft()
            request = self._requests[request_id]
            if request.status.terminal:
                continue
            cost = (
                request.prompt_tokens
                if request.status is RequestStatus.WAITING
                else 1
            )
            fits = cost <= remaining_budget
            if fits or not selected:
                selected.append(request)
                remaining_budget = max(0, remaining_budget - cost)
            self._order.append(request_id)
            if (
                len(selected) >= self._config.max_batch_size
                or remaining_budget == 0
            ):
                break
        return selected

    def _group_selected(
        self,
        selected: Sequence[_Request],
    ) -> list[list[_Request]]:
        """Group requests that can share one model tensor batch."""
        grouped: dict[tuple[object, ...], list[_Request]] = {}
        for request in selected:
            if request.status is RequestStatus.WAITING:
                key = (
                    request.status,
                    request.input_ids.shape[1],
                    request.attention_mask is not None,
                )
            elif request.status is RequestStatus.RUNNING:
                if request.state is None:
                    raise RuntimeError("running request has no generation state")
                key = (
                    request.status,
                    request.state.cache.sequence_length,
                    request.state.attention_mask is not None,
                )
            else:
                continue
            grouped.setdefault(key, []).append(request)
        return list(grouped.values())

    def _prefill_group(
        self,
        requests: Sequence[_Request],
        torch: object,
    ) -> list[StreamEvent]:
        """Move and prefill compatible prompts in one model call."""
        inputs = []
        for request in requests:
            request.input_ids = request.input_ids.to(self._model.device)
            if request.attention_mask is not None:
                request.attention_mask = request.attention_mask.to(
                    self._model.device
                )
            _validate_input_ids(request.input_ids, torch)
            _validate_attention_mask(
                request.attention_mask,
                request.input_ids,
            )
            request.generator = _make_generator(
                torch,
                self._model.device,
                request.params.seed,
            )
            inputs.append((request.input_ids, request.attention_mask))
        states = self._core.prefill_batch(inputs)
        for request, state in zip(requests, states, strict=True):
            request.state = state
            request.status = RequestStatus.RUNNING
        events: list[StreamEvent] = []
        for request in requests:
            assert request.state is not None
            _validate_step(request.state, request.input_ids, torch)
            events.extend(self._consume_logits(request))
        return events

    def _decode_group(
        self,
        requests: Sequence[_Request],
        torch: object,
    ) -> list[StreamEvent]:
        """Decode compatible running requests in one model call."""
        inputs = []
        states = []
        for request in requests:
            if request.state is None or request.pending_token is None:
                raise RuntimeError("running request has incomplete decode state")
            inputs.append(request.pending_token.unsqueeze(-1))
            states.append(request.state)
        decoded_states = self._core.decode_batch(inputs, states)
        events: list[StreamEvent] = []
        for request, decode_ids, state in zip(
            requests,
            inputs,
            decoded_states,
            strict=True,
        ):
            request.state = state
            _validate_step(request.state, decode_ids, torch)
            events.extend(self._consume_logits(request))
        return events

    def _consume_logits(self, request: _Request) -> list[StreamEvent]:
        """Sample, detokenize, and transition after one model output."""
        assert request.state is not None
        next_token = sample_token(
            request.state.logits[:, -1, :],
            request.params,
            generator=request.generator,
        )
        token_id = int(next_token.item())
        request.sampled_tokens += 1
        fragments: list[str] = []
        reached_eos = token_id in request.eos_ids
        should_finish = reached_eos
        if not should_finish:
            fragment = request.detokenizer.push(token_id)
            if fragment:
                fragments.append(fragment)
            should_finish = request.detokenizer.stopped
        stopped = request.detokenizer.stopped
        reached_limit = (
            request.sampled_tokens >= request.params.max_new_tokens
        )
        should_finish = should_finish or reached_limit
        if should_finish:
            final_fragment = request.detokenizer.finish()
            if final_fragment:
                fragments.append(final_fragment)
            request.status = RequestStatus.FINISHED
            request.finish_reason = (
                "stop" if reached_eos or stopped else "length"
            )
            self._cleanup(request)
        else:
            request.pending_token = next_token

        events = [
            StreamEvent(
                request_id=request.request_id,
                text=fragment,
                status=request.status,
            )
            for fragment in fragments
        ]
        if request.status.terminal and not events:
            events.append(
                StreamEvent(
                    request_id=request.request_id,
                    text="",
                    status=request.status,
                )
            )
        return events

    def _cleanup(self, request: _Request) -> None:
        """Release physical and reserved capacity exactly once."""
        if request.state is not None:
            request.state.cache.clear()
            request.state = None
        request.pending_token = None
        request.generator = None
        request.input_ids = None
        request.attention_mask = None
        if not request.reservation_released:
            self._reserved_blocks -= request.reserved_blocks
            request.reservation_released = True

    def _discard_terminal_order_entries(self) -> None:
        """Remove completed IDs while retaining immutable request history."""
        self._order = deque(
            request_id
            for request_id in self._order
            if not self._requests[request_id].status.terminal
        )

    def _request(self, request_id: str) -> _Request:
        try:
            return self._requests[request_id]
        except KeyError as error:
            raise KeyError(f"unknown request_id: {request_id}") from error

    def _new_request_id(self) -> str:
        while True:
            request_id = f"req-{self._next_request_number}"
            self._next_request_number += 1
            if request_id not in self._requests:
                return request_id


def _ceil_div(value: int, divisor: int) -> int:
    """Return mathematical ceiling division for positive integers."""
    return (value + divisor - 1) // divisor
