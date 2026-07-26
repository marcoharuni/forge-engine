"""Single-owner scheduler runtime and HTTP serving interfaces."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from forge_engine import __version__
from forge_engine.config import (
    DEFAULT_MODEL_ID,
    SUPPORTED_MODEL_REVISION,
    EngineConfig,
)
from forge_engine.model import ChatMessage
from forge_engine.sampling import SamplingParams
from forge_engine.scheduler import (
    ConcurrentGenerationEngine,
    OverloadedError,
    RequestStatus,
    RequestView,
    SchedulerConfig,
    StreamEvent,
)


@dataclass(slots=True)
class _RequestMetrics:
    """Timing state retained only for the lifetime of one process."""

    admitted_at: float
    first_text_at: float | None = None
    last_text_at: float | None = None
    sampled_tokens: int = 0
    terminal_recorded: bool = False


class ServingRuntime(Protocol):
    """Small runtime surface consumed by the transport layer."""

    def start(self) -> None:
        """Start scheduler execution."""
        ...

    def stop(self) -> None:
        """Stop scheduler execution and reclaim active work."""
        ...

    def submit(
        self,
        messages: Sequence[ChatMessage],
        sampling_params: SamplingParams,
    ) -> str:
        """Admit one request and return its stable ID."""
        ...

    async def stream(self, request_id: str) -> AsyncIterator[StreamEvent]:
        """Yield ordered request events until terminal."""
        ...

    def request(self, request_id: str) -> RequestView:
        """Return immutable terminal or active state."""
        ...

    def cancel(self, request_id: str) -> bool:
        """Cancel active work."""
        ...

    def release(self, request_id: str) -> None:
        """Discard terminal transport and scheduler history."""
        ...

    def health(self) -> dict[str, object]:
        """Return process and scheduler health."""
        ...

    def metrics(self) -> str:
        """Return Prometheus text exposition."""
        ...


class MessageBody(BaseModel):
    """One supported chat-template message."""

    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatCompletionBody(BaseModel):
    """Supported streaming subset of OpenAI chat completions."""

    model_config = ConfigDict(extra="forbid")
    model: Literal[DEFAULT_MODEL_ID]
    messages: list[MessageBody] = Field(min_length=1)
    stream: Literal[True] = True
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    seed: int | None = None
    top_k: int | None = Field(default=None, ge=1)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)


class SchedulerRuntime:
    """Serialize all CUDA scheduler access through one worker thread."""

    _END = object()

    def __init__(self, engine: ConcurrentGenerationEngine) -> None:
        self._engine = engine
        self._condition = threading.Condition()
        self._channels: dict[str, queue.Queue[StreamEvent | object]] = {}
        self._request_metrics: dict[str, _RequestMetrics] = {}
        self._worker: threading.Thread | None = None
        self._stopping = False
        self._started = False
        self._requests_total = 0
        self._rejected_total = 0
        self._completed_total = 0
        self._cancelled_total = 0
        self._failed_total = 0
        self._generated_tokens_total = 0
        self._ttft_sum = 0.0
        self._ttft_count = 0
        self._itl_sum = 0.0
        self._itl_count = 0
        self._duration_sum = 0.0
        self._duration_count = 0

    def start(self) -> None:
        """Start the sole scheduler worker exactly once."""
        with self._condition:
            if self._started:
                return
            if self._stopping:
                raise RuntimeError("scheduler runtime cannot be restarted")
            self._worker = threading.Thread(
                target=self._run,
                name="forge-scheduler",
                daemon=True,
            )
            self._started = True
            self._worker.start()

    def stop(self) -> None:
        """Cancel admitted work, signal consumers, and join the worker."""
        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            for request_id in tuple(self._channels):
                self._cancel_locked(request_id)
            self._condition.notify_all()
            worker = self._worker
        if worker is not None:
            worker.join(timeout=30.0)
            if worker.is_alive():
                raise RuntimeError("scheduler worker did not stop")

    def submit(
        self,
        messages: Sequence[ChatMessage],
        sampling_params: SamplingParams,
    ) -> str:
        """Admit one request before exposing its stream to a client."""
        with self._condition:
            if not self._started or self._stopping:
                raise RuntimeError("scheduler runtime is not running")
            try:
                request_id = self._engine.submit(
                    messages,
                    sampling_params=sampling_params,
                )
            except OverloadedError:
                self._rejected_total += 1
                raise
            self._channels[request_id] = queue.Queue()
            self._request_metrics[request_id] = _RequestMetrics(
                admitted_at=time.monotonic()
            )
            self._requests_total += 1
            self._condition.notify()
            return request_id

    async def stream(self, request_id: str) -> AsyncIterator[StreamEvent]:
        """Bridge the worker's blocking channel into an async response."""
        with self._condition:
            try:
                channel = self._channels[request_id]
            except KeyError as error:
                raise KeyError(f"unknown request_id: {request_id}") from error
        while True:
            try:
                item = channel.get_nowait()
            except queue.Empty:
                # Polling avoids occupying one executor thread per open SSE
                # request while the CUDA worker is between token events.
                await asyncio.sleep(0.001)
                continue
            if item is self._END:
                return
            if not isinstance(item, StreamEvent):
                raise RuntimeError("invalid scheduler stream event")
            yield item

    def request(self, request_id: str) -> RequestView:
        """Read request state under the same lock as scheduler mutation."""
        with self._condition:
            return self._engine.request(request_id)

    def cancel(self, request_id: str) -> bool:
        """Cancel active work and wake its stream consumer."""
        with self._condition:
            return self._cancel_locked(request_id)

    def release(self, request_id: str) -> None:
        """Drop terminal per-request objects after the response is closed."""
        with self._condition:
            view = self._engine.request(request_id)
            if not view.status.terminal:
                raise ValueError("cannot release an active request")
            self._channels.pop(request_id, None)
            self._request_metrics.pop(request_id, None)
            self._engine.forget(request_id)

    def health(self) -> dict[str, object]:
        """Expose bounded scheduler state without model tensor details."""
        with self._condition:
            snapshot = self._engine.snapshot()
            return {
                "status": "ok" if self._started and not self._stopping else "stopping",
                "model": DEFAULT_MODEL_ID,
                "revision": SUPPORTED_MODEL_REVISION,
                "scheduler": {
                    "waiting": snapshot.waiting,
                    "running": snapshot.running,
                    "reserved_blocks": snapshot.reserved_blocks,
                    "allocated_blocks": snapshot.allocated_blocks,
                    "iteration": snapshot.iteration,
                },
            }

    def metrics(self) -> str:
        """Render dependency-free Prometheus text metrics."""
        with self._condition:
            snapshot = self._engine.snapshot()
            active = snapshot.waiting + snapshot.running
            cuda_allocated = 0
            cuda_reserved = 0
            cuda_peak = 0
            import torch

            if torch.cuda.is_available():
                cuda_allocated = torch.cuda.memory_allocated()
                cuda_reserved = torch.cuda.memory_reserved()
                cuda_peak = torch.cuda.max_memory_allocated()
            lines = [
                "# HELP forge_requests_total Admitted chat requests.",
                "# TYPE forge_requests_total counter",
                f"forge_requests_total {self._requests_total}",
                "# HELP forge_requests_rejected_total Requests rejected by admission control.",
                "# TYPE forge_requests_rejected_total counter",
                f"forge_requests_rejected_total {self._rejected_total}",
                "# HELP forge_requests_terminal_total Terminal requests by status.",
                "# TYPE forge_requests_terminal_total counter",
                f'forge_requests_terminal_total{{status="finished"}} {self._completed_total}',
                f'forge_requests_terminal_total{{status="cancelled"}} {self._cancelled_total}',
                f'forge_requests_terminal_total{{status="failed"}} {self._failed_total}',
                "# HELP forge_requests_active Currently waiting or running requests.",
                "# TYPE forge_requests_active gauge",
                f"forge_requests_active {active}",
                "# HELP forge_scheduler_requests Scheduler requests by active state.",
                "# TYPE forge_scheduler_requests gauge",
                f'forge_scheduler_requests{{status="waiting"}} {snapshot.waiting}',
                f'forge_scheduler_requests{{status="running"}} {snapshot.running}',
                "# HELP forge_scheduler_iterations_total Completed scheduler iterations.",
                "# TYPE forge_scheduler_iterations_total counter",
                f"forge_scheduler_iterations_total {snapshot.iteration}",
                "# HELP forge_kv_blocks KV cache blocks by ownership state.",
                "# TYPE forge_kv_blocks gauge",
                f'forge_kv_blocks{{state="reserved"}} {snapshot.reserved_blocks}',
                f'forge_kv_blocks{{state="allocated"}} {snapshot.allocated_blocks}',
                "# HELP forge_cuda_memory_bytes CUDA memory by allocator state.",
                "# TYPE forge_cuda_memory_bytes gauge",
                f'forge_cuda_memory_bytes{{state="allocated"}} {cuda_allocated}',
                f'forge_cuda_memory_bytes{{state="reserved"}} {cuda_reserved}',
                f'forge_cuda_memory_bytes{{state="peak_allocated"}} {cuda_peak}',
                "# HELP forge_generated_tokens_total Sampled output tokens.",
                "# TYPE forge_generated_tokens_total counter",
                f"forge_generated_tokens_total {self._generated_tokens_total}",
                "# HELP forge_time_to_first_text_seconds Time from admission to first text fragment.",
                "# TYPE forge_time_to_first_text_seconds summary",
                f"forge_time_to_first_text_seconds_sum {self._ttft_sum:.9f}",
                f"forge_time_to_first_text_seconds_count {self._ttft_count}",
                "# HELP forge_inter_text_latency_seconds Time between streamed text fragments.",
                "# TYPE forge_inter_text_latency_seconds summary",
                f"forge_inter_text_latency_seconds_sum {self._itl_sum:.9f}",
                f"forge_inter_text_latency_seconds_count {self._itl_count}",
                "# HELP forge_request_duration_seconds Time from admission to terminal state.",
                "# TYPE forge_request_duration_seconds summary",
                f"forge_request_duration_seconds_sum {self._duration_sum:.9f}",
                f"forge_request_duration_seconds_count {self._duration_count}",
            ]
            return "\n".join(lines) + "\n"

    def _run(self) -> None:
        """Advance the scheduler whenever admitted work exists."""
        while True:
            with self._condition:
                while not self._stopping and self._engine.idle:
                    self._condition.wait()
                if self._stopping:
                    return
                events = self._engine.step()
                now = time.monotonic()
                for event in events:
                    channel = self._channels[event.request_id]
                    self._record_event_locked(event, now)
                    channel.put(event)
                    if event.status.terminal:
                        channel.put(self._END)
            time.sleep(0)

    def _record_event_locked(
        self,
        event: StreamEvent,
        now: float,
    ) -> None:
        """Update counters once for sampled progress and terminal state."""
        metrics = self._request_metrics[event.request_id]
        view = self._engine.request(event.request_id)
        token_delta = view.sampled_tokens - metrics.sampled_tokens
        if token_delta < 0:
            raise RuntimeError("sampled token counter moved backwards")
        self._generated_tokens_total += token_delta
        metrics.sampled_tokens = view.sampled_tokens
        if event.text:
            if metrics.first_text_at is None:
                metrics.first_text_at = now
                self._ttft_sum += now - metrics.admitted_at
                self._ttft_count += 1
            elif metrics.last_text_at is not None:
                self._itl_sum += now - metrics.last_text_at
                self._itl_count += 1
            metrics.last_text_at = now
        if event.status.terminal:
            self._record_terminal_locked(event.request_id, now)

    def _record_terminal_locked(
        self,
        request_id: str,
        now: float,
    ) -> None:
        """Record terminal metrics exactly once."""
        metrics = self._request_metrics[request_id]
        if metrics.terminal_recorded:
            return
        view = self._engine.request(request_id)
        if view.status is RequestStatus.FINISHED:
            self._completed_total += 1
        elif view.status is RequestStatus.CANCELLED:
            self._cancelled_total += 1
        elif view.status is RequestStatus.FAILED:
            self._failed_total += 1
        else:
            raise RuntimeError("terminal metrics requested for active work")
        self._duration_sum += now - metrics.admitted_at
        self._duration_count += 1
        metrics.terminal_recorded = True

    def _cancel_locked(self, request_id: str) -> bool:
        """Cancel while holding the scheduler condition."""
        cancelled = self._engine.cancel(request_id)
        if cancelled:
            now = time.monotonic()
            self._record_terminal_locked(request_id, now)
            channel = self._channels[request_id]
            channel.put(
                StreamEvent(
                    request_id=request_id,
                    text="",
                    status=RequestStatus.CANCELLED,
                )
            )
            channel.put(self._END)
        return cancelled


def create_app(runtime: ServingRuntime) -> FastAPI:
    """Build the FastAPI transport around an already loaded runtime."""

    @asynccontextmanager
    async def lifespan(_: object) -> AsyncIterator[None]:
        runtime.start()
        try:
            yield
        finally:
            await asyncio.to_thread(runtime.stop)

    app = FastAPI(
        title="ForgeEngine",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def browser_chat() -> str:
        return _BROWSER_CHAT_HTML

    @app.get("/health")
    async def health() -> dict[str, object]:
        return await asyncio.to_thread(runtime.health)

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [
                {
                    "id": DEFAULT_MODEL_ID,
                    "object": "model",
                    "created": 0,
                    "owned_by": "Qwen",
                }
            ],
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(
            await asyncio.to_thread(runtime.metrics),
            media_type="text/plain; version=0.0.4",
        )

    @app.post("/v1/requests/{request_id}/cancel")
    async def cancel_request(request_id: str) -> dict[str, str]:
        """Cancel one admitted request by its transport-visible ID."""
        try:
            cancelled = await asyncio.to_thread(
                runtime.cancel,
                request_id,
            )
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail=str(error),
            ) from error
        if not cancelled:
            raise HTTPException(
                status_code=409,
                detail=f"request is already terminal: {request_id}",
            )
        return {"request_id": request_id, "status": "cancelled"}

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatCompletionBody,
    ) -> StreamingResponse:
        stops = (
            ()
            if body.stop is None
            else (body.stop,)
            if isinstance(body.stop, str)
            else tuple(body.stop)
        )
        try:
            params = SamplingParams(
                temperature=body.temperature,
                top_k=body.top_k,
                top_p=body.top_p,
                min_p=body.min_p,
                max_new_tokens=body.max_tokens,
                stop_strings=stops,
                seed=body.seed,
            )
            messages = [
                ChatMessage(role=message.role, content=message.content)
                for message in body.messages
            ]
            request_id = await asyncio.to_thread(
                runtime.submit,
                messages,
                params,
            )
        except OverloadedError as error:
            raise HTTPException(status_code=429, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        created = int(time.time())

        async def event_stream() -> AsyncIterator[str]:
            terminal = False
            yield _sse_chunk(
                request_id,
                created,
                {"role": "assistant", "content": ""},
                None,
            )
            try:
                async for event in runtime.stream(request_id):
                    if event.text:
                        yield _sse_chunk(
                            request_id,
                            created,
                            {"content": event.text},
                            None,
                        )
                    if event.status.terminal:
                        terminal = True
                        view = await asyncio.to_thread(
                            runtime.request,
                            request_id,
                        )
                        if view.status is RequestStatus.FAILED:
                            yield _sse_data(
                                {
                                    "error": {
                                        "message": view.error or "generation failed",
                                        "type": "server_error",
                                    }
                                }
                            )
                        else:
                            yield _sse_chunk(
                                request_id,
                                created,
                                {},
                                view.finish_reason or "stop",
                            )
                        yield "data: [DONE]\n\n"
                        return
            finally:
                if not terminal:
                    await asyncio.to_thread(runtime.cancel, request_id)
                await asyncio.to_thread(runtime.release, request_id)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Forge-Request-ID": request_id,
            },
        )

    return app


def run_server(
    engine_config: EngineConfig,
    scheduler_config: SchedulerConfig,
    *,
    host: str,
    port: int,
) -> None:
    """Load one engine and serve it until process shutdown."""
    import uvicorn

    engine = ConcurrentGenerationEngine.from_config(
        engine_config,
        scheduler_config,
    )
    runtime = SchedulerRuntime(engine)
    uvicorn.run(
        create_app(runtime),
        host=host,
        port=port,
        log_level="info",
    )


def _sse_chunk(
    request_id: str,
    created: int,
    delta: dict[str, str],
    finish_reason: str | None,
) -> str:
    """Encode one OpenAI-compatible chat completion chunk."""
    return _sse_data(
        {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": DEFAULT_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


def _sse_data(value: dict[str, object]) -> str:
    """Encode one compact UTF-8 JSON Server-Sent Event."""
    return f"data: {json.dumps(value, separators=(',', ':'))}\n\n"


_BROWSER_CHAT_HTML = (
    files("forge_engine").joinpath("static/index.html").read_text(encoding="utf-8")
)
