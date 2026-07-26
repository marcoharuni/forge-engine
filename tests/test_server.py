"""Tests for the M6 scheduler runtime and HTTP transport."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from unittest import TestCase

from fastapi.testclient import TestClient

from forge_engine.config import DEFAULT_MODEL_ID, SUPPORTED_MODEL_REVISION
from forge_engine.model import ChatMessage
from forge_engine.sampling import SamplingParams
from forge_engine.scheduler import (
    OverloadedError,
    RequestStatus,
    RequestView,
    SchedulerSnapshot,
    StreamEvent,
)
from forge_engine.server import SchedulerRuntime, create_app


class FakeServingRuntime:
    """Deterministic transport fake that records submitted conversations."""

    def __init__(self, *, overload: bool = False) -> None:
        self.overload = overload
        self.started = 0
        self.stopped = 0
        self.cancelled: list[str] = []
        self.submissions: list[tuple[list[dict[str, str]], SamplingParams]] = []

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def submit(
        self,
        messages: Sequence[ChatMessage],
        sampling_params: SamplingParams,
    ) -> str:
        if self.overload:
            raise OverloadedError("request limit is full")
        self.submissions.append(
            ([dict(message) for message in messages], sampling_params)
        )
        return "req-test"

    async def stream(self, request_id: str) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(request_id, "Hello", RequestStatus.RUNNING)
        yield StreamEvent(request_id, " world", RequestStatus.FINISHED)

    def request(self, request_id: str) -> RequestView:
        return RequestView(
            request_id=request_id,
            status=RequestStatus.FINISHED,
            prompt_tokens=4,
            sampled_tokens=2,
            reserved_blocks=0,
            finish_reason="stop",
            error=None,
        )

    def cancel(self, request_id: str) -> bool:
        self.cancelled.append(request_id)
        return True

    def release(self, request_id: str) -> None:
        return None

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "model": DEFAULT_MODEL_ID,
            "revision": SUPPORTED_MODEL_REVISION,
        }

    def metrics(self) -> str:
        return "forge_requests_total 1\n"


class FakeSchedulerEngine:
    """Minimal engine used to exercise the real worker/thread bridge."""

    def __init__(self) -> None:
        self._status: dict[str, RequestStatus] = {}
        self._iteration = 0

    @property
    def idle(self) -> bool:
        return all(status.terminal for status in self._status.values())

    def submit(
        self,
        messages: Sequence[ChatMessage],
        *,
        sampling_params: SamplingParams,
    ) -> str:
        assert messages and sampling_params.max_new_tokens == 2
        request_id = f"req-{len(self._status) + 1}"
        self._status[request_id] = RequestStatus.WAITING
        return request_id

    def step(self) -> list[StreamEvent]:
        self._iteration += 1
        request_id = next(
            request_id
            for request_id, status in self._status.items()
            if not status.terminal
        )
        self._status[request_id] = RequestStatus.FINISHED
        return [
            StreamEvent(
                request_id=request_id,
                text=f"done:{request_id}",
                status=RequestStatus.FINISHED,
            )
        ]

    def request(self, request_id: str) -> RequestView:
        status = self._status[request_id]
        return RequestView(
            request_id=request_id,
            status=status,
            prompt_tokens=3,
            sampled_tokens=1 if status.terminal else 0,
            reserved_blocks=0,
            finish_reason="stop" if status is RequestStatus.FINISHED else None,
            error=None,
        )

    def cancel(self, request_id: str) -> bool:
        if self._status[request_id].terminal:
            return False
        self._status[request_id] = RequestStatus.CANCELLED
        return True

    def forget(self, request_id: str) -> None:
        if not self._status[request_id].terminal:
            raise ValueError("cannot forget an active request")
        del self._status[request_id]

    def snapshot(self) -> SchedulerSnapshot:
        statuses = tuple(self._status.values())
        return SchedulerSnapshot(
            iteration=self._iteration,
            waiting=sum(status is RequestStatus.WAITING for status in statuses),
            running=sum(status is RequestStatus.RUNNING for status in statuses),
            terminal=sum(status.terminal for status in statuses),
            reserved_blocks=0,
            allocated_blocks=0,
            last_batch=(),
            model_batches=(),
        )


class SchedulerRuntimeTests(TestCase):
    """The worker owns scheduling and exposes ordered async streams."""

    def test_worker_routes_concurrent_streams_and_reports_metrics(self) -> None:
        runtime = SchedulerRuntime(FakeSchedulerEngine())  # type: ignore[arg-type]
        runtime.start()
        try:
            request_ids = [
                runtime.submit(
                    [{"role": "user", "content": prompt}],
                    SamplingParams(max_new_tokens=2),
                )
                for prompt in ("hello", "again")
            ]

            async def collect(request_id: str) -> list[StreamEvent]:
                return [event async for event in runtime.stream(request_id)]

            async def collect_all() -> list[list[StreamEvent]]:
                return await asyncio.gather(
                    *(collect(request_id) for request_id in request_ids)
                )

            streams = asyncio.run(collect_all())
            self.assertEqual(
                [[event.text for event in stream] for stream in streams],
                [["done:req-1"], ["done:req-2"]],
            )
            self.assertTrue(
                all(stream[-1].status is RequestStatus.FINISHED for stream in streams)
            )
            self.assertEqual(runtime.health()["status"], "ok")
            metrics = runtime.metrics()
            self.assertIn("forge_requests_total 2", metrics)
            self.assertIn(
                'forge_requests_terminal_total{status="finished"} 2',
                metrics,
            )
            self.assertIn("forge_generated_tokens_total 2", metrics)
            for request_id in request_ids:
                runtime.release(request_id)
        finally:
            runtime.stop()


class ServerTests(TestCase):
    """Protocol shape, lifecycle, validation, and observability routes."""

    def test_streaming_chat_uses_openai_chunk_shape(self) -> None:
        runtime = FakeServingRuntime()
        app = create_app(runtime)
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": DEFAULT_MODEL_ID,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                    "max_tokens": 7,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["END"],
                    "seed": 731,
                },
            ) as response:
                self.assertEqual(response.status_code, 200)
                self.assertTrue(
                    response.headers["content-type"].startswith("text/event-stream")
                )
                self.assertEqual(
                    response.headers["x-forge-request-id"],
                    "req-test",
                )
                data_lines = [
                    line.removeprefix("data: ")
                    for line in response.iter_lines()
                    if line.startswith("data: ")
                ]

        self.assertEqual(data_lines[-1], "[DONE]")
        chunks = [json.loads(line) for line in data_lines[:-1]]
        self.assertEqual(
            chunks[0]["choices"][0]["delta"],
            {"role": "assistant", "content": ""},
        )
        self.assertEqual(
            "".join(
                chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
            ),
            "Hello world",
        )
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        self.assertTrue(all(chunk["model"] == DEFAULT_MODEL_ID for chunk in chunks))
        self.assertEqual(
            runtime.submissions[0][0],
            [{"role": "user", "content": "Hi"}],
        )
        params = runtime.submissions[0][1]
        self.assertEqual(params.max_new_tokens, 7)
        self.assertEqual(params.stop_strings, ("END",))
        self.assertEqual(params.seed, 731)
        self.assertEqual(runtime.started, 1)
        self.assertEqual(runtime.stopped, 1)

    def test_admitted_request_can_be_cancelled_by_id(self) -> None:
        runtime = FakeServingRuntime()
        with TestClient(create_app(runtime)) as client:
            response = client.post("/v1/requests/req-test/cancel")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"request_id": "req-test", "status": "cancelled"},
        )
        self.assertEqual(runtime.cancelled, ["req-test"])

    def test_health_metrics_and_browser_chat_are_served(self) -> None:
        runtime = FakeServingRuntime()
        with TestClient(create_app(runtime)) as client:
            health = client.get("/health")
            metrics = client.get("/metrics")
            browser = client.get("/")
            models = client.get("/v1/models")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["model"], DEFAULT_MODEL_ID)
        self.assertEqual(
            health.json()["revision"],
            SUPPORTED_MODEL_REVISION,
        )
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("forge_requests_total", metrics.text)
        self.assertIn("text/plain", metrics.headers["content-type"])
        self.assertEqual(browser.status_code, 200)
        self.assertIn("ForgeEngine Chat", browser.text)
        self.assertIn("/v1/chat/completions", browser.text)
        self.assertEqual(models.status_code, 200)
        self.assertEqual(models.json()["object"], "list")
        self.assertEqual(
            models.json()["data"],
            [
                {
                    "id": DEFAULT_MODEL_ID,
                    "object": "model",
                    "created": 0,
                    "owned_by": "Qwen",
                }
            ],
        )

    def test_invalid_requests_are_rejected_before_submission(self) -> None:
        runtime = FakeServingRuntime()
        with TestClient(create_app(runtime)) as client:
            wrong_model = client.post(
                "/v1/chat/completions",
                json={
                    "model": "unsupported/model",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            non_streaming = client.post(
                "/v1/chat/completions",
                json={
                    "model": DEFAULT_MODEL_ID,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": False,
                },
            )
            empty_stop = client.post(
                "/v1/chat/completions",
                json={
                    "model": DEFAULT_MODEL_ID,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stop": [""],
                },
            )

        self.assertEqual(wrong_model.status_code, 422)
        self.assertEqual(non_streaming.status_code, 422)
        self.assertEqual(empty_stop.status_code, 400)
        self.assertEqual(runtime.submissions, [])

    def test_admission_overload_is_http_429(self) -> None:
        runtime = FakeServingRuntime(overload=True)
        with TestClient(create_app(runtime)) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": DEFAULT_MODEL_ID,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

        self.assertEqual(response.status_code, 429)
        self.assertIn("request limit is full", response.text)
