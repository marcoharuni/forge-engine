"""Run Milestone 5 concurrent-engine acceptance on one Modal L4."""

from __future__ import annotations

import subprocess
import sys

import modal

from forge_engine.config import SUPPORTED_MODEL_REVISION
from tools.modal_l4_validate import (
    REMOTE_REPOSITORY,
    hf_cache_volume,
    image,
)

APP_NAME = "forge-engine-m5-validation"
PROMPT = "Write the first ten positive integers separated by spaces."
EXPECTED_TEXT = "1 2 3 4 "
MAX_NEW_TOKENS = 8

app = modal.App(APP_NAME)


def _record(
    condition: bool,
    message: str,
    failures: list[str],
) -> None:
    """Collect a failed condition while allowing independent checks to run."""
    if not condition:
        failures.append(message)
        print(f"M5_CHECK_FAILURE={message}", flush=True)


@app.function(
    image=image,
    gpu="L4",
    timeout=30 * 60,
    volumes={"/cache": hf_cache_volume},
)
def validate_m5() -> dict[str, object]:
    """Validate concurrent progress, cancellation, and bounded admission."""
    import torch

    from forge_engine.engine import _normalize_tokenizer_output
    from forge_engine.sampling import SamplingParams
    from forge_engine.scheduler import (
        ConcurrentGenerationEngine,
        OverloadedError,
        RequestStatus,
        SchedulerConfig,
    )
    from forge_engine.weights import (
        download_supported_snapshot,
        load_staged_model,
        load_supported_tokenizer,
    )

    failures: list[str] = []
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one GPU, found {torch.cuda.device_count()}"
        )
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    if "L4" not in device_name:
        raise RuntimeError(f"expected an L4 GPU, found {device_name!r}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("M5 acceptance requires L4 BF16 support")
    print(f"GPU={device_name}; torch={torch.__version__}", flush=True)

    subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REMOTE_REPOSITORY,
        check=True,
    )

    snapshot = download_supported_snapshot()
    _record(
        snapshot.name == SUPPORTED_MODEL_REVISION,
        f"snapshot revision is {snapshot.name}",
        failures,
    )
    tokenizer, _ = load_supported_tokenizer(snapshot)
    model, _ = load_staged_model(
        snapshot,
        device=device,
        dtype=torch.bfloat16,
    )
    sampling = SamplingParams(max_new_tokens=MAX_NEW_TOKENS)

    concurrency_evidence: dict[str, object] = {}
    try:
        engine = ConcurrentGenerationEngine(
            tokenizer,
            model,
            sampling,
            SchedulerConfig(
                max_requests=4,
                max_batch_size=3,
                token_budget=128,
                block_size=16,
                block_capacity=64,
            ),
        )
        messages = [{"role": "user", "content": PROMPT}]
        engine.submit(messages, request_id="alpha")
        engine.submit(messages, request_id="beta")
        events = engine.step()
        first_snapshot = engine.snapshot()
        first_batch = first_snapshot.last_batch
        _record(
            first_batch == ("alpha", "beta"),
            f"first work batch was {first_batch}",
            failures,
        )
        _record(
            first_snapshot.model_batches == (("alpha", "beta"),),
            f"first tensor batches were {first_snapshot.model_batches}",
            failures,
        )

        engine.submit(messages, request_id="gamma")
        events.extend(engine.step())
        second_snapshot = engine.snapshot()
        second_batch = second_snapshot.last_batch
        _record(
            second_batch == ("alpha", "beta", "gamma"),
            f"new request did not join running work: {second_batch}",
            failures,
        )
        _record(
            second_snapshot.model_batches
            == (("alpha", "beta"), ("gamma",)),
            f"continuous tensor batches were "
            f"{second_snapshot.model_batches}",
            failures,
        )
        allocated_before_cancel = engine.cache_pool.allocated_block_count
        _record(
            engine.cancel("gamma"),
            "running gamma request was not cancelled",
            failures,
        )
        allocated_after_cancel = engine.cache_pool.allocated_block_count
        _record(
            allocated_after_cancel < allocated_before_cancel,
            "cancellation did not immediately release physical blocks",
            failures,
        )
        events.extend(engine.run_until_idle())

        outputs = {
            request_id: "".join(
                event.text
                for event in events
                if event.request_id == request_id
            )
            for request_id in ("alpha", "beta")
        }
        for request_id, output in outputs.items():
            print(f"CONCURRENT_{request_id.upper()}={output!r}", flush=True)
            _record(
                output == EXPECTED_TEXT,
                f"{request_id} output was {output!r}",
                failures,
            )
            view = engine.request(request_id)
            _record(
                view.status is RequestStatus.FINISHED,
                f"{request_id} ended in {view.status}",
                failures,
            )
            _record(
                view.sampled_tokens == MAX_NEW_TOKENS,
                f"{request_id} sampled {view.sampled_tokens} tokens",
                failures,
            )
        _record(
            engine.request("gamma").status is RequestStatus.CANCELLED,
            f"gamma ended in {engine.request('gamma').status}",
            failures,
        )
        final_snapshot = engine.snapshot()
        _record(
            final_snapshot.allocated_blocks == 0,
            "concurrent completion leaked physical blocks",
            failures,
        )
        _record(
            final_snapshot.reserved_blocks == 0,
            "concurrent completion leaked reservations",
            failures,
        )
        concurrency_evidence = {
            "first_batch": first_batch,
            "continuous_batch": second_batch,
            "first_model_batches": first_snapshot.model_batches,
            "continuous_model_batches": second_snapshot.model_batches,
            "outputs": outputs,
            "cancelled": engine.request("gamma").status.value,
            "allocated_before_cancel": allocated_before_cancel,
            "allocated_after_cancel": allocated_after_cancel,
            "iterations": final_snapshot.iteration,
        }
        print("M5_CONCURRENT_PHASE=COMPLETE", flush=True)
    except BaseException as error:
        failures.append(f"concurrent phase raised {error!r}")
        print(f"M5_CONCURRENT_EXCEPTION={error!r}", flush=True)

    admission_evidence: dict[str, object] = {}
    try:
        encoding = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        prompt_ids, _ = _normalize_tokenizer_output(encoding)
        prompt_tokens = int(prompt_ids.shape[1])
        blocks_per_request = (
            prompt_tokens + MAX_NEW_TOKENS + 15
        ) // 16
        reservation_capacity = blocks_per_request * 2
        bounded = ConcurrentGenerationEngine(
            tokenizer,
            model,
            sampling,
            SchedulerConfig(
                max_requests=8,
                max_batch_size=2,
                token_budget=128,
                block_size=16,
                block_capacity=reservation_capacity,
            ),
        )
        messages = [{"role": "user", "content": PROMPT}]
        bounded.submit(messages, request_id="reserved-a")
        bounded.submit(messages, request_id="reserved-b")
        overload_message = ""
        try:
            bounded.submit(messages, request_id="rejected")
        except OverloadedError as error:
            overload_message = str(error)
        _record(
            bool(overload_message),
            "third reservation was not rejected",
            failures,
        )
        before_work = bounded.snapshot()
        _record(
            before_work.allocated_blocks == 0,
            "overload check happened after physical allocation",
            failures,
        )
        _record(
            before_work.reserved_blocks == reservation_capacity,
            f"reserved {before_work.reserved_blocks} blocks; "
            f"expected {reservation_capacity}",
            failures,
        )
        bounded.cancel("reserved-a")
        bounded.cancel("reserved-b")
        _record(
            bounded.snapshot().reserved_blocks == 0,
            "cancelling queued work leaked reservations",
            failures,
        )
        admission_evidence = {
            "prompt_tokens": prompt_tokens,
            "blocks_per_request": blocks_per_request,
            "capacity": reservation_capacity,
            "overload_message": overload_message,
            "allocated_before_work": before_work.allocated_blocks,
        }
        print(f"M5_OVERLOAD_REJECTION={overload_message!r}", flush=True)
        print("M5_ADMISSION_PHASE=COMPLETE", flush=True)
    except BaseException as error:
        failures.append(f"admission phase raised {error!r}")
        print(f"M5_ADMISSION_EXCEPTION={error!r}", flush=True)

    torch.cuda.synchronize()
    hf_cache_volume.commit()
    if failures:
        print(f"M5_FAILURES={failures!r}", flush=True)
        raise RuntimeError(
            f"M5 acceptance found {len(failures)} failure(s); "
            "see M5_FAILURES above"
        )
    return {
        "gpu": device_name,
        "revision": snapshot.name,
        "concurrency": concurrency_evidence,
        "admission": admission_evidence,
    }


@app.local_entrypoint()
def main() -> None:
    """Invoke M5 validation and print retained evidence."""
    result = validate_m5.remote()
    print("M5_ACCEPTANCE=PASS")
    for name, value in result.items():
        print(f"{name}={value}")
