"""Run Milestone 4 paged-state acceptance checks on one Modal L4."""

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

APP_NAME = "forge-engine-m4-validation"
PROMPT = "Write the first ten positive integers separated by spaces."
DECODE_STEPS = 8
PROBABILITY_TV_LIMIT = 0.05

app = modal.App(APP_NAME)


def _record(
    condition: bool,
    message: str,
    failures: list[str],
) -> None:
    """Collect a failed acceptance condition without stopping later checks."""
    if not condition:
        failures.append(message)
        print(f"M4_CHECK_FAILURE={message}", flush=True)


def _numbered_layers(
    torch: object,
    *,
    sequence_length: int,
    device: object,
    layer_count: int = 2,
) -> tuple[tuple[object, object], ...]:
    """Build stable BF16 CUDA cache data for exact block-pool checks."""
    positions = (
        torch.arange(sequence_length, device=device).view(1, 1, -1, 1) * 10
    )
    heads = torch.arange(2, device=device).view(1, 2, 1, 1) * 100
    dimensions = torch.arange(4, device=device).view(1, 1, 1, 4)
    base = (positions + heads + dimensions).to(torch.bfloat16)
    return tuple(
        (
            base + layer_index * 1_000,
            base + layer_index * 1_000 + 500,
        )
        for layer_index in range(layer_count)
    )


def _exact_layers_equal(
    torch: object,
    actual: tuple[tuple[object, object], ...],
    expected: tuple[tuple[object, object], ...],
) -> bool:
    """Return exact equality for every key and value tensor."""
    return len(actual) == len(expected) and all(
        torch.equal(actual_key, expected_key)
        and torch.equal(actual_value, expected_value)
        for (actual_key, actual_value), (expected_key, expected_value) in zip(
            actual,
            expected,
            strict=True,
        )
    )


def _probability_tv(torch: object, left: object, right: object) -> float:
    """Return maximum total variation between row-wise token distributions."""
    left_probabilities = torch.softmax(left.float(), dim=-1)
    right_probabilities = torch.softmax(right.float(), dim=-1)
    return float(
        (
            0.5
            * (left_probabilities - right_probabilities)
            .abs()
            .sum(dim=-1)
        )
        .max()
        .item()
    )


@app.function(
    image=image,
    gpu="L4",
    timeout=30 * 60,
    volumes={"/cache": hf_cache_volume},
)
def validate_m4() -> dict[str, object]:
    """Validate paged allocation and reference decode on a real L4."""
    import torch

    from forge_engine.cache import (
        KVCacheCapacityError,
        PagedKVBlockPool,
        PagedKVCache,
    )
    from forge_engine.engine import GenerationCore, _normalize_tokenizer_output
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
        raise RuntimeError("M4 acceptance requires L4 BF16 support")
    print(f"GPU={device_name}; torch={torch.__version__}", flush=True)

    subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REMOTE_REPOSITORY,
        check=True,
    )

    synthetic_evidence: dict[str, object] = {}
    try:
        pool = PagedKVBlockPool(block_size=4, capacity=6)
        first = PagedKVCache(pool)
        second = PagedKVCache(pool)
        five_tokens = _numbered_layers(
            torch,
            sequence_length=5,
            device=device,
        )
        first.replace(five_tokens)
        second.replace(five_tokens)
        _record(
            first.block_table == (0, 1),
            f"unexpected first block table: {first.block_table}",
            failures,
        )
        _record(
            second.block_table == (2, 3),
            f"unexpected second block table: {second.block_table}",
            failures,
        )
        _record(
            _exact_layers_equal(
                torch,
                first.as_model_input(),
                five_tokens,
            ),
            "paged prefill did not gather exactly",
            failures,
        )

        first.clear()
        for length in (6, 7, 8, 9):
            second.append(
                _numbered_layers(
                    torch,
                    sequence_length=length,
                    device=device,
                )
            )
        _record(
            second.block_table == (2, 3, 0),
            f"fragmented extension did not reuse block 0: "
            f"{second.block_table}",
            failures,
        )
        nine_tokens = _numbered_layers(
            torch,
            sequence_length=9,
            device=device,
        )
        _record(
            _exact_layers_equal(
                torch,
                second.as_model_input(),
                nine_tokens,
            ),
            "fragmented paged gather changed tensor values",
            failures,
        )

        materialized_before_cleanup = pool.materialized_block_count
        second.clear()
        _record(
            pool.allocated_block_count == 0,
            "normal cleanup leaked synthetic blocks",
            failures,
        )
        replacement = PagedKVCache(pool)
        replacement.replace(
            _numbered_layers(
                torch,
                sequence_length=4,
                device=device,
            )
        )
        _record(
            replacement.block_table == (2,),
            f"cleanup did not reuse the most recent block: "
            f"{replacement.block_table}",
            failures,
        )
        _record(
            pool.materialized_block_count == materialized_before_cleanup,
            "reuse allocated an unnecessary physical tensor block",
            failures,
        )
        replacement.clear()

        small_pool = PagedKVBlockPool(block_size=2, capacity=1)
        oversized = PagedKVCache(small_pool)
        capacity_rejected = False
        try:
            oversized.replace(
                _numbered_layers(
                    torch,
                    sequence_length=3,
                    device=device,
                )
            )
        except KVCacheCapacityError:
            capacity_rejected = True
        _record(
            capacity_rejected,
            "oversized allocation was not rejected",
            failures,
        )
        _record(
            small_pool.allocated_block_count == 0
            and small_pool.materialized_block_count == 0,
            "capacity failure was not transactional",
            failures,
        )
        synthetic_evidence = {
            "fragmented_block_table": (2, 3, 0),
            "materialized_blocks": materialized_before_cleanup,
            "capacity_failure_transactional": capacity_rejected,
        }
        print("M4_SYNTHETIC_PAGING=COMPLETE", flush=True)
    except BaseException as error:
        failures.append(f"synthetic paging phase raised {error!r}")
        print(f"M4_SYNTHETIC_EXCEPTION={error!r}", flush=True)

    model_evidence: dict[str, object] = {}
    try:
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
        model_pool = PagedKVBlockPool(block_size=16, capacity=64)
        core = GenerationCore(model, model_pool)
        encoding = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids, tokenizer_mask = _normalize_tokenizer_output(encoding)
        complete_ids = input_ids.to(device)
        attention_mask = (
            torch.ones_like(complete_ids)
            if tokenizer_mask is None
            else tokenizer_mask.to(device)
        )
        generated: list[int] = []
        probability_distances: list[float] = []
        state = None
        try:
            with torch.inference_mode():
                state = core.prefill(complete_ids, attention_mask)
                expected_blocks = (
                    complete_ids.shape[1] + model_pool.block_size - 1
                ) // model_pool.block_size
                _record(
                    len(state.cache.block_table) == expected_blocks,
                    f"prefill used {len(state.cache.block_table)} blocks; "
                    f"expected {expected_blocks}",
                    failures,
                )
                for step in range(DECODE_STEPS):
                    uncached = core.uncached(complete_ids, attention_mask)
                    paged_logits = state.logits[:, -1, :]
                    uncached_logits = uncached.logits[:, -1, :]
                    paged_next = paged_logits.argmax(dim=-1)
                    uncached_next = uncached_logits.argmax(dim=-1)
                    _record(
                        torch.equal(paged_next, uncached_next),
                        f"paged/uncached greedy mismatch at step {step}: "
                        f"{paged_next.tolist()} != {uncached_next.tolist()}",
                        failures,
                    )
                    distance = _probability_tv(
                        torch,
                        paged_logits,
                        uncached_logits,
                    )
                    probability_distances.append(distance)
                    print(
                        f"paged_vs_uncached_step_{step}: "
                        f"probability_tv={distance:.8f}, "
                        f"block_table={state.cache.block_table}",
                        flush=True,
                    )
                    generated.append(int(paged_next.item()))
                    complete_ids = torch.cat(
                        (complete_ids, paged_next.unsqueeze(1)),
                        dim=1,
                    )
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            torch.ones(
                                (attention_mask.shape[0], 1),
                                dtype=attention_mask.dtype,
                                device=device,
                            ),
                        ),
                        dim=1,
                    )
                    if step + 1 < DECODE_STEPS:
                        state = core.decode(paged_next.unsqueeze(1), state)
        finally:
            if state is not None:
                state.cache.clear()
        maximum_probability_tv = max(probability_distances)
        _record(
            maximum_probability_tv <= PROBABILITY_TV_LIMIT,
            f"maximum paged probability_tv="
            f"{maximum_probability_tv:.8f} exceeds "
            f"{PROBABILITY_TV_LIMIT}",
            failures,
        )
        _record(
            model_pool.allocated_block_count == 0,
            "model decode cleanup leaked physical blocks",
            failures,
        )
        decoded = tokenizer.decode(
            generated,
            skip_special_tokens=False,
        )
        model_evidence = {
            "revision": snapshot.name,
            "greedy_token_ids": generated,
            "greedy_text": decoded,
            "maximum_probability_tv": maximum_probability_tv,
            "materialized_blocks": model_pool.materialized_block_count,
        }
        print(f"PAGED_GREEDY_TOKEN_IDS={generated}", flush=True)
        print(f"PAGED_GREEDY_TEXT={decoded!r}", flush=True)
        print("M4_MODEL_PAGING=COMPLETE", flush=True)
    except BaseException as error:
        failures.append(f"model paging phase raised {error!r}")
        print(f"M4_MODEL_EXCEPTION={error!r}", flush=True)

    torch.cuda.synchronize()
    hf_cache_volume.commit()
    if failures:
        print(f"M4_FAILURES={failures!r}", flush=True)
        raise RuntimeError(
            f"M4 acceptance found {len(failures)} failure(s); "
            "see M4_FAILURES above"
        )
    return {
        "gpu": device_name,
        "synthetic": synthetic_evidence,
        "model": model_evidence,
    }


@app.local_entrypoint()
def main() -> None:
    """Invoke M4 validation and print retained evidence."""
    result = validate_m4.remote()
    print("M4_ACCEPTANCE=PASS")
    for name, value in result.items():
        print(f"{name}={value}")
