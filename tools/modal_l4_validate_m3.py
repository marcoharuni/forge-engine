"""Run the Milestone 3 generation-core checks on one Modal L4 GPU."""

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

APP_NAME = "forge-engine-m3-validation"
PROMPT = "Write the first ten positive integers separated by spaces."
MODEL_CODE = "COBALT-731"
DECODE_STEPS = 8
PROBABILITY_TV_LIMIT = 0.05

app = modal.App(APP_NAME)


def _measure_decode_agreement(
    label: str,
    cached: object,
    uncached: object,
) -> dict[str, float]:
    """Measure raw and generation-relevant BF16 decode differences."""
    import torch

    cached_float = cached.float()
    uncached_float = uncached.float()
    difference = (cached_float - uncached_float).abs()
    maximum = float(difference.max().item())
    mean = float(difference.mean().item())
    cached_centered = cached_float - cached_float.max(
        dim=-1,
        keepdim=True,
    ).values
    uncached_centered = uncached_float - uncached_float.max(
        dim=-1,
        keepdim=True,
    ).values
    centered_difference = (cached_centered - uncached_centered).abs()
    centered_maximum = float(centered_difference.max().item())
    centered_mean = float(centered_difference.mean().item())
    cached_probabilities = torch.softmax(cached_float, dim=-1)
    uncached_probabilities = torch.softmax(uncached_float, dim=-1)
    total_variation = float(
        (
            0.5
            * (cached_probabilities - uncached_probabilities)
            .abs()
            .sum(dim=-1)
        )
        .max()
        .item()
    )
    print(
        f"{label}: max_abs={maximum:.8f}, mean_abs={mean:.8f}, "
        f"centered_max_abs={centered_maximum:.8f}, "
        f"centered_mean_abs={centered_mean:.8f}, "
        f"probability_tv={total_variation:.8f}",
        flush=True,
    )
    return {
        "max_abs": maximum,
        "mean_abs": mean,
        "centered_max_abs": centered_maximum,
        "centered_mean_abs": centered_mean,
        "probability_tv": total_variation,
    }


@app.function(
    image=image,
    gpu="L4",
    timeout=30 * 60,
    volumes={"/cache": hf_cache_volume},
)
def validate_m3() -> dict[str, object]:
    """Validate cached generation, sampling, and multi-turn chat on an L4."""
    import torch
    import transformers

    from forge_engine.engine import (
        GenerationCore,
        GenerationEngine,
        _normalize_tokenizer_output,
    )
    from forge_engine.sampling import SamplingParams, filter_logits, sample_token
    from forge_engine.weights import (
        download_supported_snapshot,
        load_staged_model,
        load_supported_tokenizer,
    )

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
        raise RuntimeError("M3 acceptance requires L4 BF16 support")
    print(
        f"GPU={device_name}; torch={torch.__version__}; "
        f"transformers={transformers.__version__}",
        flush=True,
    )

    subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=REMOTE_REPOSITORY,
        check=True,
    )

    snapshot = download_supported_snapshot()
    if snapshot.name != SUPPORTED_MODEL_REVISION:
        raise RuntimeError(
            f"snapshot resolved to {snapshot.name}, "
            f"expected {SUPPORTED_MODEL_REVISION}"
        )
    tokenizer, _ = load_supported_tokenizer(snapshot)
    model, _ = load_staged_model(
        snapshot,
        device=device,
        dtype=torch.bfloat16,
    )
    core = GenerationCore(model)
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
    agreement: list[dict[str, float]] = []
    with torch.inference_mode():
        state = core.prefill(complete_ids, attention_mask)
        for step in range(DECODE_STEPS):
            if not all(
                key.is_contiguous() and value.is_contiguous()
                for key, value in state.cache.layers
            ):
                raise RuntimeError("KV cache contains non-contiguous tensors")
            uncached = core.uncached(complete_ids, attention_mask)
            cached_logits = state.logits[:, -1, :]
            uncached_logits = uncached.logits[:, -1, :]
            cached_next = cached_logits.argmax(dim=-1)
            uncached_next = uncached_logits.argmax(dim=-1)
            if not torch.equal(cached_next, uncached_next):
                raise RuntimeError(
                    f"cached/uncached greedy mismatch at step {step}: "
                    f"{cached_next.tolist()} != {uncached_next.tolist()}"
                )
            agreement.append(
                _measure_decode_agreement(
                    f"cached_vs_uncached_step_{step}",
                    cached_logits,
                    uncached_logits,
                )
            )
            generated.append(int(cached_next.item()))
            complete_ids = torch.cat(
                (complete_ids, cached_next.unsqueeze(1)),
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
                state = core.decode(cached_next.unsqueeze(1), state)

    synthetic = torch.tensor(
        [[5.0, 4.0, 1.0, 0.0, -2.0]],
        device=device,
    )
    if sample_token(synthetic, SamplingParams()).item() != 0:
        raise RuntimeError("GPU greedy sampling did not select argmax")
    top_k = filter_logits(
        synthetic,
        SamplingParams(temperature=0.8, top_k=2),
    )
    if torch.isfinite(top_k).sum().item() != 2:
        raise RuntimeError("GPU top-k filter retained the wrong token count")
    top_p = filter_logits(
        synthetic,
        SamplingParams(temperature=1.0, top_p=0.7),
    )
    if torch.isfinite(top_p).sum().item() != 1:
        raise RuntimeError("GPU top-p filter retained the wrong token count")
    min_p = filter_logits(
        synthetic,
        SamplingParams(temperature=1.0, min_p=0.2),
    )
    if torch.isfinite(min_p).sum().item() != 2:
        raise RuntimeError("GPU min-p filter retained the wrong token count")
    left = torch.Generator(device=device).manual_seed(731)
    right = torch.Generator(device=device).manual_seed(731)
    sampled_params = SamplingParams(temperature=1.0)
    left_tokens = [
        int(sample_token(synthetic, sampled_params, generator=left).item())
        for _ in range(16)
    ]
    right_tokens = [
        int(sample_token(synthetic, sampled_params, generator=right).item())
        for _ in range(16)
    ]
    if left_tokens != right_tokens:
        raise RuntimeError("seeded GPU sampling is not reproducible")
    print("GPU_SAMPLING=PASS", flush=True)

    engine = GenerationEngine(
        tokenizer,
        model,
        SamplingParams(max_new_tokens=64),
    )
    messages = [
        {
            "role": "user",
            "content": (
                f"Remember this code exactly: {MODEL_CODE}. "
                "Reply only with STORED."
            ),
        }
    ]
    first_answer = "".join(engine.stream(messages))
    messages.append({"role": "assistant", "content": first_answer})
    messages.append(
        {
            "role": "user",
            "content": "What code did I ask you to remember?",
        }
    )
    second_answer = "".join(engine.stream(messages))
    print(f"MULTI_TURN_FIRST={first_answer!r}", flush=True)
    print(f"MULTI_TURN_SECOND={second_answer!r}", flush=True)
    if MODEL_CODE not in second_answer:
        raise RuntimeError(
            f"second answer did not contain {MODEL_CODE}: "
            f"{second_answer!r}"
        )
    maximum_probability_tv = max(
        metrics["probability_tv"] for metrics in agreement
    )
    if maximum_probability_tv > PROBABILITY_TV_LIMIT:
        raise AssertionError(
            f"maximum cached/uncached probability_tv="
            f"{maximum_probability_tv:.8f} exceeds "
            f"{PROBABILITY_TV_LIMIT}"
        )
    print(
        f"CACHED_UNCACHED_PROBABILITY_TV_MAX="
        f"{maximum_probability_tv:.8f}",
        flush=True,
    )

    torch.cuda.synchronize()
    decoded = tokenizer.decode(generated, skip_special_tokens=False)
    print(f"CACHED_GREEDY_TOKEN_IDS={generated}", flush=True)
    print(f"CACHED_GREEDY_TEXT={decoded!r}", flush=True)
    hf_cache_volume.commit()
    return {
        "gpu": device_name,
        "revision": snapshot.name,
        "cached_greedy_token_ids": generated,
        "cached_greedy_text": decoded,
        "cached_uncached_agreement": agreement,
        "multi_turn_second": second_answer,
    }


@app.local_entrypoint()
def main() -> None:
    """Invoke the M3 validator and print its retained evidence."""
    result = validate_m3.remote()
    print("M3_ACCEPTANCE=PASS")
    for name, value in result.items():
        print(f"{name}={value}")
