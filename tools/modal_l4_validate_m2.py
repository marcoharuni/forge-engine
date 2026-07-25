"""Run the Milestone 2 oracle checks on one Modal L4 GPU."""

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

APP_NAME = "forge-engine-m2-validation"
SELECTED_LAYERS = (0, 17, 35)
HIDDEN_ATOL = 0.02
HIDDEN_RTOL = 0.02
LOGITS_ATOL = 0.125
LOGITS_RTOL = 0.02
GREEDY_STEPS = 8
PROMPT = "Write the first ten positive integers separated by spaces."

app = modal.App(APP_NAME)


def _assert_close(
    label: str,
    actual: object,
    expected: object,
    *,
    atol: float,
    rtol: float,
) -> None:
    """Assert tensor closeness and print auditable error statistics."""
    import torch

    actual_float = actual.float()
    expected_float = expected.float()
    difference = (actual_float - expected_float).abs()
    maximum = difference.max().item()
    mean = difference.mean().item()
    print(
        f"{label}: max_abs={maximum:.8f}, mean_abs={mean:.8f}, "
        f"atol={atol}, rtol={rtol}",
        flush=True,
    )
    torch.testing.assert_close(
        actual_float,
        expected_float,
        atol=atol,
        rtol=rtol,
    )


@app.function(
    image=image,
    gpu="L4",
    timeout=30 * 60,
    volumes={"/cache": hf_cache_volume},
)
def validate_m2() -> dict[str, object]:
    """Compare the staged Forge runner with Transformers on a real L4."""
    import torch
    import transformers

    from forge_engine.engine import _normalize_tokenizer_output
    from forge_engine.model import load_transformers_oracle
    from forge_engine.weights import (
        EXPECTED_PARAMETER_BYTES,
        download_supported_snapshot,
        load_staged_model,
        load_supported_tokenizer,
        parameter_bytes,
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
        raise RuntimeError("M2 acceptance requires L4 BF16 support")
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
    tokenizer, tokenizer_inspection = load_supported_tokenizer(snapshot)
    forge_model, package_inspection = load_staged_model(
        snapshot,
        device=device,
        dtype=torch.bfloat16,
    )
    if parameter_bytes(forge_model) != EXPECTED_PARAMETER_BYTES:
        raise RuntimeError("Forge model allocation does not match BF16 package bytes")

    oracle_tokenizer, oracle_model = load_transformers_oracle(torch.bfloat16)
    oracle_model.to(device)
    oracle_model.eval()

    messages = [{"role": "user", "content": PROMPT}]
    forge_encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    oracle_encoding = oracle_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    forge_ids, forge_attention_mask = _normalize_tokenizer_output(
        forge_encoding
    )
    oracle_ids, oracle_attention_mask = _normalize_tokenizer_output(
        oracle_encoding
    )
    if not torch.equal(forge_ids, oracle_ids):
        raise RuntimeError("Forge and oracle chat-template token IDs differ")
    if (forge_attention_mask is None) != (oracle_attention_mask is None):
        raise RuntimeError("Forge and oracle tokenizer mask presence differs")
    if (
        forge_attention_mask is not None
        and not torch.equal(forge_attention_mask, oracle_attention_mask)
    ):
        raise RuntimeError("Forge and oracle tokenizer masks differ")
    input_ids = forge_ids.to(device)
    attention_mask = (
        torch.ones_like(input_ids)
        if forge_attention_mask is None
        else forge_attention_mask.to(device)
    )

    oracle_layers: dict[int, torch.Tensor] = {}
    oracle_final: list[torch.Tensor] = []

    def capture_layer(index: int):
        """Build a hook retaining one selected decoder output."""

        def hook(
            _module: object,
            _inputs: tuple[object, ...],
            output: torch.Tensor,
        ) -> None:
            oracle_layers[index] = output.detach().clone()

        return hook

    handles = [
        oracle_model.model.layers[index].register_forward_hook(
            capture_layer(index)
        )
        for index in SELECTED_LAYERS
    ]
    handles.append(
        oracle_model.model.norm.register_forward_hook(
            lambda _module, _inputs, output: oracle_final.append(
                output.detach().clone()
            )
        )
    )
    try:
        with torch.inference_mode():
            forge_output = forge_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            oracle_output = oracle_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    for index in SELECTED_LAYERS:
        _assert_close(
            f"layer_{index}",
            forge_output.layer_hidden_states[index],
            oracle_layers[index],
            atol=HIDDEN_ATOL,
            rtol=HIDDEN_RTOL,
        )
    if len(oracle_final) != 1:
        raise RuntimeError("oracle final hidden state was not captured exactly once")
    _assert_close(
        "final_hidden_state",
        forge_output.last_hidden_state,
        oracle_final[0],
        atol=HIDDEN_ATOL,
        rtol=HIDDEN_RTOL,
    )
    _assert_close(
        "prompt_logits",
        forge_output.logits,
        oracle_output.logits,
        atol=LOGITS_ATOL,
        rtol=LOGITS_RTOL,
    )

    forge_parameters = dict(forge_model.named_parameters())
    oracle_parameters = dict(oracle_model.named_parameters())
    checked_parameters = (
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.17.mlp.down_proj.weight",
        "model.norm.weight",
    )
    for name in checked_parameters:
        if not torch.equal(forge_parameters[name], oracle_parameters[name]):
            raise RuntimeError(f"loaded parameter values differ for {name}")

    generated: list[int] = []
    with torch.inference_mode():
        for _ in range(GREEDY_STEPS):
            forge_next = forge_output.logits[:, -1].argmax(dim=-1)
            oracle_next = oracle_output.logits[:, -1].argmax(dim=-1)
            if not torch.equal(forge_next, oracle_next):
                raise RuntimeError(
                    f"greedy token mismatch: "
                    f"Forge={forge_next.tolist()}, "
                    f"oracle={oracle_next.tolist()}"
                )
            generated.append(int(forge_next.item()))
            next_input = forge_next.unsqueeze(1)
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
            forge_output = forge_model(
                input_ids=next_input,
                attention_mask=attention_mask,
                past_key_values=forge_output.past_key_values,
                use_cache=True,
            )
            oracle_output = oracle_model(
                input_ids=next_input,
                attention_mask=attention_mask,
                past_key_values=oracle_output.past_key_values,
                use_cache=True,
            )
    torch.cuda.synchronize()
    decoded = tokenizer.decode(generated, skip_special_tokens=False)
    print(f"GREEDY_TOKEN_IDS={generated}", flush=True)
    print(f"GREEDY_TEXT={decoded!r}", flush=True)

    hf_cache_volume.commit()
    print("Committed forge-engine-hf-cache", flush=True)
    return {
        "gpu": device_name,
        "revision": snapshot.name,
        "package": package_inspection,
        "tokenizer": tokenizer_inspection,
        "greedy_token_ids": generated,
        "greedy_text": decoded,
    }


@app.local_entrypoint()
def main() -> None:
    """Invoke the remote M2 validator and print its retained evidence."""
    result = validate_m2.remote()
    print("M2_ACCEPTANCE=PASS")
    for name, value in result.items():
        print(f"{name}={value}")
