"""Pinned package download, inspection, and staged SafeTensors loading."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from math import prod
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn
from transformers import AutoTokenizer

from forge_engine.config import DEFAULT_MODEL_ID, SUPPORTED_MODEL_REVISION
from forge_engine.qwen3 import Qwen3Config, Qwen3ForCausalLM

INDEX_NAME = "model.safetensors.index.json"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
TOKENIZER_MODEL_NAME = "tokenizer.json"
EXPECTED_TENSOR_COUNT = 398
EXPECTED_SHARD_COUNT = 3
EXPECTED_TOTAL_SIZE = 8_045_591_552
EXPECTED_PARAMETER_BYTES = 8_044_936_192
EXPECTED_TOKENIZER_SIZE = 151_669
EXPECTED_PAD_TOKEN_ID = 151_643
EXPECTED_IM_START_TOKEN_ID = 151_644
EXPECTED_EOS_TOKEN_ID = 151_645
EXPECTED_CHAT_RENDER = "<|im_start|>user\nping<|im_end|>\n<|im_start|>assistant\n"


@dataclass(frozen=True, slots=True)
class PackageInspection:
    """Validated package facts recorded by M2 acceptance."""

    tensor_count: int
    shard_count: int
    total_size: int
    tensor_bytes: int
    dtype_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class TokenizerInspection:
    """Validated tokenizer facts recorded by M2 acceptance."""

    vocabulary_size: int
    pad_token_id: int
    eos_token_id: int
    im_start_token_id: int


def download_supported_snapshot() -> Path:
    """Download only files needed by the pinned supported model."""
    return Path(
        snapshot_download(
            repo_id=DEFAULT_MODEL_ID,
            revision=SUPPORTED_MODEL_REVISION,
            allow_patterns=(
                "*.json",
                "*.jinja",
                "*.model",
                "*.safetensors",
                "merges.txt",
                "vocab.json",
            ),
        )
    )


def load_supported_tokenizer(
    snapshot: Path,
) -> tuple[object, TokenizerInspection]:
    """Load and validate the tokenizer from the pinned local snapshot."""
    tokenizer_config_path = snapshot / TOKENIZER_CONFIG_NAME
    tokenizer_model_path = snapshot / TOKENIZER_MODEL_NAME
    if not tokenizer_config_path.is_file():
        raise ValueError(f"missing {TOKENIZER_CONFIG_NAME}")
    if not tokenizer_model_path.is_file():
        raise ValueError(f"missing {TOKENIZER_MODEL_NAME}")
    raw = json.loads(tokenizer_config_path.read_text())
    expected_config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "add_bos_token": False,
        "clean_up_tokenization_spaces": False,
    }
    for name, expected in expected_config.items():
        if raw.get(name) != expected:
            raise ValueError(
                f"unsupported tokenizer {name}={raw.get(name)!r}; expected {expected!r}"
            )
    if not isinstance(raw.get("chat_template"), str):
        raise ValueError("tokenizer chat_template must be a string")

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
    )
    facts = {
        "vocabulary_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "im_start_token_id": tokenizer.convert_tokens_to_ids("<|im_start|>"),
    }
    expected_facts = {
        "vocabulary_size": EXPECTED_TOKENIZER_SIZE,
        "pad_token_id": EXPECTED_PAD_TOKEN_ID,
        "eos_token_id": EXPECTED_EOS_TOKEN_ID,
        "im_start_token_id": EXPECTED_IM_START_TOKEN_ID,
    }
    for name, expected in expected_facts.items():
        if facts[name] != expected:
            raise ValueError(
                f"unsupported tokenizer {name}={facts[name]!r}; expected {expected!r}"
            )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "ping"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if rendered != EXPECTED_CHAT_RENDER:
        raise ValueError("tokenizer chat template rendered unexpected text")
    return tokenizer, TokenizerInspection(**facts)


def inspect_package(snapshot: Path) -> PackageInspection:
    """Validate index coverage and every SafeTensors name, shape, and dtype."""
    config = Qwen3Config.from_file(snapshot / "config.json")
    expected_shapes = _expected_parameter_shapes(config)
    index = json.loads((snapshot / INDEX_NAME).read_text())
    weight_map: dict[str, str] = index["weight_map"]
    total_size = int(index["metadata"]["total_size"])
    shards = sorted(set(weight_map.values()))
    if len(weight_map) != EXPECTED_TENSOR_COUNT:
        raise ValueError(f"expected {EXPECTED_TENSOR_COUNT} indexed tensors")
    if len(shards) != EXPECTED_SHARD_COUNT:
        raise ValueError(f"expected {EXPECTED_SHARD_COUNT} SafeTensors shards")
    if total_size != EXPECTED_TOTAL_SIZE:
        raise ValueError(f"unexpected package tensor bytes: {total_size}")
    if set(weight_map) != set(expected_shapes):
        missing = sorted(set(expected_shapes) - set(weight_map))
        extra = sorted(set(weight_map) - set(expected_shapes))
        raise ValueError(
            f"parameter/index mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    names: set[str] = set()
    dtypes: Counter[str] = Counter()
    tensor_bytes = 0
    for shard_name in shards:
        with safe_open(snapshot / shard_name, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if weight_map.get(name) != shard_name:
                    raise ValueError(f"tensor {name} is assigned to the wrong shard")
                tensor = shard.get_slice(name)
                shape = tensor.get_shape()
                if tuple(shape) != expected_shapes[name]:
                    raise ValueError(
                        f"shape mismatch for {name}: "
                        f"{tuple(shape)} != {expected_shapes[name]}"
                    )
                tensor_bytes += prod(shape) * 2
                dtypes[str(tensor.get_dtype())] += 1
                names.add(name)
    if names != set(weight_map):
        missing = sorted(set(weight_map) - names)
        raise ValueError(f"SafeTensors index mismatch; missing={missing[:3]}")
    if dtypes != {"BF16": EXPECTED_TENSOR_COUNT}:
        raise ValueError(f"expected only BF16 tensors, found {dict(dtypes)}")
    return PackageInspection(
        tensor_count=len(names),
        shard_count=len(shards),
        total_size=total_size,
        tensor_bytes=tensor_bytes,
        dtype_counts=dict(dtypes),
    )


def load_staged_model(
    snapshot: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Qwen3ForCausalLM, PackageInspection]:
    """Allocate the model once, then copy one SafeTensor at a time."""
    if device.type != "cuda":
        raise ValueError("staged model loading requires a CUDA device")
    inspection = inspect_package(snapshot)
    config = Qwen3Config.from_file(snapshot / "config.json")
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config)
    model.to(dtype=dtype)
    model.to_empty(device=device)
    load_safetensors_into_model(model, snapshot, device=device, dtype=dtype)
    model.eval()
    return model, inspection


def load_safetensors_into_model(
    model: nn.Module,
    snapshot: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Copy indexed tensors into an allocated model one tensor at a time."""
    parameters = dict(model.named_parameters())
    index = json.loads((snapshot / INDEX_NAME).read_text())
    weight_map: dict[str, str] = index["weight_map"]
    if set(parameters) != set(weight_map):
        missing = sorted(set(parameters) - set(weight_map))
        extra = sorted(set(weight_map) - set(parameters))
        raise ValueError(
            f"parameter/index mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    with torch.no_grad():
        for shard_name in sorted(set(weight_map.values())):
            with safe_open(
                snapshot / shard_name, framework="pt", device="cpu"
            ) as shard:
                for name in shard.keys():
                    source = shard.get_tensor(name)
                    target = parameters[name]
                    if not _device_matches(target.device, device):
                        raise ValueError(f"{name} is allocated on the wrong device")
                    if target.dtype != dtype:
                        raise ValueError(
                            f"{name} target dtype {target.dtype} != {dtype}"
                        )
                    if tuple(source.shape) != tuple(target.shape):
                        raise ValueError(
                            f"shape mismatch for {name}: "
                            f"{tuple(source.shape)} != {tuple(target.shape)}"
                        )
                    if source.dtype != torch.bfloat16:
                        raise ValueError(f"{name} must use torch.bfloat16")
                    if not torch.isfinite(source).all():
                        raise ValueError(f"{name} contains non-finite values")
                    target.copy_(source.to(device=device, dtype=dtype))
                    del source


def parameter_bytes(model: nn.Module) -> int:
    """Return allocated parameter bytes without double-counting tensors."""
    return sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )


def _expected_parameter_shapes(
    config: Qwen3Config,
) -> dict[str, tuple[int, ...]]:
    """Return the complete parameter schema for the supported Qwen3 model."""
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    query = config.num_attention_heads * config.head_dim
    key_value = config.num_key_value_heads * config.head_dim
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (config.vocab_size, hidden),
        "model.norm.weight": (hidden,),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.self_attn.q_norm.weight": (config.head_dim,),
                f"{prefix}.self_attn.k_norm.weight": (config.head_dim,),
                f"{prefix}.self_attn.q_proj.weight": (query, hidden),
                f"{prefix}.self_attn.k_proj.weight": (key_value, hidden),
                f"{prefix}.self_attn.v_proj.weight": (key_value, hidden),
                f"{prefix}.self_attn.o_proj.weight": (hidden, query),
                f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
            }
        )
    return shapes


def _device_matches(
    actual: torch.device,
    requested: torch.device,
) -> bool:
    """Treat an unspecified device index as the active device."""
    if actual.type != requested.type:
        return False
    return requested.index is None or actual.index == requested.index
