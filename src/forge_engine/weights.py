"""Pinned package download, inspection, and staged SafeTensors loading."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn

from forge_engine.config import DEFAULT_MODEL_ID, SUPPORTED_MODEL_REVISION
from forge_engine.qwen3 import Qwen3Config, Qwen3ForCausalLM

INDEX_NAME = "model.safetensors.index.json"
EXPECTED_TENSOR_COUNT = 398
EXPECTED_TOTAL_SIZE = 8_045_591_552


@dataclass(frozen=True, slots=True)
class PackageInspection:
    """Validated package facts recorded by M2 acceptance."""

    tensor_count: int
    shard_count: int
    total_size: int
    dtype_counts: dict[str, int]


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


def inspect_package(snapshot: Path) -> PackageInspection:
    """Validate index coverage and every SafeTensors header."""
    index = json.loads((snapshot / INDEX_NAME).read_text())
    weight_map: dict[str, str] = index["weight_map"]
    total_size = int(index["metadata"]["total_size"])
    shards = sorted(set(weight_map.values()))
    if len(weight_map) != EXPECTED_TENSOR_COUNT:
        raise ValueError(f"expected {EXPECTED_TENSOR_COUNT} indexed tensors")
    if total_size != EXPECTED_TOTAL_SIZE:
        raise ValueError(f"unexpected package tensor bytes: {total_size}")
    names: set[str] = set()
    dtypes: Counter[str] = Counter()
    for shard_name in shards:
        with safe_open(snapshot / shard_name, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if weight_map.get(name) != shard_name:
                    raise ValueError(f"tensor {name} is assigned to the wrong shard")
                tensor = shard.get_slice(name)
                shape = tensor.get_shape()
                if not shape or any(dimension < 1 for dimension in shape):
                    raise ValueError(f"tensor {name} has invalid shape {shape}")
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
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
