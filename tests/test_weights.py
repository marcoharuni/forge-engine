"""Tests for pinned package inspection and staged tensor loading."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from torch import nn

from forge_engine.config import DEFAULT_MODEL_ID, SUPPORTED_MODEL_REVISION
from forge_engine.weights import (
    EXPECTED_TENSOR_COUNT,
    EXPECTED_TOTAL_SIZE,
    download_supported_snapshot,
    inspect_package,
    load_safetensors_into_model,
)


class WeightsTests(TestCase):
    """Package metadata and one-tensor-at-a-time copy behavior."""

    def test_snapshot_download_is_locked(self) -> None:
        """The package download cannot drift from the supported revision."""
        with patch(
            "forge_engine.weights.snapshot_download",
            return_value="/cache/snapshot",
        ) as download:
            path = download_supported_snapshot()

        self.assertEqual(path, Path("/cache/snapshot"))
        self.assertEqual(download.call_args.kwargs["repo_id"], DEFAULT_MODEL_ID)
        self.assertEqual(
            download.call_args.kwargs["revision"], SUPPORTED_MODEL_REVISION
        )

    def test_inspects_all_indexed_bfloat16_headers(self) -> None:
        """Every indexed tensor is found and its header dtype is checked."""
        with TemporaryDirectory() as directory:
            snapshot = Path(directory)
            shard_name = "model-00001-of-00001.safetensors"
            tensors = {
                f"tensor.{index}": torch.tensor(index, dtype=torch.bfloat16)
                for index in range(EXPECTED_TENSOR_COUNT)
            }
            save_file(tensors, snapshot / shard_name)
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": EXPECTED_TOTAL_SIZE},
                        "weight_map": {
                            name: shard_name for name in tensors
                        },
                    }
                )
            )

            inspection = inspect_package(snapshot)

        self.assertEqual(inspection.tensor_count, EXPECTED_TENSOR_COUNT)
        self.assertEqual(inspection.shard_count, 1)
        self.assertEqual(inspection.dtype_counts, {"BF16": 398})

    def test_staged_copy_checks_names_shapes_dtypes_and_values(self) -> None:
        """A staged tensor is copied exactly into its allocated parameter."""

        class TinyModel(nn.Module):
            """One-parameter load target."""

            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.empty((2, 3), dtype=torch.bfloat16))

        source = torch.arange(6, dtype=torch.bfloat16).view(2, 3)
        with TemporaryDirectory() as directory:
            snapshot = Path(directory)
            shard_name = "model.safetensors"
            save_file({"weight": source}, snapshot / shard_name)
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 12},
                        "weight_map": {"weight": shard_name},
                    }
                )
            )
            model = TinyModel()
            load_safetensors_into_model(
                model,
                snapshot,
                device=torch.device("cpu"),
                dtype=torch.bfloat16,
            )

        torch.testing.assert_close(model.weight, source)
