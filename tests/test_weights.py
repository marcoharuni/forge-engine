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
from forge_engine.qwen3 import Qwen3Config
from forge_engine.weights import (
    EXPECTED_CHAT_RENDER,
    EXPECTED_PARAMETER_BYTES,
    EXPECTED_TENSOR_COUNT,
    EXPECTED_TOTAL_SIZE,
    _device_matches,
    _expected_parameter_shapes,
    download_supported_snapshot,
    inspect_package,
    load_safetensors_into_model,
    load_supported_tokenizer,
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

    def test_device_guard_accepts_canonical_cuda_index(self) -> None:
        """A requested default CUDA device may resolve to explicit index zero."""
        self.assertTrue(
            _device_matches(
                torch.device("cuda:0"),
                torch.device("cuda"),
            )
        )
        self.assertTrue(
            _device_matches(
                torch.device("cuda:1"),
                torch.device("cuda:1"),
            )
        )
        self.assertFalse(
            _device_matches(
                torch.device("cuda:1"),
                torch.device("cuda:0"),
            )
        )
        self.assertFalse(
            _device_matches(
                torch.device("cpu"),
                torch.device("cuda"),
            )
        )

    def test_inspects_all_indexed_bfloat16_headers(self) -> None:
        """Every indexed tensor is found and its header dtype is checked."""
        with TemporaryDirectory() as directory:
            snapshot = Path(directory)
            shard_names = (
                "model-00001-of-00003.safetensors",
                "model-00002-of-00003.safetensors",
                "model-00003-of-00003.safetensors",
            )
            tensors = {
                f"tensor.{index}": torch.tensor([index], dtype=torch.bfloat16)
                for index in range(EXPECTED_TENSOR_COUNT)
            }
            weight_map = {
                name: shard_names[index % len(shard_names)]
                for index, name in enumerate(tensors)
            }
            for shard_name in shard_names:
                save_file(
                    {
                        name: tensor
                        for name, tensor in tensors.items()
                        if weight_map[name] == shard_name
                    },
                    snapshot / shard_name,
                )
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": EXPECTED_TOTAL_SIZE},
                        "weight_map": weight_map,
                    }
                )
            )

            with (
                patch(
                    "forge_engine.weights.Qwen3Config.from_file",
                    return_value=object(),
                ),
                patch(
                    "forge_engine.weights._expected_parameter_shapes",
                    return_value={name: (1,) for name in tensors},
                ),
            ):
                inspection = inspect_package(snapshot)

        self.assertEqual(inspection.tensor_count, EXPECTED_TENSOR_COUNT)
        self.assertEqual(inspection.shard_count, 3)
        self.assertEqual(inspection.tensor_bytes, EXPECTED_TENSOR_COUNT * 2)
        self.assertEqual(inspection.dtype_counts, {"BF16": 398})

    def test_parameter_schema_matches_pinned_bfloat16_storage(self) -> None:
        """The complete expected schema has the measured parameter byte size."""
        config = Qwen3Config(
            vocab_size=151936,
            hidden_size=2560,
            intermediate_size=9728,
            num_hidden_layers=36,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=262144,
            rms_norm_eps=1e-6,
            rope_theta=5_000_000.0,
            eos_token_id=151645,
            tie_word_embeddings=True,
            attention_bias=False,
        )
        shapes = _expected_parameter_shapes(config)
        storage = sum(
            torch.Size(shape).numel() * torch.bfloat16.itemsize
            for shape in shapes.values()
        )

        self.assertEqual(len(shapes), EXPECTED_TENSOR_COUNT)
        self.assertEqual(storage, EXPECTED_PARAMETER_BYTES)

    def test_loads_and_validates_pinned_tokenizer_snapshot(self) -> None:
        """Tokenizer metadata, special IDs, and chat rendering are locked."""

        class FakeTokenizer:
            """Small observable stand-in for the pinned tokenizer."""

            pad_token_id = 151643
            eos_token_id = 151645

            def __len__(self) -> int:
                return 151669

            def convert_tokens_to_ids(self, token: str) -> int:
                self.assert_token = token
                return 151644

            def apply_chat_template(
                self,
                messages: list[dict[str, str]],
                *,
                tokenize: bool,
                add_generation_prompt: bool,
            ) -> str:
                self.template_call = (
                    messages,
                    tokenize,
                    add_generation_prompt,
                )
                return EXPECTED_CHAT_RENDER

        with TemporaryDirectory() as directory:
            snapshot = Path(directory)
            (snapshot / "tokenizer.json").write_text("{}")
            (snapshot / "tokenizer_config.json").write_text(
                json.dumps(
                    {
                        "tokenizer_class": "Qwen2Tokenizer",
                        "eos_token": "<|im_end|>",
                        "pad_token": "<|endoftext|>",
                        "add_bos_token": False,
                        "clean_up_tokenization_spaces": False,
                        "chat_template": "pinned template",
                    }
                )
            )
            fake = FakeTokenizer()
            with patch(
                "forge_engine.weights.AutoTokenizer.from_pretrained",
                return_value=fake,
            ) as loader:
                tokenizer, inspection = load_supported_tokenizer(snapshot)

        self.assertIs(tokenizer, fake)
        self.assertEqual(inspection.vocabulary_size, 151669)
        self.assertEqual(inspection.eos_token_id, 151645)
        self.assertEqual(fake.assert_token, "<|im_start|>")
        self.assertEqual(
            fake.template_call,
            (
                [{"role": "user", "content": "ping"}],
                False,
                True,
            ),
        )
        loader.assert_called_once_with(snapshot, local_files_only=True)

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

    def test_staged_copy_rejects_wrong_target_dtype(self) -> None:
        """The allocated target must already use the requested dtype."""
        model = nn.Linear(2, 2, bias=False, dtype=torch.float32)
        with TemporaryDirectory() as directory:
            snapshot = Path(directory)
            save_file(
                {"weight": torch.ones((2, 2), dtype=torch.bfloat16)},
                snapshot / "model.safetensors",
            )
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 8},
                        "weight_map": {"weight": "model.safetensors"},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "target dtype"):
                load_safetensors_into_model(
                    model,
                    snapshot,
                    device=torch.device("cpu"),
                    dtype=torch.bfloat16,
                )
