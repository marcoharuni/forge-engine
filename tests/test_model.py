"""Tests for CUDA model loading and the pinned correctness oracle."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from forge_engine import weights
from forge_engine.config import EngineConfig
from forge_engine.model import (
    CUDAOutOfMemoryError,
    CUDAUnavailableError,
    load_model,
    load_transformers_oracle,
)


class FakeLoadedModel:
    """Observable model returned by fake loaders."""


def fake_modules(
    *,
    cuda_available: bool,
    bf16_supported: bool,
) -> tuple[ModuleType, ModuleType, FakeLoadedModel, list[object]]:
    """Build fake torch and Transformers modules."""
    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = object()
    fake_torch.float16 = object()
    fake_torch.device = lambda name: name
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: cuda_available,
        is_bf16_supported=lambda: bf16_supported,
        OutOfMemoryError=RuntimeError,
    )

    loaded_model = FakeLoadedModel()
    selected_dtypes: list[object] = []
    loader_calls: list[tuple[str, str, str]] = []

    class AutoTokenizer:
        """Fake tokenizer factory."""

        @classmethod
        def from_pretrained(
            cls,
            model_id: str,
            *,
            revision: str,
        ) -> object:
            """Return a tokenizer marker."""
            loader_calls.append(("tokenizer", model_id, revision))
            return {"model_id": model_id}

    class AutoModelForCausalLM:
        """Fake oracle model factory."""

        @classmethod
        def from_pretrained(
            cls,
            model_id: str,
            *,
            dtype: object,
            revision: str,
            attn_implementation: str,
        ) -> FakeLoadedModel:
            """Record the pinned oracle load."""
            assert attn_implementation == "eager"
            loader_calls.append(("model", model_id, revision))
            selected_dtypes.append(dtype)
            return loaded_model

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = AutoTokenizer
    fake_transformers.AutoModelForCausalLM = AutoModelForCausalLM
    fake_transformers.loader_calls = loader_calls
    return fake_torch, fake_transformers, loaded_model, selected_dtypes


class ModelLoadingTests(TestCase):
    """CUDA loading, dtype selection, and oracle locking."""

    def test_prefers_bfloat16_for_staged_runner(self) -> None:
        """BF16 is selected when CUDA reports support."""
        torch, transformers, model, _ = fake_modules(
            cuda_available=True,
            bf16_supported=True,
        )

        with (
            patch.dict(
                sys.modules,
                {"torch": torch, "transformers": transformers},
            ),
            patch.object(
                weights,
                "download_supported_snapshot",
                return_value="snapshot",
            ),
            patch.object(
                weights,
                "load_staged_model",
                return_value=(model, object()),
            ) as staged,
        ):
            loaded = load_model(EngineConfig())

        self.assertIs(loaded.model, model)
        self.assertIs(staged.call_args.kwargs["dtype"], torch.bfloat16)
        self.assertEqual(
            transformers.loader_calls,
            [
                (
                    "tokenizer",
                    "Qwen/Qwen3-4B-Instruct-2507",
                    "cdbee75f17c01a7cc42f958dc650907174af0554",
                )
            ],
        )

    def test_falls_back_to_float16(self) -> None:
        """FP16 is selected when BF16 is unsupported."""
        torch, transformers, model, _ = fake_modules(
            cuda_available=True,
            bf16_supported=False,
        )

        with (
            patch.dict(
                sys.modules,
                {"torch": torch, "transformers": transformers},
            ),
            patch.object(
                weights,
                "download_supported_snapshot",
                return_value="snapshot",
            ),
            patch.object(
                weights,
                "load_staged_model",
                return_value=(model, object()),
            ) as staged,
        ):
            load_model(EngineConfig())

        self.assertIs(staged.call_args.kwargs["dtype"], torch.float16)

    def test_requires_cuda_before_loading(self) -> None:
        """A clear error is raised without importing the staged runner."""
        torch, transformers, _, _ = fake_modules(
            cuda_available=False,
            bf16_supported=False,
        )

        with (
            patch.dict(
                sys.modules,
                {"torch": torch, "transformers": transformers},
            ),
            self.assertRaisesRegex(CUDAUnavailableError, "CUDA is required"),
        ):
            load_model(EngineConfig())

    def test_reports_cuda_out_of_memory_while_loading(self) -> None:
        """Staged allocation failures become concise user-facing errors."""

        class FakeOOM(RuntimeError):
            """Fake Torch CUDA out-of-memory error."""

        torch, transformers, _, _ = fake_modules(
            cuda_available=True,
            bf16_supported=True,
        )
        torch.cuda.OutOfMemoryError = FakeOOM

        with (
            patch.dict(
                sys.modules,
                {"torch": torch, "transformers": transformers},
            ),
            patch.object(
                weights,
                "download_supported_snapshot",
                return_value="snapshot",
            ),
            patch.object(
                weights,
                "load_staged_model",
                side_effect=FakeOOM,
            ),
            self.assertRaisesRegex(
                CUDAOutOfMemoryError,
                "CUDA out of memory while loading",
            ),
        ):
            load_model(EngineConfig())

    def test_transformers_oracle_loaders_are_pinned(self) -> None:
        """Both oracle loaders receive the locked model and revision."""
        _, transformers, model, dtypes = fake_modules(
            cuda_available=True,
            bf16_supported=True,
        )
        dtype = object()

        with patch.dict(sys.modules, {"transformers": transformers}):
            _, loaded_model = load_transformers_oracle(dtype)

        self.assertIs(loaded_model, model)
        self.assertIs(dtypes[0], dtype)
        self.assertEqual(
            transformers.loader_calls,
            [
                (
                    "tokenizer",
                    "Qwen/Qwen3-4B-Instruct-2507",
                    "cdbee75f17c01a7cc42f958dc650907174af0554",
                ),
                (
                    "model",
                    "Qwen/Qwen3-4B-Instruct-2507",
                    "cdbee75f17c01a7cc42f958dc650907174af0554",
                ),
            ],
        )
