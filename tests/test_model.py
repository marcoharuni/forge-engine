"""Tests for CUDA model loading."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from forge_engine.config import EngineConfig
from forge_engine.model import (
    CUDAOutOfMemoryError,
    CUDAUnavailableError,
    load_model,
)


class FakeLoadedModel:
    """Observable model returned by the fake auto-model loader."""

    def __init__(self) -> None:
        self.device: str | None = None
        self.evaluating = False

    def to(self, device: str) -> FakeLoadedModel:
        """Record the selected device."""
        self.device = device
        return self

    def eval(self) -> FakeLoadedModel:
        """Record evaluation mode."""
        self.evaluating = True
        return self


def fake_modules(
    *,
    cuda_available: bool,
    bf16_supported: bool,
) -> tuple[ModuleType, ModuleType, FakeLoadedModel, list[object]]:
    """Build fake torch and transformers modules."""
    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = object()
    fake_torch.float16 = object()
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
        """Fake model factory."""

        @classmethod
        def from_pretrained(
            cls,
            model_id: str,
            *,
            dtype: object,
            revision: str,
        ) -> FakeLoadedModel:
            """Record dtype selection and return the fake model."""
            loader_calls.append(("model", model_id, revision))
            selected_dtypes.append(dtype)
            return loaded_model

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = AutoTokenizer
    fake_transformers.AutoModelForCausalLM = AutoModelForCausalLM
    fake_transformers.loader_calls = loader_calls
    return fake_torch, fake_transformers, loaded_model, selected_dtypes


class ModelLoadingTests(TestCase):
    """CUDA loading and dtype selection."""

    def test_prefers_bfloat16_and_uses_eval_mode(self) -> None:
        """BF16 is selected when CUDA reports support."""
        torch, transformers, model, dtypes = fake_modules(
            cuda_available=True,
            bf16_supported=True,
        )

        with patch.dict(
            sys.modules,
            {"torch": torch, "transformers": transformers},
        ):
            loaded = load_model(EngineConfig())

        self.assertIs(loaded.model, model)
        self.assertIs(dtypes[0], torch.bfloat16)
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
        self.assertEqual(model.device, "cuda")
        self.assertTrue(model.evaluating)

    def test_falls_back_to_float16(self) -> None:
        """FP16 is selected when BF16 is unsupported."""
        torch, transformers, _, dtypes = fake_modules(
            cuda_available=True,
            bf16_supported=False,
        )

        with patch.dict(
            sys.modules,
            {"torch": torch, "transformers": transformers},
        ):
            load_model(EngineConfig())

        self.assertIs(dtypes[0], torch.float16)

    def test_requires_cuda_before_loading(self) -> None:
        """A clear error is raised without CUDA."""
        torch, transformers, _, _ = fake_modules(
            cuda_available=False,
            bf16_supported=False,
        )

        with patch.dict(
            sys.modules,
            {"torch": torch, "transformers": transformers},
        ):
            with self.assertRaisesRegex(
                CUDAUnavailableError,
                "CUDA is required",
            ):
                load_model(EngineConfig())

    def test_reports_cuda_out_of_memory_while_loading(self) -> None:
        """Model allocation failures become concise user-facing errors."""

        class FakeOOM(RuntimeError):
            """Fake Torch CUDA out-of-memory error."""

        torch, transformers, _, _ = fake_modules(
            cuda_available=True,
            bf16_supported=True,
        )
        torch.cuda.OutOfMemoryError = FakeOOM

        class OutOfMemoryAutoModel:
            """Fake auto-model loader that exhausts CUDA memory."""

            @classmethod
            def from_pretrained(
                cls,
                model_id: str,
                *,
                dtype: object,
                revision: str,
            ) -> object:
                """Raise the fake CUDA allocation failure."""
                raise FakeOOM

        transformers.AutoModelForCausalLM = OutOfMemoryAutoModel

        with (
            patch.dict(
                sys.modules,
                {"torch": torch, "transformers": transformers},
            ),
            self.assertRaisesRegex(
                CUDAOutOfMemoryError,
                "CUDA out of memory while loading",
            ),
        ):
            load_model(EngineConfig())
