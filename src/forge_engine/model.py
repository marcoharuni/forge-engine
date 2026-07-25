"""CUDA model loading and its small typed interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypedDict

from forge_engine.config import (
    DEFAULT_MODEL_ID,
    SUPPORTED_MODEL_REVISION,
    EngineConfig,
)

if TYPE_CHECKING:
    from torch import Tensor


class ForgeEngineError(RuntimeError):
    """Base error for user-facing ForgeEngine failures."""


class CUDAUnavailableError(ForgeEngineError):
    """Raised when CUDA is required but unavailable."""


class CUDAOutOfMemoryError(ForgeEngineError):
    """Raised when model loading or inference exhausts CUDA memory."""


class ChatMessage(TypedDict):
    """One message consumed by the tokenizer chat template."""

    role: str
    content: str


class ModelOutput(Protocol):
    """Output required by the greedy decoding loop."""

    logits: Tensor
    past_key_values: object


class Tokenizer(Protocol):
    """Tokenizer operations required by ForgeEngine."""

    eos_token_id: int | list[int] | None

    def apply_chat_template(
        self,
        conversation: list[ChatMessage],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
    ) -> object:
        """Apply the model-provided chat template."""
        ...

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        """Decode generated token IDs."""
        ...


class LanguageModel(Protocol):
    """Causal language model operations required by ForgeEngine."""

    device: object

    def eval(self) -> LanguageModel:
        """Switch to evaluation mode."""
        ...

    def to(self, device: str) -> LanguageModel:
        """Move the model to a device."""
        ...

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        past_key_values: object | None,
        use_cache: bool,
    ) -> ModelOutput:
        """Run one cached causal-language-model step."""
        ...


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Tokenizer and model prepared for CUDA inference."""

    tokenizer: Tokenizer
    model: LanguageModel


def load_transformers_oracle(dtype: object) -> tuple[object, object]:
    """Load the pinned Transformers pair for correctness checks only."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_ID,
        revision=SUPPORTED_MODEL_REVISION,
    )
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_ID,
        dtype=dtype,
        revision=SUPPORTED_MODEL_REVISION,
        attn_implementation="eager",
    )
    return tokenizer, model


def load_model(config: EngineConfig) -> LoadedModel:
    """Load the tokenizer and staged ForgeEngine Qwen3 runner on CUDA."""
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise CUDAUnavailableError(
            "CUDA is required, but no CUDA device is available."
        )

    from forge_engine.weights import (
        download_supported_snapshot,
        load_staged_model,
    )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_MODEL_ID,
            revision=SUPPORTED_MODEL_REVISION,
        )
        snapshot = download_supported_snapshot()
        model, _ = load_staged_model(
            snapshot,
            device=torch.device("cuda"),
            dtype=dtype,
        )
    except torch.cuda.OutOfMemoryError as error:
        raise CUDAOutOfMemoryError(
            "CUDA out of memory while loading the model. "
            "Close other GPU workloads or use a GPU with more memory."
        ) from error

    return LoadedModel(tokenizer=tokenizer, model=model)
