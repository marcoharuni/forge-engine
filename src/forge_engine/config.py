"""Engine configuration types."""

from dataclasses import dataclass

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SUPPORTED_MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Configuration for a single-GPU ForgeEngine instance."""

    max_new_tokens: int = 256

    def __post_init__(self) -> None:
        """Validate generation limits."""
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
