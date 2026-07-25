"""Engine configuration types."""

from dataclasses import dataclass

from forge_engine.sampling import SamplingParams

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
SUPPORTED_MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """Configuration for a single-GPU ForgeEngine instance."""

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float = 1.0
    min_p: float = 0.0
    stop_strings: tuple[str, ...] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate generation settings through their runtime value object."""
        self.sampling_params()

    def sampling_params(self) -> SamplingParams:
        """Build immutable sampling parameters used by generation."""
        return SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            min_p=self.min_p,
            max_new_tokens=self.max_new_tokens,
            stop_strings=self.stop_strings,
            seed=self.seed,
        )
