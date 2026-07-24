"""Key-value cache interfaces."""

from typing import Protocol


class KeyValueCache(Protocol):
    """Interface for an inference key-value cache."""

    @property
    def sequence_length(self) -> int:
        """Return the number of cached token positions."""
        ...

    def clear(self) -> None:
        """Discard all cached state."""
        ...
