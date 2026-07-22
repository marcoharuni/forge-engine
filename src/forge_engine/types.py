"""Common public type declarations for ForgeEngine interfaces."""

from typing import NewType

RequestId = NewType("RequestId", str)
SequenceId = NewType("SequenceId", str)
WorkerId = NewType("WorkerId", str)

# TODO: Add tensor-independent protocol and lifecycle types as interfaces stabilize.
