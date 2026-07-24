"""Triton kernel interfaces."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from torch import Tensor

TritonKernel: TypeAlias = Callable[["Tensor"], "Tensor"]
