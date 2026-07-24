"""HTTP server interfaces."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

from forge_engine.engine import InferenceEngine

if TYPE_CHECKING:
    from fastapi import FastAPI

AppFactory: TypeAlias = Callable[[InferenceEngine], "FastAPI"]
