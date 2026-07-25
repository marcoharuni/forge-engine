"""Import smoke tests for the package skeleton."""

from importlib import import_module
from unittest import TestCase


class ImportTests(TestCase):
    """Import smoke tests for public modules."""

    def test_public_modules_import(self) -> None:
        """All public skeleton modules are importable."""
        modules = (
            "forge_engine",
            "forge_engine.cache",
            "forge_engine.cli",
            "forge_engine.config",
            "forge_engine.engine",
            "forge_engine.kernels",
            "forge_engine.kernels.triton_ops",
            "forge_engine.model",
            "forge_engine.qwen3",
            "forge_engine.sampling",
            "forge_engine.server",
            "forge_engine.weights",
        )

        for module in modules:
            import_module(module)
