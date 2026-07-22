"""Exception hierarchy shared by future ForgeEngine components."""


class ForgeEngineError(Exception):
    """Base exception for ForgeEngine failures."""


class ConfigurationError(ForgeEngineError):
    """Raised when ForgeEngine configuration is invalid."""


# TODO: Add narrow error types as engine boundaries are implemented.

