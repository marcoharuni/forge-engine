#!/usr/bin/env bash
set -euo pipefail

# TODO: Add opt-in GPU and integration suites when implementations exist.
uv run pytest -m "not gpu and not integration"

