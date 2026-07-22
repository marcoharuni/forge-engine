#!/usr/bin/env bash
set -euo pipefail

# TODO: Add platform-specific prerequisite checks when supported environments stabilize.
uv sync --all-extras --dev

