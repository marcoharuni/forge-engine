#!/usr/bin/env bash
set -euo pipefail

uv run ruff format .
uv run ruff check --fix .
cargo fmt --manifest-path rust/Cargo.toml --all

