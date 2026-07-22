.PHONY: setup format lint test check-python check-rust configure-native check clean

setup:
	uv sync --all-extras --dev

format:
	./scripts/format.sh

lint:
	./scripts/lint.sh

test:
	./scripts/test.sh

check-python:
	uv run python -m compileall -q src tests benchmarks examples scripts
	uv run ruff check .
	uv run mypy src

check-rust:
	cargo fmt --manifest-path rust/Cargo.toml --all -- --check
	cargo check --manifest-path rust/Cargo.toml --workspace
	cargo clippy --manifest-path rust/Cargo.toml --workspace --all-targets -- -D warnings

configure-native:
	cmake -S native -B .tmp/native-build -DFORGE_ENABLE_CUDA=OFF

check: check-python check-rust configure-native test

clean:
	@echo "Remove .tmp, caches, and build outputs manually after confirming their contents."

