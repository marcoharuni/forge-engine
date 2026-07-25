# Development

## CPU-safe environment

The committed lock file intentionally uses CPU PyTorch for development and CI.
It does not download model weights.

```bash
uv sync --frozen --extra dev
uv run --frozen --extra dev python -m pytest -q
```

Run the Rust client checks:

```bash
cargo fmt --check --manifest-path rust/streamer/Cargo.toml
cargo test --locked --manifest-path rust/streamer/Cargo.toml
cargo clippy --locked --manifest-path rust/streamer/Cargo.toml -- -D warnings
```

## GPU environment

The release GPU environment is pinned separately because `uv.lock` is
CPU-safe:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python -m pip install --no-deps .
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

The integrated CUDA extension is JIT-compiled. Install a compatible CUDA
development toolkit, `g++`, and Ninja to exercise it. Normal serving
automatically uses the PyTorch reference when a custom kernel is unavailable or
its guards reject the input.

## Test organization

- `tests/test_model.py`, `test_qwen3.py`, and `test_weights.py` cover the pinned
  architecture and loading.
- `tests/test_generation_core.py`, `test_sampling.py`, and `test_cache.py`
  cover explicit generation behavior.
- `tests/test_scheduler.py` and `test_server.py` cover concurrent serving,
  cancellation, SSE, health, and metrics.
- `tests/test_kernels.py`, `test_cuda_paged.py`, and `test_cute_swiglu.py`
  cover reference paths, guards, and fallbacks.
- `tools/modal_l4_validate*.py` provide real-L4 acceptance evidence.

## Documentation

- [architecture.md](architecture.md): request flow and ownership.
- [benchmarks.md](benchmarks.md): measured L4 results and methodology.
- [kernels.md](kernels.md): low-level implementation and limitations.
- [validation.md](validation.md): acceptance commands and evidence boundaries.
- [`rust/streamer/README.md`](../rust/streamer/README.md): Rust client usage.
