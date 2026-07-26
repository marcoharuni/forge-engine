# ForgeEngine

[![CI](https://github.com/marcoharuni/forge-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/marcoharuni/forge-engine/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
![GPU](https://img.shields.io/badge/validated-NVIDIA%20L4-76B900)

A small, readable, single-GPU inference engine for `Qwen/Qwen3-4B-Instruct-2507`.

ForgeEngine implements the serving path explicitly: staged SafeTensors loading,
Qwen3 forward execution, prefill and token-by-token decode, paged KV caching,
continuous batching, sampling, incremental detokenization, and streamed chat.
It does not call `model.generate` or hide inference behind a pipeline.

## What it implements

| Component | Implementation |
| --- | --- |
| Model | Pinned Qwen3 4B Instruct |
| Weight loading | Staged SafeTensors loading |
| Prefill/decode | Explicit engine paths |
| KV cache | Paged block allocator with deterministic cleanup |
| Scheduling | Continuous, bounded, token-budgeted batching |
| Sampling | Greedy, temperature, top-k, top-p, and min-p |
| Kernels | Triton residual/RMSNorm and CUDA paged GQA |
| API | Streaming OpenAI-compatible chat subset |
| Clients | Terminal, browser, and Rust SSE/load client |

| Status | Paths |
| --- | --- |
| Integrated | Qwen3, paged cache, scheduler, sampling, Triton RMSNorm, CUDA paged GQA, browser, CLI, and SSE |
| Experimental | Restricted Triton prefill and hardware-gated CuTe SwiGLU |

## Architecture

```mermaid
flowchart LR
    A[Chat messages] --> B[Qwen chat template]
    B --> C[Bounded scheduler]
    C --> D[Explicit prefill / decode]
    D <--> E[Paged KV cache]
    D --> F[Sampling + stop rules]
    F --> C
    C --> G[Incremental detokenizer]
    G --> H[Terminal text or SSE]
```

The HTTP event loop never performs CUDA work directly. One scheduler worker
owns model execution and KV-cache mutation, making admission, batching,
cancellation, and cleanup easy to trace. See
[docs/architecture.md](docs/architecture.md).

## Verified status

The release path was validated on one NVIDIA L4 24 GB using BF16, CUDA 13.0,
PyTorch `2.13.0+cu130`, Transformers `5.14.1`, and model revision
`cdbee75f17c01a7cc42f958dc650907174af0554`.

| Measurement | Verified result |
| --- | ---: |
| Python tests | 91 passed |
| Parameter-validation subtests | 16 passed |
| Rust tests | 3 passed |
| End-to-end throughput | 19.31 generated tokens/s |
| Measured concurrency | 4 |
| Peak process CUDA allocation | 8.88 GB |
| Failed requests | 0 |
| Final allocated KV blocks | 0 |

The workload used eight measured requests: six finished and two were
deliberately cancelled. These numbers describe that exact workload, not general
capacity or a comparison with another engine. Full methodology and latency
distributions are in [docs/benchmarks.md](docs/benchmarks.md).

## Quickstart

Requirements: Linux, `uv`, one compatible NVIDIA GPU, and enough memory for the
4B BF16 model and KV cache. The first run downloads the pinned model; later
runs reuse `HF_HOME`.

```bash
git clone https://github.com/marcoharuni/forge-engine.git
cd forge-engine

uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements-gpu.txt
uv pip install --no-deps .

export HF_HOME="$PWD/.hf-cache"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
forge-engine serve --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` for browser chat.

## Send a streaming request

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "messages": [{"role": "user", "content": "Explain paged KV caching."}],
    "stream": true,
    "max_tokens": 128
  }'
```

Health and Prometheus metrics are available at `/health` and `/metrics`.

## Terminal chat

Run terminal chat instead of the server:

```bash
forge-engine chat --max-new-tokens 128
```

## Docker

Docker requires the NVIDIA Container Toolkit:

```bash
docker build --pull -t forge-engine:0.1.0 .
docker run --rm --gpus all \
  -p 8000:8000 \
  -v forge-hf-cache:/cache \
  forge-engine:0.1.0
```

The actual Dockerfile passed health, CUDA, pinned-model, SSE chat, error-log,
and final KV-cleanup checks on an NVIDIA L4.

## Repository structure

```text
src/forge_engine/  model, cache, scheduler, kernels, server, and CLI
tests/             CPU-safe numerical and behavior tests
rust/streamer/     asynchronous SSE client and load generator
tools/             real-GPU and Docker validation commands
docs/              architecture, benchmarks, kernels, and validation
```

## Supported scope

- Only `Qwen/Qwen3-4B-Instruct-2507` at the pinned revision is supported.
- Execution is one process on one NVIDIA CUDA GPU.
- There is no distributed inference, quantization, speculative decoding, prefix
  caching, multimodal input, or non-NVIDIA backend.
- The streaming API has no built-in authentication, TLS, persistence, quotas,
  or multi-tenant isolation.
- Custom kernels fall back to PyTorch; JIT compilation needs a CUDA development
  toolkit, `g++`, and Ninja.

## Documentation

- [Architecture](docs/architecture.md)
- [Benchmarks and release evidence](docs/benchmarks.md)
- [Kernel implementation and limits](docs/kernels.md)
- [Validation](docs/validation.md)
- [Development](docs/development.md)
- [Rust client](rust/streamer/README.md)
- [Roadmap](ROADMAP.md) and [definition of done](DEFINITION_OF_DONE.md)

## License

ForgeEngine and its Rust client use the [Apache License 2.0](LICENSE). The
pinned Qwen model is also Apache 2.0 and retains its snapshot license.
