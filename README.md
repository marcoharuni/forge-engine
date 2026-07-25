# ForgeEngine

ForgeEngine is a small, readable, single-GPU inference engine for
`Qwen/Qwen3-4B-Instruct-2507`.

Milestones 1 through 8 have accepted real-L4 evidence. The engine has a trusted single-user
baseline, a verified minimal Qwen3 runner, and staged SafeTensors loading. It
uses the official chat template and explicit prefill and single-token decode
paths. It does not use `model.generate` or a pipeline.

The engine is locked to:

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Hugging Face revision: `cdbee75f17c01a7cc42f958dc650907174af0554`

## Quickstart on a Linux NVIDIA host

Use Linux, Python 3.11, an NVIDIA GPU with a compatible driver, and
enough GPU memory for the 4B BF16 model and KV cache. NVIDIA L4 24 GB is the
validated default. The first run downloads the pinned model snapshot; later
runs reuse `HF_HOME`.

```bash
nvidia-smi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python -m pip install --no-deps .
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
forge-engine serve --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/`, or stream from another terminal with the curl
request below. Use `forge-engine chat` for a multi-turn terminal session.

The integrated CUDA C++ kernel is JIT compiled and needs the CUDA development
toolkit, `g++`, and Ninja. If they are missing or the guarded kernel does not
support the input, the engine automatically uses its PyTorch reference.

## Reproducible development environment

Install [`uv`](https://docs.astral.sh/uv/) and use the committed lock file:

```bash
uv sync --frozen --extra dev
uv run --frozen --extra dev python -m pytest -q
```

The CPU-safe tests use small fakes and never download model weights. The lock
file intentionally selects CPU PyTorch for development and CI; the
`requirements-gpu.txt` path is the runtime GPU environment validated by the
Modal commands.

## Docker

Docker inference requires an NVIDIA host, a compatible NVIDIA driver, and the
NVIDIA Container Toolkit. Build once, then preserve the Hugging Face cache in
a named volume:

```bash
docker build --pull -t forge-engine:0.1.0 .
docker run --rm --gpus all --entrypoint nvidia-smi forge-engine:0.1.0
docker run --rm --gpus all \
  -p 8000:8000 \
  -v forge-hf-cache:/cache \
  forge-engine:0.1.0
```

In another terminal:

```bash
curl --fail http://127.0.0.1:8000/health
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "messages": [{"role": "user", "content": "Count from 1 to 8."}],
    "stream": true,
    "max_tokens": 32
  }'
```

The image uses the same package, CLI, model revision, scheduler, cache, and
fallback behavior as the local path. It is based on a CUDA development image
so the integrated CUDA extension can compile at runtime.

The actual Dockerfile was built and smoke-tested on an NVIDIA L4. Its health,
pinned revision, streaming completion, deterministic `1 2 3 4 ` output,
error-free service log, and final KV cleanup passed:

```bash
uv run --extra dev modal run tools/modal_docker_validate_m9.py
```

The accepted run printed `M9_DOCKER_ACCEPTANCE=PASS`.

## Compatible CUDA cloud

The local commands are vendor-neutral. On any Linux cloud instance with one
compatible NVIDIA GPU, install the driver and either follow the local
quickstart or the Docker commands. Modal is used only by the repository's
validation scripts; it is not imported by the engine package.

## Architecture

[`ARCHITECTURE.md`](ARCHITECTURE.md) contains the component diagram, ownership
boundaries, failure cleanup, and a direct trace through request → chat template
→ scheduler → model → paged KV cache → sampler → detokenizer → stream.

## Benchmarks and release evidence

[`RELEASE_EVIDENCE.md`](RELEASE_EVIDENCE.md) records only measurements that
actually ran, their software and hardware metadata, exact workload, warm-up,
iterations, synchronization, latency, throughput, and memory boundaries.
Kernel-specific evidence and restrictions are in
[`M7_KERNEL_NOTES.md`](M7_KERNEL_NOTES.md).

## Milestone 1 validation

Milestone 1 was validated on an NVIDIA L4:

- `16` tests passed.
- The terminal streamed a successful two-turn conversation.
- The first turn stored `COBALT-731`; the second turn returned `COBALT-731`.
- The validated model revision was
  `cdbee75f17c01a7cc42f958dc650907174af0554`.

Exact validation command:

```bash
uv run --extra dev modal run tools/modal_l4_validate.py
```

ForgeEngine prefers BF16 on supported GPUs and otherwise uses FP16.

## Milestone 2 validation

Milestone 2 was validated on an NVIDIA L4 with PyTorch `2.13.0+cu130` and
Transformers `5.14.1`:

- `28` tests passed.
- All `398` indexed tensors across `3` shards were present with their expected
  names, shapes, and BF16 dtype.
- Tokenizer size and special-token IDs matched the pinned package.
- Decoder layers `0`, `17`, and `35`, the final hidden state, and prompt logits
  had zero measured absolute difference from the Transformers eager oracle.
- Eight cached greedy tokens agreed exactly: `1 2 3 4 `.
- The validated model revision was
  `cdbee75f17c01a7cc42f958dc650907174af0554`.

The enforced tolerances remain `atol=0.02`, `rtol=0.02` for hidden states and
`atol=0.125`, `rtol=0.02` for logits.

Exact validation command:

```bash
uv run --extra dev modal run tools/modal_l4_validate_m2.py
```

## Milestone 3 validation

Milestone 3 was validated on an NVIDIA L4 with PyTorch `2.13.0+cu130` and
Transformers `5.14.1`:

- `46` tests and `11` parameter-validation subtests passed.
- separate prefill, cached single-token decode, and uncached reference paths;
- a validated contiguous key-value cache correctness baseline;
- greedy, temperature, top-k, top-p, and min-p sampling;
- EOS, maximum-token, stop-string, and incremental-text stopping;
- eight cached and uncached greedy tokens agreed exactly;
- maximum probability total variation was below `1.4e-14`;
- CUDA sampling checks passed;
- the first chat turn returned `STORED`, and the second returned `COBALT-731`.

Sampling can be configured from terminal chat, for example:

```bash
forge-engine chat \
  --max-new-tokens 128 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.95 \
  --min-p 0.05 \
  --seed 731
```

Exact validation command:

```bash
uv run --extra dev modal run tools/modal_l4_validate_m3.py
```

## Milestone 4 validation

Milestone 4 was validated on an NVIDIA L4 with PyTorch `2.13.0+cu130`:

- `55` tests and `11` parameter-validation subtests passed;
- exact CUDA paging, gathering, fragmented reuse, and capacity checks passed;
- the fragmented block table reused physical block `0` as `(2, 3, 0)`;
- capacity rejection was transactional;
- eight real-model paged and uncached greedy tokens agreed exactly;
- maximum probability total variation was below `1.4e-14`;
- all model blocks were reclaimed after decode.

The serving generation core stores KV state in lazy, fixed-size physical
blocks. Each sequence owns a logical block table, freed holes are reusable
without moving live sequences, capacity failures are transactional, and the
engine releases all blocks on normal finish, error, or stream cancellation.

M4 uses a readable correctness path: it gathers paged blocks into contiguous
layer tensors before calling PyTorch attention, then appends only the new KV
position to paged storage. Direct paged CUDA decode attention is intentionally
deferred to M7.

Exact validation command:

```bash
uv run --extra dev modal run tools/modal_l4_validate_m4.py
```

## Milestone 5 validation

The concurrent engine provides explicit waiting, running, finished, cancelled,
and failed request states. Admission reserves each request's worst-case KV
blocks before prefill and enforces request, batch, token, and block limits.
Every scheduler iteration selects a fresh round-robin work batch, allowing new
requests to join active decoding. Finish, failure, and cancellation release
physical blocks and admission reservations exactly once.

Within an iteration, requests with matching prompt or cache lengths share one
real tensor-batched model call. Incompatible shapes remain separate reference
sub-batches. This keeps batching correct over M4 paging without padding or
position-ID shortcuts, while later optimized kernels remain separate work.

The L4 validator completes two concurrent deterministic requests, admits a
third while they are running, cancels it with immediate block reclamation, and
verifies that worst-case reservation rejects overload before any model forward
or physical KV allocation. Independent phases report all failures together.

Milestone 5 was validated on an NVIDIA L4 with PyTorch `2.13.0+cu130`:

- `65` tests and `16` parameter-validation subtests passed;
- `alpha` and `beta` both streamed the exact expected `1 2 3 4 `;
- the first real model batch contained `alpha` and `beta` together;
- `gamma` joined the continuous work batch and was then cancelled;
- cancellation reduced allocated blocks from `6` to `4`;
- overload was rejected with `0` physical blocks allocated before work; and
- all reservations and physical blocks were reclaimed.

Exact validation command:

```bash
uv run --extra dev modal run tools/modal_l4_validate_m5.py
```

The accepted run printed `M5_ACCEPTANCE=PASS`.

## Milestone 6 service

Start the single-GPU service with one command:

```bash
forge-engine serve --host 0.0.0.0 --port 8000
```

Then open `http://127.0.0.1:8000/` for browser chat. Health is available at
`/health`, and dependency-free Prometheus text metrics are available at
`/metrics`. Metrics expose waiting/running requests, scheduler iterations,
admission and terminal counters, generated tokens, TTFT, inter-text latency,
request duration, paged KV block ownership, and CUDA allocator memory.

The supported OpenAI-compatible transport is streaming chat completions:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "max_tokens": 128
  }'
```

The service accepts only the pinned model and `stream: true`. It intentionally
has no authentication or TLS termination; bind to loopback for local use or
place it behind an authenticated reverse proxy.

Milestone 6 was validated on an NVIDIA L4 with PyTorch `2.13.0+cu130`:

- `72` tests and `16` parameter-validation subtests passed;
- one `forge-engine serve` command started the real service;
- `/health`, the browser chat page, and `/metrics` responded successfully;
- three SSE clients remained active concurrently, with a measured maximum of
  `3` active requests;
- all three clients returned identical text through OpenAI-compatible chunks
  and finished with the `length` reason;
- the final scheduler state had zero waiting/running requests, reservations,
  and allocated KV blocks; and
- request, TTFT, inter-text latency, duration, token, KV-block, and CUDA-memory
  metrics were exposed.

Exact accepted L4 command:

```bash
uv run --extra dev modal run tools/modal_l4_validate_m6.py
```

The accepted run printed `M6_ACCEPTANCE=PASS`. Its reported server exit code
was `-15` because the validator deliberately sends `SIGTERM` after all checks
finish.

## Milestone 7 focused GPU work

M7 adds the integrated Triton residual-add + RMSNorm path and integrated CUDA
C++ single-token paged GQA decode path, both with guarded PyTorch fallbacks. A
restricted causal Triton prefill kernel remains a lab rather than replacing the
general model attention path. The CuTe fused gate/up SwiGLU experiment is
hardware-gated and remains an optional H100/B200 lab.

Required L4 validation:

```bash
uv run --extra dev modal run tools/modal_l4_validate_m7.py
```

Optional CuTe H100 validation:

```bash
uv run --extra dev modal run tools/modal_h100_validate_cute_m7.py
```

The exact restrictions, benchmark protocol, artifact inspection, and current
evidence status are recorded in [M7_KERNEL_NOTES.md](M7_KERNEL_NOTES.md).

The required M7 run passed on an NVIDIA L4 with PyTorch `2.13.0+cu130`:

- `90` tests and `16` parameter-validation subtests passed;
- all three strict L4 kernel paths passed numerical checks;
- strict integrated Qwen decode produced the exact text `1 2 3 4 `;
- maximum probability total variation was below `5.3e-14`; and
- generated PTX and SASS were retained and inspected for every L4 kernel.

The accepted run printed `M7_ACCEPTANCE=PASS`. The CuTe kernel remains an
optional, unmeasured H100/B200 lab; only its L4 fallback and hardware guard were
validated in the required run.

## Milestone 8 Rust client and load test

The small asynchronous Rust client lives in `rust/streamer`. It contains no
inference code. Its `chat` command streams one OpenAI-compatible SSE response;
its `load` command runs bounded concurrent requests, deliberately disconnects
selected streams, collects client TTFT/ITL/duration statistics, scrapes server
metrics before and after the workload, and emits one JSON report.

Build and test it locally:

```bash
cargo test --locked --manifest-path rust/streamer/Cargo.toml
cargo build --release --locked --manifest-path rust/streamer/Cargo.toml
```

Usage and metric-boundary details are documented in
[`rust/streamer/README.md`](rust/streamer/README.md).

Milestone 8 was validated on an NVIDIA L4 with PyTorch `2.13.0+cu130`:

- `91` Python tests, `16` parameter-validation subtests, and `3` Rust tests
  passed;
- Rust chat streamed the expected `1 2 3 4`;
- six load requests produced `4` finished, `2` cancelled, and `0` failed on
  both the client report and server counters;
- cancellation used the server-provided `X-Forge-Request-ID` and explicit
  `/v1/requests/{request_id}/cancel` endpoint, then consumed the authoritative
  terminal SSE event;
- client wall time was `7.098964s`, mean TTFT was `0.431646s`, mean ITL was
  `0.111612s`, p50 duration was `2.382233s`, and p95 duration was `5.009042s`;
- server mean TTFT was `0.169494s`, mean ITL was `0.170646s`, and mean request
  duration was `2.871565s`;
- the client/server mean-duration difference was `0.234794s`, within the
  declared `1.435782s` boundary tolerance; and
- final waiting/running requests, reservations, and allocated KV blocks were
  all zero.

The scheduler can record text timing before an explicit cancellation request
is processed. The load tool permits at most one such TTFT-count difference per
cancelled request, while terminal statuses come from acknowledged cancellation
and the terminal SSE event rather than being inferred from a dropped stream.

Exact accepted L4 command:

```bash
uv run --extra dev modal run tools/modal_l4_validate_m8.py
```

The accepted run printed `M8_ACCEPTANCE=PASS`.

## Milestone 9 release evidence

The consolidated M9 validator passed on an NVIDIA L4 with compute capability
8.9, driver `580.95.05`, BF16, PyTorch `2.13.0+cu130`, CUDA `13.0`, and
Transformers `5.14.1`:

- `91` Python tests, `16` parameter-validation subtests, and `3` Rust tests
  passed in the clean validation image;
- eight requests at concurrency four produced `6` finished, `2` cancelled,
  and `0` failed, with exact client/server status agreement;
- all completed deterministic outputs began with `1 2 3 4`;
- maximum observed active requests was `4`;
- `199` generated tokens over `10.306798 s` measured
  `19.307646 generated tokens/s` for the declared workload;
- peak process CUDA allocation was `8,879,752,192` bytes; and
- final requests, reservations, and allocated KV blocks were zero.

Latency distributions, the synchronization boundary, complete workload
metadata, earlier kernel evidence, and outstanding external checks are in
[`RELEASE_EVIDENCE.md`](RELEASE_EVIDENCE.md).

Exact accepted command:

```bash
uv run --extra dev modal run tools/modal_l4_validate_m9.py
```

The accepted run printed `M9_ACCEPTANCE=PASS`.

## Limitations

- Only `Qwen/Qwen3-4B-Instruct-2507` at the pinned revision is supported.
- Execution is one process on one NVIDIA CUDA GPU. There is no tensor/pipeline
  parallelism, distributed serving, CPU inference, or non-NVIDIA backend.
- The final path uses BF16 when supported and FP16 otherwise. Quantization,
  speculative decoding, prefix caching, and multimodal inputs are out of
  scope.
- The API implements the documented streaming chat subset, not the complete
  OpenAI API. It has no built-in authentication, TLS, persistence, quotas, or
  multi-tenant isolation.
- The readable paging path gathers layer KV where a guarded direct CUDA decode
  path is unavailable. The restricted Triton prefill and CuTe SwiGLU are labs,
  not general integrated attention/MLP replacements.
- Benchmarks describe their exact workloads and hardware only. They are not
  production capacity guarantees or comparisons with vLLM, SGLang, or
  TensorRT-LLM.
- Model download requires Hugging Face access on first use. The cache volume or
  `HF_HOME` must have enough disk space.

## Licenses

ForgeEngine source and the Rust client are licensed under the
[Apache License 2.0](LICENSE). The pinned Qwen model is also published under
Apache 2.0; its snapshot contains the model's own license. Installed Python,
Rust, CUDA, Triton, and optional CuTe dependencies retain their respective
licenses. Review those dependency licenses before redistribution.

## Troubleshooting

`torch.cuda.is_available()` is false:

- Confirm `nvidia-smi` works on the host.
- Install the CUDA 13.0 PyTorch wheel through `requirements-gpu.txt`, not the
  CPU environment selected by `uv.lock`.
- In Docker, install/configure NVIDIA Container Toolkit and include
  `--gpus all`.

CUDA extension compilation fails:

- Install a CUDA development toolkit compatible with the driver, `g++`, and
  Ninja; confirm `nvcc --version`.
- Check free disk space in the PyTorch extension cache.
- Normal serving falls back to PyTorch. Use the strict M7 validator only when
  debugging the custom path.

The process is rejected or runs out of KV capacity:

- Lower `--max-requests`, `--max-new-tokens`, or client `max_tokens`.
- Raise `--block-capacity` only after measuring available GPU memory.
- Inspect `/metrics` for active requests, reservations, allocated blocks, and
  CUDA memory.

The first startup is slow:

- Initial startup downloads the pinned snapshot and loads BF16 weights.
- Preserve `HF_HOME` locally or mount `/cache` in Docker so subsequent starts
  reuse the snapshot.

An HTTP request returns 400, 409, or 429:

- 400 means the request is outside the supported streaming schema.
- 409 means an explicit cancellation targeted an already terminal request.
- 429 is bounded admission rejecting work before unsafe memory use; retry only
  after active work completes or reduce the requested token budget.
