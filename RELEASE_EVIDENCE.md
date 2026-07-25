# ForgeEngine v0.1.0 release evidence

This file separates measurements that actually ran from commands that remain
to be run. Results are not copied from another engine or extrapolated to other
hardware.

## Accepted L4 correctness and kernel evidence

The M7 validator ran on one NVIDIA L4 (compute capability 8.9) using BF16,
PyTorch `2.13.0+cu130`, CUDA `13.0`, Transformers `5.14.1`, and Triton `3.7.1`.
It used model revision
`cdbee75f17c01a7cc42f958dc650907174af0554`.

- `90` Python tests and `16` parameter-validation subtests passed.
- Strict integrated decode produced exact token IDs
  `[16, 220, 17, 220, 18, 220, 19, 220]`, or `1 2 3 4 `.
- Maximum integrated probability total variation was
  `5.296194459012381e-14`.
- Triton and CUDA numerical checks, fallbacks, and PTX/SASS inspection passed.

The synchronized kernel measurements used 10 warm-ups, 50 measured
iterations, CUDA events, and an explicit synchronization:

| Workload | Custom median | PyTorch median | Custom peak increment |
| --- | ---: | ---: | ---: |
| Triton residual + RMSNorm, `[512, 2560]` BF16 | `0.067680 ms` | `0.110512 ms` | `5,242,880 B` |
| Triton causal prefill lab, `[1, 8, 128, 128]` BF16 | `0.072192 ms` | `0.253472 ms` | `262,144 B` |
| CUDA paged GQA, past 127, Q32/KV8/D128 BF16 | `0.099328 ms` | `0.312320 ms` | `9,216 B` |

The complete latency distributions, throughput units, numerical tolerances,
generated-code observations, and limitations are in
[`M7_KERNEL_NOTES.md`](M7_KERNEL_NOTES.md).

## Accepted L4 serving and client evidence

The M8 validator ran on one NVIDIA L4 with PyTorch `2.13.0+cu130`.

- `91` Python tests, `16` parameter-validation subtests, and `3` Rust tests
  passed.
- Six concurrent load requests produced `4` finished, `2` cancelled, and `0`
  failed in both client and server reports.
- Client wall time was `7.098964 s`; mean TTFT was `0.431646 s`; mean ITL was
  `0.111612 s`; p50 duration was `2.382233 s`; and p95 duration was
  `5.009042 s`.
- Server mean TTFT was `0.169494 s`; mean ITL was `0.170646 s`; and mean
  duration was `2.871565 s`.
- Client/server agreement passed and final requests, reservations, and
  allocated KV blocks were zero.

This is one controlled validation workload, not a capacity claim or comparison
with another engine.

## M9 consolidated serving measurement

The consolidated validator passed on one NVIDIA L4 (compute capability 8.9)
with driver `580.95.05`, BF16, PyTorch `2.13.0+cu130`, CUDA `13.0`,
Transformers `5.14.1`, and the pinned model revision.

- `91` Python tests, `16` parameter-validation subtests, and `3` Rust tests
  passed.
- The warm-up chat returned the deterministic `1 2 3 4`.
- Eight measured requests at concurrency four produced `6` finished, `2`
  cancelled, and `0` failed in both client and server reports.
- Client mean TTFT was `0.396415 s`, mean ITL was `0.122788 s`, p50 duration
  was `4.239049 s`, and p95 duration was `6.272952 s`.
- Server mean TTFT was `0.136788 s`, mean ITL was `0.165396 s`, and mean
  request duration was `4.085908 s`.
- The workload sampled `199` output tokens in `10.306798 s`, or
  `19.307646 generated tokens/s`.
- Maximum observed active requests was `4`; peak CUDA allocated memory for the
  loaded process was `8,879,752,192 B`.
- Client/server agreement passed, metric polling had no failures, and final
  waiting/running requests, reservations, and allocated KV blocks were zero.

This end-to-end measurement uses one warm-up request and eight measured
requests. Each terminal SSE event follows the synchronous scheduler/CUDA step;
metrics and health are fetched after all requests reach a terminal state. It
is a fixed serving workload, not a CUDA-event kernel microbenchmark.

Exact accepted command:

```bash
uv run --extra dev modal run tools/modal_l4_validate_m9.py
```

The accepted run printed `M9_ACCEPTANCE=PASS`. Modal run:
`ap-84H39cJBXlRUR01x3rswR8`.

## Optional prepared hardware lab

Run the CuTe lab only on its prepared H100:

```bash
uv run --extra dev modal run tools/modal_h100_validate_cute_m7.py
```

The image and CuTe dependencies built on 2026-07-26, but the H100 function did
not start. The CuTe result therefore remains unmeasured and no H100 performance
claim is made. Modal run: `ap-rz0k3BMRpj38oJ21mFFM1G`. There is no prepared
B200 experiment in v0.1.0.

## CPU, clean-install, and container evidence

CPU-safe commands:

```bash
uv sync --frozen --extra dev
uv run --frozen --extra dev python -m pytest -q
cargo fmt --check --manifest-path rust/streamer/Cargo.toml
cargo test --locked --manifest-path rust/streamer/Cargo.toml
cargo clippy --locked --manifest-path rust/streamer/Cargo.toml -- -D warnings
```

`.github/workflows/ci.yml` runs those checks on Ubuntu. A workflow result is
evidence only after the repository is pushed and the workflow itself passes.

The release Dockerfile built successfully from
`nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`, installed
`torch 2.13.0+cu130`, Transformers `5.14.1`, and ForgeEngine `0.1.0`, then ran
on one NVIDIA L4. The pinned health identity, streaming SSE completion,
deterministic `1 2 3 4 ` output, and final scheduler/KV cleanup passed. The
validator rejects compilation errors and the accepted service log contained
none. Exact command:

```bash
uv run --extra dev modal run tools/modal_docker_validate_m9.py
```

The accepted run printed `M9_DOCKER_ACCEPTANCE=PASS`. Modal run:
`ap-Pxmw8seSpFwN6mRvSmTyMJ`.

`.github/workflows/ci.yml` is prepared, but hosted CI cannot be marked passed
until the current uncommitted work is committed/pushed and GitHub executes the
workflow.
