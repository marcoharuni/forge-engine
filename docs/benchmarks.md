# Benchmarks and release evidence

This file separates measurements that actually ran from commands that remain
to be run. Results are not copied from another engine or extrapolated to other
hardware.

## Accepted L4 correctness and kernel evidence

The strict kernel validator ran on one NVIDIA L4 (compute capability 8.9) using BF16,
PyTorch `2.13.0+cu130`, CUDA `13.0`, Transformers `5.14.1`, and Triton `3.7.1`.
It used model revision
`cdbee75f17c01a7cc42f958dc650907174af0554`.

- `93` Python tests, `16` parameter-validation subtests, and `3` Rust tests
  passed.
- Strict integrated decode produced exact token IDs
  `[16, 220, 17, 220, 18, 220, 19, 220]`, or `1 2 3 4 `.
- Maximum integrated probability total variation was
  `5.296194459012381e-14`.
- Triton and CUDA numerical checks, fallbacks, and PTX/SASS inspection passed.

The synchronized kernel measurements used 10 warm-ups, 50 measured
iterations, CUDA events, and an explicit synchronization:

| Workload | Custom median | PyTorch median | Result |
| --- | ---: | ---: | ---: |
| Triton residual + RMSNorm, `[512, 2560]` BF16 | `0.084992 ms` | `0.148992 ms` | **1.75× faster** |
| Restricted Triton prefill lab, `[1, 8, 128, 128]` BF16 | `0.089088 ms` | `0.368640 ms` | **4.14× faster** |
| CUDA paged GQA, past 127, Q32/KV8/D128 BF16 | `0.100352 ms` | `0.462304 ms` | **4.61× faster** |

Custom peak increments were `5,242,880 B`, `262,144 B`, and `9,216 B`
respectively. The machine-readable release result is
[`results/l4-v0.1.0.json`](../results/l4-v0.1.0.json).

The complete latency distributions, throughput units, numerical tolerances,
generated-code observations, and limitations are in
[`kernels.md`](kernels.md).

## Accepted L4 serving and client evidence

The consolidated validator passed on one NVIDIA L4 (compute capability 8.9)
with driver `580.95.05`, BF16, PyTorch `2.13.0+cu130`, CUDA `13.0`,
Transformers `5.14.1`, and the pinned model revision.

- `93` Python tests, `16` parameter-validation subtests, and `3` Rust tests
  passed.
- The warm-up chat returned the deterministic `1 2 3 4`.
- Eight measured requests at concurrency four produced `6` finished, `2`
  cancelled, and `0` failed in both client and server reports.
- Client mean TTFT was `0.527762 s`, mean ITL was `0.153137 s`, p50 duration
  was `5.194081 s`, and p95 duration was `7.736030 s`.
- Server mean TTFT was `0.178531 s`, mean ITL was `0.205523 s`, and mean
  request duration was `5.085837 s`.
- The workload sampled `199` output tokens in `12.739534 s`, or
  `15.620666 generated tokens/s`.
- Maximum observed active requests was `4`; peak CUDA allocated memory for the
  loaded process was `8,879,752,192 B`.
- Client/server agreement passed, metric polling had no failures, and final
  waiting/running requests, reservations, and allocated KV blocks were zero.

This end-to-end measurement uses one warm-up request and eight measured
requests. Each terminal SSE event follows the synchronous scheduler/CUDA step;
metrics and health are fetched after all requests reach a terminal state. It
is a fixed serving workload, not a CUDA-event kernel microbenchmark.

Current reproduction command:

```bash
uv run --extra dev modal run tools/validate_l4.py
```

Accepted Modal run: `ap-ETY99zS1VmyYLnbuFpyGa7`.

## Optional prepared hardware lab

Run the CuTe lab only on its prepared H100:

```bash
uv run --extra dev modal run tools/validate_cute_h100.py
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

The repository's `CPU-safe checks` GitHub Actions workflow ran these checks on
Ubuntu 24.04 and passed on 2026-07-25 (run `30177148761`).

The release Dockerfile built successfully from
`nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04`, installed
`torch 2.13.0+cu130`, Transformers `5.14.1`, and ForgeEngine `0.1.0`, then ran
on one NVIDIA L4. The pinned health identity, streaming SSE completion,
deterministic `1 2 3 4 ` output, and final scheduler/KV cleanup passed. The
validator rejects compilation errors and the accepted service log contained
none. Exact command:

```bash
uv run --extra dev modal run tools/validate_docker_l4.py
```

Accepted Modal run: `ap-lb7G8O0g25V1SNUo3rpEiz`.
