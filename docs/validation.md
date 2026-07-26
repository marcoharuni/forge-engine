# Validation

ForgeEngine separates CPU-safe tests, real-GPU correctness checks, kernel
microbenchmarks, and end-to-end serving measurements. Transformers is used only
as the trusted correctness oracle; the engine never calls `model.generate` or
`pipeline`.

This document is maintainer evidence. Users do not need to run these commands
to install ForgeEngine or chat with the model.

## Current verified status

| Area | Hardware | Result |
| --- | --- | --- |
| Python tests | NVIDIA L4 validation image | 91 passed |
| Parameter validation | NVIDIA L4 validation image | 16 subtests passed |
| Rust client tests | NVIDIA L4 validation image | 3 passed |
| Pinned-model correctness | NVIDIA L4, BF16 | passed |
| Concurrent serving | NVIDIA L4, concurrency 4 | 6 finished, 2 cancelled, 0 failed |
| Final cleanup | NVIDIA L4 | 0 active requests and 0 allocated KV blocks |
| Docker smoke test | NVIDIA L4 | `M9_DOCKER_ACCEPTANCE=PASS` |
| Hosted CPU CI | Ubuntu 24.04 | passed |

The hosted `CPU-safe checks` workflow passed on 2026-07-25 as GitHub Actions
run `30177148761`.

## Milestone validators

Each validator builds a clean Modal image, mounts the repository without local
environments or model files, reuses the Hugging Face cache volume, and requests
one NVIDIA L4.

| Validator | Main acceptance result |
| --- | --- |
| `tools/modal_l4_validate.py` | baseline explicit forward and two-turn chat |
| `tools/modal_l4_validate_m2.py` | weights, hidden states, logits, and greedy tokens |
| `tools/modal_l4_validate_m3.py` | cached generation, sampling, stops, and multi-turn chat |
| `tools/modal_l4_validate_m4.py` | paged allocation, reuse, fragmentation, and cleanup |
| `tools/modal_l4_validate_m5.py` | continuous batching, admission, overload, cancellation |
| `tools/modal_l4_validate_m6.py` | browser, health, metrics, SSE, and concurrent clients |
| `tools/modal_l4_validate_m7.py` | L4 kernels, fallbacks, benchmarks, PTX, and SASS |
| `tools/modal_l4_validate_m8.py` | Rust chat/load client and metric agreement |
| `tools/modal_l4_validate_m9.py` | consolidated correctness and serving evidence |

Run the consolidated validation:

```bash
uv run --extra dev modal run tools/modal_l4_validate_m9.py
```

The accepted run printed `M9_ACCEPTANCE=PASS`.

Run the actual release Dockerfile on L4:

```bash
uv run --extra dev modal run tools/modal_docker_validate_m9.py
```

The accepted run printed `M9_DOCKER_ACCEPTANCE=PASS`.

## Evidence boundaries

- Measurements apply only to their recorded workload and hardware.
- L4 results are not extrapolated to other GPUs or compared with other engines.
- The restricted Triton prefill and CuTe SwiGLU paths are experiments, not
  general serving replacements.
- The optional CuTe hardware result remains unmeasured; no H100/B200
  performance claim is made.
- The project is complete only when the required checks in
  [`DEFINITION_OF_DONE.md`](../DEFINITION_OF_DONE.md) are true.

Detailed numbers and methodology are in [benchmarks.md](benchmarks.md).
Kernel guards, fallbacks, and limitations are in [kernels.md](kernels.md).
