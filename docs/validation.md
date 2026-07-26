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
| Docker smoke test | NVIDIA L4 | passed |
| Hosted CPU CI | Ubuntu 24.04 | passed |

The hosted `CPU-safe checks` workflow passed on 2026-07-25 as GitHub Actions
run `30177148761`.

## Maintainer validators

The validators build clean images, mount the repository without local
environments or model files, and reuse the Hugging Face cache volume.

| Validator | Coverage |
| --- | --- |
| `tools/validate_l4.py` | Python/Rust tests, strict kernels, PTX/SASS, model integration, serving, cancellation, metrics, and cleanup |
| `tools/validate_docker_l4.py` | actual release Dockerfile, health, SSE, CUDA, logs, and cleanup |
| `tools/validate_cute_h100.py` | optional CuTe numerical, benchmark, PTX, cubin, and SASS checks |

Run the consolidated validation:

```bash
uv run --extra dev modal run tools/validate_l4.py
```

The successful command prints `L4_ACCEPTANCE=PASS`.

Run the actual release Dockerfile on L4:

```bash
uv run --extra dev modal run tools/validate_docker_l4.py
```

The successful command prints `DOCKER_L4_ACCEPTANCE=PASS`.

## Evidence boundaries

- Measurements apply only to their recorded workload and hardware.
- L4 results are not extrapolated to other GPUs or compared with other engines.
- The restricted Triton prefill and CuTe SwiGLU paths are experiments, not
  general serving replacements.
- The optional CuTe hardware result remains unmeasured; no H100/B200
  performance claim is made.
- Historical release criteria are retained in
  [history/release-checklist.md](history/release-checklist.md).

Detailed numbers and methodology are in [benchmarks.md](benchmarks.md).
Kernel guards, fallbacks, and limitations are in [kernels.md](kernels.md).
