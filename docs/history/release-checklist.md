# Historical v0.1.0 development checklist

This document preserves the original development criteria. It is not the
current release status. Current verified status is recorded in
[`docs/validation.md`](../validation.md).

## Run and use

- [x] A clean Linux machine can install it from the README.
- [x] The supported model and revision are pinned.
- [x] Cached weights are reused.
- [x] One documented command starts the service.
- [x] Terminal chat works.
- [x] Browser chat works.
- [x] The streaming API works.
- [x] Multi-turn chat, stop conditions, and cancellation work.
- [ ] The same engine runs locally, in Docker, and on a compatible CUDA cloud.
- [x] No Modal dependency exists in the engine.

## Correct inference

- [x] Chat-template and token IDs match the trusted reference.
- [x] Loaded tensor names, shapes, dtypes, and values are validated.
- [x] Selected hidden states and final logits meet documented tolerances.
- [x] Greedy output tokens agree with Transformers.
- [x] Prefill and decode are explicit.
- [x] The final serving path uses a paged KV cache.
- [x] Sampling and stop strings are tested across token boundaries.

## Concurrent serving

- [x] Several requests can remain active concurrently.
- [x] Continuous batching and token-budget scheduling are used.
- [x] Admission control rejects unsafe work before OOM.
- [x] Cancellation and failures reclaim all KV blocks.
- [x] Queueing, TTFT, ITL, throughput, memory, and request status are observable.

## Low-level work

- [x] Triton residual-add + RMSNorm is integrated and has a fallback.
- [x] Restricted Triton FlashAttention prefill is correct and clearly labeled as a lab.
- [x] CUDA C++ paged GQA decode attention is integrated and has a fallback.
- [ ] CuTe DSL fused gate/up SwiGLU lab runs on prepared H100/B200 hardware.
- [x] Selected PTX/SASS is inspected and documented.
- [x] Rust SSE chat/load tooling works without duplicating the Python engine.
- [ ] Every custom kernel has a PyTorch reference, guards, tests, benchmark, and limitations.

## Evidence and quality

- [x] CPU-safe unit tests pass in CI.
- [x] Real L4 correctness and smoke tests pass.
- [x] Benchmarks record GPU, software versions, precision, model revision, workload, warm-up, iterations, synchronization, latency statistics, throughput, and peak memory.
- [x] H100/B200 results are reported only when actually measured.
- [x] No fabricated numbers or unsupported performance claims exist.
- [x] The repository is small, navigable, typed, documented, and free of dead code.
- [x] The README contains quickstart, architecture, benchmarks, limitations, licenses, and troubleshooting.
- [x] A recruiter can trace request → model → cache → scheduler → sampler → stream without hidden magic.
- [x] No required feature is a stub or unresolved TODO.
- [x] `v0.1.0` can be reproduced from the documented commands.
