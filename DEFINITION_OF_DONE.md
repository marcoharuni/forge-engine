# ForgeEngine Definition of Done

ForgeEngine is complete only when every required item below is true.

## Run and use

- [ ] A clean Linux machine can install it from the README.
- [ ] The supported model and revision are pinned.
- [ ] Cached weights are reused.
- [ ] One documented command starts the service.
- [ ] Terminal chat works.
- [ ] Browser chat works.
- [ ] The streaming API works.
- [ ] Multi-turn chat, stop conditions, and cancellation work.
- [ ] The same engine runs locally, in Docker, and on a compatible CUDA cloud.
- [ ] No Modal dependency exists in the engine.

## Correct inference

- [ ] Chat-template and token IDs match the trusted reference.
- [ ] Loaded tensor names, shapes, dtypes, and values are validated.
- [ ] Selected hidden states and final logits meet documented tolerances.
- [ ] Greedy output tokens agree with Transformers.
- [ ] Prefill and decode are explicit.
- [ ] The final serving path uses a paged KV cache.
- [ ] Sampling and stop strings are tested across token boundaries.

## Concurrent serving

- [ ] Several requests can remain active concurrently.
- [ ] Continuous batching and token-budget scheduling are used.
- [ ] Admission control rejects unsafe work before OOM.
- [ ] Cancellation and failures reclaim all KV blocks.
- [ ] Queueing, TTFT, ITL, throughput, memory, and request status are observable.

## Low-level work

- [ ] Triton residual-add + RMSNorm is integrated and has a fallback.
- [ ] Restricted Triton FlashAttention prefill is correct and clearly labeled as a lab.
- [ ] CUDA C++ paged GQA decode attention is integrated and has a fallback.
- [ ] CuTe DSL fused gate/up SwiGLU lab runs on prepared H100/B200 hardware.
- [ ] Selected PTX/SASS is inspected and documented.
- [ ] Rust SSE chat/load tooling works without duplicating the Python engine.
- [ ] Every custom kernel has a PyTorch reference, guards, tests, benchmark, and limitations.

## Evidence and quality

- [ ] CPU-safe unit tests pass in CI.
- [ ] Real L4 correctness and smoke tests pass.
- [ ] Benchmarks record GPU, software versions, precision, model revision, workload, warm-up, iterations, synchronization, latency statistics, throughput, and peak memory.
- [ ] H100/B200 results are reported only when actually measured.
- [ ] No fabricated numbers or unsupported performance claims exist.
- [ ] The repository is small, navigable, typed, documented, and free of dead code.
- [ ] The README contains quickstart, architecture, benchmarks, limitations, licenses, and troubleshooting.
- [ ] A recruiter can trace request → model → cache → scheduler → sampler → stream without hidden magic.
- [ ] No required feature is a stub or unresolved TODO.
- [ ] `v0.1.0` can be reproduced from the documented commands.
