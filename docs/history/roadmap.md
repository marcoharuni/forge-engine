# ForgeEngine Roadmap

Complete milestones in order. Do not begin the next milestone until the current one passes its acceptance checks.

## M1 — Validate trusted baseline

- Load the supported model on a real L4.
- Apply the official chat template.
- Run explicit `forward` decoding with `past_key_values`.
- Stream multi-turn terminal chat.

**Accept when:** unit tests pass and a real L4 conversation succeeds without `model.generate`.

## M2 — Model package and verified runner

- Inspect config, tokenizer, SafeTensors index, names, shapes, and dtypes.
- Add staged loading without unnecessary full-model copies.
- Implement the minimal Qwen3 model path.
- Compare hidden states, logits, and greedy tokens with Transformers.

**Accept when:** documented tolerances and greedy-token agreement pass on real GPU.

## M3 — Explicit generation core

- Separate prefill and decode.
- Add a contiguous KV cache as the correctness baseline.
- Add greedy, temperature, top-k, top-p, and min-p sampling.
- Add EOS, token limit, stop strings, and incremental detokenization.

**Accept when:** cached and uncached paths agree and multi-turn chat is correct.

## M4 — Paged serving state

- Add a small paged KV block pool.
- Add a correct reference paged decode path.
- Reclaim every block on finish, error, or cancellation.

**Accept when:** allocation, reuse, fragmentation, and cleanup tests pass.

## M5 — Concurrent engine

- Add request states.
- Add iteration-level continuous batching.
- Add a simple token-budget scheduler.
- Add bounded admission and cancellation.

**Accept when:** concurrent requests stream correctly and overload is rejected before OOM.

## M6 — Useful product surface

- Add an OpenAI-compatible streaming chat endpoint.
- Add `/health` and `/metrics`.
- Add a clean browser chat UI.
- Preserve terminal chat.

**Accept when:** one command starts the service and several clients can chat concurrently.

## M7 — Focused GPU work

1. Integrate Triton residual-add + RMSNorm with fallback.
2. Add restricted Triton FlashAttention prefill as a lab.
3. Integrate CUDA C++ single-token paged GQA decode attention with fallback.
4. Add CuTe DSL fused gate/up SwiGLU experiment for H100/B200.
5. Inspect selected PTX/SASS and save observations.

**Accept when:** each path has reference tests, real-GPU results, benchmarks, and limits.

## M8 — Rust client and load test

- Add a small asynchronous SSE chat client.
- Add concurrent load generation, cancellation, and latency collection.

**Accept when:** client results agree with server metrics within explained differences.

## M9 — Release evidence

- Add Docker and vendor-neutral local instructions.
- Add CI for CPU-safe tests.
- Add L4 correctness, latency, throughput, memory, and concurrency reports.
- Run only prepared H100/B200 labs.
- Add architecture diagram, limitations, licenses, and reproducibility instructions.
- Remove dead code, fake stubs, and unnecessary abstractions.

**Accept when:** every item in `release-checklist.md` is checked and `v0.1.0` is runnable.
