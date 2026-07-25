# ForgeEngine Agent Rules

Read `AGENTS.md`, `ROADMAP.md`, and `DEFINITION_OF_DONE.md` before every task.

## Fixed scope

- Build a small, readable, single-GPU-first inference engine.
- Support one model only: `Qwen/Qwen3-4B-Instruct-2507`.
- Run on a local or cloud NVIDIA CUDA GPU. Modal is only a test provider.
- Keep the default path portable; L4 is the normal target. H100/B200 are optional labs.
- Do not turn ForgeEngine into a vLLM, SGLang, or TensorRT-LLM clone.

## Required behavior

- Work only on the current roadmap milestone.
- Do not add unrequested features, directories, dependencies, abstractions, or backends.
- Prefer the smallest correct design.
- Keep modules focused, typed, documented, and easy to trace.
- Never use `model.generate` or `pipeline` in the engine path.
- Use Transformers only as the trusted correctness oracle.
- Never fabricate benchmarks, GPU results, compatibility, or completion claims.
- Never call the project production-ready.
- Do not commit or push unless explicitly asked.
- Do not modify `.git`, `.agents`, or `.codex`.

## Correctness and performance

Every custom path must have:

1. a readable PyTorch reference;
2. shape, dtype, and device guards;
3. numerical tests;
4. a real-GPU validation command;
5. synchronized benchmarks where performance is claimed;
6. an automatic fallback;
7. documented limitations.

## Locked low-level scope

- Triton, integrated: fused residual-add + RMSNorm.
- Triton, lab: restricted causal FlashAttention-style prefill.
- CUDA C++, integrated: single-token paged GQA decode attention.
- CuTe DSL, lab: fused gate/up projection plus `SiLU(gate) * up` on H100/B200.
- Rust: small SSE chat client and concurrent load generator.
- PTX/SASS: inspect generated kernels; do not build the engine in assembly.

## Completion report for every task

Return only:

1. changed files;
2. tests run and exact results;
3. exact real-GPU command still required;
4. known limitations or failures;
5. `git status` summary.
