# Kernel implementation and evidence

ForgeEngine keeps three low-level paths separate: one small Triton fusion is integrated,
one Triton attention kernel is a restricted lab, and one CUDA C++ paged decode
kernel is integrated. The CuTe DSL experiment remains an optional H100/B200
lab. Every path has a readable PyTorch reference and rejects unsupported
metadata in strict mode; the normal engine path falls back automatically.

## Implemented paths

| Path | Scope | Static code observation |
| --- | --- | --- |
| Triton residual-add + RMSNorm | Integrated | One program owns one flattened row, adds in FP32, reduces the squared sum, and writes both the residual result and normalized output. |
| Triton causal prefill | Lab only | One program owns one query/head, visits keys in 32-position tiles, applies a causal mask, and maintains online-softmax state without materializing the complete score matrix. |
| CUDA C++ paged GQA decode | Integrated single-token path | One 128-thread block owns one query head, maps it to a KV head, follows logical page pointers directly, and maintains online-softmax state. |
| CuTe gate/up SwiGLU | Optional H100/B200 lab | One thread computes one output element and fuses both dot products with the SiLU product. It is deliberately readable and does not claim tensor-core performance. |

The L4 validator retains generated Triton cubins, extracts PTX and SASS from
the Triton and CUDA C++ binaries, checks the expected entry points, and reports
instruction, global-load/store, and barrier counts.

## Accepted L4 evidence

The required command passed on an NVIDIA L4 with compute capability 8.9,
PyTorch `2.13.0+cu130`, CUDA `13.0`, Transformers `5.14.1`, Triton `3.7.1`,
and BF16. The pinned model revision was
`cdbee75f17c01a7cc42f958dc650907174af0554`.

- `90` tests and `16` parameter-validation subtests passed.
- Triton residual-add + RMSNorm, restricted causal prefill, and CUDA paged GQA
  decode passed their strict custom-kernel numerical comparisons.
- Eight strict integrated decode tokens were exactly
  `[16, 220, 17, 220, 18, 220, 19, 220]`, decoding to `1 2 3 4 `.
- Maximum integrated probability total variation was
  `5.296194459012381e-14`.
- The CuTe fallback and hardware guard passed on L4; the optional CuTe kernel
  itself was correctly reported as `SKIP_L4_HARDWARE`.

Synchronized benchmark medians:

| Workload | Custom median | PyTorch median | Custom peak increment |
| --- | ---: | ---: | ---: |
| Triton residual + RMSNorm, `[512, 2560]` BF16 | `0.067680 ms` | `0.110512 ms` | `5,242,880` bytes |
| Triton causal prefill, `[1, 8, 128, 128]` BF16 | `0.072192 ms` | `0.253472 ms` | `262,144` bytes |
| CUDA paged GQA, past 127, Q32/KV8/D128 BF16 | `0.099328 ms` | `0.312320 ms` | `9,216` bytes |

Every result used 10 warm-up iterations, 50 measured iterations, CUDA events,
and an explicit synchronization. Maximum absolute error was `0.001953125` for
the prefill lab and `0.000244140625` for paged GQA decode.

Generated-code observations:

| Kernel | PTX entries | PTX loads/stores | PTX barriers | SASS instructions | SASS loads/stores | SASS barriers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Triton residual + RMSNorm | 1 | 12 / 8 | 2 | 312 | 9 / 6 | 2 |
| Triton restricted prefill | 1 | 9 / 1 | 12 | 530 | 9 / 1 | 12 |
| CUDA paged GQA decode | 4 | 16 / 4 | 44 | 1,104 | 16 / 4 | 44 |

## Reproducible validation

Run the required L4 acceptance:

```bash
uv run --extra dev modal run tools/validate_l4.py
```

The command runs the complete CPU-safe suite, strict BF16 numerical comparisons,
strict real-model integration, synchronized CUDA-event benchmarks, automatic
fallback checks, serving evidence, and PTX/SASS inspection. A successful run
prints `L4_ACCEPTANCE=PASS`.

Run the optional CuTe hardware lab on an H100:

```bash
uv run --extra dev modal run tools/validate_cute_h100.py
```

That command installs the optional CuTe dependency only in its dedicated image,
uses the exact Qwen hidden and intermediate dimensions, benchmarks the fused
lab against PyTorch, and retains its PTX, cubin, and SASS. An L4 run records the
CuTe path as `SKIP_L4_HARDWARE` and verifies its fallback and strict guard.

Each benchmark records GPU and software versions, precision, model revision or
exact tensor workload, 10 warm-up iterations, 50 measured iterations, explicit
CUDA synchronization, minimum/p10/median/p90/maximum latency, throughput, and
peak allocated memory.

## Limits

- The Triton residual fusion supports contiguous CUDA FP16/BF16 rows whose
  power-of-two working width fits the documented 64 KiB row limit.
- The causal prefill lab supports only contiguous batch-one FP16/BF16 tensors,
  head dimension 128, and sequence lengths through 2048. It is not wired into
  the model and is not a general FlashAttention implementation.
- The CUDA C++ path supports only contiguous single-token FP16/BF16 decode with
  head dimension 128, no padding mask, and query-head counts divisible by the
  KV-head count. Multi-request decode remains on the readable batched path.
- The CUDA extension is JIT-compiled and therefore needs a CUDA development
  toolkit, C++ compiler, and Ninja to use the custom path. Missing build tools
  or compilation failures trigger the PyTorch fallback in normal mode.
- The current CUDA kernel prioritizes traceability over speed: it copies page
  pointers for each call and performs the sequence loop inside each query-head
  block. Benchmark results must be read before making any performance claim.
- The CuTe lab is not integrated, supports only BF16 Qwen-sized tensors with at
  most 128 tokens, and is hardware-gated to H100/B200. It is intentionally
  naive and may be slower than PyTorch.
