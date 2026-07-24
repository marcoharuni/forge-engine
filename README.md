# ForgeEngine

ForgeEngine is a small, readable, single-GPU inference engine for
`Qwen/Qwen3-4B-Instruct-2507`.

Milestone 1 is complete. Its scope is a trusted single-user baseline: CUDA-only
model loading, the official chat template, and an explicit greedy token loop
with key-value caching. It does not use `model.generate` or a pipeline.

The engine is locked to:

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Hugging Face revision: `cdbee75f17c01a7cc42f958dc650907174af0554`

## Install

Python 3.11 or newer and a CUDA-capable PyTorch installation are required.

```bash
python -m pip install -e ".[dev]"
```

## Unit tests

```bash
python -m pytest -q
```

The tests use small fakes and never download model weights.

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
