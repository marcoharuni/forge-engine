"""Tests for the explicit greedy token loop."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from forge_engine.cache import PagedKVBlockPool
from forge_engine.engine import GenerationEngine
from forge_engine.model import CUDAOutOfMemoryError
from forge_engine.sampling import SamplingParams

FAKE_LONG = object()
FAKE_FLOAT = object()


class FakeDevice:
    """Tiny device stand-in exposing the Torch device type."""

    def __init__(self, device_type: str) -> None:
        self.type = device_type

    def __eq__(self, other: object) -> bool:
        """Compare fake devices by type."""
        return isinstance(other, FakeDevice) and self.type == other.type


class FakeTensor:
    """Tiny tensor stand-in for prompt and token tensors."""

    def __init__(
        self,
        token_id: int | None = None,
        *,
        shape: tuple[int, ...] = (1, 1),
        dtype: object = FAKE_LONG,
        device: str = "cpu",
    ) -> None:
        self.token_id = token_id
        self.shape = shape
        self.dtype = dtype
        self.device = FakeDevice(device)

    @property
    def ndim(self) -> int:
        """Return the number of fake tensor dimensions."""
        return len(self.shape)

    def to(self, device: object) -> FakeTensor:
        """Record the requested device."""
        device_type = getattr(device, "type", device)
        self.device = FakeDevice(str(device_type))
        return self

    def item(self) -> int:
        """Return the scalar token ID."""
        assert self.token_id is not None
        return self.token_id

    def unsqueeze(self, dimension: int) -> FakeTensor:
        """Add a leading batch or trailing decode sequence dimension."""
        if dimension == 0:
            self.shape = (1, *self.shape)
        elif dimension == -1:
            self.shape = (*self.shape, 1)
        else:
            raise AssertionError(f"unsupported fake dimension: {dimension}")
        return self

    def is_floating_point(self) -> bool:
        """Report whether this fake uses the floating dtype marker."""
        return self.dtype is FAKE_FLOAT

    def contiguous(self) -> FakeTensor:
        """Return contiguous fake storage."""
        return self

    def is_contiguous(self) -> bool:
        """Report contiguous fake storage."""
        return True

    def __getitem__(self, key: object) -> FakeTensor:
        """Return a fake tensor view for cache block slicing."""
        selectors = key if isinstance(key, tuple) else (key,)
        shape: list[int] = []
        for dimension, size in enumerate(self.shape):
            selector = (
                selectors[dimension]
                if dimension < len(selectors)
                else slice(None)
            )
            if isinstance(selector, int):
                continue
            if not isinstance(selector, slice):
                raise AssertionError(
                    f"unsupported fake selector: {selector!r}"
                )
            start, stop, step = selector.indices(size)
            shape.append(len(range(start, stop, step)))
        return FakeTensor(
            self.token_id,
            shape=tuple(shape),
            dtype=self.dtype,
            device=self.device.type,
        )

    def copy_(self, source: FakeTensor) -> FakeTensor:
        """Accept a metadata-compatible fake tensor copy."""
        assert self.shape == source.shape
        return self


class FakeLogits:
    """Logits stand-in returning a predetermined greedy token."""

    def __init__(self, token_id: int, sequence_length: int) -> None:
        self.token_id = token_id
        self.shape = (1, sequence_length, 8)
        self.dtype = FAKE_FLOAT
        self.device = FakeDevice("cuda")

    @property
    def ndim(self) -> int:
        """Return the rank of the fake logits."""
        return len(self.shape)

    def __getitem__(self, key: object) -> FakeLogits:
        """Accept last-token slicing."""
        self.shape = (self.shape[0], self.shape[-1])
        return self

    def is_floating_point(self) -> bool:
        """Report that logits use the floating dtype marker."""
        return True

    def argmax(self, dim: int) -> FakeTensor:
        """Return the predetermined token."""
        assert dim == -1
        return FakeTensor(self.token_id, shape=(1,), device="cuda")


class FakeBatchEncoding(dict[str, FakeTensor]):
    """Mapping-shaped tokenizer result matching Transformers BatchEncoding."""


class FakeTokenizer:
    """Tokenizer fake with an observable chat-template call."""

    eos_token_id = 0

    def __init__(self, template_output: object | None = None) -> None:
        self.prompt_tensor = FakeTensor(shape=(1, 3))
        self.template_output = (
            self.prompt_tensor if template_output is None else template_output
        )
        self.template_call: tuple[object, ...] | None = None

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
    ) -> object:
        """Record the official chat-template inputs."""
        self.template_call = (
            conversation,
            tokenize,
            add_generation_prompt,
            return_tensors,
        )
        return self.template_output

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        """Decode the two fake generated tokens."""
        assert skip_special_tokens
        return "".join({1: "Hello", 2: "!"}[token] for token in token_ids)


class FakeModel:
    """Model fake exposing only forward."""

    def __init__(self) -> None:
        self.device = FakeDevice("cuda")
        self.tokens = iter((1, 2, 0))
        self.calls: list[dict[str, object]] = []
        self.returned_caches: list[object] = []
        self.sequence_length = 0

    def forward(self, **kwargs: object) -> SimpleNamespace:
        """Return one fake token and cache value."""
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        self.sequence_length += input_ids.shape[1]
        token = next(self.tokens)
        cache = (
            (
                FakeTensor(
                    shape=(1, 2, self.sequence_length, 4),
                    dtype=FAKE_FLOAT,
                    device="cuda",
                ),
                FakeTensor(
                    shape=(1, 2, self.sequence_length, 4),
                    dtype=FAKE_FLOAT,
                    device="cuda",
                ),
            ),
        )
        self.returned_caches.append(cache)
        return SimpleNamespace(
            logits=FakeLogits(token, input_ids.shape[1]),
            past_key_values=cache,
        )


def fake_torch_module(oom_error: type[BaseException] = RuntimeError) -> ModuleType:
    """Build the Torch surface used by the guarded decoder."""
    fake_torch = ModuleType("torch")
    fake_torch.long = FAKE_LONG
    fake_torch.cuda = SimpleNamespace(OutOfMemoryError=oom_error)
    fake_torch.is_floating_point = lambda tensor: tensor.is_floating_point()

    def ones(
        shape: tuple[int, ...],
        *,
        dtype: object,
        device: object,
    ) -> FakeTensor:
        """Create a fake all-ones attention-mask extension."""
        device_type = getattr(device, "type", device)
        return FakeTensor(
            shape=shape,
            dtype=dtype,
            device=str(device_type),
        )

    def cat(tensors: tuple[FakeTensor, ...], dim: int) -> FakeTensor:
        """Concatenate fake token or cache tensors."""
        assert dim in (1, 2)
        first = tensors[0]
        shape = list(first.shape)
        shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
        return FakeTensor(
            shape=tuple(shape),
            dtype=first.dtype,
            device=first.device.type,
        )

    def empty(
        shape: tuple[int, ...],
        *,
        dtype: object,
        device: object,
    ) -> FakeTensor:
        """Allocate fake physical KV block storage."""
        device_type = getattr(device, "type", device)
        return FakeTensor(
            shape=shape,
            dtype=dtype,
            device=str(device_type),
        )

    fake_torch.ones = ones
    fake_torch.cat = cat
    fake_torch.empty = empty
    @contextmanager
    def inference_mode() -> object:
        yield

    fake_torch.inference_mode = inference_mode
    return fake_torch


class EngineTests(TestCase):
    """Greedy decoding behavior."""

    def test_stream_uses_template_forward_cache_and_inference_mode(self) -> None:
        """The loop forwards the prompt once and then one token at a time."""
        inference_entries: list[bool] = []
        fake_torch = fake_torch_module()

        @contextmanager
        def inference_mode() -> object:
            inference_entries.append(True)
            yield

        fake_torch.inference_mode = inference_mode
        tokenizer = FakeTokenizer()
        model = FakeModel()
        pool = PagedKVBlockPool(block_size=2, capacity=8)
        engine = GenerationEngine(
            tokenizer,
            model,
            SamplingParams(max_new_tokens=8),
            pool,
        )
        messages = [{"role": "user", "content": "Hi"}]

        with patch.dict(sys.modules, {"torch": fake_torch}):
            fragments = list(engine.stream(messages))

        self.assertEqual(fragments, ["Hello", "!"])
        self.assertEqual(
            tokenizer.template_call,
            (messages, True, True, "pt"),
        )
        self.assertEqual(tokenizer.prompt_tensor.device.type, "cuda")
        self.assertEqual(inference_entries, [True])
        self.assertEqual(len(model.calls), 3)
        self.assertIsNone(model.calls[0]["past_key_values"])
        self.assertEqual(
            model.calls[1]["past_key_values"][0][0].shape,
            model.returned_caches[0][0][0].shape,
        )
        self.assertTrue(model.calls[0]["use_cache"])
        self.assertEqual(model.calls[1]["input_ids"].token_id, 1)
        self.assertEqual(pool.allocated_block_count, 0)

    def test_stream_adds_batch_dimension_to_one_dimensional_tokens(self) -> None:
        """A rank-one tokenizer tensor becomes one batched prompt."""
        prompt = FakeTensor(shape=(3,))
        model = FakeModel()
        engine = GenerationEngine(
            FakeTokenizer(prompt),
            model,
            SamplingParams(max_new_tokens=1),
        )

        with patch.dict(sys.modules, {"torch": fake_torch_module()}):
            list(engine.stream([{"role": "user", "content": "Hi"}]))

        self.assertIs(model.calls[0]["input_ids"], prompt)
        self.assertEqual(prompt.shape, (1, 3))
        self.assertEqual(prompt.device.type, "cuda")

    def test_stream_keeps_two_dimensional_tokens(self) -> None:
        """A batched tokenizer tensor keeps its batch dimension unchanged."""
        prompt = FakeTensor(shape=(1, 3))
        model = FakeModel()
        engine = GenerationEngine(
            FakeTokenizer(prompt),
            model,
            SamplingParams(max_new_tokens=1),
        )

        with patch.dict(sys.modules, {"torch": fake_torch_module()}):
            list(engine.stream([{"role": "user", "content": "Hi"}]))

        self.assertIs(model.calls[0]["input_ids"], prompt)
        self.assertEqual(prompt.shape, (1, 3))

    def test_stream_extracts_batch_encoding_and_attention_mask(self) -> None:
        """BatchEncoding-like output supplies aligned IDs and mask tensors."""
        prompt = FakeTensor(shape=(1, 3))
        attention_mask = FakeTensor(shape=(1, 3))
        tokenizer_output = FakeBatchEncoding(
            input_ids=prompt,
            attention_mask=attention_mask,
        )
        model = FakeModel()
        engine = GenerationEngine(
            FakeTokenizer(tokenizer_output),
            model,
            SamplingParams(max_new_tokens=2),
        )

        with patch.dict(sys.modules, {"torch": fake_torch_module()}):
            list(engine.stream([{"role": "user", "content": "Hi"}]))

        self.assertIs(model.calls[0]["input_ids"], prompt)
        self.assertIs(model.calls[0]["attention_mask"], attention_mask)
        self.assertEqual(attention_mask.device.type, "cuda")
        self.assertEqual(model.calls[1]["attention_mask"].shape, (1, 4))

    def test_stream_rejects_unsupported_tokenizer_shape(self) -> None:
        """Tokenizer tensors outside rank one or two fail clearly."""
        engine = GenerationEngine(
            FakeTokenizer(FakeTensor(shape=(1, 2, 3))),
            FakeModel(),
            SamplingParams(max_new_tokens=1),
        )

        with (
            patch.dict(sys.modules, {"torch": fake_torch_module()}),
            self.assertRaisesRegex(
                ValueError,
                r"must have shape \[sequence\] or \[batch, sequence\]",
            ),
        ):
            list(engine.stream([{"role": "user", "content": "Hi"}]))

    def test_stream_rejects_mismatched_attention_mask(self) -> None:
        """A tokenizer mask must align exactly with its input IDs."""
        tokenizer_output = FakeBatchEncoding(
            input_ids=FakeTensor(shape=(1, 3)),
            attention_mask=FakeTensor(shape=(1, 2)),
        )
        engine = GenerationEngine(
            FakeTokenizer(tokenizer_output),
            FakeModel(),
            SamplingParams(max_new_tokens=1),
        )

        with (
            patch.dict(sys.modules, {"torch": fake_torch_module()}),
            self.assertRaisesRegex(
                ValueError,
                "attention_mask shape must match input_ids shape",
            ),
        ):
            list(engine.stream([{"role": "user", "content": "Hi"}]))

    def test_stream_rejects_non_cuda_logits(self) -> None:
        """The explicit loop guards the model output device."""

        class CPUOutputModel(FakeModel):
            """Return otherwise valid logits on the CPU."""

            def forward(self, **kwargs: object) -> SimpleNamespace:
                output = super().forward(**kwargs)
                output.logits.device = FakeDevice("cpu")
                return output

        engine = GenerationEngine(
            FakeTokenizer(),
            CPUOutputModel(),
            SamplingParams(max_new_tokens=1),
        )

        with (
            patch.dict(sys.modules, {"torch": fake_torch_module()}),
            self.assertRaisesRegex(ValueError, "logits must be on a CUDA device"),
        ):
            list(engine.stream([{"role": "user", "content": "Hi"}]))

    def test_stream_rejects_inconsistent_cache_shape(self) -> None:
        """The explicit loop guards cached sequence length."""

        class BadCacheModel(FakeModel):
            """Return a key/value cache with the wrong sequence length."""

            def forward(self, **kwargs: object) -> SimpleNamespace:
                output = super().forward(**kwargs)
                output.past_key_values[0][0].shape = (1, 2, 1, 4)
                return output

        engine = GenerationEngine(
            FakeTokenizer(),
            BadCacheModel(),
            SamplingParams(max_new_tokens=1),
        )

        with (
            patch.dict(sys.modules, {"torch": fake_torch_module()}),
            self.assertRaisesRegex(
                ValueError,
                "key/value shapes must match",
            ),
        ):
            list(engine.stream([{"role": "user", "content": "Hi"}]))

    def test_stop_string_crosses_token_boundaries(self) -> None:
        """A stop string spanning two tokens is withheld from output."""
        engine = GenerationEngine(
            FakeTokenizer(),
            FakeModel(),
            SamplingParams(
                max_new_tokens=8,
                stop_strings=("lo!",),
            ),
        )

        with patch.dict(sys.modules, {"torch": fake_torch_module()}):
            fragments = list(
                engine.stream([{"role": "user", "content": "Hi"}])
            )

        self.assertEqual(fragments, ["Hel"])

    def test_token_limit_stops_without_an_extra_decode(self) -> None:
        """The configured token limit avoids unnecessary model work."""
        model = FakeModel()
        engine = GenerationEngine(
            FakeTokenizer(),
            model,
            SamplingParams(max_new_tokens=1),
        )

        with patch.dict(sys.modules, {"torch": fake_torch_module()}):
            fragments = list(
                engine.stream([{"role": "user", "content": "Hi"}])
            )

        self.assertEqual(fragments, ["Hello"])
        self.assertEqual(len(model.calls), 1)

    def test_stream_error_reclaims_every_paged_block(self) -> None:
        """A decode failure releases the request's entire block table."""

        class DecodeFailureModel(FakeModel):
            """Fail after a successful prefill."""

            def forward(self, **kwargs: object) -> SimpleNamespace:
                if self.calls:
                    raise ValueError("decode failed")
                return super().forward(**kwargs)

        pool = PagedKVBlockPool(block_size=2, capacity=8)
        engine = GenerationEngine(
            FakeTokenizer(),
            DecodeFailureModel(),
            SamplingParams(max_new_tokens=8),
            pool,
        )

        with (
            patch.dict(sys.modules, {"torch": fake_torch_module()}),
            self.assertRaisesRegex(ValueError, "decode failed"),
        ):
            list(engine.stream([{"role": "user", "content": "Hi"}]))

        self.assertEqual(pool.allocated_block_count, 0)

    def test_stream_cancellation_reclaims_every_paged_block(self) -> None:
        """Closing a suspended stream releases its physical blocks."""
        pool = PagedKVBlockPool(block_size=2, capacity=8)
        engine = GenerationEngine(
            FakeTokenizer(),
            FakeModel(),
            SamplingParams(max_new_tokens=8),
            pool,
        )
        stream = engine.stream([{"role": "user", "content": "Hi"}])

        with patch.dict(sys.modules, {"torch": fake_torch_module()}):
            self.assertEqual(next(stream), "Hello")
            self.assertGreater(pool.allocated_block_count, 0)
            stream.close()

        self.assertEqual(pool.allocated_block_count, 0)

    def test_stream_reports_cuda_out_of_memory(self) -> None:
        """CUDA allocation failures become concise user-facing errors."""

        class FakeOOM(RuntimeError):
            """Fake Torch CUDA out-of-memory error."""

        class OutOfMemoryModel:
            """Model fake that cannot allocate its first forward pass."""

            device = FakeDevice("cuda")

            def forward(self, **kwargs: object) -> object:
                """Raise the fake CUDA allocation failure."""
                raise FakeOOM

        fake_torch = fake_torch_module(FakeOOM)
        engine = GenerationEngine(
            FakeTokenizer(),
            OutOfMemoryModel(),
            SamplingParams(max_new_tokens=1),
        )

        with (
            patch.dict(sys.modules, {"torch": fake_torch}),
            self.assertRaisesRegex(
                CUDAOutOfMemoryError,
                "CUDA out of memory during inference",
            ),
        ):
            list(engine.stream([{"role": "user", "content": "Hi"}]))
