"""Contiguous and reference paged key-value cache implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

CacheLayer = tuple["Tensor", "Tensor"]


class KVCacheCapacityError(RuntimeError):
    """Raised when a paged cache cannot reserve another physical block."""


@dataclass(frozen=True, slots=True)
class KVLayerLayout:
    """Tensor metadata needed to allocate one layer in a physical block."""

    batch_size: int
    head_count: int
    head_dim: int
    dtype: object
    device: object


@dataclass(frozen=True, slots=True)
class PagedLayerView:
    """Direct logical pages for one decoder layer."""

    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]
    sequence_length: int
    block_size: int


@dataclass(slots=True)
class _PhysicalBlock:
    """Key/value storage for every layer at one physical block ID."""

    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]


class ContiguousKVCache:
    """Own validated contiguous key/value tensors for every decoder layer."""

    def __init__(self, layer_count: int | None = None) -> None:
        """Create an empty cache with an optional fixed layer count."""
        if layer_count is not None and layer_count < 1:
            raise ValueError("layer_count must be at least 1")
        self._layer_count = layer_count
        self._layers: tuple[CacheLayer, ...] = ()
        self._sequence_length = 0

    @property
    def sequence_length(self) -> int:
        """Return the number of cached token positions."""
        return self._sequence_length

    @property
    def layer_count(self) -> int:
        """Return the fixed or discovered number of decoder layers."""
        return (
            self._layer_count
            if self._layer_count is not None
            else len(self._layers)
        )

    @property
    def empty(self) -> bool:
        """Return whether no model state is currently cached."""
        return not self._layers

    @property
    def layers(self) -> tuple[CacheLayer, ...]:
        """Expose the immutable layer tuple for validation and model input."""
        return self._layers

    def replace(self, past_key_values: object) -> None:
        """Replace state with validated contiguous model outputs."""
        layers = _legacy_layers(past_key_values)
        if not isinstance(layers, (tuple, list)) or not layers:
            raise ValueError("past_key_values must contain cache layers")
        if self._layer_count is None:
            self._layer_count = len(layers)
        elif len(layers) != self._layer_count:
            raise ValueError(
                f"cache has {len(layers)} layers; "
                f"expected {self._layer_count}"
            )

        normalized: list[CacheLayer] = []
        sequence_length: int | None = None
        common_device: object | None = None
        common_dtype: object | None = None
        for index, layer in enumerate(layers):
            if not isinstance(layer, (tuple, list)) or len(layer) != 2:
                raise ValueError(
                    f"cache layer {index} must contain exactly key and value"
                )
            key, value = layer
            if getattr(key, "ndim", None) != 4 or getattr(
                value, "ndim", None
            ) != 4:
                raise ValueError(f"cache layer {index} tensors must be rank four")
            if key.shape != value.shape:
                raise ValueError(
                    f"cache layer {index} key/value shapes must match"
                )
            if key.shape[0] < 1 or key.shape[1] < 1 or key.shape[-1] < 1:
                raise ValueError(f"cache layer {index} has an invalid shape")
            current_length = key.shape[-2]
            if current_length < 1:
                raise ValueError("cache sequence length must be non-empty")
            if sequence_length is None:
                sequence_length = current_length
            elif current_length != sequence_length:
                raise ValueError("all cache layers must share a sequence length")
            if key.dtype != value.dtype or key.device != value.device:
                raise ValueError(
                    f"cache layer {index} key/value metadata must match"
                )
            is_floating = getattr(key, "is_floating_point", None)
            if not callable(is_floating) or not is_floating():
                raise ValueError(f"cache layer {index} must be floating point")
            if common_device is None:
                common_device = key.device
                common_dtype = key.dtype
            elif key.device != common_device or key.dtype != common_dtype:
                raise ValueError("all cache layers must share device and dtype")
            key = key.contiguous()
            value = value.contiguous()
            if not key.is_contiguous() or not value.is_contiguous():
                raise ValueError(f"cache layer {index} could not be contiguous")
            normalized.append((key, value))

        assert sequence_length is not None
        self._layers = tuple(normalized)
        self._sequence_length = sequence_length

    def as_model_input(self) -> tuple[CacheLayer, ...]:
        """Return populated cache state in the model's tuple format."""
        if self.empty:
            raise ValueError("cannot decode with an empty cache")
        return self._layers

    def clear(self) -> None:
        """Discard all cached tensor references."""
        self._layers = ()
        self._sequence_length = 0


class PagedKVBlockPool:
    """Lazily allocate and reuse fixed-size physical KV blocks."""

    def __init__(self, *, block_size: int = 16, capacity: int = 16_384) -> None:
        """Create an unconfigured pool with a fixed logical capacity."""
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._block_size = block_size
        self._capacity = capacity
        self._free_ids = list(range(capacity - 1, -1, -1))
        self._allocated_ids: set[int] = set()
        self._blocks: dict[int, _PhysicalBlock] = {}
        self._layouts: tuple[KVLayerLayout, ...] | None = None

    @property
    def block_size(self) -> int:
        """Return token positions stored in each physical block."""
        return self._block_size

    @property
    def capacity(self) -> int:
        """Return the maximum number of simultaneously allocated blocks."""
        return self._capacity

    @property
    def allocated_block_count(self) -> int:
        """Return the number of blocks currently leased to caches."""
        return len(self._allocated_ids)

    @property
    def free_block_count(self) -> int:
        """Return the remaining logical block capacity."""
        return len(self._free_ids)

    @property
    def materialized_block_count(self) -> int:
        """Return the number of lazily created reusable tensor blocks."""
        return len(self._blocks)

    @property
    def layouts(self) -> tuple[KVLayerLayout, ...] | None:
        """Return the configured per-layer tensor layouts."""
        return self._layouts

    def configure(self, layers: tuple[CacheLayer, ...]) -> None:
        """Lock the pool to the model metadata seen during first prefill."""
        layouts = tuple(
            KVLayerLayout(
                batch_size=key.shape[0],
                head_count=key.shape[1],
                head_dim=key.shape[3],
                dtype=key.dtype,
                device=key.device,
            )
            for key, _value in layers
        )
        if self._layouts is None:
            self._layouts = layouts
        elif layouts != self._layouts:
            raise ValueError("cache tensor layout does not match the block pool")

    def allocate(self, count: int) -> tuple[int, ...]:
        """Lease physical block IDs transactionally."""
        if count < 1:
            raise ValueError("allocation count must be at least 1")
        if self._layouts is None:
            raise ValueError("block pool must be configured before allocation")
        if count > len(self._free_ids):
            raise KVCacheCapacityError(
                f"KV block pool needs {count} blocks but only "
                f"{len(self._free_ids)} are free"
            )

        selected = tuple(reversed(self._free_ids[-count:]))
        created: list[int] = []
        try:
            for block_id in selected:
                if block_id not in self._blocks:
                    self._blocks[block_id] = self._new_block()
                    created.append(block_id)
        except BaseException:
            for block_id in created:
                del self._blocks[block_id]
            raise

        del self._free_ids[-count:]
        self._allocated_ids.update(selected)
        return selected

    def release(self, block_ids: tuple[int, ...]) -> None:
        """Return leased IDs without discarding their reusable tensors."""
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("cannot release duplicate block IDs")
        unknown = set(block_ids) - self._allocated_ids
        if unknown:
            raise ValueError(f"cannot release unallocated block IDs: {unknown}")
        self._allocated_ids.difference_update(block_ids)
        self._free_ids.extend(reversed(block_ids))

    def block(self, block_id: int) -> _PhysicalBlock:
        """Return a block only while its ID is actively leased."""
        if block_id not in self._allocated_ids:
            raise ValueError(f"block ID {block_id} is not allocated")
        return self._blocks[block_id]

    def _new_block(self) -> _PhysicalBlock:
        """Allocate storage for every configured decoder layer."""
        assert self._layouts is not None
        import torch

        keys = tuple(
            torch.empty(
                (
                    layout.batch_size,
                    layout.head_count,
                    self._block_size,
                    layout.head_dim,
                ),
                dtype=layout.dtype,
                device=layout.device,
            )
            for layout in self._layouts
        )
        values = tuple(
            torch.empty(
                (
                    layout.batch_size,
                    layout.head_count,
                    self._block_size,
                    layout.head_dim,
                ),
                dtype=layout.dtype,
                device=layout.device,
            )
            for layout in self._layouts
        )
        return _PhysicalBlock(keys=keys, values=values)


class PagedKVCache:
    """Map one sequence onto reusable physical blocks in a shared pool."""

    def __init__(self, pool: PagedKVBlockPool) -> None:
        self._pool = pool
        self._block_table: list[int] = []
        self._sequence_length = 0
        self._layer_count = 0

    @property
    def sequence_length(self) -> int:
        """Return the number of stored token positions."""
        return self._sequence_length

    @property
    def layer_count(self) -> int:
        """Return the number of decoder layers stored per block."""
        return self._layer_count

    @property
    def empty(self) -> bool:
        """Return whether this sequence owns no physical blocks."""
        return not self._block_table

    @property
    def block_table(self) -> tuple[int, ...]:
        """Expose the logical-to-physical mapping for validation."""
        return tuple(self._block_table)

    @property
    def layers(self) -> tuple[CacheLayer, ...]:
        """Materialize contiguous layers for the reference model path."""
        return self.as_model_input()

    def replace(self, past_key_values: object) -> None:
        """Populate an empty sequence cache from prefill output."""
        if not self.empty:
            raise ValueError("paged cache prefill requires an empty cache")
        layers = _validated_layers(past_key_values)
        self._pool.configure(layers)
        sequence_length = layers[0][0].shape[2]
        required = _ceil_div(sequence_length, self._pool.block_size)
        allocated = self._pool.allocate(required)
        try:
            for logical_index, block_id in enumerate(allocated):
                start = logical_index * self._pool.block_size
                count = min(
                    self._pool.block_size,
                    sequence_length - start,
                )
                self._copy_into_block(
                    block_id,
                    destination_offset=0,
                    layers=layers,
                    source_start=start,
                    count=count,
                )
        except BaseException:
            self._pool.release(allocated)
            raise
        self._block_table.extend(allocated)
        self._sequence_length = sequence_length
        self._layer_count = len(layers)

    def append(self, past_key_values: object) -> None:
        """Append exactly one model-output position to paged storage."""
        if self.empty:
            raise ValueError("cannot append to an empty paged cache")
        layers = _validated_layers(past_key_values)
        self._pool.configure(layers)
        if len(layers) != self._layer_count:
            raise ValueError("cache layer count changed during decode")
        output_length = layers[0][0].shape[2]
        if output_length != self._sequence_length + 1:
            raise ValueError("decode cache must add exactly one token")

        destination_offset = self._sequence_length % self._pool.block_size
        allocated: tuple[int, ...] = ()
        if destination_offset == 0:
            allocated = self._pool.allocate(1)
            block_id = allocated[0]
        else:
            block_id = self._block_table[-1]
        try:
            self._copy_into_block(
                block_id,
                destination_offset=destination_offset,
                layers=layers,
                source_start=output_length - 1,
                count=1,
            )
        except BaseException:
            if allocated:
                self._pool.release(allocated)
            raise
        if allocated:
            self._block_table.append(block_id)
        self._sequence_length = output_length

    def append_token(self, token_key_values: object) -> None:
        """Append one position when a direct paged model returns only new KV."""
        if self.empty:
            raise ValueError("cannot append to an empty paged cache")
        layers = _validated_layers(token_key_values)
        self._pool.configure(layers)
        if len(layers) != self._layer_count:
            raise ValueError("cache layer count changed during decode")
        if layers[0][0].shape[2] != 1:
            raise ValueError("direct paged decode must return exactly one token")
        destination_offset = self._sequence_length % self._pool.block_size
        allocated: tuple[int, ...] = ()
        if destination_offset == 0:
            allocated = self._pool.allocate(1)
            block_id = allocated[0]
        else:
            block_id = self._block_table[-1]
        try:
            self._copy_into_block(
                block_id,
                destination_offset=destination_offset,
                layers=layers,
                source_start=0,
                count=1,
            )
        except BaseException:
            if allocated:
                self._pool.release(allocated)
            raise
        if allocated:
            self._block_table.append(block_id)
        self._sequence_length += 1

    def layer_view(self, layer_index: int) -> PagedLayerView:
        """Expose one layer's physical pages without concatenating them."""
        if self.empty:
            raise ValueError("cannot decode with an empty paged cache")
        if not 0 <= layer_index < self._layer_count:
            raise IndexError("paged cache layer index is out of range")
        blocks = tuple(
            self._pool.block(block_id) for block_id in self._block_table
        )
        return PagedLayerView(
            keys=tuple(block.keys[layer_index] for block in blocks),
            values=tuple(block.values[layer_index] for block in blocks),
            sequence_length=self._sequence_length,
            block_size=self._pool.block_size,
        )

    def as_model_input(self) -> tuple[CacheLayer, ...]:
        """Gather paged state into contiguous tensors for reference attention."""
        if self.empty:
            raise ValueError("cannot decode with an empty paged cache")
        import torch

        layers: list[CacheLayer] = []
        for layer_index in range(self._layer_count):
            key_parts = []
            value_parts = []
            remaining = self._sequence_length
            for block_id in self._block_table:
                count = min(self._pool.block_size, remaining)
                block = self._pool.block(block_id)
                key_parts.append(block.keys[layer_index][:, :, :count, :])
                value_parts.append(block.values[layer_index][:, :, :count, :])
                remaining -= count
            key = torch.cat(tuple(key_parts), dim=2).contiguous()
            value = torch.cat(tuple(value_parts), dim=2).contiguous()
            layers.append((key, value))
        return tuple(layers)

    def clear(self) -> None:
        """Release every physical block owned by this sequence."""
        if self._block_table:
            self._pool.release(tuple(self._block_table))
        self._block_table.clear()
        self._sequence_length = 0
        self._layer_count = 0

    def _copy_into_block(
        self,
        block_id: int,
        *,
        destination_offset: int,
        layers: tuple[CacheLayer, ...],
        source_start: int,
        count: int,
    ) -> None:
        """Copy one contiguous source range into a physical block."""
        block = self._pool.block(block_id)
        destination_end = destination_offset + count
        source_end = source_start + count
        for layer_index, (key, value) in enumerate(layers):
            block.keys[layer_index][
                :, :, destination_offset:destination_end, :
            ].copy_(key[:, :, source_start:source_end, :])
            block.values[layer_index][
                :, :, destination_offset:destination_end, :
            ].copy_(value[:, :, source_start:source_end, :])


def _validated_layers(past_key_values: object) -> tuple[CacheLayer, ...]:
    """Normalize and validate cache layers without changing their storage."""
    layers = _legacy_layers(past_key_values)
    if not isinstance(layers, (tuple, list)) or not layers:
        raise ValueError("past_key_values must contain cache layers")
    normalized: list[CacheLayer] = []
    sequence_length: int | None = None
    common_dtype: object | None = None
    common_device: object | None = None
    for index, layer in enumerate(layers):
        if not isinstance(layer, (tuple, list)) or len(layer) != 2:
            raise ValueError(
                f"cache layer {index} must contain exactly key and value"
            )
        key, value = layer
        if getattr(key, "ndim", None) != 4 or getattr(
            value, "ndim", None
        ) != 4:
            raise ValueError(f"cache layer {index} tensors must be rank four")
        if key.shape != value.shape:
            raise ValueError(f"cache layer {index} key/value shapes must match")
        if (
            key.shape[0] < 1
            or key.shape[1] < 1
            or key.shape[2] < 1
            or key.shape[3] < 1
        ):
            raise ValueError(f"cache layer {index} has an invalid shape")
        if key.dtype != value.dtype or key.device != value.device:
            raise ValueError(
                f"cache layer {index} key/value metadata must match"
            )
        if not key.is_floating_point() or not value.is_floating_point():
            raise ValueError(f"cache layer {index} must be floating point")
        if common_dtype is None:
            common_dtype = key.dtype
            common_device = key.device
        elif key.dtype != common_dtype or key.device != common_device:
            raise ValueError("all cache layers must share device and dtype")
        current_length = key.shape[2]
        if sequence_length is None:
            sequence_length = current_length
        elif sequence_length != current_length:
            raise ValueError("all cache layers must share a sequence length")
        normalized.append((key, value))
    return tuple(normalized)


def _ceil_div(value: int, divisor: int) -> int:
    """Return mathematical ceiling division for positive integers."""
    return (value + divisor - 1) // divisor


def _legacy_layers(cache: object) -> object:
    """Normalize a Transformers cache or an existing layer tuple."""
    to_legacy_cache = getattr(cache, "to_legacy_cache", None)
    return to_legacy_cache() if callable(to_legacy_cache) else cache
