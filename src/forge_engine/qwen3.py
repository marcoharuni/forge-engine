"""Minimal PyTorch Qwen3 model used by the ForgeEngine runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    """Qwen3 dimensions required by the supported model."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    eos_token_id: int
    tie_word_embeddings: bool
    attention_bias: bool

    @classmethod
    def from_file(cls, path: Path) -> Qwen3Config:
        """Read and validate the pinned Hugging Face config."""
        raw = json.loads(path.read_text())
        if raw.get("model_type") != "qwen3":
            raise ValueError("supported package must declare model_type=qwen3")
        if raw.get("hidden_act") != "silu":
            raise ValueError("supported package must use SiLU")
        if raw.get("sliding_window") is not None:
            raise ValueError("sliding-window attention is not supported")
        config = cls(
            vocab_size=int(raw["vocab_size"]),
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw["head_dim"]),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            rope_theta=float(raw["rope_theta"]),
            eos_token_id=int(raw["eos_token_id"]),
            tie_word_embeddings=bool(raw["tie_word_embeddings"]),
            attention_bias=bool(raw["attention_bias"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Reject architecture variants outside the one supported package."""
        expected = {
            "vocab_size": 151936,
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5_000_000.0,
            "eos_token_id": 151645,
            "tie_word_embeddings": True,
            "attention_bias": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"unsupported Qwen3 config {name}={getattr(self, name)!r}; "
                    f"expected {value!r}"
                )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be divisible by KV heads")


class RMSNorm(nn.Module):
    """Qwen3 RMSNorm with float32 variance accumulation."""

    def __init__(self, size: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.epsilon = epsilon

    def forward(self, inputs: Tensor) -> Tensor:
        """Normalize the final dimension."""
        dtype = inputs.dtype
        normalized = inputs.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.epsilon)
        return self.weight * normalized.to(dtype)


class Qwen3Attention(nn.Module):
    """Grouped-query causal self-attention for the supported Qwen3 model."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        query_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size, query_size, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, kv_size, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, kv_size, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            query_size, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.rope_theta = config.rope_theta

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None,
        past_key_value: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Apply attention and return the extended contiguous KV tensors."""
        batch, query_length, _ = hidden_states.shape
        query = self.q_norm(
            self.q_proj(hidden_states).view(
                batch, query_length, self.num_heads, self.head_dim
            )
        ).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden_states).view(
                batch, query_length, self.num_kv_heads, self.head_dim
            )
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch, query_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        cosine, sine = _rotary_embeddings(
            position_ids, self.head_dim, self.rope_theta, hidden_states.dtype
        )
        query, key = _apply_rotary(query, key, cosine, sine)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            _validate_past(past_key, past_value, batch, self.num_kv_heads)
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        key_length = key.shape[2]
        past_length = key_length - query_length

        repeated_key = key.repeat_interleave(self.groups, dim=1)
        repeated_value = value.repeat_interleave(self.groups, dim=1)
        weights = torch.matmul(query, repeated_key.transpose(2, 3))
        weights = weights * self.scaling
        weights = weights + _causal_mask(
            batch=batch,
            query_length=query_length,
            key_length=key_length,
            past_length=past_length,
            dtype=weights.dtype,
            device=weights.device,
            attention_mask=attention_mask,
        )
        probabilities = F.softmax(weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        output = torch.matmul(probabilities, repeated_value)
        output = output.transpose(1, 2).contiguous().view(
            batch, query_length, self.num_heads * self.head_dim
        )
        return self.o_proj(output), (key, value)


class Qwen3MLP(nn.Module):
    """Qwen3 gated SiLU feed-forward block."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the gated feed-forward projection."""
        return self.down_proj(
            F.silu(self.gate_proj(inputs)) * self.up_proj(inputs)
        )


class Qwen3DecoderLayer(nn.Module):
    """One pre-normalized Qwen3 decoder layer."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None,
        past_key_value: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Run attention and MLP residual blocks."""
        attention, present = self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids,
            attention_mask,
            past_key_value,
        )
        hidden_states = hidden_states + attention
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present


@dataclass(frozen=True, slots=True)
class Qwen3Output:
    """Output consumed by generation and M2 correctness validation."""

    logits: Tensor
    past_key_values: tuple[tuple[Tensor, Tensor], ...]
    last_hidden_state: Tensor
    layer_hidden_states: tuple[Tensor, ...]


class Qwen3ForCausalLM(nn.Module):
    """Minimal tied-embedding Qwen3 causal language model."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.model.layers = nn.ModuleList(
            Qwen3DecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        )
        self.model.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    @property
    def device(self) -> torch.device:
        """Return the embedding device used by all staged parameters."""
        return self.model.embed_tokens.weight.device

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = True,
    ) -> Qwen3Output:
        """Run explicit Qwen3 forward inference."""
        _validate_inputs(input_ids, attention_mask, self.device)
        past = _normalize_past(past_key_values, len(self.model.layers))
        past_length = 0 if past[0] is None else past[0][0].shape[2]
        position_ids = torch.arange(
            past_length,
            past_length + input_ids.shape[1],
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)
        hidden_states = self.model.embed_tokens(input_ids)
        captured: list[Tensor] = []
        presents: list[tuple[Tensor, Tensor]] = []
        for layer, layer_past in zip(self.model.layers, past, strict=True):
            hidden_states, present = layer(
                hidden_states, position_ids, attention_mask, layer_past
            )
            captured.append(hidden_states)
            if use_cache:
                presents.append(present)
        last_hidden_state = self.model.norm(hidden_states)
        logits = F.linear(
            last_hidden_state, self.model.embed_tokens.weight
        ).float()
        return Qwen3Output(
            logits=logits,
            past_key_values=tuple(presents),
            last_hidden_state=last_hidden_state,
            layer_hidden_states=tuple(captured),
        )


def _rotary_embeddings(
    position_ids: Tensor,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Compute default Qwen3 rotary cosine and sine in float32."""
    frequencies = 1.0 / (
        theta
        ** (
            torch.arange(
                0, head_dim, 2, dtype=torch.float32, device=position_ids.device
            )
            / head_dim
        )
    )
    phases = position_ids.float().unsqueeze(-1) * frequencies
    embeddings = torch.cat((phases, phases), dim=-1)
    return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


def _rotate_half(inputs: Tensor) -> Tensor:
    """Rotate the two halves of each attention head."""
    first, second = inputs.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    query: Tensor, key: Tensor, cosine: Tensor, sine: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply broadcast rotary embeddings to query and key."""
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


def _causal_mask(
    *,
    batch: int,
    query_length: int,
    key_length: int,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: Tensor | None,
) -> Tensor:
    """Build the additive lower-right causal and padding mask."""
    query_positions = torch.arange(query_length, device=device) + past_length
    key_positions = torch.arange(key_length, device=device)
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    allowed = allowed.view(1, 1, query_length, key_length).expand(
        batch, 1, query_length, key_length
    )
    if attention_mask is not None:
        if tuple(attention_mask.shape) != (batch, key_length):
            raise ValueError(
                "attention_mask must match the complete cached sequence"
            )
        allowed = allowed & attention_mask[:, None, None, :].bool()
    mask = torch.zeros(
        (batch, 1, query_length, key_length), dtype=dtype, device=device
    )
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)


def _validate_inputs(
    input_ids: Tensor,
    attention_mask: Tensor | None,
    device: torch.device,
) -> None:
    """Guard token tensor shape, dtype, and device."""
    if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
        raise ValueError("input_ids must have shape [batch, nonempty sequence]")
    if input_ids.dtype != torch.long:
        raise ValueError("input_ids must use torch.long")
    if input_ids.device != device:
        raise ValueError("input_ids must be on the model device")
    if attention_mask is not None and attention_mask.device != device:
        raise ValueError("attention_mask must be on the model device")


def _validate_past(
    key: Tensor, value: Tensor, batch: int, num_kv_heads: int
) -> None:
    """Guard a contiguous KV pair before appending new tokens."""
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError("cached key/value tensors must have matching rank-4 shapes")
    if key.shape[0] != batch or key.shape[1] != num_kv_heads:
        raise ValueError("cached key/value batch or head count is invalid")
    if value.device != key.device:
        raise ValueError("cached key/value tensors must share a device")
    if not key.is_floating_point() or key.dtype != value.dtype:
        raise ValueError("cached key/value tensors must share a floating dtype")


def _normalize_past(
    past_key_values: object | None, layer_count: int
) -> tuple[tuple[Tensor, Tensor] | None, ...]:
    """Normalize the runner's legacy tuple cache."""
    if past_key_values is None:
        return (None,) * layer_count
    if not isinstance(past_key_values, (tuple, list)):
        raise ValueError("past_key_values must be a layer tuple")
    if len(past_key_values) != layer_count:
        raise ValueError("past_key_values must contain one entry per layer")
    normalized: list[tuple[Tensor, Tensor]] = []
    for entry in past_key_values:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError("each cache layer must contain key and value")
        normalized.append((entry[0], entry[1]))
    return tuple(normalized)
