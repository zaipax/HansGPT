"""Conditional autoregression over 128 lossless bytes of one binary glyph grid.

The 256 output classes enumerate every eight-bit pattern, not Unicode characters
or a font inventory. One completed 32x32 grid remains one outer language token.
Only this independent decoder is implemented here; no outer model is changed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

GRID_BYTES = 128
BYTE_VALUES = 256
BYTE_BOS = 256
INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def pack_glyph_bytes(tiles: Tensor) -> Tensor:
    """[...,1,32,32] binary pixels -> [...,128] uint8, row-major and MSB first."""
    if tiles.ndim < 3 or tuple(tiles.shape[-3:]) != (1, 32, 32) or tiles.numel() == 0:
        raise ValueError("Glyphs must have nonempty shape [...,1,32,32]")
    if not bool(((tiles == 0) | (tiles == 1)).all()):
        raise ValueError("Glyph pixels must be strictly binary 0/1")
    groups = tiles.to(torch.uint8).reshape(*tiles.shape[:-3], GRID_BYTES, 8)
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=tiles.device)
    return (groups << shifts).sum(-1, dtype=torch.int16).to(torch.uint8)


def _validate_bytes(values: Tensor, *, length: int | None = None, allow_bos: bool = False):
    if values.ndim < 1 or values.numel() == 0 or values.dtype not in INTEGER_DTYPES:
        raise ValueError("Byte values must be a nonempty integer tensor")
    if length is not None and values.shape[-1] != length:
        raise ValueError(f"Expected exactly {length} byte positions")
    maximum = BYTE_BOS if allow_bos else BYTE_VALUES - 1
    comparable = values.to(torch.int64)
    if not bool(((comparable >= 0) & (comparable <= maximum)).all()):
        raise ValueError(f"Byte values must lie between zero and {maximum}")


def unpack_glyph_bytes(values: Tensor) -> Tensor:
    """[...,128] bytes -> [...,1,32,32] strict uint8 0/1 pixels, with no glyph lookup."""
    _validate_bytes(values, length=GRID_BYTES)
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=values.device)
    bits = (values.to(torch.uint8).unsqueeze(-1) >> shifts) & 1
    return bits.reshape(*values.shape[:-1], 1, 32, 32)


def byte_grid_nll(logits: Tensor, byte_targets: Tensor) -> Tensor:
    """Joint byte CE sum / 1024, nats per pixel; each byte has all 256 outcomes."""
    _validate_bytes(byte_targets, length=GRID_BYTES)
    if logits.shape != (*byte_targets.shape, BYTE_VALUES) or not logits.is_floating_point():
        raise ValueError("Byte logits must have shape [...,128,256]")
    if logits.device != byte_targets.device:
        raise ValueError("Byte logits and targets must share a device")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Byte likelihood requires finite logits")
    losses = F.cross_entropy(
        logits.float().reshape(-1, BYTE_VALUES), byte_targets.long().reshape(-1), reduction="none"
    )
    return losses.reshape(*byte_targets.shape).sum(-1) / 1024


@dataclass(frozen=True)
class ByteDecoderCache:
    """Ephemeral cache for one decoder, one fixed condition batch and one byte prefix."""

    owner: int
    condition: Tensor
    condition_embedding: Tensor
    leading_shape: tuple[int, ...]
    length: int
    layers: tuple[tuple[Tensor, Tensor], ...]


class _CausalByteBlock(nn.Module):
    def __init__(self, inner_dim: int, heads: int, intermediate_size: int):
        super().__init__()
        self.heads = heads
        self.head_dim = inner_dim // heads
        self.attention_norm = nn.RMSNorm(inner_dim, eps=1e-6)
        self.qkv = nn.Linear(inner_dim, inner_dim * 3, bias=False)
        self.attention_output = nn.Linear(inner_dim, inner_dim, bias=False)
        self.mlp_norm = nn.RMSNorm(inner_dim, eps=1e-6)
        self.gate = nn.Linear(inner_dim, intermediate_size, bias=False)
        self.up = nn.Linear(inner_dim, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, inner_dim, bias=False)

    def forward(self, hidden: Tensor, previous: tuple[Tensor, Tensor] | None = None):
        batch, length, width = hidden.shape
        qkv = self.qkv(self.attention_norm(hidden))
        query, key, value = qkv.reshape(batch, length, 3, self.heads, self.head_dim).unbind(2)
        query, key, value = (part.transpose(1, 2) for part in (query, key, value))
        past_length = 0
        if previous is not None:
            old_key, old_value = previous
            if old_key.dtype != key.dtype or old_key.device != key.device:
                raise ValueError("Cache dtype/device changed; reset the inner decoder cache")
            past_length = old_key.shape[-2]
            key = torch.cat((old_key, key), dim=-2)
            value = torch.cat((old_value, value), dim=-2)
        mask = None
        if past_length:
            # SDPA is_causal=True uses top-left alignment for a short query against
            # a long KV sequence. Absolute positions instead allow the full past.
            positions = torch.arange(length, device=hidden.device) + past_length
            keys = torch.arange(key.shape[-2], device=hidden.device)
            mask = keys.unsqueeze(0) <= positions.unsqueeze(1)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=0.0, is_causal=not past_length
        )
        hidden = hidden + self.attention_output(
            attended.transpose(1, 2).reshape(batch, length, width)
        )
        normalized = self.mlp_norm(hidden)
        hidden = hidden + self.down(F.silu(self.gate(normalized)) * self.up(normalized))
        return hidden, (key, value)


@dataclass(frozen=True)
class ByteGlyphDistribution:
    """One conditional distribution for each h supplied by the outer prefix model."""

    decoder: ConditionalByteDecoder
    hidden: Tensor

    def nll(self, targets: Tensor) -> Tensor:
        """Return [...] nats/pixel, with only shifted target-byte prefixes as inputs."""
        logits = self.decoder.teacher_forced_logits(self.hidden, targets)
        return byte_grid_nll(logits, pack_glyph_bytes(targets))

    def decode(
        self,
        *,
        strategy: str = "greedy",
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        use_cache: bool = True,
    ) -> Tensor:
        """Return [...,1,32,32] uint8 binary grids; no pixel threshold parameter exists."""
        return self.decoder.generate(
            self.hidden,
            strategy=strategy,
            temperature=temperature,
            generator=generator,
            use_cache=use_cache,
        )


class ConditionalByteDecoder(nn.Module):
    """A small causal transformer inside each outer glyph position.

    Input embedding: bytes 0..255 and byte-BOS=256. Output: bytes 0..255 only.
    Positions 0..127 identify the byte being predicted. h must come exclusively
    from the outer prefix; this class never derives it from target images.
    Callers control target-grid batching and autocast; no font assets are read.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        inner_dim: int = 128,
        layers: int = 2,
        heads: int = 4,
        intermediate_size: int = 384,
    ):
        super().__init__()
        dimensions = (hidden_size, inner_dim, layers, heads, intermediate_size)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in dimensions
        ):
            raise ValueError("Decoder dimensions must be positive integers")
        if inner_dim % heads:
            raise ValueError("inner_dim must be divisible by heads")
        self.hidden_size = hidden_size
        self.inner_dim = inner_dim
        self.num_layers = layers
        self.heads = heads
        self.intermediate_size = intermediate_size
        self.byte_embedding = nn.Embedding(BYTE_VALUES + 1, inner_dim)
        self.position_embedding = nn.Embedding(GRID_BYTES, inner_dim)
        self.condition_projection = nn.Linear(hidden_size, inner_dim)
        self.blocks = nn.ModuleList(
            [_CausalByteBlock(inner_dim, heads, intermediate_size) for _ in range(layers)]
        )
        self.final_norm = nn.RMSNorm(inner_dim, eps=1e-6)
        self.byte_head = nn.Linear(inner_dim, BYTE_VALUES)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_config(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "inner_dim": self.inner_dim,
            "layers": self.num_layers,
            "heads": self.heads,
            "intermediate_size": self.intermediate_size,
        }

    def _validate_hidden(self, hidden: Tensor) -> tuple[int, ...]:
        if (
            hidden.ndim < 1
            or hidden.shape[-1] != self.hidden_size
            or hidden.numel() == 0
            or not hidden.is_floating_point()
        ):
            raise ValueError("Condition must be nonempty floating point [...,hidden_size]")
        if hidden.device != self.condition_projection.weight.device:
            raise ValueError("Condition and decoder parameters must share a device")
        if not bool(torch.isfinite(hidden).all()):
            raise FloatingPointError("Condition contains nonfinite values")
        return tuple(hidden.shape[:-1])

    def distribution(self, hidden: Tensor) -> ByteGlyphDistribution:
        self._validate_hidden(hidden)
        return ByteGlyphDistribution(self, hidden)

    def teacher_forced_logits(self, hidden: Tensor, targets: Tensor) -> Tensor:
        """Predict all 128 bytes from [BOS,b0,...,b126], never the unshifted target."""
        leading = self._validate_hidden(hidden)
        byte_targets = pack_glyph_bytes(targets)
        if byte_targets.shape != (*leading, GRID_BYTES) or targets.device != hidden.device:
            raise ValueError("Targets must match the condition batch and device")
        beginning = torch.full((*leading, 1), BYTE_BOS, dtype=torch.long, device=hidden.device)
        shifted = torch.cat((beginning, byte_targets[..., :-1].long()), dim=-1)
        return self(hidden, shifted)

    def forward(
        self,
        hidden: Tensor,
        input_bytes: Tensor,
        *,
        cache: ByteDecoderCache | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, ByteDecoderCache]:
        """Consume already shifted byte inputs; cached decoding is local to one complete grid.

        With no cache, the first input must be BOS. With a cache, inputs are the
        subsequent realized bytes. Output position j predicts byte j. No padding
        or attention between different flattened condition rows is introduced.
        """
        leading = self._validate_hidden(hidden)
        _validate_bytes(input_bytes, allow_bos=True)
        if tuple(input_bytes.shape[:-1]) != leading or input_bytes.device != hidden.device:
            raise ValueError("Byte inputs must match the condition batch and device")
        tokens = input_bytes.long()
        offset = 0 if cache is None else cache.length
        length = input_bytes.shape[-1]
        if offset < 0 or offset + length > GRID_BYTES or (cache is not None and offset == 0):
            raise ValueError("Each glyph has exactly 128 inner prediction positions")
        if cache is None:
            if not bool((tokens[..., 0] == BYTE_BOS).all()):
                raise ValueError("An inner sequence must start with byte-BOS")
            remainder = tokens[..., 1:]
            condition = self.condition_projection(
                hidden.reshape(-1, self.hidden_size).to(self.condition_projection.weight.dtype)
            )
        else:
            if (
                cache.owner != id(self)
                or cache.leading_shape != leading
                or cache.condition.device != hidden.device
                or not torch.equal(cache.condition, hidden.detach())
                or len(cache.layers) != len(self.blocks)
                or any(
                    key.shape[-2] != offset or value.shape != key.shape
                    for key, value in cache.layers
                )
            ):
                raise ValueError("Cache belongs to a different decoder, condition or prefix")
            remainder = tokens
            condition = cache.condition_embedding
        if not bool((remainder < BYTE_VALUES).all()):
            raise ValueError("Byte-BOS is permitted only at inner position zero")
        positions = torch.arange(offset, offset + length, device=hidden.device)
        embeddings = self.byte_embedding(tokens.reshape(-1, length))
        embeddings = embeddings + self.position_embedding(positions) + condition.unsqueeze(1)
        current = []
        for index, block in enumerate(self.blocks):
            embeddings, keys_values = block(
                embeddings, None if cache is None else cache.layers[index]
            )
            if use_cache:
                current.append(keys_values)
        logits = self.byte_head(self.final_norm(embeddings)).reshape(*leading, length, BYTE_VALUES)
        if not use_cache:
            return logits
        next_cache = ByteDecoderCache(
            owner=id(self),
            condition=hidden.detach().clone() if cache is None else cache.condition,
            condition_embedding=condition,
            leading_shape=leading,
            length=offset + length,
            layers=tuple(current),
        )
        return logits, next_cache

    @torch.no_grad()
    def generate(
        self,
        hidden: Tensor,
        *,
        strategy: str = "greedy",
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        use_cache: bool = True,
    ) -> Tensor:
        """Generate exactly 128 bytes, then unpack one binary grid per condition.

        Sampling uses a categorical draw for the whole eight-bit block. Greedy
        decoding requires temperature=1 because a sampling temperature would be
        ignored by argmax. A fresh cache is created for every call; no outer LM
        cache, glyph labels, inventory projection or font rendering participates.
        """
        leading = self._validate_hidden(hidden)
        if strategy not in {"greedy", "sample"}:
            raise ValueError("Byte decode strategy must be greedy or sample")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Sampling temperature must be finite and positive")
        if strategy == "greedy" and temperature != 1.0:
            raise ValueError("Temperature applies only to sample strategy")
        was_training = self.training
        self.eval()
        try:
            prefix = torch.full((*leading, 1), BYTE_BOS, dtype=torch.long, device=hidden.device)
            generated = []
            cache = None
            for _ in range(GRID_BYTES):
                if use_cache:
                    logits, cache = self(hidden, prefix[..., -1:], cache=cache, use_cache=True)
                else:
                    logits = self(hidden, prefix)
                scores = logits[..., -1, :].float()
                if not bool(torch.isfinite(scores).all()):
                    raise FloatingPointError("Cannot sample a byte from nonfinite logits")
                if strategy == "greedy":
                    value = scores.argmax(-1, keepdim=True)
                else:
                    probabilities = (scores / temperature).softmax(-1).reshape(-1, BYTE_VALUES)
                    value = torch.multinomial(probabilities, 1, generator=generator).reshape(
                        *leading, 1
                    )
                generated.append(value.to(torch.uint8))
                prefix = torch.cat((prefix, value), dim=-1)
            return unpack_glyph_bytes(torch.cat(generated, dim=-1))
        finally:
            self.train(was_training)
