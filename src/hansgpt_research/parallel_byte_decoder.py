"""Predict all 128 glyph bytes together from learned queries and prefix condition."""

import math

import torch
from torch import nn

from hansgpt_research.byte_glyph_decoder import (
    BYTE_VALUES,
    GRID_BYTES,
    ByteGlyphDistribution,
    ConditionalByteDecoder,
    _CausalByteBlock,
    pack_glyph_bytes,
    unpack_glyph_bytes,
)


class ParallelByteDecoder(nn.Module):
    """Independent categorical outputs after joint bidirectional query attention.

    Targets never enter the network. One learned query identifies each byte slot.
    A projected outer prefix state conditions every query before their interaction.
    """

    def __init__(self, hidden_size=768, inner_dim=256, layers=1, heads=8, intermediate_size=768):
        super().__init__()
        dimensions = (hidden_size, inner_dim, layers, heads, intermediate_size)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("Decoder dimensions must be positive integers")
        if inner_dim % heads:
            raise ValueError("inner_dim must be divisible by heads")
        self.hidden_size = hidden_size
        self.inner_dim = inner_dim
        self.num_layers = layers
        self.heads = heads
        self.intermediate_size = intermediate_size
        self.position_embedding = nn.Embedding(GRID_BYTES, inner_dim)
        self.condition_projection = nn.Linear(hidden_size, inner_dim)
        self.blocks = nn.ModuleList(
            [_CausalByteBlock(inner_dim, heads, intermediate_size) for _ in range(layers)]
        )
        self.final_norm = nn.RMSNorm(inner_dim, eps=1e-6)
        self.byte_head = nn.Linear(inner_dim, BYTE_VALUES)
        self.apply(ConditionalByteDecoder._initialize)

    _validate_hidden = ConditionalByteDecoder._validate_hidden

    def get_config(self):
        return dict(
            kind="parallel",
            hidden_size=self.hidden_size,
            inner_dim=self.inner_dim,
            layers=self.num_layers,
            heads=self.heads,
            intermediate_size=self.intermediate_size,
        )

    def parallel_logits(self, hidden):
        """Unchecked fixed-shape core shared by compiled training and inference."""
        condition = self.condition_projection(
            hidden.reshape(-1, self.hidden_size).to(self.condition_projection.weight.dtype)
        )
        positions = torch.arange(GRID_BYTES, device=hidden.device)
        x = self.position_embedding(positions)[None] + condition[:, None]
        for block in self.blocks:
            x = block(x, causal=False)[0]
        return self.byte_head(self.final_norm(x)).reshape(
            *hidden.shape[:-1], GRID_BYTES, BYTE_VALUES
        )

    def forward(self, hidden):
        self._validate_hidden(hidden)
        return self.parallel_logits(hidden)

    def distribution(self, hidden):
        self._validate_hidden(hidden)
        return ByteGlyphDistribution(self, hidden)

    def teacher_forced_logits(self, hidden, targets):
        leading = self._validate_hidden(hidden)
        byte_targets = pack_glyph_bytes(targets)
        if byte_targets.shape != (*leading, GRID_BYTES) or targets.device != hidden.device:
            raise ValueError("Targets must match the condition batch and device")
        return self.parallel_logits(hidden)

    @torch.no_grad()
    def generate(
        self, hidden, *, strategy="greedy", temperature=1.0, generator=None, use_cache=True
    ):
        if strategy not in {"greedy", "sample"}:
            raise ValueError("Byte decode strategy must be greedy or sample")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Sampling temperature must be finite and positive")
        if strategy == "greedy" and temperature != 1.0:
            raise ValueError("Temperature applies only to sample strategy")
        scores = self(hidden).float()
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("Cannot decode nonfinite logits")
        if strategy == "greedy":
            values = scores.argmax(-1)
        else:
            probabilities = (scores / temperature).softmax(-1).reshape(-1, BYTE_VALUES)
            values = torch.multinomial(probabilities, 1, generator=generator).reshape(
                *hidden.shape[:-1], GRID_BYTES
            )
        return unpack_glyph_bytes(values.to(torch.uint8))
