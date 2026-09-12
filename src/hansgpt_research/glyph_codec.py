"""Persistent continuous glyph codec with one explicit encode/decode interface."""

from __future__ import annotations

import copy

import torch
from torch import nn

from hansgpt_research.glyph_lm import validate_binary_tiles


class SpatialQueryEncoder(nn.Module):
    """Four learned queries read patch features; no raw-pixel shortcut to the decoder."""

    def __init__(self, original, slots=4, width=256):
        super().__init__()
        self.patch_projection = copy.deepcopy(original.patch_projection)
        self.row = nn.Parameter(original.row.detach().clone())
        self.column = nn.Parameter(original.column.detach().clone())
        self.glyph_token = nn.Parameter(original.glyph_token.detach().clone())
        self.blocks = copy.deepcopy(original.blocks)
        self.norm = copy.deepcopy(original.norm)
        inner = self.row.shape[-1]
        self.queries = nn.Parameter(torch.randn(1, slots, inner) * 0.02)
        self.pool = nn.TransformerDecoderLayer(
            inner, 4, inner * 4, dropout=0, activation="gelu", batch_first=True, norm_first=True
        )
        self.projection = nn.Linear(inner, width)
        self.output_norm = nn.RMSNorm(width)

    def forward(self, tiles):
        patches = tiles.reshape(-1, 8, 4, 8, 4).permute(0, 1, 3, 2, 4).reshape(-1, 64, 16)
        patches = self.patch_projection(patches.to(self.patch_projection.weight.dtype))
        positions = (self.row[:, None] + self.column[None, :]).reshape(1, 64, -1)
        hidden = torch.cat((self.glyph_token.expand(len(tiles), -1, -1), patches + positions), 1)
        for layer in self.blocks:
            hidden = layer(hidden)
        memory = self.norm(hidden[:, 1:])
        queries = self.pool(self.queries.expand(len(tiles), -1, -1), memory)
        return self.output_norm(self.projection(queries))


class GlyphCodec(nn.Module):
    """All reconstruction and future semantic conditions use [...,4,256] latents.

    This is a glyph module, not a fluent language model. Semantic alignment is a
    separate stage; changing this encoder must never replace a GPT input encoder
    implicitly. The fixed arm's adapter is permanent and checkpointed.
    """

    interface_version = "glyph_latents_4x256_v1"

    def __init__(self, arm, encoder, decoder):
        super().__init__()
        if arm not in {"fixed", "spatial"}:
            raise ValueError("Unknown codec arm")
        if decoder.config.semantic_slots != 4 or decoder.config.width != 256:
            raise ValueError("Codec requires four width-256 slots")
        self.arm = arm
        self.decoder = copy.deepcopy(decoder)
        if arm == "fixed":
            self.encoder = copy.deepcopy(encoder).requires_grad_(False).eval()
            self.adapter = nn.Sequential(nn.Linear(1024, 1024), nn.RMSNorm(1024))
        else:
            self.encoder = SpatialQueryEncoder(encoder)
            self.adapter = nn.Identity()

    def train(self, mode=True):
        super().train(mode)
        if self.arm == "fixed":
            self.encoder.eval()
        return self

    def encode(self, tiles):
        if tiles.ndim < 4 or tiles.shape[-3:] != (1, 32, 32):
            raise ValueError("Expected glyph bitmaps")
        leading = tiles.shape[:-3]
        flat = tiles.reshape(-1, 1, 32, 32)
        validate_binary_tiles(flat.unsqueeze(1))
        if self.arm == "fixed":
            with torch.no_grad():
                features = self.encoder(flat)
            latents = self.adapter(features).reshape(-1, 4, 256)
        else:
            latents = self.encoder(flat)
        return latents.reshape(*leading, 4, 256)

    def decode(self, latents):
        if latents.shape[-2:] != (4, 256):
            raise ValueError("Expected [...,4,256] glyph latents")
        return self.decoder(latents.flatten(-2))

    def forward(self, tiles):
        return self.decode(self.encode(tiles))
