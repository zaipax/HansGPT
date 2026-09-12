"""Matched A/B/C glyph experiments: patch attention and conditional byte attention."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from hansgpt_research.byte_glyph_decoder import ConditionalByteDecoder
from hansgpt_research.glyph_lm import ModelConfig, validate_binary_tiles
from hansgpt_research.structured_glyph_lm import StructuredGlyphGPT

GPU_BY_VARIANT = {"A": 4, "B": 5, "C": 6, "D": 5}


class AttentionGlyphEncoder(nn.Module):
    """Independent bidirectional 4x4 patch attention per tile; no convolution or IDs."""

    def __init__(self, hidden_size, width=128, layers=4, heads=4):
        super().__init__()
        self.patch_projection = nn.Linear(16, width)
        self.row = nn.Parameter(torch.randn(8, width) * 0.02)
        self.column = nn.Parameter(torch.randn(8, width) * 0.02)
        self.glyph_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                width,
                heads,
                width * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(layers)
        )
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, hidden_size, bias=False)
        self.output_norm = nn.RMSNorm(hidden_size)
        self.checkpoint_blocks = False

    def forward(self, tiles: Tensor) -> Tensor:
        # [N,row,patch-row,column,patch-column] -> row-major patch sequence.
        patches = tiles.reshape(-1, 8, 4, 8, 4).permute(0, 1, 3, 2, 4).reshape(-1, 64, 16)
        patches = self.patch_projection(patches.to(self.patch_projection.weight.dtype))
        positions = (self.row[:, None] + self.column[None, :]).reshape(1, 64, -1)
        hidden = torch.cat((self.glyph_token.expand(len(tiles), -1, -1), patches + positions), 1)
        for block in self.blocks:
            if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return self.output_norm(self.projection(self.norm(hidden[:, 0])))


class AttentionGlyphGPT(StructuredGlyphGPT):
    """A=attention/pixels, B=CNN/bytes, C=attention/bytes; all jointly trainable."""

    def __init__(self, config: ModelConfig, variant: str, encoder=None, decoder=None):
        if variant not in {"A", "B", "C"}:
            raise ValueError("variant must be A, B or C")
        # Identical seeds give every variant the identical initial outer backbone.
        super().__init__(config, components=1, init_noise=0.0)
        self.variant = variant
        # Isolate replacement-module RNG so B/C byte decoders also start identically.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed() + 101)
            if variant in {"A", "C"}:
                self.glyph_encoder = AttentionGlyphEncoder(config.hidden_size, **(encoder or {}))
        self.byte_decoder = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed() + 211)
            if variant in {"B", "C"}:
                self.pixel_head = None
                self.byte_decoder = ConditionalByteDecoder(config.hidden_size, **(decoder or {}))

    def gradient_checkpointing_enable(self):
        super().gradient_checkpointing_enable()
        if isinstance(self.glyph_encoder, AttentionGlyphEncoder):
            self.glyph_encoder.checkpoint_blocks = True

    def distribution(self, hidden):
        if self.byte_decoder is not None:
            return self.byte_decoder.distribution(hidden)
        return super().distribution(hidden)

    def decode_grid(self, hidden, threshold=0.5):
        distribution = self.distribution(hidden)
        if self.byte_decoder is not None:
            return distribution.decode(strategy="greedy")
        return distribution.decode(threshold=threshold, strategy="mode_threshold")

    @torch.no_grad()
    def generate(
        self, prompt_glyphs, max_new_tokens, *, threshold=0.5, eos_glyph=None, use_cache=True
    ):
        validate_binary_tiles(prompt_glyphs, name="prompt_glyphs")
        if max_new_tokens < 0 or not 0 < threshold < 1:
            raise ValueError("Invalid generation length or threshold")
        if eos_glyph is not None:
            eos_glyph = eos_glyph.to(prompt_glyphs.device).reshape(1, 1, 1, 32, 32)
            validate_binary_tiles(eos_glyph)
        was_training = self.training
        self.eval()
        try:
            context = prompt_glyphs.to(torch.uint8)
            current = context[:, -self.config.max_position_embeddings :]
            cache = None
            generated = []
            finished = torch.zeros(len(context), dtype=torch.bool, device=context.device)
            for _ in range(max_new_tokens):
                hidden, next_cache = self.forward_hidden(
                    current, past_key_values=cache, use_cache=use_cache, return_cache=True
                )
                if not bool(torch.isfinite(hidden).all()):
                    raise FloatingPointError("Nonfinite generation condition")
                tile = self.decode_grid(hidden[:, -1:], threshold)
                tile[finished] = 0
                generated.append(tile)
                if eos_glyph is not None:
                    finished |= (tile == eos_glyph).flatten(1).all(1)
                    if bool(finished.all()):
                        break
                context = torch.cat((context, tile), 1)
                if use_cache and context.shape[1] <= self.config.max_position_embeddings:
                    cache, current = next_cache, tile
                else:
                    cache, current = None, context[:, -self.config.max_position_embeddings :]
            return torch.cat(generated, 1) if generated else context[:, :0]
        finally:
            self.train(was_training)
