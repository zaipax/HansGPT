"""Semantic and spatial Transformer decoders jointly predict a whole binary glyph."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import LlamaConfig, LlamaModel

from hansgpt_research.attention_glyph_lm import AttentionGlyphGPT
from hansgpt_research.glyph_lm import ModelConfig
from hansgpt_research.structured_glyph_lm import GlyphDistribution


@dataclass(frozen=True)
class DualDecoderConfig:
    width: int = 256
    semantic_layers: int = 2
    semantic_slots: int = 4
    glyph_layers: int = 3
    part_slots: int = 16
    heads: int = 8
    intermediate_size: int = 768

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in vars(self).values()):
            raise ValueError("All decoder dimensions must be positive integers")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("Decoder width needs an even, integral attention head dimension")


def patches_to_grid(patches: Tensor) -> Tensor:
    """Row-major 64x(4x4) output patches -> one [1,32,32] grid, no lookup/iteration."""
    if patches.shape[-2:] != (64, 16):
        raise ValueError("Expected 64 spatial positions with 16 pixel logits each")
    leading = patches.shape[:-2]
    flat = patches.reshape(-1, 8, 8, 4, 4).permute(0, 1, 3, 2, 4)
    return flat.reshape(*leading, 1, 32, 32)


class SemanticDecoder(nn.Module):
    """Causal refinement across outer GPT states; expose several semantic condition slots.

    Own KV cache, shared weights across all outer positions. Current/previous GPT
    states only; no target pixels or target identity enter this module.
    """

    def __init__(self, config: ModelConfig, decoder: DualDecoderConfig):
        super().__init__()
        self.input_projection = nn.Linear(config.hidden_size, decoder.width, bias=False)
        self.transformer = LlamaModel(
            LlamaConfig(
                vocab_size=1,
                hidden_size=decoder.width,
                num_hidden_layers=decoder.semantic_layers,
                num_attention_heads=decoder.heads,
                num_key_value_heads=decoder.heads,
                head_dim=decoder.width // decoder.heads,
                intermediate_size=decoder.intermediate_size,
                max_position_embeddings=config.max_position_embeddings,
                attention_dropout=0.0,
                use_cache=False,
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=None,
                _attn_implementation="sdpa",
            )
        )
        self.transformer.embed_tokens = None
        self.transformer.main_input_name = "inputs_embeds"
        self.output_projection = nn.Linear(config.hidden_size + decoder.width, config.hidden_size)
        self.output_norm = nn.RMSNorm(config.hidden_size)

    def forward(self, hidden, attention_mask=None, past_key_values=None, use_cache=False):
        output = self.transformer(
            inputs_embeds=self.input_projection(hidden),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        # Preserve outer context alongside semantic refinement, without another vocabulary.
        slots = self.output_norm(
            self.output_projection(torch.cat((hidden, output.last_hidden_state), -1))
        )
        return slots, output.past_key_values


class SpatialGlyphDecoder(nn.Module):
    """Part slots and 64 spatial queries jointly attend to semantic slots in one forward.

    Part slots are learned latent vectors, not claimed to be identified radicals.
    Self-attention is bidirectional only within this next glyph. Cross-attention
    reads semantic conditions. No autoregressive bytes, masking schedule or GAN.
    """

    def __init__(self, decoder: DualDecoderConfig):
        super().__init__()
        self.config = decoder
        self.parts = nn.Parameter(torch.randn(1, decoder.part_slots, decoder.width) * 0.02)
        self.rows = nn.Parameter(torch.randn(8, decoder.width) * 0.02)
        self.columns = nn.Parameter(torch.randn(8, decoder.width) * 0.02)
        self.slot_positions = nn.Parameter(
            torch.randn(1, decoder.semantic_slots, decoder.width) * 0.02
        )
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    decoder.width,
                    decoder.heads,
                    decoder.intermediate_size,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(decoder.glyph_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(decoder.width)
        self.pixel_projection = nn.Linear(decoder.width, 16)
        nn.init.normal_(self.pixel_projection.weight, std=0.02)
        nn.init.zeros_(self.pixel_projection.bias)

    def forward(self, semantic_vectors: Tensor):
        cfg = self.config
        memory = semantic_vectors.reshape(-1, cfg.semantic_slots, cfg.width)
        memory = memory + self.slot_positions
        spatial = (self.rows[:, None] + self.columns[None, :]).reshape(1, 64, cfg.width)
        queries = torch.cat((self.parts, spatial), 1).expand(len(memory), -1, -1)
        # A shared global condition initializes every query; spatial IDs distinguish their jobs.
        queries = queries + memory.mean(1, keepdim=True)
        for layer in self.layers:
            queries = layer(queries, memory)
        patches = self.pixel_projection(self.final_norm(queries[:, cfg.part_slots :]))
        return patches_to_grid(patches).reshape(*semantic_vectors.shape[:-1], 1, 32, 32)


class DualDecoderGlyphGPT(AttentionGlyphGPT):
    """Pure Transformers, two decoder stages, one complete next-glyph prediction per step."""

    def __init__(self, config: ModelConfig, encoder=None, decoders=None):
        decoder = DualDecoderConfig(**(decoders or {}))
        if decoder.width * decoder.semantic_slots != config.hidden_size:
            raise ValueError("semantic_slots * width must match the outer hidden width")
        super().__init__(config, "A", encoder=encoder)
        self.variant = "D"
        self.pixel_head = None
        self.semantic_decoder = SemanticDecoder(config, decoder)
        self.glyph_decoder = SpatialGlyphDecoder(decoder)

    def gradient_checkpointing_enable(self):
        super().gradient_checkpointing_enable()
        self.semantic_decoder.transformer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def forward_hidden(
        self, glyphs, attention_mask=None, past_key_values=None, use_cache=False, return_cache=False
    ):
        if past_key_values is not None and not isinstance(past_key_values, tuple):
            raise ValueError("Dual decoder requires an outer/semantic cache pair")
        outer_cache, semantic_cache = past_key_values or (None, None)
        hidden, outer_cache = super().forward_hidden(
            glyphs,
            attention_mask=attention_mask,
            past_key_values=outer_cache,
            use_cache=use_cache,
            return_cache=True,
        )
        semantic, semantic_cache = self.semantic_decoder(
            hidden, attention_mask, semantic_cache, use_cache
        )
        return (semantic, (outer_cache, semantic_cache)) if return_cache else semantic

    def distribution(self, hidden):
        pixels = self.glyph_decoder(hidden).reshape(*hidden.shape[:-1], 1, 1024)
        return GlyphDistribution(pixels, pixels.new_zeros((*hidden.shape[:-1], 1)))


def make_dual_model(config):
    """Keep model construction identical in training, benchmarks and later evaluation."""
    return DualDecoderGlyphGPT(
        ModelConfig.from_dict(copy.deepcopy(config["model"])),
        encoder=config["encoder"],
        decoders=config["decoders"],
    )
