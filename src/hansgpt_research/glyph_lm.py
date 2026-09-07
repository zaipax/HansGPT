"""A language model whose only content inputs and targets are binary glyph tiles.

The uint16 values in the corpus are *asset addresses*, never model features. The
dataset resolves them to pixels before returning a batch. Every model position
predicts the next full 32 by 32 tile, including document control tiles.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import Dataset, default_collate
from transformers import LlamaConfig, LlamaModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    intermediate_size: int = 2048
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    deduplicate_glyphs: bool = True
    glyph_encode_chunk_size: int = 256

    def __post_init__(self) -> None:
        dimensions = (
            self.hidden_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.intermediate_size,
            self.max_position_embeddings,
            self.glyph_encode_chunk_size,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("All model dimensions must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Query head count must be divisible by KV head count")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelConfig:
        return cls(**values)


def validate_binary_tiles(tiles: Tensor, *, name: str = "glyphs") -> None:
    """Reject grayscale, NaN, wrong-sized and empty sequence image inputs."""
    if tiles.ndim != 5 or tuple(tiles.shape[-3:]) != (1, 32, 32):
        raise ValueError(f"{name} must have shape [batch, sequence, 1, 32, 32]")
    if not tiles.shape[0] or not tiles.shape[1]:
        raise ValueError(f"{name} must contain at least one tile")
    if not bool(((tiles == 0) | (tiles == 1)).all()):
        raise ValueError(f"{name} must contain only binary 0/1 pixels")


class GlyphEncoder(nn.Module):
    """Shared per-tile CNN; GroupNorm never aggregates across sequence positions."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for input_channels, output_channels, stride in (
            (1, 32, 1),
            (32, 64, 2),
            (64, 128, 2),
            (128, 128, 2),
        ):
            layers.extend(
                [
                    nn.Conv2d(input_channels, output_channels, 3, stride, 1, bias=False),
                    nn.GroupNorm(8, output_channels),
                    nn.SiLU(),
                ]
            )
        self.convolutions = nn.Sequential(*layers)
        self.projection = nn.Linear(2048, config.hidden_size, bias=False)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, tiles: Tensor) -> Tensor:
        features = self.convolutions(tiles.to(dtype=self.projection.weight.dtype))
        return self.norm(self.projection(features.flatten(1)))


class GlyphGPT(nn.Module):
    """Randomly initialized glyph CNN + causal GQA Transformer + pixel head.

    ``forward`` returns logits [B,T,1,32,32], without shifting labels. There is no
    character embedding or classification head. The same CNN handles teacher
    forcing, cached generation, and generated images absent from the glyph bank.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.glyph_encoder = GlyphEncoder(config)
        transformer_config = LlamaConfig(
            vocab_size=1,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            initializer_range=config.initializer_range,
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=False,
            use_cache=False,
            _attn_implementation="sdpa",
        )
        self.backbone = LlamaModel(transformer_config)
        # Passing inputs_embeds alone would leave an unused character embedding.
        # Remove the module itself so checkpoints and parameter counts are honest.
        self.backbone.embed_tokens = None
        self.pixel_head = nn.Linear(config.hidden_size, 1024, bias=True)
        nn.init.normal_(self.pixel_head.weight, std=config.initializer_range)
        nn.init.zeros_(self.pixel_head.bias)

    def gradient_checkpointing_enable(self) -> None:
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def encode_glyphs(self, glyphs: Tensor) -> Tensor:
        validate_binary_tiles(glyphs)
        batch_size, length = glyphs.shape[:2]
        flat_tiles = glyphs.reshape(-1, 1, 32, 32)
        inverse = None
        if self.config.deduplicate_glyphs and not glyphs.requires_grad:
            # Uniqueness is decided from pixels, not corpus IDs. Gather backward
            # sums every occurrence's gradient into the shared CNN computation.
            # This cache lives for this forward only; it never survives an update.
            unique_pixels, inverse = torch.unique(
                flat_tiles.reshape(-1, 1024).to(torch.uint8), dim=0, return_inverse=True
            )
            flat_tiles = unique_pixels.reshape(-1, 1, 32, 32)
        encoded = torch.cat(
            [
                self.glyph_encoder(tiles)
                for tiles in flat_tiles.split(self.config.glyph_encode_chunk_size)
            ],
            dim=0,
        )
        if inverse is not None:
            encoded = encoded[inverse]
        return encoded.reshape(batch_size, length, self.config.hidden_size)

    def forward(
        self,
        glyphs: Tensor,
        attention_mask: Tensor | None = None,
        *,
        past_key_values: Any = None,
        use_cache: bool = False,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, Any]:
        embeddings = self.encode_glyphs(glyphs)
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != glyphs.shape[0]:
                raise ValueError("attention_mask must be [batch, total_context_length]")
            if past_key_values is None and attention_mask.shape[1] != glyphs.shape[1]:
                raise ValueError("attention_mask must match the input sequence length")
            if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
                raise ValueError("attention_mask must be binary")
        outputs = self.backbone(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        logits = self.pixel_head(outputs.last_hidden_state).reshape(*glyphs.shape)
        if return_cache:
            return logits, outputs.past_key_values
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_glyphs: Tensor,
        max_new_tokens: int,
        *,
        threshold: float = 0.5,
        eos_glyph: Tensor | None = None,
        use_cache: bool = True,
    ) -> Tensor:
        """Return generated binary uint8 tiles only, with shape [B,N,1,32,32].

        Prompts must be unpadded. EOS is recognized by exact bitmap equality;
        finished batch rows are padded with zero tiles. If context capacity is
        reached, recompute a sliding context and discard the obsolete KV cache.
        No nearest-glyph projection, OCR, or corpus lookup enters feedback.
        """
        validate_binary_tiles(prompt_glyphs, name="prompt_glyphs")
        if max_new_tokens < 0 or not 0 < threshold < 1:
            raise ValueError("max_new_tokens must be nonnegative and threshold in (0, 1)")
        if eos_glyph is not None:
            eos_glyph = eos_glyph.to(device=prompt_glyphs.device).reshape(1, 1, 1, 32, 32)
            validate_binary_tiles(eos_glyph, name="eos_glyph")
        was_training = self.training
        self.eval()
        try:
            context = prompt_glyphs.to(torch.uint8)
            generated: list[Tensor] = []
            finished = torch.zeros(context.shape[0], dtype=torch.bool, device=context.device)
            cache = None
            model_input = context[:, -self.config.max_position_embeddings :]
            for _ in range(max_new_tokens):
                logits, next_cache = self.forward(
                    model_input,
                    past_key_values=cache,
                    use_cache=use_cache,
                    return_cache=True,
                )
                tile = (logits[:, -1:].float().sigmoid() >= threshold).to(torch.uint8)
                tile[finished] = 0
                generated.append(tile)
                if eos_glyph is not None:
                    finished |= (tile == eos_glyph).flatten(1).all(1)
                    if bool(finished.all()):
                        break
                context = torch.cat((context, tile), dim=1)
                if use_cache and context.shape[1] <= self.config.max_position_embeddings:
                    cache = next_cache
                    model_input = tile
                else:
                    # Correct recomputation fallback also supports contexts longer
                    # than the configured RoPE window without stale cache positions.
                    cache = None
                    model_input = context[:, -self.config.max_position_embeddings :]
            if not generated:
                return context[:, :0]
            return torch.cat(generated, dim=1)
        finally:
            self.train(was_training)


def pixel_bce_loss(logits: Tensor, targets: Tensor, loss_mask: Tensor) -> Tensor:
    """Unweighted mean pixel BCE over valid next-tile targets, with no shift."""
    validate_binary_tiles(targets, name="targets")
    if logits.shape != targets.shape or loss_mask.shape != targets.shape[:2]:
        raise ValueError("Logits, targets and loss mask shapes do not align")
    if not bool(((loss_mask == 0) | (loss_mask == 1)).all()):
        raise ValueError("loss_mask must be binary")
    count = loss_mask.sum()
    if not bool(count > 0):
        raise ValueError("At least one valid next-tile target is required")
    loss_per_tile = F.binary_cross_entropy_with_logits(
        logits.float(), targets.float(), reduction="none"
    ).mean(dim=(-3, -2, -1))
    return (loss_per_tile * loss_mask).sum() / count


def collate_glyph_sequences(
    samples: list[dict[str, Tensor]], *, pad_to_multiple_of: int = 8
) -> dict[str, Tensor]:
    """Stack document chunks and trim only padding shared by the whole batch.

    Rounding to eight preserves tensor-core-friendly sizes without extending the
    dataset context cap. Input and target masks both determine the final used
    position; no valid target or context tile is dropped and documents stay rows.
    """
    if not samples or pad_to_multiple_of < 1:
        raise ValueError("Need a nonempty batch and a positive padding multiple")
    batch = default_collate(samples)
    attention_mask, loss_mask = batch["attention_mask"], batch["loss_mask"]
    if attention_mask.ndim != 2 or loss_mask.shape != attention_mask.shape:
        raise ValueError("Batch attention and loss masks must be aligned [batch, sequence]")
    used_positions = (attention_mask.bool() | loss_mask.bool()).any(dim=0).nonzero()
    if not len(used_positions):
        raise ValueError("Cannot collate a batch containing only padding")
    context_cap = attention_mask.shape[1]
    used_length = int(used_positions[-1, 0]) + 1
    rounded_length = (used_length + pad_to_multiple_of - 1) // pad_to_multiple_of
    length = min(rounded_length * pad_to_multiple_of, context_cap)
    return {
        name: value[:, :length].contiguous()
        if value.ndim >= 2 and value.shape[1] == context_cap
        else value
        for name, value in batch.items()
    }


class GlyphSequenceDataset(Dataset[dict[str, Tensor]]):
    """Document-isolated fixed-length chunks of an on-disk glyph asset stream.

    Every document is stored as [BOS, content..., EOS]. A document of N tiles
    contributes exactly N-1 next-tile targets. Adjacent chunks overlap the one
    tile needed as the next chunk's first input, so no chunk-edge target is lost.
    Only the first chunk has BOS; no context or target crosses documents.
    """

    def __init__(self, data_dir: str | Path, split: str, sequence_length: int) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.token_path = self.data_dir / f"{split}.uint16"
        with np.load(self.data_dir / "glyph_bank.npz", allow_pickle=False) as bank:
            bitmaps = bank["bitmaps"]
        if bitmaps.ndim == 3:
            bitmaps = bitmaps[:, None]
        if bitmaps.shape[1:] != (1, 32, 32) or not np.isin(bitmaps, [0, 1]).all():
            raise ValueError("glyph_bank must contain [assets, 1, 32, 32] binary tiles")
        self.glyph_bank = torch.from_numpy(np.array(bitmaps, dtype=np.uint8, copy=True))
        self.inventory = json.loads((self.data_dir / "glyph_inventory.json").read_text("utf-8"))
        self.control_ids = {name: int(index) for index, name in self.inventory["controls"].items()}
        if any(name not in self.control_ids for name in ("PAD", "BOS", "EOS")):
            raise ValueError("Corpus must define PAD, BOS and EOS control tiles")
        controls = set(self.control_ids.values())
        if not controls.issubset(range(len(self.glyph_bank))):
            raise ValueError("Control asset address lies outside glyph_bank")
        self.gallery_ids = [index for index in range(len(self.glyph_bank)) if index not in controls]
        manifest_path = self.data_dir / "manifest.json"
        self.metadata = (
            json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
        )
        self.offsets = np.load(self.data_dir / f"{split}.offsets.npy", allow_pickle=False)
        if self.token_path.stat().st_size % 2:
            raise ValueError("Asset stream length must be divisible by uint16 width")
        self.tokens = np.memmap(self.token_path, mode="r", dtype="<u2")
        if (
            self.offsets.ndim != 1
            or self.offsets.dtype.kind not in "iu"
            or len(self.offsets) < 2
            or self.offsets[0] != 0
            or self.offsets[-1] != len(self.tokens)
            or np.any(np.diff(self.offsets) < 2)
        ):
            raise ValueError("Offsets must delimit complete nonempty BOS/EOS documents")
        if int(self.tokens.max()) >= len(self.glyph_bank):
            raise ValueError("Asset stream contains an address outside glyph_bank")
        if not np.all(self.tokens[self.offsets[:-1]] == self.control_ids["BOS"]):
            raise ValueError("Every stored document must begin with BOS")
        if not np.all(self.tokens[self.offsets[1:] - 1] == self.control_ids["EOS"]):
            raise ValueError("Every stored document must end with EOS")
        self.target_count = int(np.sum(np.diff(self.offsets) - 1))
        chunk_counts = (np.diff(self.offsets) - 2) // sequence_length + 1
        self.chunk_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(chunk_counts)))

    def __len__(self) -> int:
        return int(self.chunk_offsets[-1])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        document = int(np.searchsorted(self.chunk_offsets, index, side="right") - 1)
        chunk = index - int(self.chunk_offsets[document])
        start = int(self.offsets[document]) + chunk * self.sequence_length
        stop = min(start + self.sequence_length + 1, int(self.offsets[document + 1]))
        ids = np.array(self.tokens[start:stop], dtype=np.int64, copy=True)
        valid = len(ids) - 1
        input_ids = torch.full((self.sequence_length,), self.control_ids["PAD"], dtype=torch.long)
        target_ids = input_ids.clone()
        input_ids[:valid] = torch.from_numpy(ids[:-1])
        target_ids[:valid] = torch.from_numpy(ids[1:])
        mask = torch.arange(self.sequence_length) < valid
        return {
            "glyphs": self.glyph_bank[input_ids],
            "targets": self.glyph_bank[target_ids],
            "attention_mask": mask,
            "loss_mask": mask.clone(),
            # Asset addresses are for scoring/frequency audits, never model input.
            "target_ids": target_ids,
        }
