"""Align a semantic predictor to a pinned immutable glyph codec."""

import torch
import torch.nn.functional as F
from torch import nn

from hansgpt_research.structured_glyph_lm import GlyphDistribution, StructuredGlyphGPT


class CodecAlignedGlyphGPT(nn.Module):
    """Keep the original input encoder/GPT fixed; learn the semantic output interface.

    Future glyphs are supervision only and never arguments to prediction/generation.
    The codec remains frozen but propagates gradients to predicted latent vectors.
    """

    def __init__(self, base, codec):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.codec = codec.requires_grad_(False)
        self.base.semantic_decoder.requires_grad_(True)
        self.config = base.config
        self.byte_decoder = None
        self.mapping = nn.Linear(self.config.hidden_size, codec.slots * 256)
        self.latent_norm = nn.RMSNorm(256)

    def train(self, mode=True):
        super().train(mode)
        self.base.glyph_encoder.eval()
        self.base.backbone.eval()
        self.codec.eval()
        return self

    def forward_hidden(self, *args, **kwargs):
        return self.base.forward_hidden(*args, **kwargs)

    def latents(self, hidden):
        shape = (*hidden.shape[:-1], self.codec.slots, 256)
        return self.latent_norm(self.mapping(hidden).reshape(shape))

    def distribution(self, hidden):
        pixels = self.codec.decode(self.latents(hidden)).reshape(*hidden.shape[:-1], 1, 1024)
        return GlyphDistribution(pixels, pixels.new_zeros((*hidden.shape[:-1], 1)))

    generate = StructuredGlyphGPT.generate

    def decode_grid(self, hidden, threshold=0.5):
        return self.distribution(hidden).decode(threshold=threshold, strategy="mode_threshold")


def latent_contrastive_loss(predicted, targets, groups, temperature=0.1):
    """In-batch image-latent supervision, with bitmap aliases as multiple positives.

    No vocabulary or glyph lookup participates in inference. Targets must be
    detached frozen-codec representations. Repeated glyphs are not false negatives.
    """
    left = F.normalize(predicted.float().flatten(1), dim=-1)
    right = F.normalize(targets.detach().float().flatten(1), dim=-1)
    scores = left @ right.T / temperature
    positives = groups[:, None] == groups[None, :]
    return torch.logsumexp(scores, dim=1) - torch.logsumexp(
        scores.masked_fill(~positives, -torch.inf), dim=1
    )
