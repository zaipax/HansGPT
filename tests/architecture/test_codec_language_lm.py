from types import SimpleNamespace

import torch
from torch import nn

from hansgpt_research.codec_language_lm import CodecAlignedGlyphGPT, latent_contrastive_loss


def test_repeated_glyphs_are_not_false_negatives_and_targets_are_detached():
    predicted = torch.randn(3, 4, 256, requires_grad=True)
    targets = torch.randn(3, 4, 256, requires_grad=True)
    groups = torch.zeros(3, dtype=torch.long)
    loss = latent_contrastive_loss(predicted, targets, groups)
    torch.testing.assert_close(loss, torch.zeros_like(loss), rtol=0, atol=0)
    loss.sum().backward()
    assert targets.grad is None


def test_matching_latents_score_better_than_permuted_targets():
    torch.manual_seed(4)
    targets = torch.randn(4, 4, 256)
    groups = torch.arange(4)
    aligned = latent_contrastive_loss(targets, targets, groups).mean()
    wrong = latent_contrastive_loss(targets.roll(1, 0), targets, groups).mean()
    assert aligned < wrong


def test_alignment_gradients_reach_only_semantic_path():
    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.glyph_encoder = nn.Linear(4, 4)
            self.backbone = nn.Linear(4, 4)
            self.semantic_decoder = nn.Linear(4, 4)

        def forward_hidden(self, x):
            return self.semantic_decoder(self.backbone(self.glyph_encoder(x)))

    class Codec(nn.Module):
        slots = 4

        def __init__(self):
            super().__init__()
            self.decoder = nn.Linear(1024, 1024)

        def decode(self, z):
            return self.decoder(z.flatten(-2)).reshape(-1, 1, 32, 32)

    model = CodecAlignedGlyphGPT(Base(), Codec()).train()
    hidden = model.forward_hidden(torch.randn(2, 4))
    model.distribution(hidden).pixel_logits.square().mean().backward()
    assert all(p.grad is None for p in model.base.glyph_encoder.parameters())
    assert all(p.grad is None for p in model.base.backbone.parameters())
    assert all(p.grad is None for p in model.codec.parameters())
    assert model.mapping.weight.grad.abs().sum() > 0
    assert model.base.semantic_decoder.weight.grad.abs().sum() > 0
