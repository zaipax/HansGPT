import torch

from hansgpt_research.codec_language_lm import latent_contrastive_loss


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
