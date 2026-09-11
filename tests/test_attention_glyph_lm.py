"""Attention ablation boundaries, causal gradients and raw-image feedback."""

import copy

import pytest
import torch
from torch import nn

from hansgpt_research.attention_glyph_lm import AttentionGlyphGPT
from hansgpt_research.glyph_lm import ModelConfig
from hansgpt_research.train_structured_glyph_lm import generator_backward


def model_for(variant, deduplicate=True):
    return AttentionGlyphGPT(
        ModelConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            max_position_embeddings=16,
            deduplicate_glyphs=deduplicate,
            glyph_deduplication_strategy="packed",
        ),
        variant,
        {"width": 16, "layers": 1, "heads": 2},
        {"inner_dim": 16, "layers": 1, "heads": 2, "intermediate_size": 32},
    )


def tiles(length=3):
    return torch.randint(0, 2, (1, length, 1, 32, 32), dtype=torch.uint8)


def test_matched_initialization_and_no_cnn_or_unused_pixel_head():
    models = []
    for variant in "ABC":
        torch.manual_seed(71)
        models.append(model_for(variant))
    a, b, c = models
    for name, value in a.backbone.state_dict().items():
        torch.testing.assert_close(value, b.backbone.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(value, c.backbone.state_dict()[name], rtol=0, atol=0)
    for name, value in b.byte_decoder.state_dict().items():
        torch.testing.assert_close(value, c.byte_decoder.state_dict()[name], rtol=0, atol=0)
    for model in (a, c):
        assert not any(isinstance(module, nn.Conv2d) for module in model.modules())
        assert model.backbone.embed_tokens is None
    assert any(isinstance(module, nn.Conv2d) for module in b.modules())
    assert b.pixel_head is None and c.pixel_head is None


@pytest.mark.parametrize("variant", list("ABC"))
def test_outer_prefix_is_causal_and_encoder_receives_likelihood_gradient(variant):
    model = model_for(variant).eval()
    inputs = tiles()
    changed = inputs.clone()
    changed[:, -1] = 1 - changed[:, -1]
    original = model.forward_hidden(inputs)
    altered = model.forward_hidden(changed)
    torch.testing.assert_close(original[:, :-1], altered[:, :-1], atol=2e-5, rtol=2e-5)
    model.distribution(original).nll(tiles()).mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.glyph_encoder.parameters()
    )
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())


@pytest.mark.parametrize("variant", list("ABC"))
def test_chunked_head_backward_matches_full_likelihood(variant):
    model = model_for(variant).train()
    reference = copy.deepcopy(model)
    batch = {
        "glyphs": tiles(),
        "targets": tiles(),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "loss_mask": torch.tensor([[True, True, False]]),
    }
    hidden = reference.forward_hidden(batch["glyphs"], batch["attention_mask"])
    expected = (
        reference.distribution(hidden[batch["loss_mask"]])
        .nll(batch["targets"][batch["loss_mask"]])
        .mean()
    )
    expected.backward()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    result = generator_backward(
        model,
        None,
        [batch],
        {"head_chunk_size": 1, "precision": "fp32", "adversarial_weight": 0},
        torch.device("cpu"),
        scaler,
        2,
        frozen=False,
    )
    assert result["nll_sum"] / 2 == pytest.approx(float(expected.detach()), rel=1e-5)
    for (name, actual), (_, wanted) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert actual.grad is not None, name
        torch.testing.assert_close(actual.grad, wanted.grad, atol=2e-5, rtol=2e-4)


def test_patch_deduplication_preserves_values_and_gradients():
    model = model_for("C")
    other = model_for("C", deduplicate=False)
    other.load_state_dict(model.state_dict())
    inputs = tiles(2).repeat(1, 2, 1, 1, 1)
    actual, expected = model.encode_glyphs(inputs), other.encode_glyphs(inputs)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for p, q in zip(
        model.glyph_encoder.parameters(), other.glyph_encoder.parameters(), strict=True
    ):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-4, rtol=2e-3)


@pytest.mark.parametrize("variant", list("ABC"))
def test_generation_caches_match_and_feedback_is_original_image(variant):
    model = model_for(variant).eval()
    prompt = tiles(2)
    observed = []
    hook = model.glyph_encoder.register_forward_pre_hook(
        lambda module, args: observed.append(args[0].detach().clone())
    )
    cached = model.generate(prompt, 2, use_cache=True)
    hook.remove()
    uncached = model.generate(prompt, 2, use_cache=False)
    assert cached.dtype == torch.uint8 and bool(((cached == 0) | (cached == 1)).all())
    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)
    torch.testing.assert_close(observed[-1], cached[:, 0], rtol=0, atol=0)
