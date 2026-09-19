"""Causal semantic states, one-pass spatial prediction, end-to-end gradients and caches."""

import copy

import pytest
import torch
from torch import nn

from hansgpt_research.dual_decoder_glyph_lm import DualDecoderGlyphGPT, patches_to_grid
from hansgpt_research.glyph_lm import ModelConfig
from hansgpt_research.train_structured_glyph_lm import generator_backward


def small_model():
    torch.manual_seed(109)
    return DualDecoderGlyphGPT(
        ModelConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            max_position_embeddings=8,
        ),
        encoder={"width": 16, "layers": 1, "heads": 2},
        decoders={
            "width": 16,
            "semantic_slots": 2,
            "semantic_layers": 1,
            "glyph_layers": 2,
            "part_slots": 3,
            "heads": 2,
            "intermediate_size": 32,
        },
    )


def tiles(length=3):
    return torch.randint(0, 2, (1, length, 1, 32, 32), dtype=torch.uint8)


def test_spatial_patch_assembly_preserves_every_pixel_coordinate():
    image = torch.arange(1024).reshape(2, 512).reshape(1, 1, 32, 32)
    patches = image.reshape(1, 8, 4, 8, 4).permute(0, 1, 3, 2, 4).reshape(1, 64, 16)
    torch.testing.assert_close(patches_to_grid(patches), image)


def test_all_networks_are_transformers_without_character_embeddings_or_byte_head():
    model = small_model()
    assert not any(isinstance(m, nn.Conv2d) for m in model.modules())
    assert model.backbone.embed_tokens is None
    assert model.semantic_decoder.transformer.embed_tokens is None
    assert model.byte_decoder is None and model.pixel_head is None
    assert model.distribution(model.forward_hidden(tiles())).pixel_logits.shape == (1, 3, 1, 1024)


def test_future_glyphs_cannot_change_earlier_predictions():
    model = small_model().eval()
    inputs = tiles(4)
    changed = inputs.clone()
    changed[:, 2:] = 1 - changed[:, 2:]
    original = model(inputs).pixel_logits
    altered = model(changed).pixel_logits
    torch.testing.assert_close(original[:, :2], altered[:, :2], atol=2e-5, rtol=2e-5)


def test_both_decoders_part_slots_and_encoder_receive_pixel_loss_gradients():
    model = small_model().train()
    model(tiles()).nll(tiles()).mean().backward()
    for label, module in (
        ("input", model.glyph_encoder),
        ("language", model.backbone),
        ("semantic", model.semantic_decoder),
        ("glyph", model.glyph_decoder),
    ):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()), (
            label
        )
    assert model.glyph_decoder.parts.grad.abs().sum() > 0
    assert model.semantic_decoder.transformer.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


def test_chunked_spatial_loss_matches_whole_head_backward():
    model = small_model().train()
    reference = copy.deepcopy(model)
    batch = {
        "glyphs": tiles(),
        "targets": tiles(),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "loss_mask": torch.tensor([[True, True, False]]),
    }
    hidden = reference.forward_hidden(batch["glyphs"], batch["attention_mask"])
    wanted = (
        reference.distribution(hidden[batch["loss_mask"]])
        .nll(batch["targets"][batch["loss_mask"]])
        .mean()
    )
    wanted.backward()
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
    assert result["nll_sum"] / 2 == pytest.approx(float(wanted.detach()), rel=1e-5)
    for (name, p), (_, q) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=3e-4)


def test_checkpointed_and_normal_gradients_match():
    model = small_model().train()
    reference = copy.deepcopy(model)
    model.gradient_checkpointing_enable()
    inputs, targets = tiles(), tiles()
    model(inputs).nll(targets).mean().backward()
    reference(inputs).nll(targets).mean().backward()
    for p, q in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=3e-4)


def test_cache_and_sliding_window_preserve_one_call_per_generated_glyph():
    model = small_model().eval()
    prompt = tiles(7)
    calls = {"semantic": 0, "glyph": 0}
    observed = []

    def count(name):
        def hook(module, args):
            calls[name] += 1

        return hook

    hooks = [
        model.semantic_decoder.register_forward_pre_hook(count("semantic")),
        model.glyph_decoder.register_forward_pre_hook(count("glyph")),
        model.glyph_encoder.register_forward_pre_hook(
            lambda module, args: observed.append(args[0].detach().clone())
        ),
    ]
    cached = model.generate(prompt, 3, use_cache=True)
    for hook in hooks:
        hook.remove()
    uncached = model.generate(prompt, 3, use_cache=False)
    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)
    assert calls == {"semantic": 3, "glyph": 3}
    assert cached.dtype == torch.uint8 and bool(((cached == 0) | (cached == 1)).all())
    torch.testing.assert_close(observed[1], cached[:, 0], rtol=0, atol=0)
