"""Run these model/gradient integration checks only in the server's uv environment."""

from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from hansgpt_research.glyph_lm import GlyphGPT, ModelConfig
from hansgpt_research.structured_glyph_lm import (
    ConditionalGlyphDiscriminator,
    GlyphDistribution,
    StructuredGlyphGPT,
    hard_binary_st,
)


def small_config():
    return ModelConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=8,
        glyph_encode_chunk_size=4,
    )


def images(batch=1, length=4, device="cpu"):
    return torch.randint(0, 2, (batch, length, 1, 32, 32), dtype=torch.uint8, device=device)


def test_single_component_nll_equals_pixel_bce_and_its_gradient():
    torch.manual_seed(3)
    logits = torch.randn((2, 3, 1, 1024), requires_grad=True)
    targets = images(2, 3)
    distribution = GlyphDistribution(logits, torch.randn((2, 3, 1)))
    actual = distribution.nll(targets)
    expected = F.binary_cross_entropy_with_logits(
        logits[..., 0, :], targets.float().flatten(-3), reduction="none"
    ).mean(-1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual_gradient = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=0, rtol=0)


def test_whole_grid_mixture_is_normalized_and_uses_weights_before_marginalizing():
    # An analytically enumerable two-bit subspace; the other pixels are certainly
    # zero at machine precision. pi=(.25,.75), component probabilities=(.8,.3)/(.2,.7).
    pixels = torch.full((4, 2, 1024), -1000.0)
    pixels[:, :, :2] = torch.logit(torch.tensor([[0.8, 0.3], [0.2, 0.7]]))
    weights = torch.tensor([0.25, 0.75]).log().repeat(4, 1).requires_grad_()
    targets = torch.zeros((4, 1, 32, 32), dtype=torch.uint8)
    targets.flatten(1)[:, :2] = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    distribution = GlyphDistribution(pixels, weights)
    probability = (-1024 * distribution.nll(targets)).exp()
    oracle = torch.tensor([0.215, 0.435, 0.185, 0.165])
    torch.testing.assert_close(probability, oracle, atol=2e-7, rtol=2e-6)
    assert float(probability.sum()) == pytest.approx(1.0, abs=2e-6)
    shifted = GlyphDistribution(pixels, weights + 10)
    torch.testing.assert_close(
        distribution.nll(targets), shifted.nll(targets), atol=1e-8, rtol=1e-5
    )
    responsibilities = distribution.responsibilities(targets)
    torch.testing.assert_close(responsibilities[0], torch.tensor([0.035, 0.180]) / 0.215)
    distribution.nll(targets)[0].backward()
    assert weights.grad is not None and weights.grad[0].abs().sum() > 0
    assert weights.grad[1:].abs().sum() == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decode_chooses_one_whole_grid_mode_and_never_averages_the_two(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA integration runs on the authorized server GPU")
    logits = torch.full((256, 2, 1024), -1000.0, device=device)
    logits[:, 0, 0] = 1000
    logits[:, 1, 1] = 1000
    distribution = GlyphDistribution(logits, torch.zeros((256, 2), device=device))
    deterministic = distribution.decode(threshold=0.3, strategy="mode_threshold")
    assert (deterministic.flatten(1)[:, :2] == torch.tensor([1, 0], device=device)).all()
    for strategy in ("sample_threshold", "sample_pixels"):
        first = distribution.decode(
            threshold=0.3,
            strategy=strategy,
            generator=torch.Generator(device=device).manual_seed(5),
        )
        repeated = distribution.decode(
            threshold=0.3,
            strategy=strategy,
            generator=torch.Generator(device=device).manual_seed(5),
        )
        torch.testing.assert_close(first, repeated, atol=0, rtol=0)
        assert first.dtype == torch.uint8 and first.shape == (256, 1, 32, 32)
        assert ((first == 0) | (first == 1)).all()
        assert (first.flatten(1).sum(-1) == 1).all()
        assert torch.unique(first.flatten(1), dim=0).shape[0] == 2
    # Averaging component maps would give11 at .3, which neither mode contains.
    marginal_threshold = (logits.sigmoid().mean(1) >= 0.3).to(torch.uint8)
    assert (marginal_threshold[:, :2] == 1).all()


def test_pixel_sampling_is_distinct_from_thresholding_and_threshold_independent():
    distribution = GlyphDistribution(torch.zeros((64, 1, 1024)), torch.zeros((64, 1)))
    first = distribution.decode(0.25, "sample_pixels", torch.Generator().manual_seed(7))
    second = distribution.decode(0.75, "sample_pixels", torch.Generator().manual_seed(7))
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert 0.49 < float(first.float().mean()) < 0.51
    assert distribution.decode(0.25).sum() == 64 * 1024
    assert distribution.decode(0.75).sum() == 0


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_distribution_rejects_nonfinite_pixel_or_component_logits(bad_value):
    for corrupted in ("pixels", "weights"):
        pixels, weights = torch.zeros((1, 2, 1024)), torch.zeros((1, 2))
        (pixels if corrupted == "pixels" else weights).flatten()[0] = bad_value
        with pytest.raises(FloatingPointError, match="nonfinite"):
            GlyphDistribution(pixels, weights).decode()


def test_targets_must_be_binary_images_and_not_asset_addresses():
    distribution = GlyphDistribution(torch.zeros((2, 2, 1024)), torch.zeros((2, 2)))
    with pytest.raises(ValueError, match="shape"):
        distribution.nll(torch.tensor([123, 456]))
    with pytest.raises(ValueError, match="binary"):
        distribution.nll(torch.full((2, 1, 32, 32), 0.5))


def test_v1_migration_k1_preserves_all_weights_logits_and_binary_generation():
    torch.manual_seed(11)
    original = GlyphGPT(small_config()).eval()
    migrated = StructuredGlyphGPT(small_config(), components=1).eval()
    migrated.load_v1_state_dict(original.state_dict())
    assert migrated.state_dict().keys() == original.state_dict().keys()
    assert migrated.component_head is None
    assert not any(isinstance(module, nn.Embedding) for module in migrated.modules())
    for name, weight in original.state_dict().items():
        torch.testing.assert_close(weight, migrated.state_dict()[name], atol=0, rtol=0)
    prompt = images(length=3)
    with torch.no_grad():
        expected = original(prompt)
        actual = migrated(prompt).pixel_logits[..., 0, :].reshape_as(expected)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        migrated.generate(prompt, 4, threshold=0.3),
        original.generate(prompt, 4, threshold=0.3),
        atol=0,
        rtol=0,
    )


def test_v1_migration_k4_clones_heads_breaks_symmetry_and_never_mutates_source():
    torch.manual_seed(13)
    original = GlyphGPT(small_config())
    source = {name: weight.clone() for name, weight in original.state_dict().items()}
    identical = StructuredGlyphGPT(small_config(), components=4, init_noise=0)
    identical.load_v1_state_dict(source)
    copied = identical.pixel_head.weight.reshape(4, 1024, small_config().hidden_size)
    for head in copied:
        torch.testing.assert_close(head, source["pixel_head.weight"], atol=0, rtol=0)
    assert identical.component_head.weight.count_nonzero() == 0
    perturbed = StructuredGlyphGPT(small_config(), components=4, init_noise=0.002)
    perturbed.load_v1_state_dict(source)
    heads = perturbed.pixel_head.weight.reshape(4, 1024, small_config().hidden_size)
    assert not torch.equal(heads[0], heads[1])
    for name, weight in source.items():
        torch.testing.assert_close(weight, original.state_dict()[name], atol=0, rtol=0)
        if not name.startswith("pixel_head"):
            torch.testing.assert_close(weight, perturbed.state_dict()[name], atol=0, rtol=0)
    dimension = small_config().hidden_size
    expected_extra = 3 * 1024 * (dimension + 1) + 4 * (dimension + 1)
    assert sum(p.numel() for p in perturbed.parameters()) == (
        sum(p.numel() for p in original.parameters()) + expected_extra
    )
    with pytest.raises(ValueError, match="checkpoint keys"):
        perturbed.load_v1_state_dict({**source, "unexpected_embedding.weight": torch.ones(1)})


def test_structured_model_preserves_causality_cache_and_cnn_loss_gradient():
    torch.manual_seed(17)
    model = StructuredGlyphGPT(small_config(), components=4).train()
    prompt = images(length=5)
    altered = prompt.clone()
    altered[:, 3:] = 1 - altered[:, 3:]
    with torch.no_grad():
        original = model.forward_hidden(prompt)
        changed = model.forward_hidden(altered)
        prefix, cache = model.forward_hidden(prompt[:, :3], use_cache=True, return_cache=True)
        incremental = model.forward_hidden(prompt[:, 3:], past_key_values=cache, use_cache=True)
    torch.testing.assert_close(original[:, :3], changed[:, :3], atol=2e-6, rtol=2e-5)
    assert not torch.allclose(original[:, 3:], changed[:, 3:])
    torch.testing.assert_close(original[:, :3], prefix, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(original[:, 3:], incremental, atol=2e-6, rtol=2e-5)
    model(prompt).nll(images(length=5)).mean().backward()
    gradient = model.glyph_encoder.convolutions[0].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_generated_raw_bits_return_through_same_cnn_and_cache_rollover_is_equivalent():
    torch.manual_seed(19)
    model = StructuredGlyphGPT(small_config(), components=4).eval()
    prompt = images(length=3)
    calls = []
    hook = model.glyph_encoder.register_forward_pre_hook(
        lambda _module, args: calls.append(args[0].detach().clone())
    )
    result = model.generate(prompt, 3)
    hook.remove()
    assert result.dtype == torch.uint8 and ((result == 0) | (result == 1)).all()
    assert len(calls) == 3
    torch.testing.assert_close(calls[1], result[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(calls[2], result[:, 1], atol=0, rtol=0)
    cached = model.generate(prompt, 8)
    recomputed = model.generate(prompt, 8, use_cache=False)
    torch.testing.assert_close(cached, recomputed, atol=0, rtol=0)


def test_eos_stops_only_on_exact_binary_pattern():
    model = StructuredGlyphGPT(small_config(), components=2, init_noise=0)
    eos = images(length=1)[0, 0]
    with torch.no_grad():
        model.pixel_head.weight.zero_()
        model.pixel_head.bias.copy_((eos.float().flatten() * 2000 - 1000).repeat(2))
    prompt = images(length=2)
    stopped = model.generate(prompt, 5, eos_glyph=eos)
    assert stopped.shape[1] == 1
    torch.testing.assert_close(stopped[0, 0], eos, atol=0, rtol=0)
    different = eos.clone()
    different[0, 0, 0] = 1 - different[0, 0, 0]
    assert model.generate(prompt, 5, eos_glyph=different).shape[1] == 5


@pytest.mark.parametrize("frozen_hidden", [True, False])
def test_chunked_recomputed_mixture_nll_preserves_head_and_hidden_gradients(frozen_hidden):
    torch.manual_seed(23)
    direct = StructuredGlyphGPT(small_config(), components=4)
    chunked = StructuredGlyphGPT(small_config(), components=4)
    chunked.load_state_dict(direct.state_dict())
    first = torch.randn((7, small_config().hidden_size), requires_grad=not frozen_hidden)
    second = first.detach().clone().requires_grad_(not frozen_hidden)
    targets = images(length=7)[0]
    expected = direct.distribution(first).nll(targets)
    actual = chunked.head_nll(second, targets, chunk_size=2, checkpoint_chunks=True)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    expected.sum().backward()
    actual.sum().backward()
    for name in ("pixel_head", "component_head"):
        for (first_name, first_weight), (second_name, second_weight) in zip(
            getattr(direct, name).named_parameters(),
            getattr(chunked, name).named_parameters(),
            strict=True,
        ):
            assert first_name == second_name
            assert second_weight.grad is not None and torch.isfinite(second_weight.grad).all()
            torch.testing.assert_close(second_weight.grad, first_weight.grad, atol=2e-7, rtol=2e-5)
    if not frozen_hidden:
        torch.testing.assert_close(second.grad, first.grad, atol=2e-7, rtol=2e-5)


def test_structured_checkpoint_roundtrip(tmp_path):
    model = StructuredGlyphGPT(small_config(), components=4).eval()
    path = tmp_path / "structured.pt"
    torch.save(
        {
            "model_config": model.config.to_dict(),
            "components": model.components,
            "model": model.state_dict(),
        },
        path,
    )
    saved = torch.load(path, weights_only=True)
    restored = StructuredGlyphGPT(ModelConfig.from_dict(saved["model_config"]), saved["components"])
    restored.load_state_dict(saved["model"], strict=True)
    restored.eval()
    prompt = images(length=3)
    with torch.no_grad():
        first, second = model(prompt), restored(prompt)
    torch.testing.assert_close(first.pixel_logits, second.pixel_logits, atol=0, rtol=0)
    torch.testing.assert_close(first.component_logits, second.component_logits, atol=0, rtol=0)


def test_st_is_exact_binary_forward_and_sigmoid_surrogate_backward():
    logits = torch.tensor([-4.0, -0.2, 0.0, 0.2, 4.0], requires_grad=True)
    hard = hard_binary_st(logits)
    torch.testing.assert_close(hard, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]), atol=0, rtol=0)
    hard.sum().backward()
    probability = logits.detach().sigmoid()
    torch.testing.assert_close(logits.grad, probability * (1 - probability), atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_conditional_gan_detach_boundaries_and_hard_st_generator_gradient(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA integration runs on the authorized server GPU")
    torch.manual_seed(29)
    discriminator = ConditionalGlyphDiscriminator(hidden_size=32, channels=8).to(device)
    generator_head = nn.Linear(32, 1024).to(device)
    context = torch.randn((3, 32), device=device, requires_grad=True)
    real = images(length=3, device=device)[0]
    autocast = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
    with autocast:
        fake_logits = generator_head(context).reshape(3, 1, 32, 32)
        fake = hard_binary_st(fake_logits)
        discriminator_loss = F.softplus(-discriminator(real, context.detach())).mean()
        discriminator_loss += F.softplus(discriminator(fake.detach(), context.detach())).mean()
    discriminator_loss.backward()
    assert context.grad is None
    assert all(parameter.grad is None for parameter in generator_head.parameters())
    assert any(parameter.grad is not None for parameter in discriminator.parameters())
    discriminator.zero_grad(set_to_none=True)
    discriminator.requires_grad_(False)
    autocast = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
    with autocast:
        fake_logits = generator_head(context).reshape(3, 1, 32, 32)
        generator_loss = -discriminator(hard_binary_st(fake_logits), context.detach()).mean()
    generator_loss.backward()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert context.grad.abs().sum() > 0  # Gradient follows image branch, not detached condition.
    assert generator_head.weight.grad is not None and generator_head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in discriminator.parameters())
    with pytest.raises(ValueError, match="binary"):
        discriminator(torch.full((3, 1, 32, 32), 0.5, device=device), context.detach())
