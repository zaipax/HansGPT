"""Structured loss, hard-image GAN boundaries, successful budgets and complete resume."""

import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from hansgpt_research.glyph_lm import ModelConfig
from hansgpt_research.structured_glyph_lm import GlyphDistribution
from hansgpt_research.train_structured_glyph_lm import (
    CHECKPOINT_FORMAT,
    IDENTITY_KEYS,
    advance_progress,
    complete_optimizer_step,
    configure_backbone,
    discriminator_backward,
    effective_config,
    frozen_at,
    generator_backward,
    initial_progress,
    optimizer_for,
    restore_checkpoint,
    save_checkpoint,
    validate,
    validate_initialization,
    validate_resume_identity,
    validation_subset,
)


class TinyStructuredModel(nn.Module):
    """Small differentiable context path; distribution is the actual K-component head law."""

    def __init__(self, components=2):
        super().__init__()
        self.components = components
        self.glyph_encoder = nn.Linear(1024, 8, bias=False)
        self.backbone = nn.Linear(8, 8)
        self.pixel_head = nn.Linear(8, components * 1024)
        self.component_head = nn.Linear(8, components) if components > 1 else None

    def forward_hidden(self, glyphs, attention_mask=None):
        value = self.glyph_encoder(glyphs.float().flatten(2)).tanh()
        return self.backbone(value).tanh()

    def distribution(self, hidden):
        pixels = self.pixel_head(hidden).reshape(*hidden.shape[:-1], self.components, 1024)
        weights = (
            self.component_head(hidden) if self.component_head is not None else pixels[..., 0] * 0
        )
        return GlyphDistribution(pixels, weights)


class TrackingDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Linear(1024, 8)
        self.context = nn.Linear(8, 8)
        self.score = nn.Linear(8, 1)
        self.observed = []

    def forward(self, images, hidden):
        assert bool(((images == 0) | (images == 1)).all())
        self.observed.append((images.requires_grad, hidden.requires_grad, self.training))
        features = self.image(images.float().flatten(1)) + self.context(hidden.float())
        return self.score(features.tanh()).squeeze(-1)


class ControlledScaler:
    """A deterministic AMP skip harness; actual CUDA AMP is checked on the server GPU."""

    def __init__(self, skips=()):
        self.skips = list(skips)
        self.index = 0
        self.value = 128.0

    def is_enabled(self):
        return True

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def get_scale(self):
        return self.value

    def step(self, optimizer):
        self.skipped = self.index < len(self.skips) and self.skips[self.index]
        if not self.skipped:
            optimizer.step()

    def update(self):
        if self.skipped:
            self.value /= 2
        self.index += 1

    def state_dict(self):
        return {"skips": list(self.skips), "index": self.index, "scale": self.value}

    def load_state_dict(self, state):
        self.skips = list(state["skips"])
        self.index = state["index"]
        self.value = state["scale"]


def settings(**overrides):
    config = {
        "model": {"max_position_embeddings": 16},
        "head": {"components": 2},
        "training": {
            "batch_size": 2,
            "sequence_length": 4,
            "learning_rate": 0.001,
            "target_tokens": 100,
            "precision": "fp32",
            "num_workers": 0,
            "head_chunk_size": 1,
            "validation_samples": 2,
            **overrides,
        },
    }
    return effective_config(config, mode="full", smoke_tokens=10)["training"]


def group():
    # Both global torch RNG and python/numpy RNG affect the next minibatch.
    offset = (random.random() + float(np.random.rand())) / 4
    images = (torch.rand(2, 3, 1, 32, 32) > 0.3 + offset).to(torch.uint8)
    targets = torch.randint(0, 2, images.shape, dtype=torch.uint8)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    return [{"glyphs": images, "targets": targets, "attention_mask": mask, "loss_mask": mask}]


@pytest.mark.parametrize("frozen", [False, True])
def test_chunked_hidden_gradient_bridge_matches_whole_head_backprop(frozen):
    torch.manual_seed(15)
    actual = TinyStructuredModel(4)
    expected = copy.deepcopy(actual)
    configure_backbone(actual, frozen=frozen)
    configure_backbone(expected, frozen=frozen)
    batches = group()
    cfg = settings()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    output = generator_backward(
        actual, None, batches, cfg, torch.device("cpu"), scaler, 3, frozen=frozen
    )
    batch = batches[0]
    hidden = expected.forward_hidden(batch["glyphs"])[batch["loss_mask"]]
    loss = expected.distribution(hidden).nll(batch["targets"][batch["loss_mask"]]).mean()
    loss.backward()
    assert output["nll_sum"] / 3 == pytest.approx(float(loss.detach()), rel=1e-6)
    for observed, reference in zip(actual.parameters(), expected.parameters(), strict=True):
        if reference.grad is None:
            assert observed.grad is None
        else:
            assert torch.allclose(observed.grad, reference.grad, atol=2e-6, rtol=2e-4)


def test_discriminator_and_generator_steps_keep_gradient_and_binary_boundaries():
    model, discriminator = TinyStructuredModel(1), TrackingDiscriminator()
    nll_only = copy.deepcopy(model)
    cfg = settings(adversarial_weight=0.2)
    batches = group()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    optimizer = torch.optim.AdamW(discriminator.parameters(), lr=0.001)
    discriminator_backward(
        model, discriminator, batches, cfg, torch.device("cpu"), scaler, 3, frozen=False
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        not image_grad and not condition_grad
        for image_grad, condition_grad, _ in discriminator.observed
    )
    result = complete_optimizer_step(optimizer, scaler, list(discriminator.parameters()), 1.0)
    assert result["succeeded"]
    optimizer.zero_grad(set_to_none=True)
    discriminator.observed.clear()
    before = copy.deepcopy(discriminator.state_dict())
    generator_backward(
        model, discriminator, batches, cfg, torch.device("cpu"), scaler, 3, frozen=False
    )
    assert all(
        image_grad and not condition_grad and not training
        for image_grad, condition_grad, training in discriminator.observed
    )
    assert all(parameter.grad is None for parameter in discriminator.parameters())
    assert all(parameter.requires_grad for parameter in discriminator.parameters())
    assert discriminator.training
    for key, value in discriminator.state_dict().items():
        assert torch.equal(value, before[key])
    generator_backward(nll_only, None, batches, cfg, torch.device("cpu"), scaler, 3, frozen=False)
    assert not torch.allclose(model.pixel_head.weight.grad, nll_only.pixel_head.weight.grad)


def test_only_generator_success_counts_toward_budget_and_overflows_are_independent():
    progress = initial_progress()
    advance_progress(progress, tokens=7, samples=2, generator_ok=False, discriminator_ok=True)
    assert progress["valid_tokens"] == 0 and progress["discriminator_valid_tokens"] == 7
    assert progress["cursor"] == 2 and progress["overflow_streak"] == 1
    advance_progress(progress, tokens=11, samples=3, generator_ok=True, discriminator_ok=False)
    assert progress["valid_tokens"] == 11 and progress["attempted_tokens"] == 18
    assert progress["optimizer_steps"] == 1 and progress["discriminator_steps"] == 1
    assert progress["overflow_streak"] == 0 and progress["discriminator_overflow_streak"] == 1
    assert progress["discriminator_attempted_tokens"] == 18 and progress["cursor"] == 5


def test_amp_skip_does_not_update_weights_or_optimizer_state():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    before = copy.deepcopy(model.state_dict())
    model(torch.ones(1, 2)).sum().backward()
    scaler = ControlledScaler([True])
    result = complete_optimizer_step(optimizer, scaler, list(model.parameters()), 1.0)
    assert result["succeeded"] is False and result["scale"] == 64
    assert not optimizer.state
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_nonfinite_cpu_gradient_is_rejected_before_optimizer_mutation():
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    before = copy.deepcopy(model.state_dict())
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("inf"))
    with pytest.raises(FloatingPointError):
        complete_optimizer_step(
            optimizer, torch.amp.GradScaler("cuda", enabled=False), list(model.parameters()), 1.0
        )
    assert not optimizer.state
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_aggregate_norm_overflow_cannot_be_mistaken_for_successful_amp_update(monkeypatch):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    before = copy.deepcopy(model.state_dict())
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 1e30)
        assert bool(torch.isfinite(parameter.grad).all())

    def overflowed_norm(parameters, max_norm):
        # Reproduce a float32 norm reduction overflow and the resulting zero clip.
        for parameter in parameters:
            parameter.grad.zero_()
        return torch.tensor(float("inf"))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", overflowed_norm)
    scaler = ControlledScaler([False])
    with pytest.raises(FloatingPointError, match="aggregate"):
        complete_optimizer_step(optimizer, scaler, list(model.parameters()), 1.0)
    assert scaler.index == 0 and not optimizer.state
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_elementwise_amp_overflow_keeps_the_normal_skip_path():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    before = copy.deepcopy(model.state_dict())
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("inf"))
    result = complete_optimizer_step(
        optimizer, ControlledScaler([True]), list(model.parameters()), 1.0
    )
    assert result["succeeded"] is False and result["grad_norm"] is None
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_scheduled_unfreeze_preserves_optimizer_groups_and_enables_backbone_updates():
    model = TinyStructuredModel(1)
    cfg = settings(unfreeze_after_tokens=3)
    optimizer = optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    identities = [
        [id(parameter) for parameter in values["params"]] for values in optimizer.param_groups
    ]
    before = copy.deepcopy(model.glyph_encoder.state_dict())
    for tokens in (0, 3):
        frozen = frozen_at(cfg, tokens)
        configure_backbone(model, frozen=frozen)
        optimizer.zero_grad(set_to_none=True)
        generator_backward(model, None, group(), cfg, torch.device("cpu"), scaler, 3, frozen=frozen)
        complete_optimizer_step(optimizer, scaler, list(model.parameters()), 1.0)
        if tokens == 0:
            assert all(
                torch.equal(value, before[key])
                for key, value in model.glyph_encoder.state_dict().items()
            )
    assert not torch.equal(model.glyph_encoder.weight, before["weight"])
    assert identities == [
        [id(parameter) for parameter in values["params"]] for values in optimizer.param_groups
    ]


@pytest.mark.parametrize(
    "key",
    [
        "git_commit",
        "config_file_sha256",
        "data_sha256",
        "init_checkpoint_sha256",
        "validation_selection",
    ],
)
def test_resume_rejects_changed_code_data_origin_or_selection(key):
    metadata = {name: name for name in IDENTITY_KEYS}
    saved = {"format": CHECKPOINT_FORMAT, "metadata": copy.deepcopy(metadata)}
    saved["metadata"][key] = "different"
    with pytest.raises(ValueError, match=key):
        validate_resume_identity(saved, metadata)


def test_initialization_compares_normalized_architecture_even_without_shape_changes():
    saved = {"metadata": {"config": {"model": {}}, "data_sha256": {"bank": "same"}}}
    validate_initialization(saved, ModelConfig(), {"bank": "same"})
    for changed in ({"rope_theta": 500000.0}, {"rms_norm_eps": 1e-5}, {"attention_dropout": 0.1}):
        with pytest.raises(ValueError, match="architecture"):
            validate_initialization(saved, ModelConfig.from_dict(changed), {"bank": "same"})
    with pytest.raises(ValueError, match="corpus"):
        validate_initialization(saved, ModelConfig(), {"bank": "changed"})


def test_gan_resume_restores_both_optimizers_scalers_overflow_streak_and_rng(tmp_path):
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    model, discriminator = TinyStructuredModel(2), TrackingDiscriminator()
    cfg = settings(adversarial_weight=0.1, freeze_backbone=False)
    optimizer = optimizer_for(model, cfg)
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=0.002)
    scaler, d_scaler = ControlledScaler([True, False]), ControlledScaler([False, True])
    progress = initial_progress()
    metadata = {name: name for name in IDENTITY_KEYS}

    def update():
        batches = group()
        optimizer.zero_grad(set_to_none=True)
        d_optimizer.zero_grad(set_to_none=True)
        discriminator_backward(
            model, discriminator, batches, cfg, torch.device("cpu"), d_scaler, 3, frozen=False
        )
        d_step = complete_optimizer_step(
            d_optimizer, d_scaler, list(discriminator.parameters()), 1.0
        )
        d_optimizer.zero_grad(set_to_none=True)
        generator_backward(
            model, discriminator, batches, cfg, torch.device("cpu"), scaler, 3, frozen=False
        )
        g_step = complete_optimizer_step(optimizer, scaler, list(model.parameters()), 1.0)
        advance_progress(
            progress,
            tokens=3,
            samples=2,
            generator_ok=g_step["succeeded"],
            discriminator_ok=d_step["succeeded"],
        )

    update()
    path = tmp_path / "resume.pt"
    save_checkpoint(
        path, model, optimizer, scaler, discriminator, d_optimizer, d_scaler, progress, metadata
    )
    assert progress["overflow_streak"] == 1 and progress["valid_tokens"] == 0
    update()
    update()
    expected_model, expected_d = (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(discriminator.state_dict()),
    )
    expected_progress = dict(progress)
    expected_scales = (scaler.state_dict(), d_scaler.state_dict())
    saved = torch.load(path, weights_only=False)
    progress = restore_checkpoint(
        saved, model, optimizer, scaler, discriminator, d_optimizer, d_scaler, metadata
    )
    assert progress["overflow_streak"] == 1 and progress["cursor"] == 2
    update()
    update()
    assert progress == expected_progress
    assert (scaler.state_dict(), d_scaler.state_dict()) == expected_scales
    for actual, expected in (
        (model.state_dict(), expected_model),
        (discriminator.state_dict(), expected_d),
    ):
        for key, value in actual.items():
            assert torch.equal(value, expected[key])


def test_validation_subset_has_fixed_selection_and_exact_target_accounting():
    class Dataset:
        sequence_length = 4
        offsets = np.array([0, 3, 7, 12, 15])
        chunk_offsets = np.array([0, 1, 2, 3, 4])
        target_count = 11

        def __len__(self):
            return 4

        def __getitem__(self, index):
            generator = torch.Generator().manual_seed(index)
            glyphs = torch.randint(0, 2, (4, 1, 32, 32), generator=generator, dtype=torch.uint8)
            mask = torch.arange(4) < (self.offsets[index + 1] - self.offsets[index] - 1)
            return {
                "glyphs": glyphs,
                "targets": glyphs.flip(-1),
                "attention_mask": mask,
                "loss_mask": mask,
            }

    cfg = settings()
    subset, selection = validation_subset(Dataset(), cfg)
    _, repeated = validation_subset(Dataset(), cfg)
    assert selection == repeated and selection["scope"] == "fixed_validation_subset"
    result = validate(TinyStructuredModel(2), subset, selection, cfg, torch.device("cpu"))
    assert result["targets"] == selection["expected_targets"]
    assert result["chunks"] == 2 and result["total_chunks"] == 4
    assert sum(result["prior_component_fraction"]) == pytest.approx(1)
    assert sum(result["posterior_component_fraction"]) == pytest.approx(1)
