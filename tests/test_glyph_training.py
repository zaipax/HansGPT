"""Numerical, scoring and checkpoint invariants; execute on the training server."""

import random

import numpy as np
import pytest
import torch

from hansgpt_research.evaluate_glyph_lm import (
    BinaryMetrics,
    nearest_glyphs,
    select_constant_threshold,
)
from hansgpt_research.train_glyph_lm import (
    EpochSampler,
    learning_rate,
    loss_sum,
    restore_rng,
    save_checkpoint,
)


def test_resume_sampler_uses_consumed_cursor():
    full = list(EpochSampler(31, seed=99, epoch=2))
    assert sorted(full) == list(range(31))
    assert list(EpochSampler(31, seed=99, epoch=2, cursor=13)) == full[13:]
    assert list(EpochSampler(31, seed=99, epoch=3)) != full


def test_accumulated_loss_weights_real_targets_not_padded_microbatches():
    logits = torch.tensor([0.0, 2.0, -4.0]).reshape(1, 3, 1, 1, 1).expand(1, 3, 1, 32, 32)
    targets = torch.ones_like(logits)
    mask = torch.tensor([[True, True, False]])
    total, count = loss_sum(logits, {"targets": targets, "loss_mask": mask})
    expected = torch.nn.functional.softplus(-torch.tensor([0.0, 2.0])).sum()
    assert count == 2
    assert torch.allclose(total, expected)
    left, _ = loss_sum(logits[:, :1], {"targets": targets[:, :1], "loss_mask": mask[:, :1]})
    right, _ = loss_sum(logits[:, 1:], {"targets": targets[:, 1:], "loss_mask": mask[:, 1:]})
    assert torch.allclose((left + right) / count, total / count)


def test_binary_metrics_foreground_and_full_grid_denominators():
    target = torch.zeros(2, 1024, dtype=torch.bool)
    prediction = target.clone()
    target[0, :4] = True
    prediction[0, 2:6] = True
    metrics = BinaryMetrics()
    metrics.add(prediction, target, torch.tensor([1024.0, 512.0]))
    result = metrics.result()
    assert result["foreground_f1"] == pytest.approx(0.5)
    assert result["iou"] == pytest.approx(1 / 3)
    assert result["dice"] == result["foreground_f1"]
    assert result["exact_bitmap_match"] == 0.5
    assert result["hamming_bits_per_grid"] == 2
    assert result["bce_per_pixel"] == 0.75
    assert result["nll_nats_per_grid"] == 768


def test_chunked_hamming_matches_exhaustive_search_including_ties():
    generator = torch.Generator().manual_seed(23)
    gallery = torch.randint(0, 2, (13, 1024), generator=generator).float()
    gallery[8] = gallery[2]
    query = torch.cat((gallery[[8]], torch.randint(0, 2, (6, 1024), generator=generator)))
    ids = torch.arange(4, 17)
    actual_ids, actual_distance = nearest_glyphs(
        query, gallery, ids, query_chunk=2, gallery_chunk=3
    )
    distances = (query[:, None].bool() != gallery[None].bool()).sum(-1)
    indices = (distances.double() + ids.double() * 1e-9).topk(5, largest=False).indices
    assert torch.equal(actual_ids, ids[indices])
    assert torch.equal(actual_distance, distances.gather(1, indices).float())
    assert actual_ids[0, 0].item() == 6


def test_checkpoint_restores_optimizer_rng_and_next_update(tmp_path):
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    def step():
        optimizer.zero_grad()
        value = torch.rand(4, 3) + random.random() + float(np.random.rand())
        model(value).square().mean().backward()
        optimizer.step()

    step()
    checkpoint = tmp_path / "resume.pt"
    save_checkpoint(checkpoint, model, optimizer, scaler, {"cursor": 7}, {"mode": "test"})
    step()
    uninterrupted = {key: value.detach().clone() for key, value in model.state_dict().items()}
    saved = torch.load(checkpoint, weights_only=False)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    restore_rng(saved["rng"])
    assert saved["progress"]["cursor"] == 7
    step()
    for key, value in model.state_dict().items():
        assert torch.equal(value, uninterrupted[key])


def test_learning_rate_reaches_declared_floor_after_budget():
    cfg = {
        "warmup_tokens": 100,
        "target_tokens": 1000,
        "learning_rate": 0.001,
        "minimum_learning_rate_ratio": 0.1,
    }
    assert learning_rate(50, cfg) == pytest.approx(0.0005)
    assert learning_rate(100, cfg) == pytest.approx(0.001)
    assert learning_rate(1000, cfg) == pytest.approx(0.0001)
    assert learning_rate(2000, cfg) == pytest.approx(0.0001)


def test_baseline_selects_own_validation_threshold_and_excludes_controls():
    bank = torch.zeros(3, 1024)
    bank[0, 0] = 1
    bank[1, 1] = 1
    bank[2] = 1
    probability = torch.zeros(1024)
    probability[0], probability[1] = 0.2, 0.4
    # A frequent all-foreground control must not affect content-F1 threshold choice.
    counts = torch.tensor([9, 1, 100000])
    actual = select_constant_threshold(probability, counts, bank, [2], [0.15, 0.25, 0.5])
    assert actual["selected_threshold"] == 0.15
    assert actual["validation_content_tiles"] == 10
    assert actual["threshold_metrics"]["0.15"]["foreground_f1"] == pytest.approx(2 / 3)
    assert actual["threshold_metrics"]["0.25"]["foreground_f1"] == pytest.approx(0.1)
    assert counts.tolist() == [9, 1, 100000]
