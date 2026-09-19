"""D1 split, image and frozen-feature contracts; run only in the server uv environment."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hansgpt_research.diagnose_glyph_reconstruction import (
    BalancedGlyphOrder,
    ReconstructionDecoder,
    encode_images,
    fixed_bitflips,
    glyph_metrics,
    module_hash,
    predict_images,
    select_decoder_glyphs,
    train_decoder,
)
from hansgpt_research.glyph_lm import GlyphEncoder, ModelConfig


def toy_assets():
    bank = torch.zeros((12, 1, 32, 32), dtype=torch.uint8)
    for index in range(1, len(bank)):
        bank[index, 0, 0, :index] = 1
    bank[5] = bank[4]  # Two training characters have exactly the same pixel image.
    bank[9] = bank[4]  # A corpus-train-unseen alias must still be excluded.
    inventory = {
        "controls": {"0": "PAD", "1": "BOS", "2": "EOS", "3": "NEWLINE"},
        "characters": dict(zip("甲乙丙丁戊己庚辛", range(4, 12), strict=True)),
    }
    counts = np.array([0, 0, 20, 0, 1000, 1, 5, 500, 1, 0, 5, 0], dtype=np.int64)
    return bank, inventory, counts


def test_decoder_split_excludes_train_zero_glyphs_and_keeps_pixel_collisions_together():
    bank, inventory, counts = toy_assets()
    selected = select_decoder_glyphs(bank, inventory, counts, seed=13)
    repeated = select_decoder_glyphs(bank, inventory, counts, seed=13)
    assert selected == repeated
    assert selected["eligible_asset_ids"] == [4, 5, 6, 7, 8, 10]
    assert selected["excluded_train_zero_asset_ids"] == [9, 11]
    assert selected["excluded_control_asset_ids"] == [0, 1, 2, 3]
    groups = [set(values) for values in selected["cohorts"].values()]
    assert all(groups)
    assert set.union(*groups) == {4, 5, 6, 7, 8, 10}
    assert all(not first & second for i, first in enumerate(groups) for second in groups[i + 1 :])
    assert any({4, 5}.issubset(group) for group in groups)
    assert selected["collision_groups"] == [[4, 5]]
    assert "not v1-unseen" in selected["heldout_scope"]


def test_balanced_sampler_covers_partial_sweeps_without_frequency_weighting():
    order = BalancedGlyphOrder([2, 5, 9], seed=17)
    first_sweep = order.take(3)
    second_sweep = order.take(3)
    assert sorted(first_sweep.tolist()) == sorted(second_sweep.tolist()) == [2, 5, 9]
    long_order = BalancedGlyphOrder([2, 5, 9], seed=19)
    draws = np.concatenate([long_order.take(7), long_order.take(11), long_order.take(5)])
    repeated = BalancedGlyphOrder([2, 5, 9], seed=19).take(23)
    np.testing.assert_array_equal(draws, repeated)
    counts = np.bincount(draws, minlength=10)
    assert counts[[2, 5, 9]].max() - counts[[2, 5, 9]].min() <= 1
    assert counts.sum() == counts[[2, 5, 9]].sum() == 23


@pytest.mark.parametrize("flips", [0, 1, 4, 32])
def test_bitflip_diagnostic_is_binary_exact_and_reproducible(flips):
    clean = torch.randint(0, 2, (7, 1, 32, 32), dtype=torch.uint8)
    original = clean.clone()
    first = fixed_bitflips(clean, flips, seed=23)
    second = fixed_bitflips(clean, flips, seed=23)
    assert first.dtype == torch.uint8 and first.shape == clean.shape
    assert ((first == 0) | (first == 1)).all()
    assert ((first != clean).flatten(1).sum(1) == flips).all()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(clean, original, atol=0, rtol=0)


def test_macro_glyph_and_micro_foreground_metrics_have_distinct_denominators():
    target = torch.zeros((2, 1, 32, 32), dtype=torch.uint8)
    target[0, 0, 0, 0] = 1
    target[1].flatten()[:100] = 1
    prediction = torch.zeros_like(target)
    prediction[0] = target[0]
    result = glyph_metrics(prediction, target, torch.tensor([0.1, 0.3]))
    assert result["per_glyph"]["foreground_f1"] == [1, 0]
    assert result["macro_foreground_f1"] == result["macro_dice"] == 0.5
    assert result["micro_foreground_f1"] == pytest.approx(2 / 102)
    assert result["exact_bitmap_match"] == 0.5
    assert result["mean_hamming_bits"] == 50
    assert result["reconstruction_bce_nats_per_pixel"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="binary"):
        glyph_metrics(prediction.float() + 0.1, target)


@pytest.mark.parametrize("kind", ["linear", "spatial"])
def test_new_decoders_predict_full_images_and_receive_reconstruction_gradients(kind):
    torch.manual_seed(29)
    decoder = ReconstructionDecoder(32, kind)
    features = torch.randn((3, 32))
    targets = torch.randint(0, 2, (3, 1, 32, 32), dtype=torch.uint8)
    before = module_hash(decoder)
    logits = decoder(features)
    assert logits.shape == targets.shape
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float())
    loss.backward()
    assert all(parameter.grad is not None for parameter in decoder.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in decoder.parameters())
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.001)
    optimizer.step()
    assert module_hash(decoder) != before
    predicted, nll = predict_images(decoder, features, targets, torch.device("cpu"), 2, "fp32")
    assert predicted.dtype == torch.uint8 and ((predicted == 0) | (predicted == 1)).all()
    assert nll.shape == (3,) and torch.isfinite(nll).all()


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_cached_v1_features_train_decoder_without_touching_encoder_or_heldout(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA integration runs only on the authorized training server")
    device = torch.device(device_name)
    precision = "fp16" if device.type == "cuda" else "fp32"
    config = ModelConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
    )
    encoder = GlyphEncoder(config).to(device).requires_grad_(False).eval()
    before = module_hash(encoder)
    clean = torch.randint(0, 2, (6, 1, 32, 32), dtype=torch.uint8)
    cached = encode_images(encoder, clean, device, 2, precision)
    assert cached.shape == (6, 32) and not cached.requires_grad
    assert not torch.is_inference(cached)
    # Poison the decoder-held-out feature rows. A mistaken fit selection would
    # immediately produce nonfinite loss; correct training never reads them.
    cached[[1, 3, 5]] = float("nan")
    decoder = ReconstructionDecoder(32, "linear").to(device)
    args = SimpleNamespace(seed=31, steps=2, batch_size=8)
    result = train_decoder(decoder, cached, clean, np.array([0, 2, 4]), args, device, precision)
    assert result["successful_steps"] == 2
    assert result["successful_glyph_presentations"] == 16
    assert result["decoder_validation_or_holdout_presentations"] == 0
    assert result["fit_exposure_max"] - result["fit_exposure_min"] <= 1
    assert module_hash(encoder) == before
    assert all(parameter.grad is None for parameter in encoder.parameters())
