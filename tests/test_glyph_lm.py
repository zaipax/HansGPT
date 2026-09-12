"""Small CPU/GPU integration tests, executed in the server's project uv environment."""

import importlib.util
import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hansgpt_research.glyph_lm import (
    GlyphGPT,
    GlyphSequenceDataset,
    ModelConfig,
    _unique_binary_pixels,
    collate_glyph_sequences,
    pixel_bce_loss,
)


def small_config(**overrides) -> ModelConfig:
    return replace(
        ModelConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            max_position_embeddings=16,
            glyph_encode_chunk_size=4,
        ),
        **overrides,
    )


def binary_input(length=5):
    return torch.randint(0, 2, (1, length, 1, 32, 32), dtype=torch.uint8)


def test_default_architecture_has_exact_agreed_parameter_count():
    model = GlyphGPT(ModelConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) == 78_118_368
    assert model.backbone.embed_tokens is None
    assert model.backbone.main_input_name == "inputs_embeds"
    assert model.backbone.config.bos_token_id is None
    assert model.backbone.config.eos_token_id is None
    assert model.backbone.config.pad_token_id is None
    assert model.pixel_head.out_features == 1024
    assert model.pixel_head.bias is not None


def test_future_tiles_cannot_change_prefix_predictions_in_training_mode():
    torch.manual_seed(13)
    model = GlyphGPT(small_config()).train()
    original = binary_input()
    changed = original.clone()
    changed[:, 3:] = 1 - changed[:, 3:]
    with torch.no_grad():
        first = model(original)
        second = model(changed)
    assert first.shape == original.shape
    torch.testing.assert_close(first[:, :3], second[:, :3], atol=2e-6, rtol=2e-5)
    assert not torch.allclose(first[:, 3:], second[:, 3:])
    assert not any(isinstance(module, nn.Embedding) for module in model.modules())
    assert not any(
        isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()
    )


def test_repeated_pixel_encoding_preserves_logits_and_cnn_gradients():
    torch.manual_seed(17)
    deduplicated = GlyphGPT(small_config(deduplicate_glyphs=True))
    direct = GlyphGPT(small_config(deduplicate_glyphs=False))
    direct.load_state_dict(deduplicated.state_dict())
    glyphs = binary_input(3)[:, [0, 1, 0, 2, 1]]
    targets = binary_input(5)
    mask = torch.ones((1, 5), dtype=torch.bool)
    logits_deduplicated = deduplicated(glyphs)
    logits_direct = direct(glyphs)
    torch.testing.assert_close(logits_deduplicated, logits_direct, atol=2e-6, rtol=2e-5)
    pixel_bce_loss(logits_deduplicated, targets, mask).backward()
    pixel_bce_loss(logits_direct, targets, mask).backward()
    for (name_a, weight_a), (name_b, weight_b) in zip(
        deduplicated.glyph_encoder.named_parameters(),
        direct.glyph_encoder.named_parameters(),
        strict=True,
    ):
        assert name_a == name_b
        assert weight_a.grad is not None and weight_a.grad.abs().sum() > 0
        torch.testing.assert_close(weight_a.grad, weight_b.grad, atol=2e-7, rtol=2e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_checkpoint_replays_decoder_preserves_cnn_gradients_and_updates_weights(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA integration case requires the authorized training server GPU")
    torch.manual_seed(31)
    direct = GlyphGPT(small_config()).to(device).train()
    checkpointed = GlyphGPT(small_config()).to(device).train()
    checkpointed.load_state_dict(direct.state_dict())
    checkpointed.gradient_checkpointing_enable()
    assert all(layer.gradient_checkpointing for layer in checkpointed.backbone.layers)
    calls = []
    hook = checkpointed.backbone.layers[0].register_forward_pre_hook(
        lambda _module, _args: calls.append("decoder execution")
    )
    # Images have no gradient: the CNN's trainable weights must create and retain
    # the graph that checkpoint replay follows back through inputs_embeds.
    glyphs = binary_input(3)[:, [0, 1, 0, 2, 1]].to(device)
    targets = binary_input(5).to(device)
    mask = torch.ones((1, 5), dtype=torch.bool, device=device)
    optimizer = torch.optim.SGD(checkpointed.parameters(), lr=0.1)
    before = checkpointed.glyph_encoder.convolutions[0].weight.detach().clone()
    context = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
    with context:
        direct_logits = direct(glyphs)
        checkpointed_logits = checkpointed(glyphs)
        direct_loss = pixel_bce_loss(direct_logits, targets, mask)
        checkpointed_loss = pixel_bce_loss(checkpointed_logits, targets, mask)
    torch.testing.assert_close(direct_logits, checkpointed_logits, atol=2e-6, rtol=2e-5)
    assert len(calls) == 1
    direct_loss.backward()
    checkpointed_loss.backward()
    hook.remove()
    assert len(calls) >= 2  # Recompute really ran; a disabled checkpoint would fail.
    for (direct_name, direct_weight), (checkpointed_name, checkpointed_weight) in zip(
        direct.glyph_encoder.named_parameters(),
        checkpointed.glyph_encoder.named_parameters(),
        strict=True,
    ):
        assert direct_name == checkpointed_name
        gradient = checkpointed_weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        torch.testing.assert_close(gradient, direct_weight.grad, atol=2e-7, rtol=2e-4)
    optimizer.step()
    assert not torch.equal(before, checkpointed.glyph_encoder.convolutions[0].weight)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_packed_pixel_uniqueness_is_lossless_for_all_byte_values(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA integration case requires the authorized training server GPU")
    values = torch.arange(256, device=device)
    shifts = torch.arange(7, -1, -1, device=device)
    # Every possible byte is tested in every byte position, including 0 and 255.
    all_patterns = ((values[:, None] >> shifts) & 1).repeat(1, 128).to(torch.uint8)
    pixels = torch.cat((all_patterns.flip(0), all_patterns[[0, 255, 127, 128, 255]]))
    unpacked, unpacked_inverse = _unique_binary_pixels(pixels, "unpacked")
    packed, packed_inverse = _unique_binary_pixels(pixels, "packed")
    assert packed.dtype == torch.uint8 and packed.shape == (256, 1024)
    assert ((packed == 0) | (packed == 1)).all()
    torch.testing.assert_close(packed, unpacked, atol=0, rtol=0)
    torch.testing.assert_close(packed_inverse, unpacked_inverse, atol=0, rtol=0)
    torch.testing.assert_close(packed[packed_inverse], pixels, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_packed_dedup_preserves_predictions_cnn_gradients_and_parameter_count(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA integration case requires the authorized training server GPU")
    torch.manual_seed(37)
    unpacked = GlyphGPT(small_config(glyph_deduplication_strategy="unpacked")).to(device)
    packed = GlyphGPT(small_config(glyph_deduplication_strategy="packed")).to(device)
    packed.load_state_dict(unpacked.state_dict())
    assert sum(p.numel() for p in packed.parameters()) == sum(
        p.numel() for p in unpacked.parameters()
    )
    assert packed.state_dict().keys() == unpacked.state_dict().keys()
    glyphs = binary_input(3)[:, [0, 1, 0, 2, 1, 0, 0]].to(device)
    targets = binary_input(7).to(device)
    mask = torch.ones((1, 7), dtype=torch.bool, device=device)
    context = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
    with context:
        unpacked_logits = unpacked(glyphs)
        packed_logits = packed(glyphs)
        unpacked_loss = pixel_bce_loss(unpacked_logits, targets, mask)
        packed_loss = pixel_bce_loss(packed_logits, targets, mask)
    torch.testing.assert_close(packed_logits, unpacked_logits, atol=2e-6, rtol=2e-5)
    unpacked_loss.backward()
    packed_loss.backward()
    for (unpacked_name, unpacked_weight), (packed_name, packed_weight) in zip(
        unpacked.glyph_encoder.named_parameters(),
        packed.glyph_encoder.named_parameters(),
        strict=True,
    ):
        assert unpacked_name == packed_name
        assert packed_weight.grad is not None and packed_weight.grad.abs().sum() > 0
        torch.testing.assert_close(packed_weight.grad, unpacked_weight.grad, atol=2e-7, rtol=2e-4)


@pytest.mark.parametrize("bad_value", [0.5, -1.0, 2.0, float("nan")])
def test_model_and_loss_reject_nonbinary_tiles(bad_value):
    model = GlyphGPT(small_config())
    invalid = binary_input(2).float()
    invalid[0, 0, 0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="binary"):
        model(invalid)
    with pytest.raises(ValueError, match="binary"):
        pixel_bce_loss(torch.zeros_like(invalid), invalid, torch.ones((1, 2), dtype=torch.bool))


def test_loss_is_next_tile_pixel_mean_and_masks_padding():
    logits = torch.full((1, 2, 1, 32, 32), -2.0, requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[:, 0, :, :16] = 1
    mask = torch.tensor([[True, False]])
    loss = pixel_bce_loss(logits, targets, mask)
    expected = (torch.nn.functional.softplus(torch.tensor(-2.0)) + 1.0).item()
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert logits.grad[:, 0].abs().sum() > 0
    assert logits.grad[:, 1].abs().sum() == 0


def test_incremental_cache_matches_full_prefix_and_generation_uses_same_cnn():
    torch.manual_seed(19)
    model = GlyphGPT(small_config()).eval()
    glyphs = binary_input(4)
    with torch.no_grad():
        full = model(glyphs)
        prefix, cache = model(glyphs[:, :3], use_cache=True, return_cache=True)
        incremental, _ = model(
            glyphs[:, 3:], past_key_values=cache, use_cache=True, return_cache=True
        )
    torch.testing.assert_close(full[:, :3], prefix, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(full[:, 3:], incremental, atol=2e-6, rtol=2e-5)
    calls = []
    hook = model.glyph_encoder.register_forward_pre_hook(
        lambda _module, args: calls.append(args[0].detach().clone())
    )
    generated = model.generate(glyphs[:, :2], max_new_tokens=3)
    hook.remove()
    assert generated.shape == (1, 3, 1, 32, 32)
    assert generated.dtype == torch.uint8
    assert ((generated == 0) | (generated == 1)).all()
    assert len(calls) == 3
    torch.testing.assert_close(calls[1], generated[:, 0])
    torch.testing.assert_close(calls[2], generated[:, 1])
    recomputed = model.generate(glyphs[:, :2], max_new_tokens=3, use_cache=False)
    torch.testing.assert_close(generated, recomputed, atol=0, rtol=0)


def test_generation_sliding_window_recompute_matches_uncached():
    torch.manual_seed(23)
    model = GlyphGPT(small_config(max_position_embeddings=3)).eval()
    prompt = binary_input(2)
    cached = model.generate(prompt, max_new_tokens=4)
    direct = model.generate(prompt, max_new_tokens=4, use_cache=False)
    torch.testing.assert_close(cached, direct, atol=0, rtol=0)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_generation_rejects_nonfinite_logits_before_binary_feedback(bad_value):
    model = GlyphGPT(small_config()).train()
    with torch.no_grad():
        model.pixel_head.bias[0] = bad_value
    encoder_calls = []
    hook = model.glyph_encoder.register_forward_pre_hook(
        lambda _module, _args: encoder_calls.append("CNN")
    )
    with pytest.raises(FloatingPointError, match="nonfinite logits"):
        model.generate(binary_input(2), max_new_tokens=3)
    hook.remove()
    assert len(encoder_calls) == 1
    assert model.training  # Failure restores the mode as well as stopping feedback.


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_masked_zero_pad_preserves_finite_cnn_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA integration case requires the authorized training server GPU")
    torch.manual_seed(41)
    model = GlyphGPT(small_config()).to(device).train()
    model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    glyphs = torch.randint(0, 2, (2, 8, 1, 32, 32), dtype=torch.uint8, device=device)
    targets = torch.randint(0, 2, glyphs.shape, dtype=torch.uint8, device=device)
    mask = torch.arange(8, device=device)[None, :] < torch.tensor([[3], [5]], device=device)
    glyphs[~mask] = 0
    targets[~mask] = 0
    context = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext()
    with context:
        logits = model(glyphs, attention_mask=mask)
        loss = pixel_bce_loss(logits, targets, mask)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert model.glyph_encoder.convolutions[0].weight.grad.abs().sum() > 0
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    assert scaler.get_scale() >= scale_before


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_capacity_sampler_has_structural_boundaries_and_never_samples_pad(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA integration case requires the authorized training server GPU")
    specification = importlib.util.spec_from_file_location(
        "stress_glyph_lm_for_test", Path(__file__).parents[1] / "scripts" / "stress_glyph_lm.py"
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    controls = {"PAD": 8, "BOS": 3, "EOS": 9, "NEWLINE": 1}
    content = torch.tensor([0, 2, 4, 5, 6, 7], device=device)
    generator = torch.Generator(device=device).manual_seed(43)
    for length in (1, 2, 64):
        addresses = module.sample_capacity_asset_addresses(content, controls, 3, length, generator)
        assert addresses.shape == (3, length + 1)
        assert (addresses[:, 0] == controls["BOS"]).all()
        assert (addresses[:, -1] == controls["EOS"]).all()
        assert torch.isin(addresses[:, 1:-1], content).all()
        assert not (addresses == controls["PAD"]).any()
        assert not (addresses == controls["NEWLINE"]).any()
        assert not (addresses[:, :-1] == controls["EOS"]).any()
        assert not (addresses[:, 1:] == controls["BOS"]).any()
    invalid_content = torch.tensor([0, controls["PAD"]], device=device)
    with pytest.raises(ValueError, match="exclude every distinct control"):
        module.sample_capacity_asset_addresses(invalid_content, controls, 3, 64, generator)


def test_model_config_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(29)
    model = GlyphGPT(small_config()).eval()
    glyphs = binary_input(3)
    payload = {"model_config": model.config.to_dict(), "model": model.state_dict()}
    checkpoint = tmp_path / "model.pt"
    torch.save(payload, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = GlyphGPT(ModelConfig.from_dict(loaded["model_config"])).eval()
    restored.load_state_dict(loaded["model"])
    with torch.no_grad():
        torch.testing.assert_close(model(glyphs), restored(glyphs), atol=0, rtol=0)


def make_corpus(tmp_path):
    bitmaps = np.zeros((9, 32, 32), dtype=np.uint8)
    for index in range(1, len(bitmaps)):
        bitmaps[index, 0, :index] = 1
    np.savez_compressed(tmp_path / "glyph_bank.npz", bitmaps=bitmaps)
    (tmp_path / "glyph_inventory.json").write_text(
        json.dumps(
            {
                "controls": {"0": "PAD", "1": "BOS", "2": "EOS", "3": "NEWLINE"},
                "characters": {"春": 4, "风": 5, "吹": 6, "过": 7, "山": 8},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    documents = [[1, 4, 5, 6, 7, 8, 2], [1, 8, 7, 2], [1, 4, 2]]
    offsets = np.concatenate(([0], np.cumsum([len(doc) for doc in documents])))
    np.save(tmp_path / "train.offsets.npy", offsets)
    np.array([asset for doc in documents for asset in doc], dtype="<u2").tofile(
        tmp_path / "train.uint16"
    )
    return documents, bitmaps


@pytest.mark.parametrize("sequence_length", [1, 2, 3, 4, 6, 8])
def test_document_chunks_cover_every_shifted_pair_exactly_once(tmp_path, sequence_length):
    documents, bitmaps = make_corpus(tmp_path)
    dataset = GlyphSequenceDataset(tmp_path, "train", sequence_length)
    pairs = []
    target_count = 0
    for example in dataset:
        valid = example["loss_mask"]
        assert torch.equal(valid, example["attention_mask"])
        assert example["glyphs"].shape == (sequence_length, 1, 32, 32)
        assert example["targets"].shape == (sequence_length, 1, 32, 32)
        assert (example["glyphs"][~valid] == 0).all()
        assert (example["targets"][~valid] == 0).all()
        for input_tile, target_id, target_tile in zip(
            example["glyphs"][valid],
            example["target_ids"][valid],
            example["targets"][valid],
            strict=True,
        ):
            input_id = int(input_tile.sum())  # Fixture pixels uniquely identify each asset.
            pairs.append((input_id, int(target_id)))
            np.testing.assert_array_equal(target_tile[0].numpy(), bitmaps[int(target_id)])
        target_count += int(valid.sum())
    expected = [(doc[index], doc[index + 1]) for doc in documents for index in range(len(doc) - 1)]
    assert pairs == expected
    assert target_count == dataset.target_count == sum(len(doc) - 1 for doc in documents)
    assert (2, 1) not in pairs  # Independent documents never form a training transition.
    assert dataset.gallery_ids == [4, 5, 6, 7, 8]
    assert dataset.control_ids == {"PAD": 0, "BOS": 1, "EOS": 2, "NEWLINE": 3}


def test_asset_renumbering_does_not_change_model_pixels(tmp_path):
    _, bitmaps = make_corpus(tmp_path)
    original = GlyphSequenceDataset(tmp_path, "train", 4)[0]
    # Swap two corpus addresses, their bitmaps, and their metadata together.
    indices = np.arange(len(bitmaps))
    indices[[4, 5]] = indices[[5, 4]]
    tokens = np.fromfile(tmp_path / "train.uint16", dtype="<u2")
    indices[tokens].astype("<u2").tofile(tmp_path / "train.uint16")
    np.savez_compressed(tmp_path / "glyph_bank.npz", bitmaps=bitmaps[indices])
    inventory_path = tmp_path / "glyph_inventory.json"
    inventory = json.loads(inventory_path.read_text("utf-8"))
    inventory["characters"] = {
        char: int(indices[asset]) for char, asset in inventory["characters"].items()
    }
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    renumbered = GlyphSequenceDataset(tmp_path, "train", 4)[0]
    assert not torch.equal(original["target_ids"], renumbered["target_ids"])
    torch.testing.assert_close(original["glyphs"], renumbered["glyphs"], atol=0, rtol=0)
    torch.testing.assert_close(original["targets"], renumbered["targets"], atol=0, rtol=0)


def test_multidomain_control_format_preserves_dataset_pixels(tmp_path):
    make_corpus(tmp_path)
    original = GlyphSequenceDataset(tmp_path, "train", 1024)
    before = original[0]
    path = tmp_path / "glyph_inventory.json"
    inventory = json.loads(path.read_text("utf-8"))
    inventory["controls"] = {name: int(index) for index, name in inventory["controls"].items()}
    path.write_text(json.dumps(inventory), encoding="utf-8")
    restored = GlyphSequenceDataset(tmp_path, "train", 1024)
    assert restored.control_ids == original.control_ids
    assert restored.target_count == original.target_count
    for key, value in before.items():
        torch.testing.assert_close(value, restored[0][key], atol=0, rtol=0)


@pytest.mark.parametrize(
    "controls",
    [
        {"0": "PAD", "BOS": 1},
        {"0": "PAD", "1": "PAD"},
        {"PAD": 0, "BOS": 0},
        {"PAD": False},
    ],
)
def test_ambiguous_control_inventory_is_rejected(controls):
    from hansgpt_research.glyph_lm import normalize_control_inventory

    with pytest.raises(ValueError):
        normalize_control_inventory(controls)


def test_dataset_rejects_wrong_document_boundaries(tmp_path):
    make_corpus(tmp_path)
    tokens = np.fromfile(tmp_path / "train.uint16", dtype="<u2")
    tokens[0] = 4
    tokens.tofile(tmp_path / "train.uint16")
    with pytest.raises(ValueError, match="begin with BOS"):
        GlyphSequenceDataset(tmp_path, "train", 4)


@pytest.mark.parametrize("padding_multiple,expected_length", [(1, 6), (4, 8), (8, 8), (32, 16)])
def test_dynamic_collation_preserves_all_targets_and_document_rows(
    tmp_path, padding_multiple, expected_length
):
    documents, _ = make_corpus(tmp_path)
    dataset = GlyphSequenceDataset(tmp_path, "train", sequence_length=16)
    samples = [dataset[index] for index in range(len(dataset))]
    batch = collate_glyph_sequences(samples, pad_to_multiple_of=padding_multiple)
    assert batch["glyphs"].shape == (3, expected_length, 1, 32, 32)
    assert batch["targets"].shape == batch["glyphs"].shape
    assert int(batch["loss_mask"].sum()) == dataset.target_count
    for index, document in enumerate(documents):
        mask = batch["loss_mask"][index]
        assert batch["target_ids"][index][mask].tolist() == document[1:]
        for name in samples[index]:
            torch.testing.assert_close(batch[name][index], samples[index][name][:expected_length])
    assert ((batch["glyphs"] == 0) | (batch["glyphs"] == 1)).all()
    assert ((batch["targets"] == 0) | (batch["targets"] == 1)).all()


def test_padding_trim_does_not_change_valid_next_tile_logits(tmp_path):
    make_corpus(tmp_path)
    dataset = GlyphSequenceDataset(tmp_path, "train", sequence_length=16)
    samples = [dataset[0], dataset[1]]
    full = collate_glyph_sequences(samples, pad_to_multiple_of=16)
    trimmed = collate_glyph_sequences(samples)
    model = GlyphGPT(small_config()).eval()
    with torch.no_grad():
        full_logits = model(full["glyphs"], attention_mask=full["attention_mask"])
        trimmed_logits = model(trimmed["glyphs"], attention_mask=trimmed["attention_mask"])
    torch.testing.assert_close(
        full_logits[full["loss_mask"]],
        trimmed_logits[trimmed["loss_mask"]],
        atol=2e-6,
        rtol=2e-5,
    )


def test_dynamic_collator_never_trims_a_masked_context_hole():
    sample = {
        "glyphs": binary_input(16)[0],
        "targets": binary_input(16)[0],
        "attention_mask": torch.tensor([True] + [False] * 9 + [True] + [False] * 5),
        "loss_mask": torch.tensor([True] + [False] * 15),
        "target_ids": torch.arange(16),
    }
    batch = collate_glyph_sequences([sample], pad_to_multiple_of=1)
    assert batch["glyphs"].shape[1] == 11
    assert int(batch["attention_mask"].sum()) == 2
    assert int(batch["loss_mask"].sum()) == 1
