"""Full coverage, exact mixture likelihood, gallery scope and checkpoint binding."""

import copy
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hansgpt_research.evaluate_glyph_lm import nearest_glyphs
from hansgpt_research.evaluate_structured_glyph_lm import (
    SplitAccumulator,
    content_candidates,
    decode_and_score,
    distribution_for_hidden,
    forward_hidden,
    score_full_split,
    validate_checkpoint_identity,
)
from hansgpt_research.glyph_lm import GlyphGPT, GlyphSequenceDataset, ModelConfig
from hansgpt_research.structured_glyph_lm import GlyphDistribution, StructuredGlyphGPT
from hansgpt_research.train_glyph_lm import canonical_hash


def corpus_fixture(directory):
    bank = np.random.default_rng(31).integers(0, 2, (8, 32, 32), dtype=np.uint8)
    bank[0] = 0
    np.savez_compressed(directory / "glyph_bank.npz", bitmaps=bank)
    inventory = {
        "controls": {"0": "PAD", "1": "BOS", "2": "EOS", "3": "NEWLINE"},
        "characters": {"甲": 4, "乙": 5, "，": 6, "。": 7},
        "collisions": [],
    }
    (directory / "glyph_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    documents = [[1, 4, 5, 2], [1, 6, 4, 2], [1, 5, 2]]
    np.array(sum(documents, []), dtype="<u2").tofile(directory / "test.uint16")
    np.save(directory / "test.offsets.npy", np.array([0, 4, 8, 11], dtype=np.int64))
    return GlyphSequenceDataset(directory, "test", 2)


def small_config():
    return ModelConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=16,
        glyph_encode_chunk_size=8,
    )


def test_exact_mixture_nll_is_not_bce_of_marginal_pixel_map():
    logits = torch.full((1, 2, 1024), -1000.0)
    logits[0, 0, 0], logits[0, 1, 1] = 1000.0, 1000.0
    distribution = GlyphDistribution(logits, torch.zeros(1, 2))
    targets = torch.zeros(1, 1, 32, 32, dtype=torch.uint8)
    targets.flatten(1)[0, 0] = 1
    binary, nll = decode_and_score(
        distribution,
        targets,
        threshold=0.45,
        strategy="mode_threshold",
        generator=torch.Generator().manual_seed(1),
    )
    assert nll.item() == pytest.approx(math.log(2) / 1024, rel=1e-6)
    marginal_probability = logits.sigmoid().mean(1)
    marginal_bce = F.binary_cross_entropy(marginal_probability, targets.float().flatten(1))
    assert marginal_bce.item() == pytest.approx(2 * nll.item(), rel=1e-6)
    assert torch.equal(binary, targets)


def test_nll_is_independent_of_threshold_and_sampling_decode():
    generator = torch.Generator().manual_seed(2)
    distribution = GlyphDistribution(
        torch.randn(3, 2, 1024, generator=generator), torch.randn(3, 2, generator=generator)
    )
    targets = torch.randint(0, 2, (3, 1, 32, 32), dtype=torch.uint8, generator=generator)
    _, first = decode_and_score(
        distribution,
        targets,
        threshold=0.3,
        strategy="mode_threshold",
        generator=torch.Generator().manual_seed(3),
    )
    _, second = decode_and_score(
        distribution,
        targets,
        threshold=0.5,
        strategy="sample_pixels",
        generator=torch.Generator().manual_seed(4),
    )
    assert torch.equal(first, second)


def test_filtered_full_gallery_candidates_match_direct_content_search():
    generator = torch.Generator().manual_seed(14)
    bank = torch.randint(0, 2, (13, 1, 32, 32), generator=generator).float()
    bank[10] = bank[4]  # Deterministic ID tie breaking must also survive filtering.
    predictions = torch.cat((bank[[0, 2, 10]], bank[[5]]))
    all_ids, all_distances = nearest_glyphs(
        predictions, bank, torch.arange(13), query_chunk=2, gallery_chunk=3, k=9
    )
    ids, distances = content_candidates(all_ids, all_distances, torch.arange(4), 5)
    oracle_ids, oracle_distances = nearest_glyphs(
        predictions, bank[4:], torch.arange(4, 13), query_chunk=2, gallery_chunk=3, k=5
    )
    assert torch.equal(ids, oracle_ids)
    assert torch.equal(distances, oracle_distances)


def test_v1_hidden_chunk_path_matches_original_forward():
    torch.manual_seed(9)
    model = GlyphGPT(small_config()).eval()
    inputs = torch.randint(0, 2, (1, 3, 1, 32, 32), dtype=torch.uint8)
    mask = torch.ones(1, 3, dtype=torch.bool)
    with torch.inference_mode():
        expected = model(inputs, attention_mask=mask)
        hidden = forward_hidden(model, inputs, mask)[mask]
        actual = distribution_for_hidden(model, hidden).logits
    torch.testing.assert_close(actual, expected[mask].float(), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("components", [1, 2])
def test_full_split_covers_chunk_edges_padding_and_all_target_categories(tmp_path, components):
    dataset = corpus_fixture(tmp_path)
    torch.manual_seed(12)
    model = StructuredGlyphGPT(small_config(), components=components, init_noise=0).eval()
    with torch.no_grad():
        model.pixel_head.weight.zero_()
        desired = dataset.glyph_bank[4].float().flatten() * 8 - 4
        model.pixel_head.bias.copy_(desired.repeat(components))
    result = score_full_split(
        model,
        dataset,
        torch.device("cpu"),
        batch_size=2,
        head_chunk_size=1,
        precision="fp32",
        query_chunk=1,
        gallery_chunk=3,
    )
    assert result["coverage"]["blocks_seen"] == 5
    assert result["coverage"]["targets_seen"] == 8
    assert result["coverage"]["full_split_verified"]
    assert result["content_only"]["tiles"] == 5
    assert result["han_only"]["tiles"] == 4
    assert result["punctuation_only"]["tiles"] == 1
    assert result["controls_only"]["tiles"] == 3
    assert result["all_targets"]["exact_bitmap_match"] == pytest.approx(2 / 8)
    assert result["content_only"]["retrieval"]["top1_accuracy"] == pytest.approx(2 / 5)
    assert "bce_per_pixel" not in result["all_targets"]
    assert result["all_targets"]["nll_nats_per_grid"] == pytest.approx(
        result["all_targets"]["nll_per_pixel"] * 1024
    )


def test_control_scoring_uses_full_gallery_and_empty_content_is_not_zero_performance(tmp_path):
    dataset = corpus_fixture(tmp_path)
    scorer = SplitAccumulator(dataset, torch.device("cpu"), query_chunk=1, gallery_chunk=3)
    eos = dataset.glyph_bank[[2]]
    scorer.add(eos, eos, torch.tensor([2]), torch.tensor([0.1]))
    result = scorer.result()
    assert result["controls_only"]["retrieval"]["top1_accuracy"] == 1
    assert result["controls_only"]["retrieval"]["gallery_size"] == 8
    assert result["content_only"]["nll_per_pixel"] is None
    assert result["content_only"]["retrieval"]["top1_accuracy"] is None


def test_false_expected_target_count_cannot_produce_full_split_completion(tmp_path):
    dataset = corpus_fixture(tmp_path)
    dataset.target_count += 1
    model = StructuredGlyphGPT(small_config(), components=1).eval()
    with pytest.raises(ValueError, match="full split"):
        score_full_split(
            model,
            dataset,
            torch.device("cpu"),
            batch_size=3,
            head_chunk_size=3,
            precision="fp32",
            gallery_chunk=3,
        )


@pytest.mark.parametrize(
    "field", ["config", "data_sha256", "data_manifest_sha256", "data_verification_sha256"]
)
def test_checkpoint_binding_rejects_changed_config_or_data(field):
    config = {"model": {"hidden_size": 32}, "head": {"components": 2}}
    original = {
        "config": config,
        "config_sha256": canonical_hash(config),
        "git_commit": "test",
        "data_sha256": {"glyph_bank.npz": "bank"},
        "data_manifest_sha256": "manifest",
        "data_verification_sha256": "verification",
    }
    saved, current = {"metadata": copy.deepcopy(original)}, copy.deepcopy(original)
    assert validate_checkpoint_identity(saved, current)["config_and_data_identity_verified"]
    if field == "config":
        saved["metadata"]["config"]["head"]["components"] = 3
    else:
        current[field] = "changed"
    with pytest.raises(ValueError):
        validate_checkpoint_identity(saved, current)
