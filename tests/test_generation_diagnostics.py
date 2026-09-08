"""Diagnostic selection, raw-feedback, binary scoring and EOS invariants."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hansgpt_research.diagnose_glyph_generation import (
    GlyphScorer,
    IndependentPrediction,
    assert_binary,
    bootstrap_mean,
    flagged_example,
    repetition_metrics,
    rollout,
    select_prompt_records,
    summarize_examples,
    trim_at_eos,
)


def records_for(pages):
    return [
        {"source_page_id": page, "sample_id": f"sample-{index}", "text_sha256": f"text-{index}"}
        for index, page in enumerate(pages)
    ]


def toy_dataset():
    bank = torch.zeros(8, 1, 32, 32, dtype=torch.uint8)
    for index in range(1, len(bank)):
        bank[index, 0, 0, index] = 1
    return SimpleNamespace(
        glyph_bank=bank,
        gallery_ids=[4, 5, 6, 7],
        control_ids={"PAD": 0, "BOS": 1, "EOS": 2, "NEWLINE": 3},
        inventory={"characters": {"甲": 4, "乙": 5, "，": 6, "。": 7}},
    )


def test_prompt_selection_is_page_disjoint_and_has_eight_real_future_targets():
    records = records_for(["short", "a", "a", "b", "c"])
    offsets = np.array([0, 71, 143, 223, 303, 383])
    selected, info = select_prompt_records(
        records, offsets, count=3, max_prompt_length=64, seed=12, split="validation"
    )
    assert {row["source_page_id"] for row in selected} == {"a", "b", "c"}
    assert all(row["stored_tiles_including_bos_eos"] >= 72 for row in selected)
    assert [row["cohort"] for row in selected] == ["development", "audit", "audit"]
    repeated, _ = select_prompt_records(
        iter(records), offsets, count=3, max_prompt_length=64, seed=12, split="validation"
    )
    assert selected == repeated
    assert info["eligible_pages"] == 3


def test_test_prompts_exclude_old_three_paragraph_pages_and_are_never_development():
    records = records_for(["old-a", "old-a", "old-b", "new-a", "old-a", "new-b", "new-c"])
    offsets = np.arange(8) * 80
    selected, info = select_prompt_records(
        records, offsets, count=3, max_prompt_length=64, seed=12, split="test"
    )
    assert {row["source_page_id"] for row in selected} == {"new-a", "new-b", "new-c"}
    assert all(row["cohort"] == "final" for row in selected)
    assert info["excluded_page_ids"] == ["old-a", "old-b"]


def test_prompt_exclusion_and_insufficient_eligibility_fail_instead_of_padding():
    records = records_for(["a", "b"])
    with pytest.raises(ValueError, match="only 1 eligible"):
        select_prompt_records(
            records,
            np.array([0, 80, 160]),
            count=2,
            max_prompt_length=64,
            seed=12,
            split="validation",
            exclude_pages={"b"},
        )
    with pytest.raises(ValueError, match="align"):
        select_prompt_records(
            records,
            np.array([0, 80, 160, 240]),
            count=1,
            max_prompt_length=64,
            seed=12,
            split="validation",
        )


def test_cycle_detection_catches_approximate_repetition_missed_by_exact_test():
    pixels = np.zeros((64, 1, 32, 32), dtype=np.uint8)
    for index in range(64):
        pixels[index].reshape(-1)[index] = 1
    result = repetition_metrics(pixels, near_hamming=8)
    assert result["adjacent_exact_repeat_rate"] == 0
    assert result["adjacent_near_repeat_rate"] == 1
    assert result["has_short_cycle"]
    assert result["unique_bitmaps"] == 64


def test_cycle_detection_catches_alternation_without_adjacent_duplicates():
    generator = np.random.default_rng(17)
    base = generator.integers(0, 2, (2, 1, 32, 32), dtype=np.uint8)
    sequence = np.tile(base, (20, 1, 1, 1))
    result = repetition_metrics(sequence, near_hamming=0)
    assert result["adjacent_exact_repeat_rate"] == 0
    assert result["has_short_cycle"]
    assert result["longest_short_cycle_tiles"] == 40


def test_eos_trim_excludes_padding_and_sentence_end_precedes_eos():
    dataset = toy_dataset()
    sequence = dataset.glyph_bank[[4, 7, 2, 0, 0]].unsqueeze(0)
    trimmed, stop = trim_at_eos(sequence, dataset.glyph_bank[2])
    assert stop == 3 and trimmed.shape[1] == 3
    result = GlyphScorer(dataset, torch.device("cpu")).score(trimmed, requested_horizon=128)
    assert result["observed_tiles"] == 3
    assert result["exact_pad_count"] == 0
    assert result["exact_control_count"] == 1
    assert result["exact_han_only_count"] == 1
    assert result["exact_punctuation_only_count"] == 1
    assert result["ends_with_exact_sentence_punctuation"]
    assert result["content_legal_rate_before_terminal_eos"] == 1
    assert result["initial_legal_content_prefix_tiles"] == 2
    assert result["legal_sentence_prefix_tiles"] == 2
    assert not result["entire_horizon_content_legal"]


def test_legal_sentence_prefix_stops_at_first_invalid_grid():
    dataset = toy_dataset()
    # PAD is an invalid body grid; a later legal period cannot repair the first sentence.
    sequence = dataset.glyph_bank[[4, 4, 0, 7, 4, 7]].unsqueeze(0)
    result = GlyphScorer(dataset, torch.device("cpu")).score(sequence, requested_horizon=6)
    assert result["initial_legal_content_prefix_tiles"] == 2
    assert result["first_illegal_body_position"] == 3
    assert result["legal_sentence_prefix_tiles"] == 0
    assert not result["has_complete_legal_sentence_prefix"]


def test_binary_dtype_and_value_checks_reject_grayscale_and_float_arrays():
    with pytest.raises(ValueError, match="uint8"):
        assert_binary(torch.zeros(1, 1, 1, 32, 32))
    invalid = torch.full((1, 1, 1, 32, 32), 2, dtype=torch.uint8)
    with pytest.raises(ValueError, match="strict binary"):
        assert_binary(invalid)


def test_independent_pixel_sampling_reproducible_and_threshold_modes_equivalent():
    prediction = IndependentPrediction(torch.zeros(1, 1, 1, 32, 32))
    sample_a = prediction.decode(
        threshold=0.3, strategy="sample_pixels", generator=torch.Generator().manual_seed(3)
    )
    sample_b = prediction.decode(
        threshold=0.5, strategy="sample_pixels", generator=torch.Generator().manual_seed(3)
    )
    assert torch.equal(sample_a, sample_b)
    assert sample_a.dtype == torch.uint8 and 0 < sample_a.sum() < 1024
    assert torch.equal(
        prediction.decode(threshold=0.3, strategy="mode_threshold"),
        prediction.decode(threshold=0.3, strategy="sample_threshold"),
    )


def test_paired_rollout_first_step_matches_and_feedback_is_really_raw():
    class ToggleModel:
        config = SimpleNamespace(max_position_embeddings=128)

        def __init__(self):
            self.inputs = []

        def __call__(self, glyphs, **kwargs):
            self.inputs.append(glyphs.clone())
            # Toggle all pixels, making the second output depend on the actual feedback.
            return (1 - 2 * glyphs.float()) * 5, object()

    prompt = torch.zeros(1, 2, 1, 32, 32, dtype=torch.uint8)
    reference = torch.zeros(1, 8, 1, 32, 32, dtype=torch.uint8)
    raw_model, teacher_model = ToggleModel(), ToggleModel()
    arguments = {
        "steps": 8,
        "threshold": 0.5,
        "strategy": "mode_threshold",
        "seed": 3,
        "precision": "fp32",
    }
    raw = rollout(raw_model, prompt, reference, **arguments)
    teacher = rollout(teacher_model, prompt, reference, teacher_forcing=True, **arguments)
    assert torch.equal(raw["tiles"][:, :1], teacher["tiles"][:, :1])
    assert not torch.equal(raw["tiles"][:, 1:2], teacher["tiles"][:, 1:2])
    assert torch.equal(raw_model.inputs[1], raw["tiles"][:, :1])
    assert torch.equal(teacher_model.inputs[1], reference[:, :1])
    assert len(raw["reference_nll_per_pixel"]) == 8


def test_natural_stop_does_not_flag_valid_shorter_sequence_as_missing_horizon():
    metrics = {
        "has_short_cycle": False,
        "entire_horizon_content_legal": False,
        "all_pre_eos_content_legal": True,
    }
    example = {
        "natural_stop": True,
        "terminated_by_eos": True,
        "early_eos_before_eight": False,
        "horizons": {"128": metrics},
    }
    assert not flagged_example(example, 128)
    example["early_eos_before_eight"] = True
    assert flagged_example(example, 128)


def test_bootstrap_is_over_prompts_and_is_seed_reproducible():
    first = bootstrap_mean([0.0, 1.0, 1.0], 8, samples=100)
    assert first == bootstrap_mean([0.0, 1.0, 1.0], 8, samples=100)
    assert first["prompts"] == 3
    assert first["mean"] == pytest.approx(2 / 3)


def test_horizon_summaries_label_stopping_as_whole_rollout_state():
    example = {
        "cohort": "audit",
        "prompt_length": 16,
        "threshold": 0.45,
        "strategy": "mode_threshold",
        "natural_stop": True,
        "terminated_by_eos": True,
        "early_eos_before_eight": False,
        "hit_length_cap": False,
        "horizons": {"32": {}, "128": {}},
    }
    summaries = summarize_examples([example], seed=3)
    assert len(summaries) == 2
    for summary in summaries:
        assert summary["natural_eos_rate"] == 1
        assert summary["hit_length_cap_rate"] == 0
        assert set(summary["stopping_aggregation_scope"]) == {
            "natural_eos_rate",
            "early_eos_before_eight_rate",
            "hit_length_cap_rate",
        }
        assert all(
            "whole rollout" in text for text in summary["stopping_aggregation_scope"].values()
        )
