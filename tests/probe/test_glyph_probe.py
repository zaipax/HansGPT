import numpy as np

from hansgpt_research.glyph_probe import (
    DatasetBundle,
    ExperimentConfig,
    _split_indices,
    choose_threshold,
    paired_f1_difference,
)


def test_split_indices_are_disjoint_and_complete() -> None:
    config = ExperimentConfig(
        experiment_name="test",
        model_id="test/model",
        model_revision=None,
        unihan_zip="unihan.zip",
        wikipedia_dump="wiki.bz2",
        font_path="font.otf",
        output_dir="artifacts/test",
        max_characters=100,
    )
    splits = _split_indices(100, config)
    assert (splits == 0).sum() == 70
    assert (splits == 1).sum() == 15
    assert (splits == 2).sum() == 15
    assert set(splits.tolist()) == {0, 1, 2}


def test_choose_threshold_prefers_separable_predictions() -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.8, 0.2]])
    targets = np.asarray([[1, 0], [1, 0]], dtype=np.uint8)
    threshold, score = choose_threshold(probabilities, targets)
    assert 0.2 < threshold <= 0.8
    assert score == 1.0


def test_dataset_bundle_has_character_identity() -> None:
    bundle = DatasetBundle(
        characters=["汉"],
        codepoints=np.asarray([ord("汉")]),
        frequencies=np.asarray([1]),
        token_ids=np.asarray([1]),
        bitmaps=np.zeros((1, 1024), dtype=np.uint8),
        splits=np.asarray([2], dtype=np.uint8),
    )
    assert chr(bundle.codepoints[0]) == bundle.characters[0]


def test_paired_f1_difference_detects_better_predictions() -> None:
    targets = np.asarray([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    perfect = targets.astype(np.float32)
    empty = np.zeros_like(perfect)
    comparison = paired_f1_difference(perfect, 0.5, empty, 0.5, targets, 100, 7)
    assert comparison["foreground_f1_difference"] == 1.0
    assert comparison["confidence_interval_95"] == [1.0, 1.0]
    assert comparison["bootstrap_probability_first_greater"] == 1.0
