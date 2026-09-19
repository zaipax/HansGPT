import torch

from hansgpt_research.glyph_readability import score_glyphs


def test_one_pixel_error_is_distinguished_from_exact_match():
    gallery = torch.zeros(2, 1, 32, 32)
    gallery[0, 0, 0, :10] = 1
    gallery[1, 0, 1, :10] = 1
    predicted = gallery[0:1].clone()
    predicted[0, 0, 0, 0] = 0
    row = score_glyphs(predicted, gallery)[0]
    assert row["min_hamming"] == 1
    assert row["best_f1"] > 0.94
    assert row["f1_margin"] > 0.9


def test_blank_near_punctuation_has_zero_foreground_similarity():
    gallery = torch.zeros(2, 1, 32, 32)
    gallery[0, 0, 10, 10] = 1
    gallery[1, 0, 11, 11] = 1
    row = score_glyphs(torch.zeros(1, 1, 32, 32), gallery)[0]
    assert row["min_hamming"] == 1 and row["foreground_pixels"] == 0
    assert row["best_f1"] == 0


def test_close_alternative_glyphs_are_marked_ambiguous():
    gallery = torch.zeros(2, 1, 32, 32)
    gallery[:, 0, 0, :10] = 1
    gallery[0, 0, 1, 0] = 1
    gallery[1, 0, 1, 1] = 1
    predicted = gallery[0:1].clone()
    predicted[0, 0, 1, 0] = 0
    row = score_glyphs(predicted, gallery)[0]
    assert row["best_f1"] > 0.95 and row["f1_margin"] == 0
