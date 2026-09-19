"""Guard evaluation against EOS/padding inflation and repeated source pages."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

helpers = runpy.run_path(str(Path(__file__).parents[1] / "scripts/evaluate_attention_abc.py"))


def grid(bits):
    value = np.zeros((1, 32, 32), dtype=np.uint8)
    value.reshape(-1)[:bits] = 1
    return value


def test_eos_and_padding_do_not_inflate_legal_output():
    key = helpers["bitmap_key"]
    glyph, eos, pad = grid(2), grid(1), grid(0)
    output = np.array([[glyph, eos, pad], [eos, pad, pad]])
    prompts = np.array([[glyph], [glyph]])
    result = helpers["generation_summary"](
        prompts, output, {key(glyph): ["中"]}, {key(eos): "EOS", key(pad): "PAD"}
    )
    assert result["summary"]["body_grids"] == 1
    assert result["summary"]["fully_legal_nonempty"] == 1
    assert result["summary"]["terminated"] == 2
    assert result["samples"][1]["exact_content_rate"] is None
    assert result["samples"][0]["exact_bitmap_transcription"] == "中"


def test_invalid_and_ambiguous_bitmaps_are_not_silently_projected():
    key = helpers["bitmap_key"]
    result = helpers["transcribe"]([grid(2), grid(3)], {key(grid(2)): ["甲", "乙"]}, {})
    assert result == "[甲/乙]□"


def test_generation_summary_catches_multiglyph_cycle_with_zero_adjacent_repeats():
    key = helpers["bitmap_key"]
    sequence = np.array([[grid(8), grid(16), grid(24)] * 12])
    result = helpers["generation_summary"](
        sequence[:, :1],
        sequence,
        {key(grid(8)): ["、"], key(grid(16)): ["公"], key(grid(24)): ["路"]},
        {},
    )
    assert result["summary"]["adjacent_repeat_rate"] == 0
    assert result["summary"]["exact_cycle_samples"] == 1
    assert result["samples"][0]["exact_repetition"]["longest_short_cycle_tiles"] == 36


def test_prompt_selection_is_deterministic_and_page_disjoint(monkeypatch):
    monkeypatch.setattr(
        pq,
        "read_table",
        lambda *a, **k: pa.table({"source_page_id": ["p1", "p1", "p2", "p2", "p3", "p4"]}),
    )
    dataset = SimpleNamespace(data_dir=Path("unused"), offsets=np.arange(7) * 30)
    first = helpers["select_documents"](dataset, 4, 123)
    assert first == helpers["select_documents"](dataset, 4, 123)
    assert len(set(first[1])) == 4


def test_phrase_cycles_longer_than_four_glyphs_are_reported_separately():
    key = helpers["bitmap_key"]
    pattern = [grid(n * 8) for n in range(1, 7)]
    sequence = np.array([pattern * 8])
    result = helpers["generation_summary"](
        sequence[:, :1], sequence, {key(g): [str(i)] for i, g in enumerate(pattern)}, {}
    )
    assert result["summary"]["exact_cycle_samples"] == 0
    assert result["summary"]["phrase_cycle_samples"] == 1
