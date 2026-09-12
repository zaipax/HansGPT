import hashlib
import runpy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
helpers = runpy.run_path(str(ROOT / "scripts/reorganize_corpus.py"))
legacy = runpy.run_path(str(ROOT / "scripts/prepare_multidomain_corpus.py"))
SPEC = {"adapter": "jsonl_text", "family": "education_web"}
ONE = "这是一段完整的中文说明文字，可以用于检验原文顺序以及段落之间的关系。"
TWO = "另一段说明文字仍然属于同一篇文章，应当在没有遗漏内容时保留连续的上下文。"
THREE = "最后一段也包含完整的中文内容，用来检查去重或者过滤之后是否正确建立断点。"


def expected(texts, units=None, parents=None):
    values = np.empty(len(texts), dtype=helpers["INDEX_DTYPE"])
    values["row"] = 0
    values["split"] = 0
    values["unit"] = list(range(len(texts))) if units is None else units
    values["parent"] = list(range(len(texts))) if parents is None else parents
    values["digest"] = [np.void(hashlib.sha256(t.encode()).digest()) for t in texts]
    return values


def recover(raw, records):
    return helpers["recover_row"](
        raw, records, SPEC, legacy["cleaned_units"], legacy["FastOpenCC"]()
    )


def test_blank_formatting_preserves_continuity_and_original_coordinates():
    raw = ONE + "\n\n　\n" + TWO
    records = expected([ONE, TWO])
    line, rank, join, opaque = recover(raw, records)
    assert line.tolist() == [0, 3] and rank.tolist() == [0, 1]
    assert join.tolist() == [False, True] and not opaque
    helpers["verify_row_edges"](
        raw, records, line, rank, join, SPEC, legacy["cleaned_units"], legacy["FastOpenCC"]()
    )


def test_filtered_line_is_a_barrier_despite_compacted_unit_numbers():
    raw = ONE + "\n包含English的内容不能保留。\n" + TWO
    line, rank, join, _ = recover(raw, expected([ONE, TWO]))
    assert line.tolist() == [0, 2] and rank.tolist() == [0, 2] and not join.any()


def test_deduplicated_middle_paragraph_is_also_a_barrier():
    _, _, join, _ = recover("\n".join([ONE, TWO, THREE]), expected([ONE, THREE], units=[0, 2]))
    assert not join.any()


def test_section_marker_breaks_even_when_it_is_a_retained_paragraph():
    heading = "篇二：这是一段完整的中文说明文字，用于测试新的一篇文章开始时的段落边界。"
    _, _, join, _ = recover("\n".join([ONE, heading, TWO]), expected([ONE, heading, TWO]))
    assert join.tolist() == [False, False, True]


def test_hash_mismatch_and_forged_join_are_rejected():
    with pytest.raises(ValueError):
        recover(ONE, expected([TWO]))
    raw = ONE + "\n存在123的被删除内容。\n" + TWO
    records = expected([ONE, TWO])
    line, rank, join, _ = recover(raw, records)
    join[1] = True
    with pytest.raises(ValueError):
        helpers["verify_row_edges"](
            raw, records, line, rank, join, SPEC, legacy["cleaned_units"], legacy["FastOpenCC"]()
        )


def test_gaps_in_parent_order_and_oversized_rows_are_not_joined():
    _, _, join, _ = recover(ONE + "\n" + TWO, expected([ONE, TWO], parents=[2, 4]))
    assert not join.any()
    line, _, join, opaque = recover(ONE * 10000, expected([ONE]))
    assert opaque and line[0] == -1 and not join.any()


def test_poetry_and_synthetic_article_excerpts_keep_original_boundaries():
    assert not helpers["join_allowed"]({"adapter": "jsonl_text"}, 3)
    assert not helpers["join_allowed"]({"adapter": "jsonl_article"}, 8)
    assert helpers["join_allowed"]({"adapter": "parquet_text"}, 14)
