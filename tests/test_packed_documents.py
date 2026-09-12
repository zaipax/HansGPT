import json
import runpy
from collections import Counter

import numpy as np
import pytest
import torch

from hansgpt_research.glyph_lm import GlyphSequenceDataset
from hansgpt_research.packed_glyph_data import PackedGlyphSequenceDataset, packing_statistics
from hansgpt_research.train_glyph_lm import sequence_lengths


@pytest.mark.parametrize("context", [1, 2, 4, 8, 1024])
def test_packing_preserves_each_target_and_masks_boundaries(tmp_path, context):
    docs = [[1, 4, 5, 2], [1, 6, 2], [1, 7, 8, 9, 4, 5, 6, 2]]
    np.array(sum(docs, []), dtype="<u2").tofile(tmp_path / "train.uint16")
    np.save(tmp_path / "train.offsets.npy", np.r_[0, np.cumsum([len(x) for x in docs])])
    np.savez(tmp_path / "glyph_bank.npz", bitmaps=np.zeros((10, 1, 32, 32), dtype=np.uint8))
    (tmp_path / "glyph_inventory.json").write_text(
        json.dumps(dict(controls=dict(PAD=0, BOS=1, EOS=2, NEWLINE=3), characters={"字": 4}))
    )
    ds = PackedGlyphSequenceDataset(tmp_path, "train", context)
    targets = []
    lengths = []
    for item in ds:
        mask = item["loss_mask"]
        targets.extend(item["target_ids"][mask].tolist())
        lengths.append(int(mask.sum()))
        assert not torch.any(mask & (item["target_ids"] == 1))
    assert targets == sum([x[1:] for x in docs], [])
    assert lengths == sequence_lengths(ds).tolist()
    assert sum(lengths) == GlyphSequenceDataset(tmp_path, "train", context).target_count
    stats = packing_statistics(ds.offsets, context)
    assert stats["effective_targets"] == sum(lengths)
    assert stats["masked_document_transitions"] == 2


def test_document_cleaning_keeps_headings_and_cross_line_quotes_but_not_mixed_script():
    code = runpy.run_path("scripts/prepare_document_corpus.py")
    legacy = runpy.run_path("scripts/prepare_multidomain_corpus.py")
    replay = runpy.run_path("scripts/reorganize_corpus.py")
    first = "春天\n他说：“春天的风吹过这片广阔的大地，\n山上的树木终于慢慢长出了新的叶子。”"
    second = "这一段描述秋天的景色，远处的山林已经染上了金色。"
    raw = first + "\nEnglish广告\n" + second
    segments = code["document_segments"](
        raw, legacy["FastOpenCC"](), set(raw), legacy, replay["SECTION"], Counter()
    )
    assert ["\n".join(t for _, t in s) for s in segments] == [first, second]


def test_heldout_near_index_checks_punctuation_variants_and_near_copies():
    code = runpy.run_path("scripts/prepare_document_corpus.py")
    idx = code["NearIndex"]()
    text = "春天的风吹过这片广阔的大地山上的树木终于慢慢长出了新的叶子"
    idx.add(text)
    assert idx.matches(text)
    assert idx.matches(text + "啊")
    assert not idx.matches("科学实验必须认真记录数据并且验证结果")


def test_explicit_new_corpus_requires_pinned_full_identity(tmp_path, monkeypatch):
    import hansgpt_research.train_glyph_lm as training

    receipt = dict(
        manifest=dict(type="chinese_document_packed_v3", mode="full"),
        manifest_sha256="a" * 64,
        verification={},
        verification_sha256="b" * 64,
        model_consumed_sha256={},
    )
    calls = []

    def verify(_path, **kwargs):
        calls.append(kwargs)
        return receipt

    monkeypatch.setattr(training, "verify_data_readiness", verify)
    monkeypatch.setattr(
        training.subprocess,
        "check_output",
        lambda args, **kwargs: "" if "--porcelain" in args else "main",
    )
    config = dict(
        training=dict(seed=7, precision="fp32"),
        data_requirements=dict(corpus_type="chinese_document_packed_v3", manifest_sha256="a" * 64),
    )
    training.runtime_metadata(config, tmp_path, torch.device("cpu"))
    assert calls[-1]["require_full_snapshot"] is False
    config["data_requirements"]["manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="exact verified full"):
        training.runtime_metadata(config, tmp_path, torch.device("cpu"))
    config["data_requirements"]["manifest_sha256"] = "a" * 64
    receipt["manifest"]["mode"] = "smoke"
    with pytest.raises(ValueError, match="exact verified full"):
        training.runtime_metadata(config, tmp_path, torch.device("cpu"))
    config["data_requirements"] = {}
    training.runtime_metadata(config, tmp_path, torch.device("cpu"))
    assert calls[-1]["require_full_snapshot"] is True
