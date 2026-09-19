import bz2
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from opencc import OpenCC

from hansgpt_research.prepare_corpus import (
    CorpusStore,
    artifact_reason,
    clean_paragraphs,
    control_tiles,
    download,
    modelscope_pages,
    plain_wikitext,
    split_for_page,
    wiki_pages,
)


def test_mixed_paragraphs_are_rejected_without_joining_neighbors():
    stats = Counter()
    result = clean_paragraphs(
        "前面的中文段落。\n这个工具支持Python语言。\n後面的繁體中文段落。\n今年是2026年。",
        OpenCC("t2s"),
        stats,
        min_han=2,
        max_length=100,
    )
    assert result == ["前面的中文段落。", "后面的繁体中文段落。"]
    assert stats["rejected_non_chinese_or_unsupported_symbols"] == 2


def test_removed_markup_keeps_a_boundary_and_visible_chinese_links_survive():
    assert plain_wikitext("前文{{模板|数据}}后文") == "前文\n后文"
    assert plain_wikitext("学习[[语言学|语言知识]]。") == "学习语言知识。"


def test_namespace_redirect_and_revision_provenance(tmp_path):
    path = tmp_path / "wiki.xml.bz2"
    xml = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
    <page><title>正文</title><ns>0</ns><id>17</id>
    <revision><id>101</id><text>这是正文内容。</text></revision></page>
    <page><title>重定向</title><ns>0</ns><id>18</id><redirect title="正文"/>
    <revision><id>102</id><text>重定向内容</text></revision></page>
    <page><title>讨论</title><ns>1</ns><id>19</id>
    <revision><id>103</id><text>讨论内容</text></revision></page></mediawiki>"""
    path.write_bytes(bz2.compress(xml.encode()))
    pages = list(wiki_pages(path, 100))
    assert len(pages) == 1
    assert pages[0]["page_id"] == "17"
    assert pages[0]["revision_id"] == "101"
    assert pages[0]["raw"] == "这是正文内容。"
    assert split_for_page("17", 20260907) == split_for_page(pages[0]["page_id"], 20260907)


def test_control_tiles_are_distinct_binary_images():
    tiles = control_tiles()
    assert all(tile.shape == (32, 32) for tile in tiles)
    assert all(np.isin(tile, [0, 1]).all() for tile in tiles)
    assert len({tile.tobytes() for tile in tiles}) == 4
    assert not tiles[0].any()


def test_modelscope_plaintext_does_not_silently_parse_markup(tmp_path):
    path = tmp_path / "train.parquet"
    raw = "这是已经提取完成的中文正文。\n前文{{模板}}后文。"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "17",
                    "url": "https://zh.wikipedia.org/wiki/正文",
                    "title": "正文",
                    "text": raw,
                },
                {
                    "id": "18",
                    "url": "https://zh.wikipedia.org/wiki/次页",
                    "title": "次页",
                    "text": "内容",
                },
            ]
        ),
        path,
    )
    pages = list(modelscope_pages([path], "pinned-file-revision", 1))
    assert len(pages) == 1
    assert pages[0]["revision_id"] is None
    assert pages[0]["file_revision"] == "pinned-file-revision"
    assert pages[0]["raw"] == raw
    stats = Counter()
    assert clean_paragraphs(
        raw, OpenCC("t2s"), stats, min_han=2, max_length=100, wikitext=False
    ) == ["这是已经提取完成的中文正文。"]
    assert stats["rejected_non_chinese_or_unsupported_symbols"] == 1
    assert len(list(modelscope_pages([path], "pinned", 0))) == 2


def test_download_requires_size_and_sha256_even_for_existing_files(tmp_path):
    path = tmp_path / "shard.parquet"
    path.write_bytes(b"complete shard")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    assert download("unused", path, size=14, sha256=checksum) == path
    with pytest.raises(ValueError, match="size mismatch"):
        download("unused", path, size=15, sha256=checksum)
    path.write_bytes(b"tampered shard")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download("unused", path, size=14, sha256=checksum)


def test_complete_partial_file_is_verified_before_publication(tmp_path):
    path = tmp_path / "shard.parquet"
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(b"good")
    checksum = hashlib.sha256(b"good").hexdigest()
    assert download("unused", path, size=4, sha256=checksum) == path
    assert path.read_bytes() == b"good" and not partial.exists()


def test_global_dedup_rejects_cross_split_exact_punctuation_and_near_copies(tmp_path):
    store = CorpusStore(tmp_path / "records.sqlite")
    stats = Counter()

    def record(text, split):
        return {
            "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "split": split,
        }

    text = "".join(chr(0x4E00 + index) for index in range(200))
    assert store.add(record(text + "。", "train"), stats)
    assert not store.add(record(text + "。", "test"), stats)
    assert not store.add(record(text + "！", "validation"), stats)
    near = text[:100] + "雪" + text[101:]
    assert not store.add(record(near + "。", "test"), stats)
    distinct = "".join(chr(0x6000 + index) for index in range(200))
    assert store.add(record(distinct + "。", "test"), stats)
    assert stats["rejected_exact_duplicate"] == 1
    assert stats["rejected_punctuation_variant"] == 1
    assert stats["rejected_near_duplicate"] == 1
    assert len(list(store.records("train"))) == len(list(store.records("test"))) == 1
    store.close()


def test_rejected_foreign_text_bypasses_opencc_and_conversion_is_rechecked():
    class Converter:
        def __init__(self):
            self.calls = []

        def convert(self, text):
            self.calls.append(text)
            return "意外出现Latin" if text == "转换异常。" else text

    converter = Converter()
    stats = Counter()
    assert clean_paragraphs(
        "混杂English。\n含有１２３数字。\n纯中文段落。\n转换异常。",
        converter,
        stats,
        min_han=2,
        max_length=100,
        wikitext=False,
    ) == ["纯中文段落。"]
    assert converter.calls == ["纯中文段落。", "转换异常。"]
    assert stats["rejected_before_opencc"] == 2
    assert stats["opencc_paragraphs_processed"] == 2
    assert stats["rejected_non_chinese_or_unsupported_symbols"] == 3


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("博南格莱加来（）面积，位于法国，是当地的一座村镇。", "empty_parentheses"),
        ("（）面积，位于法国，是当地的一座村镇。", "empty_parentheses"),
        ("格拉沃利讷（，；）是法国的一座城市。", "empty_parentheses"),
        ("长吻梅花鲈（学名：）为河鲈科的一种鱼类。", "empty_labeled_parentheses"),
        ("这个城镇（法语：；）位于法国北部。", "empty_labeled_parentheses"),
        ("这个城镇面积，位于法国北部。", "missing_numeric_slot"),
        ("这个城镇总面积约为，位于法国北部。", "missing_numeric_slot"),
        ("这个地区总人口为，主要从事农业。", "missing_numeric_slot"),
    ],
)
def test_upstream_extraction_artifacts_drop_whole_paragraphs(text, reason):
    stats = Counter()
    assert artifact_reason(text) == reason
    assert clean_paragraphs(
        f"前面的正文段落。\n{text}\n后面的正文段落。",
        OpenCC("t2s"),
        stats,
        min_han=2,
        max_length=200,
        wikitext=False,
    ) == ["前面的正文段落。", "后面的正文段落。"]
    assert stats[f"rejected_{reason}"] == 1
    assert stats["candidate_paragraphs"] == 3


@pytest.mark.parametrize(
    "text",
    [
        "这座城市（位于法国北部）拥有悠久历史。",
        "这个城镇面积为十平方公里，位于法国北部。",
        "这里讨论土地面积，位于第二章的例题介绍计算方法。",
        "长吻梅花鲈（别名：梅花鲈）为河鲈科的一种鱼类。",
        "引号也是标点（标点：，）示例中的一部分。",
        "这片土地的面积，人口数量和地理位置都已经记录。",
    ],
)
def test_artifact_filters_preserve_complete_explanations_and_values(text):
    assert artifact_reason(text) is None


def test_source_scan_evidence_counts_last_yield_and_unvisited_shards(tmp_path):
    paths = [tmp_path / f"part-{index}.parquet" for index in range(2)]
    for index, path in enumerate(paths):
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "id": str(index),
                        "url": "https://zh.wikipedia.org/wiki/正文",
                        "title": "正文",
                        "text": "正文",
                    },
                ]
            ),
            path,
        )

    def scan():
        return {
            "scanned_rows": 0,
            "files": [
                {"file": path.name, "expected_rows": 1, "scanned_rows": 0, "completed": False}
                for path in paths
            ],
        }

    partial = scan()
    iterator = modelscope_pages(paths, "revision", 0, partial)
    next(iterator)
    assert partial["scanned_rows"] == 1
    assert partial["files"][0]["completed"]
    assert not partial["files"][1]["completed"]
    iterator.close()
    complete = scan()
    assert len(list(modelscope_pages(paths, "revision", 0, complete))) == 2
    assert complete["scanned_rows"] == 2 and all(item["completed"] for item in complete["files"])


def load_verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_corpus", Path(__file__).parents[1] / "scripts" / "verify_corpus.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_verification_invalidates_previous_success_receipt(tmp_path):
    receipt = tmp_path / "verification.json"
    receipt.write_text('{"passed":true}')
    with pytest.raises(FileNotFoundError):
        load_verifier().verify(tmp_path)
    assert not receipt.exists()


def test_verifier_rejects_reused_content_addresses_before_reading_bitmaps(tmp_path):
    (tmp_path / "glyph_inventory.json").write_text(
        json.dumps(
            {
                "characters": {"一": 4, "二": 4},
                "controls": {"0": "PAD", "1": "BOS", "2": "EOS", "3": "NEWLINE"},
            }
        )
    )
    with pytest.raises(ValueError, match="inventory"):
        load_verifier().verify_glyphs(tmp_path, {})


def test_independent_sample_audit_compares_exact_shingles_without_lsh():
    module = load_verifier()
    text = "".join(chr(0x4E00 + index) for index in range(200))
    sample = {"sample_id": "heldout", "split": "test", "text": text}
    audit = module.SampledLeakageAudit([sample])
    near = text[:100] + "雪" + text[101:]
    audit.observe(
        {
            "sample_id": "training",
            "source_page_id": "17",
            "source_url": "public",
            "text": near,
            "text_sha256": hashlib.sha256(near.encode()).hexdigest(),
        },
        near,
    )
    report = audit.report()
    assert report["training_paragraphs_scanned"] == 1
    assert report["flagged_heldout_paragraphs"] == 1
    assert 0.9 <= report["sample_results"][0]["maximum_train_jaccard"] < 1


def test_verifier_compares_content_to_rerender_and_checks_control_frames(tmp_path, monkeypatch):
    module = load_verifier()
    raw = tmp_path / "raw"
    font_path = raw / "20231101" / "NotoSansCJKsc-Regular.otf"
    font_path.parent.mkdir(parents=True)
    font_path.write_bytes(b"test font identity")
    tile = np.zeros((32, 32), dtype=np.uint8)
    tile[16, 8:24] = 1
    bitmaps = np.stack([*control_tiles(), tile])
    np.savez_compressed(tmp_path / "glyph_bank.npz", bitmaps=bitmaps)
    (tmp_path / "glyph_inventory.json").write_text(
        json.dumps(
            {
                "characters": {"一": 4},
                "collisions": [],
                "controls": {"0": "PAD", "1": "BOS", "2": "EOS", "3": "NEWLINE"},
            }
        )
    )
    manifest = {
        "configuration": {"raw": str(raw), "snapshot": "20231101"},
        "source": {"font_sha256": hashlib.sha256(font_path.read_bytes()).hexdigest()},
        "glyphs": {
            "shape": [5, 32, 32],
            "font_size": 26,
            "baseline_y": 27,
            "render_threshold": 128,
            "content_characters": 1,
            "controls": 4,
            "collision_groups": [],
        },
    }

    class Font:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def getBestCmap(self):
            return {ord("一"): "glyph"}

    monkeypatch.setattr(module, "TTFont", lambda _: Font())
    monkeypatch.setattr(module.ImageFont, "truetype", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "render_binary", lambda *args: tile)
    assert module.verify_glyphs(tmp_path, manifest)[0] == {"一": 4}
    bitmaps[4, 16, 8] = 0
    np.savez_compressed(tmp_path / "glyph_bank.npz", bitmaps=bitmaps)
    with pytest.raises(ValueError, match="source character and font"):
        module.verify_glyphs(tmp_path, manifest)
    bitmaps[4] = tile
    bitmaps[1, 0, 0] = 0
    np.savez_compressed(tmp_path / "glyph_bank.npz", bitmaps=bitmaps)
    with pytest.raises(ValueError, match="Control bitmap"):
        module.verify_glyphs(tmp_path, manifest)
