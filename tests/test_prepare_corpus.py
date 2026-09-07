import bz2
import hashlib
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from opencc import OpenCC

from hansgpt_research.prepare_corpus import (
    CorpusStore,
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
