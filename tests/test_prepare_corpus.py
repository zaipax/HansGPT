import bz2
from collections import Counter

import numpy as np
from opencc import OpenCC

from hansgpt_research.prepare_corpus import (
    clean_paragraphs,
    control_tiles,
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
