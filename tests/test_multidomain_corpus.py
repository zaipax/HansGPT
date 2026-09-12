"""Pure-Chinese boundaries, complete QA pairs and transactional preparation state."""

import runpy
from collections import Counter
from pathlib import Path

from opencc import OpenCC

helpers = runpy.run_path(str(Path(__file__).parents[1] / "scripts/prepare_multidomain_corpus.py"))


def test_mixed_question_rejects_entire_pair_instead_of_retaining_answer():
    row = {"instruction": "患者今年25岁，应该如何处理？", "output": "应该到医院进行检查。"}
    assert helpers["qa_text"](row, OpenCC("t2s")) is None


def test_complete_qa_preserves_roles_and_adjacency():
    row = {"instruction": "請問什麼是股東權益？", "input": "", "output": "股東權益是所有者權益。"}
    assert (
        helpers["qa_text"](row, OpenCC("t2s"))
        == "问：请问什么是股东权益？答：股东权益是所有者权益。"
    )


def test_qa_markdown_formatting_is_removed_without_dropping_content():
    row = {
        "instruction": "请介绍这种症状。",
        "output": "## 症状\n**头痛**是常见症状。 还可能出现疲劳。",
    }
    assert (
        helpers["qa_text"](row, OpenCC("t2s"))
        == "问：请介绍这种症状。答：症状：头痛是常见症状。还可能出现疲劳。"
    )
    row["output"] += "患者年龄为25岁。"
    assert helpers["qa_text"](row, OpenCC("t2s")) is None


def test_rejected_prose_line_is_not_joined_into_surrounding_paragraphs():
    first = "这是一段完整的中文说明文字，用来测试段落边界是否能够得到保留。"
    second = "另一段说明文字也应该独立存在，不能因为中间内容被删除而拼接起来。"
    row = {"text": first + "\n夹杂English的段落必须拒绝。\n" + second}
    result = helpers["cleaned_units"](
        row, {"adapter": "jsonl_text", "family": "academic"}, OpenCC("t2s"), Counter()
    )
    assert result == [first, second]


def test_fragments_repetition_and_unbalanced_delimiters_are_rejected():
    check = helpers["quality_reason"]
    assert check("，这是一段残缺的开头。") == "leading_fragment"
    assert check("这个标题没有句末标点") == "no_sentence_ending"
    assert check("国际、" * 12 + "。") == "repetition"
    assert check("他说：“这些内容没有完成。") == "unbalanced_delimiters"
    assert check("他说：“这些内容已经完整了。”") is None


def test_source_based_domains_and_explicit_heuristic():
    result = helpers["domain_for"](
        "法院在审理诉讼后作出了判决。", {"family": "education_web", "domain_hint": "教育与行业网页"}
    )
    assert result == ("法律与公共事务", "keyword heuristic, not human labels")


def test_resume_preserves_committed_dedup_state(tmp_path):
    store = helpers["ResumableStore"](tmp_path / "corpus.sqlite")
    row = {"text": "这是一段完整的中文测试内容。", "text_sha256": "testhash", "split": "train"}
    assert store.add(row, Counter())
    state = store.state()
    state["row"] = 17
    store.save_state(state)
    store.close()
    resumed = helpers["ResumableStore"](tmp_path / "corpus.sqlite")
    assert resumed.state()["row"] == 17
    assert not resumed.add(row, Counter())
    assert list(resumed.records("train")) == [row]
    resumed.close()


def test_article_excerpts_keep_topic_and_never_join_across_rejected_paragraphs():
    first = "所有者权益反映所有者对企业资产的剩余索取权，是企业财务分析的重要内容。"
    second = "企业的资产和负债会随着经营活动发生变化，分析时应当结合完整的财务资料。"
    row = {"instruction": "请介绍所有者权益。", "output": first + "\n数值为25%的段落。\n" + second}
    result = helpers["cleaned_units"](
        row, {"adapter": "jsonl_article", "family": "finance"}, OpenCC("t2s"), Counter()
    )
    prefix = "主题：请介绍所有者权益。资料摘录："
    assert result == [prefix + first, prefix + second]
    assert all("答：" not in value for value in result)


def test_fast_converter_matches_every_dictionary_key_and_joined_contexts():
    fast = helpers["FastOpenCC"]()
    reference = OpenCC("t2s")
    keys = [key for _, _, table in fast.converter.dict_cache.values() for key in table]
    for key in keys:
        assert fast.convert(key) == reference.convert(key)
    for start in range(0, len(keys), 31):
        text = "这是上下文。" + "".join(keys[start : start + 31]) + "一句完整的话。"
        assert fast.convert(text) == reference.convert(text)


def test_article_context_prefix_cannot_hide_duplicate_body(tmp_path):
    store = helpers["ResumableStore"](tmp_path / "articles.sqlite")
    body = "企业财务分析需要结合完整的财务资料，才能更准确地了解企业的经营情况。"
    first = {"text": "主题：财务。资料摘录：" + body, "text_sha256": "first", "split": "train"}
    second = {"text": "主题：经营。资料摘录：" + body, "text_sha256": "second", "split": "test"}
    assert store.add_with_body(first, body, Counter())
    assert not store.add_with_body(second, body, Counter())
    assert list(store.records("train")) == [first]
    store.close()
