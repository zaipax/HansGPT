"""Download pinned ModelScope sources and build a separate pure-Chinese glyph corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import regex
from fontTools.ttLib import TTFont
from opencc import OpenCC
from PIL import ImageFont

from hansgpt_research.prepare_corpus import (
    ALLOWED,
    CONTROL_NAMES,
    HAN,
    PUNCTUATION_MAP,
    SPLITS,
    CorpusStore,
    artifact_reason,
    canonical_han,
    clean_paragraphs,
    control_tiles,
    digest_file,
    download,
    render_binary,
    split_for_page,
)

CAPS = {
    "education_web": 800_000_000,
    "news": 250_000_000,
    "life_qa": 150_000_000,
    "literature": 120_000_000,
    "academic": 100_000_000,
    "classical": 60_000_000,
    "finance": 30_000_000,
    "medicine": 20_000_000,
    "agriculture": 5_000_000,
}
ALIASES = {
    "YeungNLP/firefly-pretrain-dataset": "firefly",
    "AI-ModelScope/COIG-CQIA": "coig_cqia",
    "opencsg/Fineweb-Edu-Chinese-V2.1": "fineweb_edu",
}
FONT_SHA = "2c76254f6fc379fddfce0a7e84fb5385bb135d3e399294f6eeb6680d0365b74b"
WORDS = {
    "法律与公共事务": ("法院", "诉讼", "刑法", "民法", "司法", "判决", "法律", "立法"),
    "金融与经济": ("银行", "金融", "证券", "股东", "债券", "财务", "股票", "经济"),
    "医疗与健康": ("医学", "疾病", "患者", "临床", "药物", "医院", "治疗", "健康"),
    "学术与科技": ("物理", "化学", "算法", "工程", "技术", "实验", "数学", "科学"),
    "教育": ("教学", "教师", "学生", "课程", "学校", "课堂", "学习", "教育"),
    "历史与文化": ("历史", "哲学", "文学", "文化", "诗歌", "艺术", "作家", "小说"),
    "农业与环境": ("农业", "农作物", "生态", "环境", "种植", "土壤", "水稻", "气候"),
}
REPETITION = regex.compile(r"(.{1,16})\1{5,}")
SPAM = regex.compile(r"扫码关注|点击下载|关注公众号|加微信|博彩|色情网站|免费领取|免责声明")


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def space_guard(root, minimum_gib=40):
    fs = os.statvfs(root)
    if fs.f_bavail * fs.f_frsize < minimum_gib * 2**30:
        raise OSError("Free disk below reserve; current training takes priority")


def quality_reason(text):
    if text[0] in "，。！？；：、…—·":
        return "leading_fragment"
    if text[-1] not in "。！？；”’」』":
        return "no_sentence_ending"
    if REPETITION.search(text):
        return "repetition"
    if SPAM.search(text):
        return "boilerplate_or_spam"
    stack = []
    pairs = {"）": "（", "】": "【", "》": "《", "」": "「", "』": "『", "”": "“", "’": "‘"}
    for char in text:
        if char in pairs.values():
            stack.append(char)
        elif char in pairs and (not stack or stack.pop() != pairs[char]):
            return "unbalanced_delimiters"
    return "unbalanced_delimiters" if stack else None


class FastOpenCC:
    """Skip conversion only when no changed mapping can occur in the input.

    Inspect the installed, pinned OpenCC dictionaries once. Every differing source
    character is a trigger; changed-length mappings conservatively use the whole
    source key. Otherwise delegate to the original converter without approximation.
    """

    def __init__(self):
        self.converter = OpenCC("t2s")
        self.converter.convert("漢字")
        self.triggers = set()
        for _, _, mappings in self.converter.dict_cache.values():
            for source, alternatives in mappings.items():
                target = alternatives.split(" ")[0]
                if len(source) == len(target):
                    self.triggers.update(a for a, b in zip(source, target, strict=True) if a != b)
                elif source != target:
                    self.triggers.update(source)

    def convert(self, text):
        return text if self.triggers.isdisjoint(text) else self.converter.convert(text)


def normalized_lines(value):
    if not isinstance(value, str):
        return None
    parts = []
    for line in value.splitlines():
        text = unicodedata.normalize("NFC", line).translate(PUNCTUATION_MAP).strip()
        if not text:
            continue
        heading = bool(regex.match(r"^#{1,6}\s+", text))
        text = regex.sub(r"^#{1,6}\s+", "", text)
        text = regex.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = regex.sub(r"(?<=[，。！？；：、])[ \t\u3000]+", "", text)
        text = regex.sub(r"[ \t\u3000]+(?=[，。！？；：、])", "", text)
        if heading and text[-1] not in "：。！？":
            text += "："
        parts.append(text)
    return parts


def pure_field(value, converter):
    parts = normalized_lines(value)
    if parts is None or any(not ALLOWED.fullmatch(line) or artifact_reason(line) for line in parts):
        return None
    return converter.convert("".join(parts))


def qa_text(row, converter):
    """Keep the entire Q/A pair; never salvage an answer after deleting a mixed-script question."""
    values = []
    for name in ("instruction", "input", "output"):
        value = pure_field(row.get(name, ""), converter)
        if value is None:
            return None
        values.append(value)
    question, extra, answer = values
    if not question or not answer:
        return None
    return "问：" + question + ("补充：" + extra if extra else "") + "答：" + answer


def cleaned_units(row, spec, converter, stats):
    if spec["adapter"] == "jsonl_qa":
        text = qa_text(row, converter)
        stats["qa_pairs_seen"] += 1
        if text is None:
            stats["rejected_whole_qa_pair"] += 1
            return []
        candidates = [text] if len(text) <= 4096 and len(HAN.findall(text)) >= 20 else []
    elif spec["adapter"] == "jsonl_article":
        topic = pure_field(row.get("instruction", ""), converter)
        extra = pure_field(row.get("input", ""), converter)
        body = normalized_lines(row.get("output"))
        if not topic or extra is None or body is None:
            stats["rejected_article_topic"] += 1
            return []
        paragraphs = clean_paragraphs(
            "\n".join(body), converter, stats, min_han=20, max_length=3500, wikitext=False
        )
        prefix = "主题：" + topic + ("补充：" + extra if extra else "") + "资料摘录："
        candidates = [
            prefix + paragraph
            for paragraph in paragraphs
            if len(prefix + paragraph) <= 4096 and not quality_reason(paragraph)
        ]
        stats["encyclopedia_excerpts"] += len(candidates)
    else:
        raw = row.get("text")
        if not isinstance(raw, str):
            stats["rejected_schema"] += 1
            return []
        candidates = clean_paragraphs(
            raw,
            converter,
            stats,
            min_han=8 if spec["family"] == "classical" else 20,
            max_length=4096,
            wikitext=False,
        )
    accepted = []
    for text in candidates:
        if not ALLOWED.fullmatch(text):
            stats["rejected_post_normalization"] += 1
            continue
        reason = artifact_reason(text) or quality_reason(text)
        if reason:
            stats["rejected_quality_" + reason] += 1
        else:
            accepted.append(text)
    return accepted


def domain_for(text, spec):
    if spec["family"] not in {"education_web", "news"}:
        return spec["domain_hint"], "source subset"
    scores = {domain: sum(word in text for word in words) for domain, words in WORDS.items()}
    best = max(scores, key=scores.get)
    return (
        best if scores[best] >= 2 else spec["domain_hint"]
    ), "keyword heuristic, not human labels"


def raw_path(raw, spec):
    path = Path(spec["path"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe source path")
    return raw / ALIASES[spec["dataset"]] / path


def fetch_one(raw, spec):
    space_guard(raw)
    query = urlencode(
        {"Source": "SDK", "Revision": spec["revision"], "FilePath": spec["path"], "View": "false"}
    )
    url = "https://modelscope.cn/api/v1/datasets/" + spec["dataset"] + "/repo?" + query
    return download(url, raw_path(raw, spec), size=spec["size"], sha256=spec["sha256"])


def rows(path, adapter):
    if adapter == "parquet_text":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()
    else:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("Source JSONL row is not an object")
                    yield value


class ResumableStore(CorpusStore):
    def __init__(self, path):
        if path.exists():
            self.connection = sqlite3.connect(path)
            self.connection.execute("PRAGMA cache_size=-65536")
            self.connection.execute(
                "SELECT id, text_hash, canonical_hash, split FROM records LIMIT 1"
            )
        else:
            super().__init__(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, payload TEXT)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS exclusions (hash TEXT PRIMARY KEY) WITHOUT ROWID"
        )

    def state(self):
        row = self.connection.execute("SELECT payload FROM state WHERE id=1").fetchone()
        return (
            json.loads(row[0])
            if row
            else {
                "file": 0,
                "row": 0,
                "seeded": False,
                "stats": {},
                "family_han": {},
                "characters": "",
                "bad_characters": "",
                "scan": [],
            }
        )

    def save_state(self, state):
        self.connection.execute("INSERT OR REPLACE INTO state VALUES (1,?)", (json.dumps(state),))
        self.commit()

    def excluded(self, text):
        hashed = hashlib.sha256(canonical_han(text).encode()).hexdigest()
        return self.connection.execute(
            "SELECT 1 FROM exclusions WHERE hash=?", (hashed,)
        ).fetchone()

    def add_with_body(self, record, body, stats):
        if body == record["text"]:
            return self.add(record, stats)
        indexed = {**record, "text": body, "text_sha256": hashlib.sha256(body.encode()).hexdigest()}
        if not self.add(indexed, stats):
            return False
        key = hashlib.sha256(canonical_han(body).encode()).hexdigest()
        self.connection.execute(
            "UPDATE records SET payload=? WHERE canonical_hash=?",
            (json.dumps(record, ensure_ascii=False), key),
        )
        return True


def seed_exclusions(store, state, old):
    """Exclude exact copies of all old data; add held-out paragraphs to approximate near index."""
    for split in SPLITS:
        count = 0
        for batch in pq.ParquetFile(old / f"{split}.parquet").iter_batches(batch_size=1024):
            values = batch.to_pylist()
            store.connection.executemany(
                "INSERT OR IGNORE INTO exclusions VALUES (?)",
                [(hashlib.sha256(canonical_han(r["text"]).encode()).hexdigest(),) for r in values],
            )
            if split != "train":
                for r in values:
                    store.add(
                        {
                            "text": r["text"],
                            "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
                            "split": "exclude",
                        },
                        Counter(),
                    )
            store.commit()
            count += len(values)
        print("Seeded existing-corpus exclusions:", split, count, flush=True)
    state["seeded"] = True
    store.save_state(state)


def export(store, state, output, font_path, source_config, config_hash, args):
    font = ImageFont.truetype(str(font_path), size=26)
    characters = sorted(state["characters"], key=ord)
    if len(characters) + 4 > 65536:
        raise ValueError("Too many glyphs for uint16")
    bank = np.stack(control_tiles() + [render_binary(c, font) for c in characters])
    lookup = {c: i + 4 for i, c in enumerate(characters)}
    collisions = {}
    for index, tile in enumerate(bank):
        collisions.setdefault(hashlib.sha256(tile.tobytes()).hexdigest(), []).append(index)
    collision_groups = [ids for ids in collisions.values() if len(ids) > 1]
    if any(any(i < 4 for i in ids) for ids in collision_groups):
        raise ValueError("Control glyph collision")
    np.savez_compressed(output / "glyph_bank.npz", bitmaps=bank)
    atomic_json(
        output / "glyph_inventory.json",
        {
            "controls": {name: i for i, name in enumerate(CONTROL_NAMES)},
            "characters": lookup,
            "collision_ids": collision_groups,
        },
    )
    fields = [
        "sample_id",
        "source_page_id",
        "source_file",
        "source_file_revision",
        "source_url",
        "source_family",
        "domain",
        "domain_method",
        "source_metadata",
        "text",
        "text_sha256",
        "split",
    ]
    schema = pa.schema([(name, pa.string()) for name in fields] + [("han_count", pa.int64())])
    splits, domain_counts, audit = {}, Counter(), {}
    audit_seen = Counter()
    audit_rng = np.random.default_rng(20260915)
    for split in SPLITS:
        offsets, batch, pages, han_total = [0], [], set(), 0
        with (
            (output / f"{split}.uint16").open("wb") as binary,
            (output / f"{split}.txt").open("w", encoding="utf-8") as textfile,
            pq.ParquetWriter(output / f"{split}.parquet", schema, compression="zstd") as writer,
        ):
            for record in store.records(split):
                ids = np.array([1, *(lookup[c] for c in record["text"]), 2], dtype="<u2")
                binary.write(ids.tobytes())
                offsets.append(offsets[-1] + len(ids))
                textfile.write(record["text"] + "\n\n")
                pages.add(record["source_page_id"])
                han_total += record["han_count"]
                domain_counts[record["domain"]] += record["han_count"]
                samples = audit.setdefault(record["source_family"], [])
                audit_seen[record["source_family"]] += 1
                if len(samples) < 30:
                    samples.append(record)
                else:
                    replacement = int(audit_rng.integers(audit_seen[record["source_family"]]))
                    if replacement < 30:
                        samples[replacement] = record
                batch.append(record)
                if len(batch) == 1024:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        np.save(output / f"{split}.offsets.npy", np.asarray(offsets, dtype=np.int64))
        splits[split] = {
            "paragraphs": len(offsets) - 1,
            "source_pages": len(pages),
            "han_characters": han_total,
            "effective_targets": offsets[-1] - len(offsets) + 1,
        }
    if any(not s["paragraphs"] for s in splits.values()):
        raise ValueError("Need nonempty train, validation and test")
    atomic_json(output / "audit_samples.json", audit)
    report = {
        "status": "bounded_multidomain_experiment_corpus",
        "mode": args.mode,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_config_sha256": config_hash,
        "source": source_config,
        "source_scan": state["scan"],
        "splits": splits,
        "family_han": state["family_han"],
        "domain_han": dict(domain_counts),
        "filter_stats": state["stats"],
        "font_path": str(font_path),
        "font_sha256": digest_file(font_path),
        "cleaning": "whole pure-Chinese paragraphs or whole QA pairs; no internal deletion",
        "family_caps_han": CAPS,
        "existing_wiki_exclusions": state["seeded"],
        "split": "98/1/1 hash of complete original document; globally deduplicated paragraphs",
        "limitations": [
            "Keyword domain labels are approximate; facts are not independently verified.",
            "Only selected source files and per-family budgets, not a full source snapshot.",
            "Approximate near dedup; no exhaustive semantic or external-benchmark decontamination.",
            "Paragraphs can lose context; complete question-answer pairs stay together.",
            "Han-only characters do not prove the language of every quotation is Chinese.",
            "Publisher licenses do not clear every upstream copyright; retain attribution.",
            "New corpus version, never silently substituted into the current training run.",
        ],
        "output_sha256": {p.name: digest_file(p) for p in output.iterdir() if p.is_file()},
    }
    atomic_json(output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="configs/datasets/chinese_multidomain_v1.json")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/chinese_multidomain_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interim", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Prepare data only from clean committed source")
    if args.output.resolve() == Path("data/processed/modelscope_zhwiki_full_v1").resolve():
        raise ValueError("Never overwrite active training corpus")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Use fresh output or --resume")
    if (args.output / "verification.json").exists():
        raise FileExistsError("A verified corpus is immutable; use a new version")
    args.output.mkdir(parents=True, exist_ok=True)
    args.interim.mkdir(parents=True, exist_ok=True)
    args.raw.mkdir(parents=True, exist_ok=True)
    pa.set_cpu_count(2)
    source_config = json.loads(Path(args.sources).read_text("utf-8"))
    config_hash = digest_file(Path(args.sources))
    identity = {
        "sources_sha256": config_hash,
        "script_sha256": digest_file(Path(__file__)),
        "mode": args.mode,
        "raw": str(args.raw.resolve()),
        "output": str(args.output.resolve()),
    }
    run_path = args.interim / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != identity:
        raise ValueError("Resume identity mismatch")
    atomic_json(run_path, identity)
    files = [f for f in source_config["files"] if args.mode == "full" or f["smoke"]]
    font_path = Path("data/raw/hansgpt_modelscope_wikipedia/20231101/NotoSansCJKsc-Regular.otf")
    if digest_file(font_path) != FONT_SHA:
        raise ValueError("Font identity differs from existing training corpus")

    def status(phase, **details):
        atomic_json(
            args.interim / "status.json",
            {"phase": phase, "time": datetime.now(UTC).isoformat(), "pid": os.getpid(), **details},
        )

    try:
        status("downloading", files=len(files))
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fetch_one, args.raw, f) for f in files]
            for i, future in enumerate(as_completed(futures), 1):
                future.result()
                status("downloading", completed_files=i, files=len(files))
        store = ResumableStore(args.interim / "records.sqlite")
        state = store.state()
        if args.mode == "full" and not state["seeded"]:
            status("seeding_existing_corpus_exclusions")
            seed_exclusions(store, state, Path("data/processed/modelscope_zhwiki_full_v1"))
        converter = FastOpenCC()
        font = ImageFont.truetype(str(font_path), size=26)
        valid_chars, bad_chars = set(state["characters"]), set(state["bad_characters"])
        checked_chars = set(valid_chars)
        with TTFont(font_path) as source_font:
            supported = set(source_font.getBestCmap())
        stats, family_han = Counter(state["stats"]), Counter(state["family_han"])

        def commit_state():
            state.update(
                stats=dict(stats),
                family_han=dict(family_han),
                characters="".join(sorted(valid_chars)),
                bad_characters="".join(sorted(bad_chars)),
            )
            store.save_state(state)
            status(
                "processing",
                source_index=state["file"],
                sources=len(files),
                accepted_han=sum(family_han.values()),
                family_han=dict(family_han),
            )
            space_guard(args.output)

        for index in range(state["file"], len(files)):
            spec = files[index]
            family = spec["family"]
            completed, seen, reason = True, state["row"], None
            if family_han[family] < CAPS[family]:
                for row_index, row in enumerate(rows(raw_path(args.raw, spec), spec["adapter"])):
                    if row_index < state["row"]:
                        continue
                    if (
                        args.mode == "smoke"
                        and row_index >= 200
                        or family_han[family] >= CAPS[family]
                    ):
                        completed, reason = False, "smoke_or_family_budget"
                        break
                    text_identity = (
                        row.get("text")
                        if spec["adapter"] in {"jsonl_text", "parquet_text"}
                        else json.dumps(
                            {k: row.get(k, "") for k in ("instruction", "input", "output")},
                            ensure_ascii=False,
                        )
                    )
                    page_id = hashlib.sha256(str(text_identity).encode()).hexdigest()
                    for unit_index, text in enumerate(cleaned_units(row, spec, converter, stats)):
                        if family_han[family] >= CAPS[family]:
                            break
                        original_body = (
                            text.split("资料摘录：", 1)[1]
                            if spec["adapter"] == "jsonl_article"
                            else text
                        )
                        if store.excluded(text) or store.excluded(original_body):
                            stats["rejected_existing_wikipedia_exact"] += 1
                            continue
                        chars = set(text)
                        for char in chars - checked_chars - bad_chars:
                            try:
                                if ord(char) not in supported:
                                    raise ValueError("Font lacks this character")
                                render_binary(char, font)
                                checked_chars.add(char)
                            except ValueError:
                                bad_chars.add(char)
                        if chars & bad_chars:
                            stats["rejected_unrenderable"] += 1
                            continue
                        domain, method = domain_for(text, spec)
                        record = {
                            "sample_id": f"{index}:{row_index}:{unit_index}",
                            "source_page_id": page_id,
                            "source_file": spec["dataset"] + "/" + spec["path"],
                            "source_file_revision": spec["revision"],
                            "source_url": "https://modelscope.cn/datasets/" + spec["dataset"],
                            "source_family": family,
                            "domain": domain,
                            "domain_method": method,
                            "source_metadata": json.dumps(
                                {
                                    k: row[k]
                                    for k in (
                                        "source",
                                        "score",
                                        "domain",
                                        "copyright",
                                        "answer_from",
                                        "human_verified",
                                    )
                                    if k in row
                                },
                                ensure_ascii=False,
                            ),
                            "text": text,
                            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "han_count": len(HAN.findall(text)),
                            "split": split_for_page(page_id, 20260915),
                        }
                        if store.add_with_body(record, original_body, stats):
                            family_han[family] += record["han_count"]
                            valid_chars.update(chars)
                            stats["retained_paragraphs"] += 1
                    state["row"] = row_index + 1
                    seen = row_index + 1
                    if state["row"] % 1000 == 0:
                        commit_state()
            else:
                completed, reason = False, "family_budget"
            state["scan"].append(
                {
                    "dataset": spec["dataset"],
                    "path": spec["path"],
                    "rows_scanned": seen,
                    "complete_file_scan": completed,
                    "stop_reason": reason,
                }
            )
            state["file"], state["row"] = index + 1, 0
            commit_state()
        status("exporting", accepted_han=sum(family_han.values()))
        report = export(store, state, args.output, font_path, source_config, config_hash, args)
        store.close()
        status("prepared_unverified", splits=report["splits"], family_han=dict(family_han))
        print(json.dumps(report["splits"]), flush=True)
    except BaseException as error:
        status("failed", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
