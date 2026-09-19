"""Download pinned Chinese Wikipedia data and prepare binary glyphs on the server."""

from __future__ import annotations

import argparse
import bz2
import hashlib
import http.client
import importlib.metadata
import json
import sqlite3
import subprocess
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from collections import Counter
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import mwparserfromhell
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import regex
from fontTools.ttLib import TTFont
from opencc import OpenCC
from PIL import Image, ImageDraw, ImageFont

PUNCTUATION = "，。！？；：、（）《》〈〉“”‘’「」『』【】〔〕…—·"
ALLOWED = regex.compile(rf"\A[\p{{Unified_Ideograph}}〇{PUNCTUATION}]+\Z")
HAN = regex.compile(r"[\p{Unified_Ideograph}〇]")
PUNCTUATION_MAP = str.maketrans(
    {",": "，", ".": "。", "!": "！", "?": "？", ";": "；", ":": "：", "(": "（", ")": "）"}
)
INNER_PUNCTUATION = regex.escape(PUNCTUATION.replace("（", "").replace("）", ""))
EMPTY_VALUE_LABELS = "学名|原名|别名|法语|英语|拉丁语|德语|日语|西班牙语|意大利语|荷兰语|俄语"
LOCATION_PREFIXES = "法国|中国|英国|德国|日本|美国|俄罗斯|西班牙|意大利|欧洲|亚洲|非洲|美洲|大洋洲"
ARTIFACT_PATTERNS = (
    ("empty_parentheses", regex.compile(rf"（[{INNER_PUNCTUATION}]*）")),
    (
        "empty_labeled_parentheses",
        regex.compile(rf"（(?:{EMPTY_VALUE_LABELS})：[{INNER_PUNCTUATION}]*）"),
    ),
    (
        "missing_numeric_slot",
        regex.compile(
            r"(?:总面积|面积)(?:约为|为|约|达)?[，；](?:位于|地处|坐落于|位在)"
            rf"(?:{LOCATION_PREFIXES})"
            r"|(?:总面积|面积|总人口|人口|海拔)(?:约为|为|达)[，；。]"
        ),
    ),
)
FONT_URL = (
    "https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/"
    "Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
)
SPLITS = ("train", "validation", "test")
CONTROL_NAMES = ("PAD", "BOS", "EOS", "NEWLINE")


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download(
    url: str,
    destination: Path,
    *,
    size: int | None = None,
    sha1: str | None = None,
    sha256: str | None = None,
) -> Path:
    """Resume owned partial files; publish only after complete size/hash validation."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    def validate(path: Path) -> None:
        if size is not None and path.stat().st_size != size:
            raise ValueError(f"Download size mismatch: {path.name}")
        if sha1 is not None and digest_file(path, "sha1") != sha1:
            raise ValueError(f"Official SHA-1 mismatch: {path.name}")
        if sha256 is not None and digest_file(path) != sha256:
            raise ValueError(f"Pinned SHA-256 mismatch: {path.name}")

    if destination.exists():
        validate(destination)
        return destination
    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and size is not None and partial.stat().st_size == size:
        validate(partial)
        partial.rename(destination)
        return destination
    for attempt in range(8):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "HansGPT-research/0.1 (Chinese binary-glyph corpus)"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                resumed = response.status == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                ):
                    raise ValueError("Unexpected HTTP resume range")
                content_length = response.headers.get("Content-Length")
                expected = (
                    (offset if resumed else 0) + int(content_length) if content_length else None
                )
                with partial.open("ab" if resumed else "wb") as handle:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(chunk)
                if expected is not None and partial.stat().st_size != expected:
                    raise OSError("Incomplete HTTP response")
            validate(partial)
            partial.rename(destination)
            print(f"Downloaded and verified: {destination.name}", flush=True)
            return destination
        except (OSError, TimeoutError, http.client.HTTPException):
            if attempt == 7:
                raise
            time.sleep(min(2**attempt, 30))
    raise RuntimeError("Download did not finish")


def split_for_page(page_id: str, seed: int) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{page_id}".encode()).hexdigest()[:8], 16) % 1000
    return "train" if bucket < 980 else "validation" if bucket < 990 else "test"


def plain_wikitext(raw: str) -> str:
    """Preserve a hard boundary wherever non-prose markup is removed."""
    code = mwparserfromhell.parse(raw)
    nodes = list(code.filter_templates())
    nodes.extend(
        tag
        for tag in code.filter_tags()
        if str(tag.tag).lower() in {"ref", "gallery", "math", "table", "code", "syntaxhighlight"}
    )
    nodes.extend(
        link
        for link in code.filter_wikilinks()
        if str(link.title).split(":", 1)[0].strip().lower()
        in {"file", "image", "category", "文件", "檔案", "分类", "分類"}
    )
    for node in nodes:
        # A containing template/tag may already have been removed.
        with suppress(ValueError):
            code.replace(node, "\n")
    return code.strip_code(normalize=True, collapse=False)


def artifact_reason(text: str) -> str | None:
    """Reject observed upstream extraction holes without inventing replacement text."""
    return next((name for name, pattern in ARTIFACT_PATTERNS if pattern.search(text)), None)


def clean_paragraphs(
    raw: str,
    converter: OpenCC,
    stats: Counter,
    *,
    min_han: int,
    max_length: int,
    wikitext: bool = True,
) -> list[str]:
    accepted = []
    for line in (plain_wikitext(raw) if wikitext else raw).splitlines():
        text = unicodedata.normalize("NFC", line).translate(PUNCTUATION_MAP).strip()
        if not text:
            continue
        stats["candidate_paragraphs"] += 1
        stats["candidate_characters"] += len(text)
        if not ALLOWED.fullmatch(text):
            stats["rejected_non_chinese_or_unsupported_symbols"] += 1
            stats["rejected_before_opencc"] += 1
            continue
        text = converter.convert(text)
        stats["opencc_paragraphs_processed"] += 1
        if not ALLOWED.fullmatch(text):
            stats["rejected_non_chinese_or_unsupported_symbols"] += 1
            continue
        if reason := artifact_reason(text):
            stats[f"rejected_{reason}"] += 1
            continue
        if len(HAN.findall(text)) < min_han:
            stats["rejected_short"] += 1
            continue
        if len(text) > max_length:
            stats["rejected_long"] += 1
            continue
        if regex.search(r"(.)\1{9,}", text):
            stats["rejected_repeated_character"] += 1
            continue
        accepted.append(text)
    return accepted


def wiki_pages(path: Path, max_pages: int):
    with bz2.open(path, "rb") as handle:
        iterator = ET.iterparse(handle, events=("start", "end"))
        _, root = next(iterator)
        count = 0
        for event, element in iterator:
            if event != "end" or element.tag.rsplit("}", 1)[-1] != "page":
                continue
            count += 1
            if element.findtext("{*}ns") == "0" and element.find("{*}redirect") is None:
                yield {
                    "page_id": element.findtext("{*}id", ""),
                    "revision_id": element.findtext("{*}revision/{*}id", ""),
                    "title": element.findtext("{*}title", ""),
                    "raw": element.findtext("{*}revision/{*}text", ""),
                }
            root.clear()
            if max_pages and count >= max_pages:
                break


def modelscope_pages(paths: list[Path], revision: str, max_pages: int, scan: dict | None = None):
    """The mirror contains extracted plaintext, not MediaWiki source markup."""
    count = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        file_scan = (
            next((item for item in scan["files"] if item["file"] == path.name), None)
            if scan is not None
            else None
        )
        for batch in parquet.iter_batches(batch_size=512, columns=["id", "url", "title", "text"]):
            for row in batch.to_pylist():
                if file_scan is not None:
                    file_scan["scanned_rows"] += 1
                    file_scan["completed"] = file_scan["scanned_rows"] == file_scan["expected_rows"]
                    scan["scanned_rows"] += 1
                yield {
                    "page_id": str(row["id"]),
                    "revision_id": None,
                    "title": row["title"],
                    "raw": row["text"] or "",
                    "url": row["url"],
                    "file": path.name,
                    "file_revision": revision,
                }
                count += 1
                if max_pages and count >= max_pages:
                    return


def canonical_han(text: str) -> str:
    return "".join(HAN.findall(text))


def shingles(text: str) -> set[str]:
    return {text[index : index + 5] for index in range(max(1, len(text) - 4))}


def shingle_anchors(text: str) -> list[int]:
    # Hash collisions only generate candidates; exact string-set Jaccard confirms matches.
    return sorted({zlib.crc32(part.encode()) for part in shingles(text)})[:8]


class CorpusStore:
    """Disk-backed records and approximate near-duplicate candidate index."""

    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA cache_size=-65536")
        self.connection.execute("PRAGMA journal_mode=TRUNCATE")
        self.connection.executescript(
            """
            CREATE TABLE records (
                id INTEGER PRIMARY KEY, text_hash TEXT UNIQUE NOT NULL,
                canonical_hash TEXT UNIQUE NOT NULL, canonical TEXT NOT NULL,
                length INTEGER NOT NULL, split TEXT NOT NULL, payload TEXT NOT NULL
            );
            CREATE INDEX split_index ON records(split, id);
            CREATE TABLE anchors (
                anchor INTEGER NOT NULL, record_id INTEGER NOT NULL,
                PRIMARY KEY (anchor, record_id)
            ) WITHOUT ROWID;
            """
        )

    def add(self, record: dict, stats: Counter) -> bool:
        canonical = canonical_han(record["text"])
        canonical_hash = hashlib.sha256(canonical.encode()).hexdigest()
        existing = self.connection.execute(
            "SELECT text_hash FROM records WHERE canonical_hash=?", (canonical_hash,)
        ).fetchone()
        if existing:
            key = (
                "rejected_exact_duplicate"
                if existing[0] == record["text_sha256"]
                else "rejected_punctuation_variant"
            )
            stats[key] += 1
            return False
        anchors = shingle_anchors(canonical)
        placeholders = ",".join("?" for _ in anchors)
        candidates = self.connection.execute(
            f"""SELECT r.canonical FROM anchors a JOIN records r ON r.id=a.record_id
            WHERE a.anchor IN ({placeholders}) AND r.length BETWEEN ? AND ?
            GROUP BY r.id HAVING COUNT(*) >= 2 ORDER BY COUNT(*) DESC, r.id LIMIT 101""",
            [*anchors, int(len(canonical) * 0.9), int(len(canonical) / 0.9) + 1],
        ).fetchall()
        if len(candidates) > 100:
            stats["near_duplicate_candidate_limit_reached"] += 1
        parts = shingles(canonical)
        for (candidate,) in candidates[:100]:
            other = shingles(candidate)
            if len(parts & other) / len(parts | other) >= 0.9:
                stats["rejected_near_duplicate"] += 1
                return False
        cursor = self.connection.execute(
            """INSERT INTO records
            (text_hash, canonical_hash, canonical, length, split, payload)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                record["text_sha256"],
                canonical_hash,
                canonical,
                len(canonical),
                record["split"],
                json.dumps(record, ensure_ascii=False),
            ),
        )
        self.connection.executemany(
            "INSERT INTO anchors VALUES (?, ?)", [(value, cursor.lastrowid) for value in anchors]
        )
        return True

    def records(self, split: str):
        for (payload,) in self.connection.execute(
            "SELECT payload FROM records WHERE split=? ORDER BY id", (split,)
        ):
            yield json.loads(payload)

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def render_binary(character: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    # The larger canvas detects clipping rather than silently discarding strokes.
    canvas = Image.new("L", (96, 96), 0)
    draw = ImageDraw.Draw(canvas)
    x = 32 + (32 - font.getlength(character)) / 2
    draw.text((x, 32 + 27), character, font=font, fill=255, anchor="ls")
    full = (np.asarray(canvas) >= 128).astype(np.uint8)
    tile = full[32:64, 32:64].copy()
    if not tile.any() or int(full.sum()) != int(tile.sum()):
        raise ValueError("Empty or clipped glyph")
    return tile


def control_tiles() -> list[np.ndarray]:
    tiles = [np.zeros((32, 32), dtype=np.uint8)]
    for name in CONTROL_NAMES[1:]:
        tile = np.zeros((32, 32), dtype=np.uint8)
        tile[[0, -1], :] = 1
        tile[:, [0, -1]] = 1
        bits = np.unpackbits(np.frombuffer(hashlib.sha256(name.encode()).digest(), dtype=np.uint8))
        tile[8:24, 8:24] = bits.reshape(16, 16)
        tiles.append(tile)
    return tiles


def source_files(raw_dir: Path, snapshot: str) -> tuple[Path, Path, dict]:
    base = f"https://dumps.wikimedia.org/zhwiki/{snapshot}/"
    status_path = download(base + "dumpstatus.json", raw_dir / "dumpstatus.json")
    status = json.loads(status_path.read_text())
    job = status["jobs"]["articlesmultistreamdump"]
    if job["status"] != "done":
        raise ValueError("The selected official dump is not complete")
    candidates = sorted(
        name for name in job["files"] if regex.search(r"multistream1\.xml.*\.bz2$", name)
    )
    if not candidates:
        raise ValueError("No first article shard in the official manifest")
    name = candidates[0]
    if Path(name).name != name:
        raise ValueError("Unsafe official manifest filename")
    metadata = job["files"][name]
    dump_url = base + name
    dump = download(dump_url, raw_dir / name, size=metadata["size"], sha1=metadata["sha1"])
    font_path = download(FONT_URL, raw_dir / "NotoSansCJKsc-Regular.otf")
    return (
        dump,
        font_path,
        {
            "project": "zhwiki",
            "snapshot": snapshot,
            "dump_url": dump_url,
            "official_sha1": metadata["sha1"],
            "dump_sha256": digest_file(dump),
            "dump_manifest_sha256": digest_file(status_path),
            "font_url": FONT_URL,
            "font_version": "Noto Sans CJK SC Regular 2.004",
            "font_sha256": digest_file(font_path),
            "license_reference": "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
            "font_license": "SIL Open Font License 1.1",
        },
    )


def modelscope_source_files(args: argparse.Namespace) -> tuple[list[Path], Path, dict]:
    config_path = Path(args.source_config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["provider"] != "ModelScope" or config["snapshot"] != args.snapshot:
        raise ValueError("ModelScope source configuration does not match the requested snapshot")
    indices = list(range(len(config["files"]))) if args.shards == "all" else args.shards
    if not isinstance(indices, list):
        indices = [int(value) for value in indices.split(",")]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Select distinct source shards")
    raw_dir = Path(args.raw) / args.snapshot
    paths = []
    files = []
    for index in indices:
        if not 0 <= index < len(config["files"]):
            raise ValueError("Shard index outside the pinned source manifest")
        metadata = config["files"][index]
        query = urlencode(
            {
                "Source": "SDK",
                "Revision": config["revision"],
                "FilePath": metadata["path"],
                "View": "false",
            }
        )
        url = f"https://modelscope.cn/api/v1/datasets/{config['dataset']}/repo?{query}"
        path = download(
            url,
            raw_dir / Path(metadata["path"]).name,
            size=metadata["size"],
            sha256=metadata["sha256"],
        )
        paths.append(path)
        files.append(
            {
                **metadata,
                "download_url": url,
                "verified_sha256": metadata["sha256"],
                "expected_rows": pq.ParquetFile(path).metadata.num_rows,
            }
        )
    font_path = download(FONT_URL, raw_dir / "NotoSansCJKsc-Regular.otf")
    return (
        paths,
        font_path,
        {
            **{key: value for key, value in config.items() if key != "files"},
            "files": files,
            "source_config_sha256": digest_file(config_path),
            "selected_shards": indices,
            "snapshot_shard_count": len(config["files"]),
            "article_revision_available": False,
            "format": "upstream extracted plaintext Parquet: id, url, title, text",
            "font_url": FONT_URL,
            "font_version": "Noto Sans CJK SC Regular 2.004",
            "font_sha256": digest_file(font_path),
            "font_license": "SIL Open Font License 1.1",
        },
    )


def prepare(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Output already exists; choose a new versioned output directory")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Run preprocessing from a clean committed checkout")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.source == "modelscope":
        paths, font_path, source = modelscope_source_files(args)
        source_scan = {
            "all_snapshot_shards_selected": len(source["selected_shards"])
            == source["snapshot_shard_count"],
            "all_selected_rows_scanned": False,
            "expected_rows": sum(item["expected_rows"] for item in source["files"]),
            "scanned_rows": 0,
            "files": [
                {
                    "file": path.name,
                    "expected_rows": item["expected_rows"],
                    "scanned_rows": 0,
                    "completed": False,
                }
                for path, item in zip(paths, source["files"], strict=True)
            ],
        }
        pages = modelscope_pages(paths, source["revision"], args.max_pages, source_scan)
    else:
        dump, font_path, source = source_files(Path(args.raw) / args.snapshot, args.snapshot)
        pages = wiki_pages(dump, args.max_pages)
        source_scan = None
    font = ImageFont.truetype(str(font_path), size=26)
    with TTFont(font_path) as ttfont:
        supported = set(ttfont.getBestCmap() or {})
    converter = OpenCC("t2s")
    stats = Counter()
    output.mkdir(parents=True)
    spool_path = output / ".records.sqlite"
    store = CorpusStore(spool_path)
    glyph_cache: dict[str, np.ndarray] = {}
    accepted_characters = set()
    bad_characters = set()
    total_han = 0
    for page in pages:
        stats["article_pages_scanned"] += 1
        split = split_for_page(page["page_id"], args.seed)
        paragraphs = clean_paragraphs(
            page["raw"],
            converter,
            stats,
            min_han=args.min_han,
            max_length=4096,
            wikitext=args.source == "wikimedia",
        )
        for paragraph_index, paragraph in enumerate(paragraphs):
            chars = set(paragraph)
            for char in chars - glyph_cache.keys() - bad_characters:
                if ord(char) not in supported:
                    bad_characters.add(char)
                    continue
                try:
                    glyph_cache[char] = render_binary(char, font)
                except ValueError:
                    bad_characters.add(char)
            if chars & bad_characters:
                stats["rejected_unrenderable"] += 1
                continue
            text_hash = hashlib.sha256(paragraph.encode()).hexdigest()
            han_count = len(HAN.findall(paragraph))
            record = {
                "sample_id": f"{args.snapshot}:{page['page_id']}:{paragraph_index}",
                "source_page_id": page["page_id"],
                "source_revision_id": page["revision_id"],
                "source_title": page["title"],
                "source_url": page.get("url")
                or (f"https://zh.wikipedia.org/w/index.php?oldid={page['revision_id']}"),
                "source_snapshot": args.snapshot,
                "source_file": page.get("file", source.get("dump_url", "")),
                "source_file_revision": page.get("file_revision"),
                "text": paragraph,
                "text_sha256": text_hash,
                "han_count": han_count,
                "split": split,
            }
            if store.add(record, stats):
                total_han += han_count
                accepted_characters.update(chars)
                stats["retained_paragraphs"] += 1
        if stats["article_pages_scanned"] % 1000 == 0:
            store.commit()
            print(f"Pages: {stats['article_pages_scanned']}; retained Han: {total_han}", flush=True)
        if args.max_han and total_han >= args.max_han:
            break
    store.commit()
    if source_scan is not None:
        source_scan["all_selected_rows_scanned"] = all(
            item["completed"] for item in source_scan["files"]
        )
        if source_scan["scanned_rows"] != stats["article_pages_scanned"]:
            raise ValueError("Source scan row accounting mismatch")
    if any(next(store.records(split), None) is None for split in SPLITS):
        store.close()
        raise ValueError("Need nonempty train, validation and test splits; increase the sample")
    characters = sorted(accepted_characters, key=ord)
    if len(characters) + len(CONTROL_NAMES) > 65536:
        raise ValueError("The uint16 asset index is too small")
    tiles = control_tiles() + [glyph_cache[c] for c in characters]
    bitmaps = np.stack(tiles)
    if bitmaps.shape[1:] != (32, 32) or not np.isin(bitmaps, [0, 1]).all():
        raise ValueError("Invalid binary glyph bank")
    hashes: dict[str, list[str]] = {}
    for index, bitmap in enumerate(tiles):
        pixel_hash = hashlib.sha256(bitmap.tobytes()).hexdigest()
        label = CONTROL_NAMES[index] if index < 4 else characters[index - 4]
        hashes.setdefault(pixel_hash, []).append(label)
    collisions = [labels for labels in hashes.values() if len(labels) > 1]
    if any(set(labels) & set(CONTROL_NAMES) for labels in collisions):
        raise ValueError("A control tile collides with a content glyph")
    np.savez_compressed(output / "glyph_bank.npz", bitmaps=bitmaps)
    lookup = {c: i + 4 for i, c in enumerate(characters)}
    write_json(
        output / "glyph_inventory.json",
        {
            "controls": dict(enumerate(CONTROL_NAMES)),
            "characters": lookup,
            "collisions": collisions,
        },
    )
    split_stats = {}
    for split in SPLITS:
        offsets = [0]
        pages_in_split = set()
        han_in_split = 0
        batch = []
        schema = pa.schema(
            [
                ("sample_id", pa.string()),
                ("source_page_id", pa.string()),
                ("source_revision_id", pa.string()),
                ("source_title", pa.string()),
                ("source_url", pa.string()),
                ("source_snapshot", pa.string()),
                ("source_file", pa.string()),
                ("source_file_revision", pa.string()),
                ("text", pa.string()),
                ("text_sha256", pa.string()),
                ("han_count", pa.int64()),
                ("split", pa.string()),
            ]
        )
        with (
            (output / f"{split}.uint16").open("wb") as tokens_file,
            (output / f"{split}.txt").open("w", encoding="utf-8") as text_file,
            pq.ParquetWriter(output / f"{split}.parquet", schema, compression="zstd") as writer,
        ):
            for record in store.records(split):
                ids = np.array([1, *(lookup[c] for c in record["text"]), 2], dtype="<u2")
                tokens_file.write(ids.tobytes())
                offsets.append(offsets[-1] + len(ids))
                text_file.write(record["text"] + "\n\n")
                pages_in_split.add(record["source_page_id"])
                han_in_split += record["han_count"]
                batch.append(record)
                if len(batch) >= 1024:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        np.save(output / f"{split}.offsets.npy", np.array(offsets, dtype=np.int64))
        split_stats[split] = {
            "paragraphs": len(offsets) - 1,
            "source_pages": len(pages_in_split),
            "han_characters": han_in_split,
            "grid_tokens_including_bos_eos": offsets[-1],
        }
    store.close()
    spool_path.unlink()
    spool_path.with_name(spool_path.name + "-journal").unlink(missing_ok=True)
    exhaustive = bool(
        source_scan is not None
        and source_scan["all_snapshot_shards_selected"]
        and source_scan["all_selected_rows_scanned"]
    )
    report = {
        "status": "full_snapshot_experiment_corpus" if exhaustive else "bounded_experiment_corpus",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "configuration": vars(args),
        "source": source,
        "source_scan": source_scan,
        "cleaning_version": "strict_han_v3_artifact_rejection",
        "filter_stats": dict(stats),
        "splits": split_stats,
        "glyphs": {
            "content_characters": len(characters),
            "controls": len(CONTROL_NAMES),
            "dtype": "uint8 containing only 0 and 1",
            "shape": list(bitmaps.shape),
            "font_size": 26,
            "baseline_y": 27,
            "render_threshold": 128,
            "unrenderable_characters": sorted(bad_characters, key=ord),
            "collision_groups": collisions,
        },
        "deduplication": {
            "exact": "global SHA-256 of normalized text and punctuation-stripped Han text",
            "near": (
                "Eight smallest CRC32 character-five-gram hashes; at least two shared anchors, "
                "length ratio >=0.9; first 100 candidates by shared-anchor count; "
                "confirm exact character-five-gram set Jaccard >=0.9; keep first occurrence"
            ),
            "recall": "Approximate candidate generation; not exhaustive semantic deduplication",
            "split": "source-page-disjoint deterministic hash 98/1/1 after global deduplication",
        },
        "limitations": [
            "Snapshot and selected files are scanned in source order; Wikipedia is domain-biased.",
            "Mixed-script, numeric and unsupported-symbol paragraphs are rejected intact.",
            "Known empty-value extraction artifacts are rejected; source omissions can remain.",
            "Chinese Wikipedia provenance and strict character checks; no statistical language ID.",
            "Near-duplicate recall is approximate; no external benchmark decontamination.",
            "Document splits share characters; no character-disjoint generalization claim.",
            "Font collisions are reported, not treated as evidence of distinct visual identities.",
            "Retain per-page attribution and inspect page-specific rights before redistribution.",
        ],
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pyarrow",
                "pillow",
                "fonttools",
                "mwparserfromhell",
                "regex",
                "opencc-python-reimplemented",
            )
        },
        "normalization_unicode_version": unicodedata.unidata_version,
        "output_sha256": {p.name: digest_file(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    write_json(output / "manifest.json", report)
    print(
        json.dumps({"output": str(output), "splits": split_stats, "glyphs": len(tiles)}), flush=True
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["modelscope", "wikimedia"], default="modelscope")
    parser.add_argument("--source-config", default="configs/datasets/modelscope_wikipedia_zh.json")
    parser.add_argument("--snapshot", default="20231101")
    parser.add_argument("--shards", default="all", help="all or comma-separated indices, e.g. 2")
    parser.add_argument("--raw", default="data/raw/hansgpt_modelscope_wikipedia")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pages", type=int, default=0, help="0 scans all selected pages")
    parser.add_argument("--max-han", type=int, default=0, help="0 keeps all passing paragraphs")
    parser.add_argument("--min-han", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if not regex.fullmatch(r"20\d{6}", args.snapshot):
        parser.error("snapshot must be an eight-digit date")
    if min(args.max_pages, args.max_han) < 0 or args.min_han <= 0:
        parser.error("sample limits must be nonnegative and min-han must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
