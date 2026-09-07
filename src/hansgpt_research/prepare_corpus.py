"""Prepare a small, traceable Chinese Wikipedia binary-glyph corpus on the server."""

from __future__ import annotations

import argparse
import bz2
import hashlib
import importlib.metadata
import json
import subprocess
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

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
FONT_URL = "https://github.com/notofonts/noto-cjk/releases/download/Sans2.004/08_NotoSansCJKsc.zip"
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
    url: str, destination: Path, *, size: int | None = None, sha1: str | None = None
) -> Path:
    """Resume owned partial files; publish only after complete size/hash validation."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    def validate(path: Path) -> None:
        if size is not None and path.stat().st_size != size:
            raise ValueError(f"Download size mismatch: {path.name}")
        if sha1 is not None and digest_file(path, "sha1") != sha1:
            raise ValueError(f"Official SHA-1 mismatch: {path.name}")

    if destination.exists():
        validate(destination)
        return destination
    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and size is not None and partial.stat().st_size == size:
        validate(partial)
        partial.rename(destination)
        return destination
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "HansGPT-research/0.1 (small Wikipedia corpus pilot)"}
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
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
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


def clean_paragraphs(
    raw: str, converter: OpenCC, stats: Counter, *, min_han: int, max_length: int
) -> list[str]:
    accepted = []
    for line in plain_wikitext(raw).splitlines():
        text = converter.convert(unicodedata.normalize("NFC", line)).translate(PUNCTUATION_MAP)
        text = text.strip()
        if not text:
            continue
        stats["candidate_paragraphs"] += 1
        stats["candidate_characters"] += len(text)
        if not ALLOWED.fullmatch(text):
            stats["rejected_non_chinese_or_unsupported_symbols"] += 1
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
            if count >= max_pages:
                break


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
    archive = download(FONT_URL, raw_dir / "08_NotoSansCJKsc.zip")
    font_path = raw_dir / "NotoSansCJKsc-Regular.otf"
    with zipfile.ZipFile(archive) as zipped:
        members = [n for n in zipped.namelist() if n.split("/")[-1] == font_path.name]
        if len(members) != 1:
            raise ValueError("The official font archive has an unexpected structure")
        font_bytes = zipped.read(members[0])
        if not font_path.exists():
            font_path.write_bytes(font_bytes)
        elif hashlib.sha256(font_bytes).hexdigest() != digest_file(font_path):
            raise ValueError("Cached font differs from the pinned official archive")
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
            "font_archive_sha256": digest_file(archive),
            "font_sha256": digest_file(font_path),
            "license_reference": "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use",
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
    dump, font_path, source = source_files(Path(args.raw) / args.snapshot, args.snapshot)
    font = ImageFont.truetype(str(font_path), size=26)
    with TTFont(font_path) as ttfont:
        supported = set(ttfont.getBestCmap() or {})
    converter = OpenCC("t2s")
    stats = Counter()
    records = []
    seen = set()
    glyph_cache: dict[str, np.ndarray] = {}
    bad_characters = set()
    total_han = 0
    for page in wiki_pages(dump, args.max_pages):
        stats["article_pages_scanned"] += 1
        split = split_for_page(page["page_id"], args.seed)
        paragraphs = clean_paragraphs(
            page["raw"], converter, stats, min_han=args.min_han, max_length=4096
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
            if text_hash in seen:
                stats["rejected_exact_duplicate"] += 1
                continue
            seen.add(text_hash)
            han_count = len(HAN.findall(paragraph))
            total_han += han_count
            records.append(
                {
                    "sample_id": f"{page['page_id']}:{page['revision_id']}:{paragraph_index}",
                    "source_page_id": page["page_id"],
                    "source_revision_id": page["revision_id"],
                    "source_title": page["title"],
                    "source_url": f"https://zh.wikipedia.org/w/index.php?oldid={page['revision_id']}",
                    "text": paragraph,
                    "text_sha256": text_hash,
                    "han_count": han_count,
                    "split": split,
                }
            )
        if stats["article_pages_scanned"] % 1000 == 0:
            print(f"Pages: {stats['article_pages_scanned']}; retained Han: {total_han}", flush=True)
        if total_han >= args.max_han:
            break
    if not records or any(not any(r["split"] == s for r in records) for s in SPLITS):
        raise ValueError("Need nonempty train, validation and test splits; increase the sample")
    output.mkdir(parents=True)
    characters = sorted({c for record in records for c in record["text"]}, key=ord)
    if len(characters) + len(CONTROL_NAMES) > 65536:
        raise ValueError("The pilot uint16 asset index is too small")
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
        subset = [r for r in records if r["split"] == split]
        pq.write_table(
            pa.Table.from_pylist(subset), output / f"{split}.parquet", compression="zstd"
        )
        offsets = [0]
        with (output / f"{split}.uint16").open("wb") as tokens_file:
            for record in subset:
                ids = np.array([1, *(lookup[c] for c in record["text"]), 2], dtype="<u2")
                tokens_file.write(ids.tobytes())
                offsets.append(offsets[-1] + len(ids))
        np.save(output / f"{split}.offsets.npy", np.array(offsets, dtype=np.int64))
        (output / f"{split}.txt").write_text(
            "\n\n".join(r["text"] for r in subset) + "\n", encoding="utf-8"
        )
        split_stats[split] = {
            "paragraphs": len(subset),
            "source_pages": len({r["source_page_id"] for r in subset}),
            "han_characters": sum(r["han_count"] for r in subset),
            "grid_tokens_including_bos_eos": offsets[-1],
        }
    report = {
        "status": "pilot_not_formal_benchmark_corpus",
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "configuration": vars(args),
        "source": source,
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
        "deduplication": "global exact paragraph SHA-256; source-page-disjoint split",
        "limitations": [
            "One official article shard, scanned in dump order; not a representative full corpus.",
            "Mixed-script, numeric and unsupported-symbol paragraphs are rejected intact.",
            "Chinese Wikipedia provenance and strict character checks; no statistical language ID.",
            "No full near-duplicate clustering or external benchmark decontamination yet.",
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
    parser.add_argument("--snapshot", default="20260901")
    parser.add_argument("--raw", default="data/raw/hansgpt_zhwiki")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pages", type=int, default=100000)
    parser.add_argument("--max-han", type=int, default=5_000_000)
    parser.add_argument("--min-han", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if not regex.fullmatch(r"20\d{6}", args.snapshot):
        parser.error("snapshot must be an eight-digit date")
    if min(args.max_pages, args.max_han, args.min_han) <= 0:
        parser.error("sample limits must be positive")
    prepare(args)


if __name__ == "__main__":
    main()
