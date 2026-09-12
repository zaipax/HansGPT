"""Verify pure text, splits, exact deduplication and every glyph-stream alignment independently."""

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fontTools.ttLib import TTFont
from PIL import ImageFont

from hansgpt_research.prepare_corpus import (
    ALLOWED,
    CONTROL_NAMES,
    HAN,
    SPLITS,
    artifact_reason,
    canonical_han,
    control_tiles,
    digest_file,
    render_binary,
    split_for_page,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    receipt = root / "verification.json"
    receipt.unlink(missing_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["output_sha256"].items():
        if digest_file(root / name) != expected:
            raise ValueError(f"Output checksum mismatch: {name}")
    inventory = json.loads((root / "glyph_inventory.json").read_text())
    if inventory["controls"] != {name: i for i, name in enumerate(CONTROL_NAMES)}:
        raise ValueError("Control inventory mismatch")
    bank = np.load(root / "glyph_bank.npz", allow_pickle=False)["bitmaps"]
    chars = inventory["characters"]
    if (
        bank.shape != (len(chars) + 4, 32, 32)
        or bank.dtype != np.uint8
        or not np.isin(bank, [0, 1]).all()
        or sorted(chars.values()) != list(range(4, len(bank)))
    ):
        raise ValueError("Invalid binary glyph bank")
    if not np.array_equal(bank[:4], np.stack(control_tiles())):
        raise ValueError("Control pixels differ")
    font_path = Path(manifest["font_path"])
    if digest_file(font_path) != manifest["font_sha256"]:
        raise ValueError("Font identity changed")
    font = ImageFont.truetype(str(font_path), size=26)
    with TTFont(font_path) as data:
        supported = set(data.getBestCmap())
    for char, index in chars.items():
        if ord(char) not in supported or not np.array_equal(bank[index], render_binary(char, font)):
            raise ValueError("Glyph rendering mismatch")
    temporary = root / ".verify.sqlite"
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    connection.execute("CREATE TABLE canonical (hash TEXT PRIMARY KEY) WITHOUT ROWID")
    connection.execute(
        "CREATE TABLE pages (id TEXT PRIMARY KEY, split TEXT NOT NULL) WITHOUT ROWID"
    )
    family_han, domain_han, actual_splits = Counter(), Counter(), {}
    try:
        for split in SPLITS:
            offsets = np.load(root / f"{split}.offsets.npy", allow_pickle=False)
            tokens = np.memmap(root / f"{split}.uint16", dtype="<u2", mode="r")
            if (
                offsets.ndim != 1
                or offsets[0] != 0
                or offsets[-1] != len(tokens)
                or np.any(np.diff(offsets) < 3)
            ):
                raise ValueError("Invalid offsets")
            count, han_count, pages = 0, 0, set()
            for batch in pq.ParquetFile(root / f"{split}.parquet").iter_batches(batch_size=1024):
                for record in batch.to_pylist():
                    text = record["text"]
                    if not ALLOWED.fullmatch(text) or artifact_reason(text):
                        raise ValueError("Not pure Chinese or contains known extraction holes")
                    if record["text_sha256"] != hashlib.sha256(text.encode()).hexdigest():
                        raise ValueError("Text hash mismatch")
                    expected = np.array([1, *(chars[c] for c in text), 2], dtype="<u2")
                    if count + 1 >= len(offsets) or not np.array_equal(
                        tokens[offsets[count] : offsets[count + 1]], expected
                    ):
                        raise ValueError("Text does not match the binary glyph asset stream")
                    if (
                        split_for_page(record["source_page_id"], 20260915) != split
                        or record["split"] != split
                    ):
                        raise ValueError("Document split mismatch")
                    canonical = hashlib.sha256(canonical_han(text).encode()).hexdigest()
                    connection.execute("INSERT INTO canonical VALUES (?)", (canonical,))
                    prior = connection.execute(
                        "SELECT split FROM pages WHERE id=?", (record["source_page_id"],)
                    ).fetchone()
                    if prior and prior[0] != split:
                        raise ValueError("Source document crosses splits")
                    connection.execute(
                        "INSERT OR IGNORE INTO pages VALUES (?,?)",
                        (record["source_page_id"], split),
                    )
                    han = len(HAN.findall(text))
                    if han != record["han_count"]:
                        raise ValueError("Han count mismatch")
                    count += 1
                    han_count += han
                    pages.add(record["source_page_id"])
                    family_han[record["source_family"]] += han
                    domain_han[record["domain"]] += han
                connection.commit()
            actual = {
                "paragraphs": count,
                "source_pages": len(pages),
                "han_characters": han_count,
                "effective_targets": len(tokens) - count,
            }
            if count != len(offsets) - 1 or actual != manifest["splits"][split]:
                raise ValueError("Split statistics mismatch")
            actual_splits[split] = actual
            print("Verified", split, actual, flush=True)
        if dict(family_han) != manifest["family_han"] or dict(domain_han) != manifest["domain_han"]:
            raise ValueError("Family/domain counts mismatch")
        names = ["glyph_bank.npz", "glyph_inventory.json"] + [
            f"{s}.{suffix}" for s in SPLITS for suffix in ("uint16", "offsets.npy")
        ]
        result = {
            "passed": True,
            "manifest_sha256": digest_file(root / "manifest.json"),
            "model_consumed_sha256": {name: digest_file(root / name) for name in sorted(names)},
            "splits": actual_splits,
            "family_han": dict(family_han),
            "domain_han": dict(domain_han),
            "scope": "all rows: pure Chinese, exact dedup, splits, glyph pixels and alignment",
            "near_dedup_limit": "approximate preparer search, not exhaustive semantic verification",
        }
        pending = receipt.with_suffix(".tmp")
        pending.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        pending.replace(receipt)
    finally:
        connection.close()
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
