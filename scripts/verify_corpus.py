"""Read exported corpus assets independently and check text/pixel/index integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from hansgpt_research.prepare_corpus import ALLOWED, CONTROL_NAMES, canonical_han, digest_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["output_sha256"].items():
        if Path(name).name != name or digest_file(directory / name) != expected:
            raise ValueError("Output checksum verification failed")
    inventory = json.loads((directory / "glyph_inventory.json").read_text(encoding="utf-8"))
    lookup = inventory["characters"]
    with np.load(directory / "glyph_bank.npz", allow_pickle=False) as archive:
        bitmaps = archive["bitmaps"]
    if bitmaps.shape != (len(lookup) + len(CONTROL_NAMES), 32, 32):
        raise ValueError("Unexpected glyph shape")
    if bitmaps.dtype != np.uint8 or not np.isin(bitmaps, [0, 1]).all():
        raise ValueError("Glyphs are not binary uint8 tiles")
    page_splits = {}
    seen_hashes = set()
    seen_canonical_hashes = set()
    counts = {}
    examples = []
    for split in ("train", "validation", "test"):
        parquet = pq.ParquetFile(directory / f"{split}.parquet")
        records = (
            row for batch in parquet.iter_batches(batch_size=1024) for row in batch.to_pylist()
        )
        ids = np.memmap(directory / f"{split}.uint16", dtype="<u2", mode="r")
        offsets = np.load(directory / f"{split}.offsets.npy", allow_pickle=False)
        paragraphs = parquet.metadata.num_rows
        if len(offsets) != paragraphs + 1 or offsets[0] != 0 or offsets[-1] != len(ids):
            raise ValueError("Broken sequence offsets")
        if not (np.diff(offsets) > 2).all() or int(ids.max()) >= len(bitmaps):
            raise ValueError("Invalid sequence lengths or image references")
        for index, record in enumerate(records):
            text = record["text"]
            if record["split"] != split or not ALLOWED.fullmatch(text):
                raise ValueError("Noncanonical corpus text or incorrect split label")
            actual_hash = hashlib.sha256(text.encode()).hexdigest()
            if actual_hash != record["text_sha256"] or actual_hash in seen_hashes:
                raise ValueError("Text checksum mismatch or duplicate paragraph")
            seen_hashes.add(actual_hash)
            canonical_hash = hashlib.sha256(canonical_han(text).encode()).hexdigest()
            if canonical_hash in seen_canonical_hashes:
                raise ValueError("Punctuation-only duplicate paragraph")
            seen_canonical_hashes.add(canonical_hash)
            if manifest["source"].get("provider") == "ModelScope":
                if record["source_revision_id"] is not None:
                    raise ValueError("Mirror does not supply an article revision ID")
                if (
                    record["source_file_revision"] != manifest["source"]["revision"]
                    or record["source_snapshot"] != manifest["source"]["snapshot"]
                    or record["source_file"]
                    not in {Path(file["path"]).name for file in manifest["source"]["files"]}
                ):
                    raise ValueError("Source file provenance does not match manifest")
            page_id = record["source_page_id"]
            if page_id in page_splits and page_splits[page_id] != split:
                raise ValueError("Source page crosses dataset splits")
            page_splits[page_id] = split
            sequence = ids[offsets[index] : offsets[index + 1]]
            expected = [1, *(lookup[char] for char in text), 2]
            if sequence.tolist() != expected:
                raise ValueError("Binary asset references do not round-trip to the text")
            if len(examples) < 3 and split == "train":
                examples.append(text[:160])
        counts[split] = {"paragraphs": paragraphs, "grid_tokens": len(ids)}
        if (
            paragraphs != manifest["splits"][split]["paragraphs"]
            or len(ids) != manifest["splits"][split]["grid_tokens_including_bos_eos"]
        ):
            raise ValueError("Export counts do not match manifest")
    report = {
        "passed": True,
        "checks": [
            "sha256",
            "strict_chinese",
            "binary_pixels",
            "sequence_alignment",
            "page_split",
            "exact_dedup",
            "punctuation_variant_dedup",
            "source_file_provenance",
        ],
        "near_duplicate_note": (
            "Preparation uses approximate shingle candidates with exact Jaccard confirmation; "
            "this independent verifier does not claim exhaustive near-duplicate detection."
        ),
        "splits": counts,
        "glyph_shape": list(bitmaps.shape),
        "examples": examples,
    }
    (directory / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
