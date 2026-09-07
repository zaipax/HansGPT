"""Independently verify corpus exports, re-render glyphs and audit sampled leakage."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fontTools.ttLib import TTFont
from PIL import ImageFont

from hansgpt_research.prepare_corpus import (
    ALLOWED,
    CONTROL_NAMES,
    SPLITS,
    canonical_han,
    control_tiles,
    digest_file,
    render_binary,
    shingles,
    split_for_page,
)


def write_atomic(path: Path, payload: dict | list) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parquet_records(path: Path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
        yield from batch.to_pylist()


def validate_scan(manifest: dict) -> None:
    if manifest["source"].get("provider") != "ModelScope":
        return
    scan = manifest["source_scan"]
    source_files = {Path(item["path"]).name: item for item in manifest["source"]["files"]}
    if len(scan["files"]) != len(source_files) or len({f["file"] for f in scan["files"]}) != len(
        source_files
    ):
        raise ValueError("Missing or repeated source scan evidence")
    for item in scan["files"]:
        if item["file"] not in source_files:
            raise ValueError("Unexpected source scan file")
        if (
            item["expected_rows"] != source_files[item["file"]]["expected_rows"]
            or not 0 <= item["scanned_rows"] <= item["expected_rows"]
            or item["completed"] != (item["scanned_rows"] == item["expected_rows"])
        ):
            raise ValueError("Invalid per-file source scan counts")
        raw_path = (
            Path(manifest["configuration"]["raw"])
            / manifest["configuration"]["snapshot"]
            / item["file"]
        )
        if (
            raw_path.stat().st_size != source_files[item["file"]]["size"]
            or pq.ParquetFile(raw_path).metadata.num_rows != item["expected_rows"]
        ):
            raise ValueError("Source scan expectation differs from original Parquet metadata")
    if (
        scan["scanned_rows"] != sum(item["scanned_rows"] for item in scan["files"])
        or scan["expected_rows"] != sum(item["expected_rows"] for item in scan["files"])
        or scan["scanned_rows"] != manifest["filter_stats"]["article_pages_scanned"]
        or scan["all_selected_rows_scanned"] != all(item["completed"] for item in scan["files"])
        or scan["all_snapshot_shards_selected"]
        != (len(scan["files"]) == manifest["source"]["snapshot_shard_count"])
    ):
        raise ValueError("Source scan aggregate does not match per-file evidence")
    complete = scan["all_selected_rows_scanned"] and scan["all_snapshot_shards_selected"]
    if (manifest["status"] == "full_snapshot_experiment_corpus") != complete:
        raise ValueError("Full-corpus status does not match actual source coverage")


def verify_glyphs(directory: Path, manifest: dict) -> tuple[dict, np.ndarray]:
    inventory = json.loads((directory / "glyph_inventory.json").read_text(encoding="utf-8"))
    lookup = inventory["characters"]
    if (
        list(lookup.values()) != list(range(4, len(lookup) + 4))
        or list(lookup) != sorted(lookup, key=ord)
        or any(len(char) != 1 or not ALLOWED.fullmatch(char) for char in lookup)
        or inventory["controls"] != {str(index): name for index, name in enumerate(CONTROL_NAMES)}
    ):
        raise ValueError("Noncanonical character inventory or control addresses")
    with np.load(directory / "glyph_bank.npz", allow_pickle=False) as archive:
        bitmaps = archive["bitmaps"]
    expected_shape = (len(lookup) + len(CONTROL_NAMES), 32, 32)
    if bitmaps.shape != expected_shape or list(expected_shape) != manifest["glyphs"]["shape"]:
        raise ValueError("Unexpected glyph shape")
    if bitmaps.dtype != np.uint8 or not np.isin(bitmaps, [0, 1]).all():
        raise ValueError("Glyphs are not binary uint8 tiles")
    for index, expected in enumerate(control_tiles()):
        if not np.array_equal(bitmaps[index], expected):
            raise ValueError("Control bitmap differs from the defined binary frame")
    glyphs = manifest["glyphs"]
    if (
        glyphs["font_size"] != 26
        or glyphs["baseline_y"] != 27
        or glyphs["render_threshold"] != 128
        or glyphs["content_characters"] != len(lookup)
        or glyphs["controls"] != len(CONTROL_NAMES)
    ):
        raise ValueError("Unexpected glyph rendering contract")
    font_path = (
        Path(manifest["configuration"]["raw"])
        / manifest["configuration"]["snapshot"]
        / "NotoSansCJKsc-Regular.otf"
    )
    if digest_file(font_path) != manifest["source"]["font_sha256"]:
        raise ValueError("Source font checksum mismatch")
    font = ImageFont.truetype(str(font_path), size=glyphs["font_size"])
    with TTFont(font_path) as ttfont:
        supported = set(ttfont.getBestCmap() or {})
    for char, index in lookup.items():
        if ord(char) not in supported or not np.array_equal(
            bitmaps[index], render_binary(char, font)
        ):
            raise ValueError("Content bitmap does not match its source character and font")
    groups = defaultdict(list)
    labels = [*CONTROL_NAMES, *lookup]
    for bitmap, label in zip(bitmaps, labels, strict=True):
        groups[hashlib.sha256(bitmap.tobytes()).hexdigest()].append(label)
    collisions = [labels for labels in groups.values() if len(labels) > 1]
    if (
        collisions != inventory["collisions"]
        or collisions != glyphs["collision_groups"]
        or any(set(labels) & set(CONTROL_NAMES) for labels in collisions)
    ):
        raise ValueError("Glyph collision inventory is inconsistent")
    return lookup, bitmaps


def audit_samples(directory: Path, seed: int, per_split: int) -> list[dict]:
    selected = []
    for split in ("validation", "test"):
        heap = []
        for serial, record in enumerate(parquet_records(directory / f"{split}.parquet")):
            key = f"{seed}:independent-leakage-audit:{split}:{record['sample_id']}"
            rank = int(hashlib.sha256(key.encode()).hexdigest(), 16)
            heapq.heappush(heap, (-rank, serial, record))
            if len(heap) > per_split:
                heapq.heappop(heap)
        selected.extend(record for _, _, record in sorted(heap, reverse=True))
    return selected


class SampledLeakageAudit:
    """Exact five-gram Jaccard against every training paragraph for sampled holdouts."""

    def __init__(self, samples: list[dict]):
        self.samples = samples
        self.parts = [shingles(canonical_han(record["text"])) for record in samples]
        self.inverted = defaultdict(list)
        for index, parts in enumerate(self.parts):
            for part in parts:
                self.inverted[part].append(index)
        self.vocabulary = set(self.inverted)
        self.maximum = [0.0] * len(samples)
        self.matches = [0] * len(samples)
        self.nearest = [None] * len(samples)
        self.train_paragraphs = 0

    def observe(self, record: dict, canonical: str) -> None:
        self.train_paragraphs += 1
        parts = shingles(canonical)
        counts = Counter(index for part in parts & self.vocabulary for index in self.inverted[part])
        for index, intersection in counts.items():
            similarity = intersection / (len(parts) + len(self.parts[index]) - intersection)
            if similarity > self.maximum[index]:
                self.maximum[index] = similarity
                self.nearest[index] = {
                    "sample_id": record["sample_id"],
                    "source_page_id": record["source_page_id"],
                    "source_url": record["source_url"],
                    "text_sha256": record["text_sha256"],
                    "text_excerpt": record["text"][:300],
                }
            if similarity >= 0.9:
                self.matches[index] += 1

    def report(self) -> dict:
        return {
            "method": (
                "seeded heldout sample; exact Han five-gram Jaccard against all train paragraphs"
            ),
            "scope": "sampled paragraphs only; no full-heldout or semantic contamination claim",
            "sampled_heldout_paragraphs": len(self.samples),
            "training_paragraphs_scanned": self.train_paragraphs,
            "jaccard_threshold": 0.9,
            "flagged_heldout_paragraphs": sum(count > 0 for count in self.matches),
            "sample_results": [
                {
                    "sample_id": record["sample_id"],
                    "split": record["split"],
                    "maximum_train_jaccard": self.maximum[index],
                    "matching_train_paragraphs": self.matches[index],
                    "nearest_train": self.nearest[index],
                }
                for index, record in enumerate(self.samples)
            ],
        }


def verify(directory: Path, per_split: int = 32) -> dict:
    receipt = directory / "verification.json"
    receipt.unlink(missing_ok=True)
    manifest_path = directory / "manifest.json"
    manifest_sha256 = digest_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"glyph_bank.npz", "glyph_inventory.json"} | {
        f"{split}.{suffix}"
        for split in SPLITS
        for suffix in ("parquet", "txt", "uint16", "offsets.npy")
    }
    if set(manifest["output_sha256"]) != required:
        raise ValueError("Output manifest does not cover the complete required corpus assets")
    for name, expected in manifest["output_sha256"].items():
        if Path(name).name != name or digest_file(directory / name) != expected:
            raise ValueError("Output checksum verification failed")
    validate_scan(manifest)
    lookup, bitmaps = verify_glyphs(directory, manifest)
    selected = audit_samples(directory, manifest["configuration"]["seed"], per_split)
    audit = SampledLeakageAudit(selected)
    page_splits = {}
    seen_hashes = set()
    seen_canonical_hashes = set()
    counts = {}
    examples = []
    source_files = {Path(item["path"]).name for item in manifest["source"].get("files", [])}
    for split in SPLITS:
        parquet = pq.ParquetFile(directory / f"{split}.parquet")
        ids = np.memmap(directory / f"{split}.uint16", dtype="<u2", mode="r")
        offsets = np.load(directory / f"{split}.offsets.npy", allow_pickle=False)
        paragraphs = parquet.metadata.num_rows
        if (
            offsets.ndim != 1
            or offsets.dtype != np.dtype("int64")
            or len(offsets) != paragraphs + 1
            or offsets[0] != 0
            or offsets[-1] != len(ids)
        ):
            raise ValueError("Broken sequence offsets")
        if not (np.diff(offsets) > 2).all() or int(ids.max()) >= len(bitmaps):
            raise ValueError("Invalid sequence lengths or image references")
        split_pages = set()
        han_count = 0
        for index, record in enumerate(parquet_records(directory / f"{split}.parquet")):
            text = record["text"]
            if (
                record["split"] != split
                or not ALLOWED.fullmatch(text)
                or unicodedata.normalize("NFC", text) != text
            ):
                raise ValueError("Noncanonical corpus text or incorrect split label")
            actual_hash = hashlib.sha256(text.encode()).hexdigest()
            if actual_hash != record["text_sha256"] or actual_hash in seen_hashes:
                raise ValueError("Text checksum mismatch or duplicate paragraph")
            seen_hashes.add(actual_hash)
            canonical = canonical_han(text)
            canonical_hash = hashlib.sha256(canonical.encode()).hexdigest()
            if canonical_hash in seen_canonical_hashes:
                raise ValueError("Punctuation-only duplicate paragraph")
            seen_canonical_hashes.add(canonical_hash)
            if (
                len(canonical) != record["han_count"]
                or len(canonical) < manifest["configuration"]["min_han"]
                or len(text) > 4096
            ):
                raise ValueError("Han count or paragraph length contract mismatch")
            han_count += len(canonical)
            if manifest["source"].get("provider") == "ModelScope":
                if record["source_revision_id"] is not None:
                    raise ValueError("Mirror does not supply an article revision ID")
                if (
                    record["source_file_revision"] != manifest["source"]["revision"]
                    or record["source_snapshot"] != manifest["source"]["snapshot"]
                    or record["source_file"] not in source_files
                ):
                    raise ValueError("Source file provenance does not match manifest")
            page_id = record["source_page_id"]
            if split_for_page(page_id, manifest["configuration"]["seed"]) != split or (
                page_id in page_splits and page_splits[page_id] != split
            ):
                raise ValueError("Source page split differs from deterministic assignment")
            page_splits[page_id] = split
            split_pages.add(page_id)
            sequence = ids[offsets[index] : offsets[index + 1]]
            expected = [1, *(lookup[char] for char in text), 2]
            if sequence.tolist() != expected:
                raise ValueError("Binary asset references do not round-trip to the text")
            if split == "train":
                audit.observe(record, canonical)
                if len(examples) < 3:
                    examples.append(text[:160])
        counts[split] = {
            "paragraphs": paragraphs,
            "source_pages": len(split_pages),
            "han_characters": han_count,
            "grid_tokens_including_bos_eos": len(ids),
        }
        if counts[split] != manifest["splits"][split]:
            raise ValueError("Independently counted export statistics do not match manifest")
    if (
        sum(count["paragraphs"] for count in counts.values())
        != manifest["filter_stats"]["retained_paragraphs"]
    ):
        raise ValueError("Retained paragraph total does not match exports")
    filter_stats = manifest["filter_stats"]
    rejection_keys = (
        "rejected_non_chinese_or_unsupported_symbols",
        "rejected_short",
        "rejected_long",
        "rejected_repeated_character",
        "rejected_unrenderable",
        "rejected_exact_duplicate",
        "rejected_punctuation_variant",
        "rejected_near_duplicate",
    )
    if filter_stats["candidate_paragraphs"] != filter_stats["retained_paragraphs"] + sum(
        filter_stats.get(key, 0) for key in rejection_keys
    ):
        raise ValueError("Candidate paragraph accounting does not reconcile with all rejections")
    consumed = {"glyph_bank.npz", "glyph_inventory.json"} | {
        f"{split}.{suffix}" for split in SPLITS for suffix in ("uint16", "offsets.npy")
    }
    fingerprints = {name: digest_file(directory / name) for name in sorted(consumed)}
    if any(fingerprints[name] != manifest["output_sha256"][name] for name in consumed):
        raise ValueError("Model assets changed during verification")
    if digest_file(manifest_path) != manifest_sha256:
        raise ValueError("Manifest changed during verification")
    write_atomic(directory / "audit_samples.json", selected)
    report = {
        "passed": True,
        "manifest_sha256": manifest_sha256,
        "model_consumed_sha256": fingerprints,
        "audit_samples_sha256": digest_file(directory / "audit_samples.json"),
        "checks": [
            "sha256",
            "source_scan_coverage",
            "strict_chinese",
            "binary_pixels",
            "full_glyph_rerender",
            "canonical_inventory",
            "control_frames",
            "glyph_collisions",
            "sequence_alignment",
            "page_split",
            "exact_dedup",
            "punctuation_variant_dedup",
            "source_file_provenance",
            "independent_counts",
        ],
        "near_duplicate_note": "Sampled audit; preprocessing deduplication is approximate.",
        "leakage_audit": audit.report(),
        "splits": counts,
        "glyph_shape": list(bitmaps.shape),
        "examples": examples,
    }
    write_atomic(receipt, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--audit-per-split", type=int, default=32)
    args = parser.parse_args()
    if args.audit_per_split <= 0:
        parser.error("audit-per-split must be positive")
    report = verify(args.directory, args.audit_per_split)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
