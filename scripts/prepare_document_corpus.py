"""Restore long Chinese documents, retaining v2 where safe replacement is impossible."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import runpy
import shutil
import sqlite3
import subprocess
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from hansgpt_research.packed_glyph_data import packing_statistics
from hansgpt_research.prepare_corpus import (
    ALLOWED,
    HAN,
    PUNCTUATION_MAP,
    artifact_reason,
    canonical_han,
    shingle_anchors,
    shingles,
)

LEGACY = None
REPLAY = None
SETTINGS = None
HELD = None
EXPORT_STATE = None


def output_schema():
    return pa.schema(
        [(name, pa.string()) for name in ("text", "source_page_id", "source_family", "provenance")]
    )


def base_export_job(task):
    cfg, removed, covered = EXPORT_STATE
    parent = Path(cfg["parent"])
    directory = Path(cfg["interim"]) / "base_shards" / str(task["start"])
    directory.mkdir(parents=True)
    offsets = np.load(parent / "train.offsets.npy", mmap_mode="r")
    tokens = np.memmap(parent / "train.uint16", dtype="<u2", mode="r")
    pf = pq.ParquetFile(parent / "train.parquet")
    out_offsets = [0]
    count = task["start"]
    han = replaced = 0
    with (
        (directory / "data.uint16").open("wb", buffering=8 * 2**20) as binary,
        pq.ParquetWriter(directory / "data.parquet", output_schema(), compression="zstd") as writer,
    ):
        for rg in task["groups"]:
            records = []
            for r in pf.read_row_group(rg).to_pylist():
                idx = count
                count += 1
                mask = removed[r["parent_start"] : r["parent_start"] + r["parent_count"]]
                if mask.any():
                    if not mask.all() or any(
                        hashlib.sha256(t.encode()).digest() not in covered[r["source_page_id"]]
                        for t in r["text"].splitlines()
                    ):
                        raise ValueError("Independent parent coverage check failed")
                    replaced += 1
                    continue
                ids = tokens[offsets[idx] : offsets[idx + 1]]
                binary.write(ids.tobytes())
                out_offsets.append(out_offsets[-1] + len(ids))
                han += r["han_count"]
                records.append(
                    dict(
                        text=r["text"],
                        source_page_id=r["source_page_id"],
                        source_family=r["source_family"],
                        provenance=json.dumps(
                            dict(
                                kind="v2",
                                parent_start=r["parent_start"],
                                parent_count=r["parent_count"],
                            )
                        ),
                    )
                )
            if records:
                writer.write_table(pa.Table.from_pylist(records, schema=output_schema()))
    np.save(directory / "offsets.npy", np.asarray(out_offsets, dtype=np.int64))
    return dict(path=str(directory), han=han, replaced=replaced)


def parquet_jobs(path, rows_per_task=100000):
    pf = pq.ParquetFile(path)
    tasks = []
    count = start = 0
    groups = []
    for i in range(pf.num_row_groups):
        groups.append(i)
        count += pf.metadata.row_group(i).num_rows
        if count - start >= rows_per_task or i + 1 == pf.num_row_groups:
            tasks.append(dict(start=start, groups=groups))
            start, groups = count, []
    return tasks


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


class NearIndex:
    """Same five-gram/anchor criterion as v1, without its 100-candidate cap."""

    def __init__(self):
        self.texts = []
        self.exact = set()
        self.anchors = defaultdict(list)

    def add(self, canonical):
        digest = hashlib.sha256(canonical.encode()).digest()
        if digest in self.exact:
            return
        self.exact.add(digest)
        idx = len(self.texts)
        self.texts.append(canonical)
        for anchor in shingle_anchors(canonical):
            self.anchors[anchor].append(idx)

    def matches(self, canonical):
        if hashlib.sha256(canonical.encode()).digest() in self.exact:
            return True
        hits = Counter(i for a in shingle_anchors(canonical) for i in self.anchors.get(a, ()))
        parts = None
        for i, count in hits.items():
            other = self.texts[i]
            if count < 2 or not len(canonical) * 0.9 <= len(other) <= len(canonical) / 0.9 + 1:
                continue
            parts = shingles(canonical) if parts is None else parts
            candidates = shingles(other)
            if len(parts & candidates) / len(parts | candidates) >= 0.9:
                return True
        return False


def document_segments(raw, converter, inventory, legacy, section, stats):
    """Keep short lines and cross-line quotes; hard failures remain hard breaks."""
    segment = []
    result = []

    def flush():
        if segment:
            text = "\n".join(x[1] for x in segment)
            if len(HAN.findall(text)) >= 20 and not legacy["quality_reason"](text):
                result.append(list(segment))
            else:
                stats["rejected_segment_quality"] += 1
            segment.clear()

    for line_id, line in enumerate(raw.splitlines()):
        text = unicodedata.normalize("NFC", line).translate(PUNCTUATION_MAP).strip()
        if not text:
            continue
        stats["raw_nonblank_lines"] += 1
        reason = None
        if not ALLOWED.fullmatch(text):
            reason = "non_chinese"
        else:
            text = converter.convert(text)
            if not ALLOWED.fullmatch(text) or any(c not in inventory for c in text):
                reason = "unsupported_glyph"
            elif artifact_reason(text) or legacy["SPAM"].search(text):
                reason = "artifact_or_spam"
            elif legacy["REPETITION"].search(text):
                reason = "repetition"
        if reason:
            stats["hard_break_" + reason] += 1
            flush()
            continue
        if document_boundary(text, section):
            stats["section_break"] += 1
            flush()
        segment.append((line_id, text))
    flush()
    return result


def document_boundary(text, section):
    """Bracketed article titles in composition anthologies start new documents."""
    return bool(section.match(text)) or (
        text.startswith("【") and text.endswith("】") and len(text) <= 80
    )


def worker(task):
    cfg = SETTINGS
    spec = cfg["files"][task["file"]]
    out = Path(cfg["interim"]) / (task["id"] + ".parquet")
    stats = Counter()
    if not REPLAY["join_allowed"](spec, task["file"]):
        return dict(task=task["id"], records=0, stats={"isolated_source_task": 1})
    index = np.load(Path(cfg["previous"]) / "indices" / f"{task['file']:04d}.npy", mmap_mode="r")
    converter = LEGACY["FastOpenCC"]()
    path = LEGACY["raw_path"](Path(cfg["raw"]), spec)
    records = []
    for row_id, row in REPLAY["raw_rows"](task, path, spec, LEGACY):
        a, b = np.searchsorted(index["row"], [row_id, row_id + 1])
        expected = index[a:b]
        if not len(expected) or np.any(expected["split"] != 0):
            continue
        stats["anchored_train_rows"] += 1
        raw = row.get("text", "")
        if not 1024 <= len(raw) <= 200000:
            stats["outside_raw_length_range"] += 1
            continue
        segments = document_segments(
            raw, converter, cfg["characters"], LEGACY, REPLAY["SECTION"], stats
        )
        if not any(sum(len(t) + 1 for _, t in seg) >= 1024 for seg in segments):
            stats["no_long_segment"] += 1
            continue
        lines = [text for seg in segments for _, text in seg]
        digests = {hashlib.sha256(t.encode()).digest() for t in lines}
        if any(x.tobytes() not in digests for x in expected["digest"]):
            stats["parent_coverage_failed"] += 1
            continue
        # Check every restored line, plus old-cleaner units and full segments.
        # Old held-out paragraphs are paragraph-level, so whole-doc checks alone
        # would miss a held-out paragraph embedded in a much longer new document.
        candidates = set(lines)
        candidates.update(LEGACY["cleaned_units"](row, spec, converter, Counter()))
        candidates.update("\n".join(t for _, t in seg) for seg in segments)
        if any(HELD.matches(canonical_han(t)) for t in candidates):
            stats["heldout_overlap_row"] += 1
            continue
        page = hashlib.sha256(raw.encode()).hexdigest()
        for seg_id, seg in enumerate(segments):
            text = "\n".join(t for _, t in seg)
            records.append(
                dict(
                    source_index=task["file"],
                    raw_row=row_id,
                    segment=seg_id,
                    source_page_id=page,
                    source_family=spec["family"],
                    text=text,
                    first_line=seg[0][0],
                    last_line=seg[-1][0],
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    parent_indices=expected["parent"].tolist(),
                )
            )
        stats["candidate_rows"] += 1
    if records:
        pq.write_table(pa.Table.from_pylist(records), out, compression="zstd")
    return dict(
        task=task["id"],
        records=len(records),
        stats=dict(stats),
        sha256=sha(out) if records else None,
    )


def replay_or_reuse(task):
    """Reuse only pinned candidates; all intervals still undergo raw verification."""
    reuse = SETTINGS.get("reuse")
    if reuse:
        source = Path(reuse) / (task["id"] + ".parquet")
        if source.exists():
            try:
                count = pq.read_metadata(source).num_rows
            except pa.ArrowInvalid:
                return worker(task)  # Interrupted writes have no valid footer.
            if count:
                target = Path(SETTINGS["interim"]) / source.name
                shutil.copyfile(source, target)
                return dict(
                    task=task["id"],
                    records=count,
                    stats={"reused_candidate_tasks": 1},
                    sha256=sha(target),
                )
    return worker(task)


def verify_candidates(task):
    """Re-read source rows and check physical interval independently of exporter."""
    cfg = SETTINGS
    path = Path(cfg["interim"]) / (task["id"] + ".parquet")
    if not path.exists():
        return 0
    records = pq.read_table(path).to_pylist()
    by_row = defaultdict(list)
    for record in records:
        by_row[record["raw_row"]].append(record)
    spec = cfg["files"][task["file"]]
    converter = LEGACY["FastOpenCC"]()
    raw_path = LEGACY["raw_path"](Path(cfg["raw"]), spec)
    verified = 0
    for row_id, row in REPLAY["raw_rows"](task, raw_path, spec, LEGACY):
        for record in by_row.get(row_id, ()):
            lines = row["text"].splitlines()[record["first_line"] : record["last_line"] + 1]
            normalized = [
                converter.convert(
                    unicodedata.normalize("NFC", x).translate(PUNCTUATION_MAP).strip()
                )
                for x in lines
                if x.strip()
            ]
            text = "\n".join(normalized)
            if (
                text != record["text"]
                or hashlib.sha256(text.encode()).hexdigest() != record["text_sha256"]
            ):
                raise ValueError("Raw interval proof failed")
            if any(not ALLOWED.fullmatch(x) or artifact_reason(x) for x in normalized):
                raise ValueError("Invalid raw interval")
            if any(document_boundary(x, REPLAY["SECTION"]) for x in normalized[1:]):
                raise ValueError("Interval crosses article heading")
            verified += 1
    if verified != len(records):
        raise ValueError("Missing raw records")
    return verified


def export(cfg, tasks, output):
    """Replace complete anchored rows only; keep every other original run."""
    global EXPORT_STATE
    parent = Path(cfg["parent"])
    original = Path(cfg["original"])
    removed = np.zeros(len(np.load(original / "train.offsets.npy")) - 1, dtype=bool)
    accepted = []
    covered_lines = defaultdict(set)
    dedup = NearIndex()
    stats = Counter()
    # Process in source order, independent of worker completion order.
    for task in sorted(tasks, key=lambda t: (t["file"], t["start"])):
        path = Path(cfg["interim"]) / (task["id"] + ".parquet")
        if not path.exists():
            continue
        groups = defaultdict(list)
        for r in pq.read_table(path).to_pylist():
            groups[r["raw_row"]].append(r)
        for group in groups.values():
            canonicals = [canonical_han(r["text"]) for r in group]
            if any(dedup.matches(t) for t in canonicals):
                stats["duplicate_candidate_rows_kept_in_v2"] += 1
                continue
            for text in canonicals:
                dedup.add(text)
            removed[group[0]["parent_indices"]] = True
            accepted.extend(group)
            for record in group:
                covered_lines[record["source_page_id"]].update(
                    hashlib.sha256(line.encode()).digest() for line in record["text"].splitlines()
                )
            stats["replaced_source_rows"] += 1
    np.save(output / "replaced_parent_indices.npy", np.flatnonzero(removed))
    stats["replaced_parent_paragraphs"] = int(removed.sum())
    schema = output_schema()
    EXPORT_STATE = (cfg, removed, covered_lines)
    jobs = parquet_jobs(parent / "train.parquet")
    if cfg["smoke"]:
        jobs = parquet_jobs(parent / "train.parquet", 8192)[:1]
    print(json.dumps(dict(phase="parallel_base_export", tasks=len(jobs))), flush=True)
    with ProcessPoolExecutor(max_workers=cfg["workers"], mp_context=mp.get_context("fork")) as pool:
        base_shards = list(pool.map(base_export_job, jobs))
    offsets = [0]
    han = 0
    buf = []
    inventory = cfg["characters"]
    with (
        (output / "train.uint16").open("wb", buffering=8 * 2**20) as binary,
        pq.ParquetWriter(output / "train.parquet", schema, compression="zstd") as writer,
    ):

        def emit(text, page, family, provenance):
            nonlocal han
            ids = np.fromiter((3 if c == "\n" else inventory[c] for c in text), dtype="<u2")
            binary.write(np.r_[np.uint16(1), ids, np.uint16(2)].astype("<u2").tobytes())
            offsets.append(offsets[-1] + len(ids) + 2)
            han += len(HAN.findall(text))
            buf.append(
                dict(
                    text=text,
                    source_page_id=page,
                    source_family=family,
                    provenance=json.dumps(provenance),
                )
            )
            if len(buf) >= 8192:
                writer.write_table(pa.Table.from_pylist(buf, schema=schema))
                buf.clear()

        for shard in base_shards:
            directory = Path(shard["path"])
            with (directory / "data.uint16").open("rb") as source:
                shutil.copyfileobj(source, binary, 8 * 2**20)
            local = np.load(directory / "offsets.npy")
            offsets.extend((local[1:] + offsets[-1]).tolist())
            for batch in pq.ParquetFile(directory / "data.parquet").iter_batches():
                writer.write_batch(batch)
            han += shard["han"]
            stats["replaced_v2_runs"] += shard["replaced"]
        for r in accepted:
            emit(
                r["text"],
                r["source_page_id"],
                r["source_family"],
                dict(
                    kind="raw_interval",
                    source_index=r["source_index"],
                    raw_row=r["raw_row"],
                    first_line=r["first_line"],
                    last_line=r["last_line"],
                ),
            )
        if buf:
            writer.write_table(pa.Table.from_pylist(buf, schema=schema))
    np.save(output / "train.offsets.npy", np.asarray(offsets, dtype=np.int64))
    stats["train_han"] = han
    stats["restored_segments"] = len(accepted)
    for split in ("validation", "test"):
        for suffix in ("parquet", "uint16", "offsets.npy"):
            shutil.copyfile(parent / f"{split}.{suffix}", output / f"{split}.{suffix}")
    for name in ("glyph_bank.npz", "glyph_inventory.json"):
        shutil.copyfile(parent / name, output / name)
    return dict(stats)


def verify_export_job(task):
    output, split, job, characters = task
    offsets = np.load(output / f"{split}.offsets.npy", mmap_mode="r")
    tokens = np.memmap(output / f"{split}.uint16", dtype="<u2", mode="r")
    pf = pq.ParquetFile(output / f"{split}.parquet")
    count = job["start"]
    for group in job["groups"]:
        for text in pf.read_row_group(group, columns=["text"]).column(0).to_pylist():
            expected = [1] + [3 if c == "\n" else characters[c] for c in text] + [2]
            if not np.array_equal(tokens[offsets[count] : offsets[count + 1]], expected):
                raise ValueError("Text/glyph mismatch")
            count += 1
    return count - job["start"]


def verify_export(output, characters, workers=24):
    stats = {}
    for split in ("train", "validation", "test"):
        offsets = np.load(output / f"{split}.offsets.npy")
        tokens = np.memmap(output / f"{split}.uint16", dtype="<u2", mode="r")
        if offsets[0] != 0 or offsets[-1] != len(tokens) or np.any(np.diff(offsets) < 2):
            raise ValueError("Invalid offsets")
        jobs = [
            (output, split, job, characters) for job in parquet_jobs(output / f"{split}.parquet")
        ]
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as pool:
            count = sum(pool.map(verify_export_job, jobs))
        if count != len(offsets) - 1:
            raise ValueError("Parquet/offset count mismatch")
        stats[split] = packing_statistics(offsets)
        lengths = np.diff(offsets) - 1
        starts = np.r_[0, np.cumsum((lengths + 1023) // 1024)[:-1]]
        full = lengths // 1024
        ids = (
            np.concatenate(
                [
                    np.arange(a, a + b, dtype=np.int64)
                    for a, b in zip(starts[full > 0], full[full > 0], strict=True)
                ]
            )
            if full.any()
            else np.empty(0, dtype=np.int64)
        )
        np.save(output / f"{split}.ctx1024_full_chunks.npy", ids)
    return stats


def main():
    global LEGACY, REPLAY, SETTINGS, HELD
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--interim", type=Path, required=True)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--reuse-candidates", type=Path)
    args = p.parse_args()
    if not 1 <= args.workers <= 64:
        raise ValueError("Invalid worker count")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Committed clean checkout required")
    if args.output.exists() or args.interim.exists():
        raise FileExistsError("Use new output and interim directories")
    if shutil.disk_usage("data").free < 40 * 2**30:
        raise OSError("Insufficient free space reserve")
    args.output.mkdir(parents=True)
    args.interim.mkdir(parents=True)
    started = time.monotonic()

    def phase(name):
        print(
            json.dumps(dict(phase=name, elapsed=round(time.monotonic() - started, 2))), flush=True
        )

    LEGACY = runpy.run_path("scripts/prepare_multidomain_corpus.py")
    REPLAY = runpy.run_path("scripts/reorganize_corpus.py")
    parent = Path("data/processed/chinese_contiguous_v2")
    original = Path("data/processed/chinese_multidomain_v1")
    previous = Path("data/interim/chinese_contiguous_v2")
    files = json.loads(Path("configs/datasets/chinese_multidomain_v1.json").read_text())["files"]
    inventory = json.loads((parent / "glyph_inventory.json").read_text())
    SETTINGS = dict(
        reuse=str(args.reuse_candidates) if args.reuse_candidates else None,
        workers=args.workers,
        smoke=args.smoke,
        parent=str(parent),
        original=str(original),
        previous=str(previous),
        interim=str(args.interim),
        raw="data/raw/chinese_multidomain_v1",
        files=files,
        characters=inventory["characters"],
    )
    phase("verify_inputs")
    receipt = json.loads((parent / "verification.json").read_text())
    if not receipt["passed"] or receipt["manifest_sha256"] != sha(parent / "manifest.json"):
        raise ValueError("Parent identity mismatch")
    checks = [(parent / name, value) for name, value in receipt["model_consumed_sha256"].items()]
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    checks.extend(
        (parent / f"{split}.parquet", parent_manifest["output_sha256"][f"{split}.parquet"])
        for split in ("train", "validation", "test")
    )
    checks += [(LEGACY["raw_path"](Path(SETTINGS["raw"]), s), s["sha256"]) for s in files]
    with ThreadPoolExecutor(max_workers=8) as pool:
        if not all(pool.map(lambda pair: sha(pair[0]) == pair[1], checks)):
            raise ValueError("Input hash mismatch")
    reuse_identity = None
    if args.reuse_candidates:
        reuse_identity = json.loads((args.reuse_candidates / "cache_identity.json").read_text())
        producer = subprocess.check_output(["git", "rev-parse", "30a0e89"], text=True).strip()
        producer_script = subprocess.check_output(
            ["git", "show", f"{producer}:scripts/prepare_document_corpus.py"]
        )
        expected = dict(
            producer_commit=producer,
            script_sha256=hashlib.sha256(producer_script).hexdigest(),
            source_config_sha256=sha("configs/datasets/chinese_multidomain_v1.json"),
            parent_manifest_sha256=sha(parent / "manifest.json"),
            exclusion_db_sha256=sha("data/interim/chinese_multidomain_v1/records.sqlite"),
        )
        if reuse_identity != expected:
            raise ValueError("Candidate reuse identity mismatch")
    phase("build_heldout_exclusions")
    HELD = NearIndex()
    db = Path("data/interim/chinese_multidomain_v1/records.sqlite").resolve()
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        for (canonical,) in conn.execute("SELECT canonical FROM records WHERE split != 'train'"):
            HELD.add(canonical)
    # Include whole v2 held-out runs as well as the old paragraph/legacy exclusions.
    for split in ("validation", "test"):
        for batch in pq.ParquetFile(parent / f"{split}.parquet").iter_batches(columns=["text"]):
            for text in batch.column(0).to_pylist():
                HELD.add(canonical_han(text))
    tasks = json.loads((previous / "tasks.json").read_text())
    if args.smoke:
        tasks = [t for t in tasks if t["file"] in {0, 1, 14, 64}][:8]
    phase("parallel_document_recovery")
    results = []
    # Fork immutable in-memory held-out indexes; avoid copying them to 24 workers.
    pa.set_cpu_count(1)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork")) as pool:
        for i, result in enumerate(pool.map(replay_or_reuse, tasks)):
            results.append(result)
            if i % 25 == 0:
                print(json.dumps(dict(completed=i + 1, tasks=len(tasks))), flush=True)
        phase("independent_raw_interval_verification")
        verified = sum(pool.map(verify_candidates, tasks))
    save(args.interim / "tasks_report.json", results)
    phase("document_dedup_and_export")
    export_stats = export(SETTINGS, tasks, args.output)
    phase("verify_export_and_packing")
    statistics = verify_export(args.output, inventory["characters"], args.workers)
    hashes = {path.name: sha(path) for path in args.output.iterdir() if path.is_file()}
    manifest = dict(
        type="chinese_document_packed_v3",
        mode="smoke" if args.smoke else "full",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        parent_manifest_sha256=sha(parent / "manifest.json"),
        source_config_sha256=sha("configs/datasets/chinese_multidomain_v1.json"),
        workers=args.workers,
        reused_candidate_identity=reuse_identity,
        cleaning_counter_scope="recomputed tasks only; reused tasks have no cleaning counters",
        elapsed_seconds=time.monotonic() - started,
        verified_raw_segments=verified,
        export=export_stats,
        packing=statistics,
        cleaning_counters=dict(sum((Counter(r["stats"]) for r in results), Counter())),
        attention_policy="causal across EOS; EOS-to-BOS loss masked; no position reset",
        dedup_policy=(
            "v2 fallback; restored document anchor-fivegram Jaccard >=0.9; "
            "heldout line/segment exclusion"
        ),
        output_sha256=hashes,
    )
    save(args.output / "manifest.json", manifest)
    names = ["glyph_bank.npz", "glyph_inventory.json"] + [
        f"{s}.{x}" for s in ("train", "validation", "test") for x in ("uint16", "offsets.npy")
    ]
    save(
        args.output / "verification.json",
        dict(
            passed=True,
            manifest_sha256=sha(args.output / "manifest.json"),
            model_consumed_sha256={n: hashes[n] for n in sorted(names)},
            verified_raw_segments=verified,
        ),
    )
    phase("complete")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
