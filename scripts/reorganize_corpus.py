"""Recover contiguous retained excerpts without repeating global paragraph dedup."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import runpy
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SPLITS = ("train", "validation", "test")
INDEX_DTYPE = np.dtype(
    [("row", "<u8"), ("unit", "<u4"), ("split", "u1"), ("parent", "<u8"), ("digest", "V32")]
)
COPY_FILES = {3, 7, 8, 9, 10, 11, 12, 13}
SMOKE_FILES = {0, 1, 3, 4, 5, 6, 8, 14, 64, 114}
SECTION = re.compile(
    r"^\s*(?:【?篇[一二三四五六七八九十百\d]+|【?第[一二三四五六七八九十百\d]+(?:篇|章|回)|问[:：]|问题[一二三四五六七八九十百\d]*[:：])"
)
_WORKER = None


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def join_allowed(spec, index):
    return index not in COPY_FILES and spec["adapter"] in {"jsonl_text", "parquet_text"}


def recover_row(raw, expected, spec, cleaner, converter, max_chars=200000):
    """Return raw coordinates and edges; every edge crosses only blank formatting lines."""
    n = len(expected)
    lines = np.full(n, -1, dtype=np.int32)
    ranks = lines.copy()
    joins = np.zeros(n, dtype=np.bool_)
    if len(raw) > max_chars:
        return lines, ranks, joins, True
    positions = {int(v["unit"]): i for i, v in enumerate(expected)}
    unit = 0
    rank = -1
    previous = None
    matched = 0
    for line_index, line in enumerate(raw.splitlines()):
        if not line.strip():
            continue
        rank += 1
        values = cleaner({"text": line}, spec, converter, Counter())
        if not values:
            previous = None
            continue
        if len(values) != 1:
            raise ValueError("Single source line produced multiple units")
        value = values[0]
        position = positions.get(unit)
        unit += 1
        if position is None:
            previous = None
            continue
        if hashlib.sha256(value.encode()).digest() != expected[position]["digest"].tobytes():
            raise ValueError("Replayed text differs from verified parent paragraph")
        lines[position] = line_index
        ranks[position] = rank
        matched += 1
        if previous is not None:
            prior = expected[previous]
            current = expected[position]
            joins[position] = (
                rank == ranks[previous] + 1
                and current["split"] == prior["split"]
                and int(current["parent"]) == int(prior["parent"]) + 1
                and not SECTION.match(line)
            )
        previous = position
        if matched == n:
            break
    if matched != n:
        raise ValueError(f"Raw replay matched {matched}/{n} retained paragraphs")
    return lines, ranks, joins, False


def verify_row_edges(raw, expected, lines, ranks, joins, spec, cleaner, converter):
    """Independent proof: legacy full-row units plus physical intervening-line checks."""
    if not joins.any():
        return
    source_lines = raw.splitlines()
    nonblank = np.cumsum([bool(x.strip()) for x in source_lines]) - 1
    units = cleaner({"text": raw}, spec, converter, Counter())
    endpoints = set(np.flatnonzero(joins).tolist())
    endpoints.update(i - 1 for i in list(endpoints))
    for i in endpoints:
        if i < 0 or lines[i] < 0:
            raise ValueError("Joined edge lacks source coordinates")
        line = int(lines[i])
        unit = int(expected[i]["unit"])
        if unit >= len(units):
            raise ValueError("Parent ordinal absent in original cleaner output")
        values = cleaner({"text": source_lines[line]}, spec, converter, Counter())
        if values != [units[unit]]:
            raise ValueError("Original line/filtered ordinal mismatch")
        if hashlib.sha256(units[unit].encode()).digest() != expected[i]["digest"].tobytes():
            raise ValueError("Source endpoint hash mismatch")
        if nonblank[line] != ranks[i]:
            raise ValueError("Incorrect raw nonblank rank")
    for i in np.flatnonzero(joins):
        a, b = int(lines[i - 1]), int(lines[i])
        if b <= a or any(x.strip() for x in source_lines[a + 1 : b]):
            raise ValueError("Join crosses removed content")
        if SECTION.match(source_lines[b]):
            raise ValueError("Join crosses an explicit section boundary")
        if (
            expected[i]["split"] != expected[i - 1]["split"]
            or int(expected[i]["parent"]) != int(expected[i - 1]["parent"]) + 1
        ):
            raise ValueError("Join crosses a parent-order or split boundary")


def init_worker(parent, interim, raw, files):
    global _WORKER
    pa.set_cpu_count(1)
    legacy = runpy.run_path("scripts/prepare_multidomain_corpus.py")
    _WORKER = (Path(parent), Path(interim), Path(raw), files, legacy, legacy["FastOpenCC"]())


def raw_rows(task, path, spec, legacy):
    if spec["adapter"] == "parquet_text":
        for row_index, row in enumerate(legacy["rows"](path, spec["adapter"])):
            if row_index >= task["stop"]:
                break
            if row_index >= task["start"]:
                yield row_index, row
    else:
        with path.open("rb") as f:
            f.seek(task["byte_start"])
            row_index = task["start"]
            while row_index < task["stop"]:
                line = f.readline()
                if not line:
                    raise EOFError("Raw JSONL task ended early")
                if not line.strip():
                    continue
                yield row_index, json.loads(line)
                row_index += 1


def replay_task(task, folder="replay", verify=False):
    parent, interim, raw, files, legacy, converter = _WORKER
    index = task["file"]
    spec = files[index]
    arr = np.load(interim / "indices" / f"{index:04d}.npy", mmap_mode="r")
    lo = int(np.searchsorted(arr["row"], task["start"]))
    hi = int(np.searchsorted(arr["row"], task["stop"]))
    expected = arr[lo:hi]
    path = interim / folder / (task["id"] + ".npz")
    if not verify and path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if int(saved["lo"]) != lo or int(saved["hi"]) != hi:
                raise ValueError("Cached task range mismatch")
        return {"id": task["id"], "path": str(path), "parents": hi - lo, "cached": True}
    if verify:
        with np.load(path, allow_pickle=False) as saved:
            all_lines = saved["lines"]
            all_ranks = saved["ranks"]
            all_joins = saved["joins"]
        if not all_joins.any():
            return {"id": task["id"], "verified_edges": 0}
    else:
        all_lines = np.full(hi - lo, -1, dtype=np.int32)
        all_ranks = all_lines.copy()
        all_joins = np.zeros(hi - lo, dtype=np.bool_)
    matched = 0
    opaque = 0
    started = time.monotonic()
    raw_path = legacy["raw_path"](raw, spec)
    for row_index, row in raw_rows(task, raw_path, spec, legacy):
        a = int(np.searchsorted(expected["row"], row_index))
        b = int(np.searchsorted(expected["row"], row_index, side="right"))
        if a == b:
            continue
        text = row.get("text")
        if not isinstance(text, str):
            raise ValueError("Retained source row has no text")
        if verify:
            verify_row_edges(
                text,
                expected[a:b],
                all_lines[a:b],
                all_ranks[a:b],
                all_joins[a:b],
                spec,
                legacy["cleaned_units"],
                converter,
            )
        else:
            lines, ranks, joins, large = recover_row(
                text, expected[a:b], spec, legacy["cleaned_units"], converter
            )
            all_lines[a:b] = lines
            all_ranks[a:b] = ranks
            all_joins[a:b] = joins
            opaque += int(large)
        matched += b - a
    if matched != len(expected):
        raise ValueError("Task did not visit every retained source row")
    if verify:
        return {"id": task["id"], "verified_edges": int(all_joins.sum())}
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp")
    with pending.open("wb") as f:
        np.savez_compressed(f, lo=lo, hi=hi, lines=all_lines, ranks=all_ranks, joins=all_joins)
    pending.replace(path)
    return {
        "id": task["id"],
        "path": str(path),
        "parents": len(expected),
        "joins": int(all_joins.sum()),
        "opaque_rows": opaque,
        "seconds": time.monotonic() - started,
        "content_sha256": hashlib.sha256(
            all_lines.tobytes() + all_ranks.tobytes() + all_joins.tobytes()
        ).hexdigest(),
    }


def merged_ranges(ranges):
    result = []
    for start, stop in sorted(ranges):
        if result and result[-1][1] == start:
            result[-1][1] = stop
        else:
            result.append([start, stop])
    return result


def build_indices(parent, interim, files, smoke):
    pieces = defaultdict(list)
    ranges = {s: [] for s in SPLITS}
    counts = {}
    for split_id, split in enumerate(SPLITS):
        position = 0
        selected = Counter()
        pf = pq.ParquetFile(parent / f"{split}.parquet")
        counts[split] = pf.metadata.num_rows
        for batch in pf.iter_batches(batch_size=131072, columns=["sample_id", "text_sha256"]):
            parts = pc.split_pattern(batch.column(0), pattern=":")
            fids = pc.cast(pc.list_element(parts, 0), pa.int32()).to_numpy()
            rows = pc.cast(pc.list_element(parts, 1), pa.uint64()).to_numpy()
            units = pc.cast(pc.list_element(parts, 2), pa.uint32()).to_numpy()
            hashes = batch.column(1).to_pylist()
            for index in np.unique(fids):
                if smoke and index not in SMOKE_FILES:
                    continue
                ids = np.flatnonzero(fids == index)
                if smoke:
                    ids = ids[: max(0, 300 - selected[int(index)])]
                if not len(ids):
                    continue
                if np.any(np.diff(ids) != 1):
                    raise ValueError("Parent files are not source-ordered")
                arr = np.empty(len(ids), dtype=INDEX_DTYPE)
                arr["row"] = rows[ids]
                arr["unit"] = units[ids]
                arr["split"] = split_id
                arr["parent"] = ids + position
                arr["digest"] = [np.void(bytes.fromhex(hashes[i])) for i in ids]
                pieces[int(index)].append(arr)
                selected[int(index)] += len(ids)
                ranges[split].append([int(position + ids[0]), int(position + ids[-1] + 1)])
            position += len(batch)
    directory = interim / "indices"
    directory.mkdir(parents=True, exist_ok=True)
    summary = {}
    for index, values in pieces.items():
        arr = np.concatenate(values)
        arr = arr[np.lexsort((arr["unit"], arr["row"]))]
        if len(arr) > 1 and np.any(
            (arr["row"][1:] == arr["row"][:-1]) & (arr["unit"][1:] == arr["unit"][:-1])
        ):
            raise ValueError("Duplicate original sample ID")
        np.save(directory / f"{index:04d}.npy", arr)
        summary[str(index)] = {
            "parents": len(arr),
            "max_row": int(arr["row"][-1]),
            "replay": join_allowed(files[index], index),
        }
    plan = {
        "counts": counts,
        "ranges": {s: merged_ranges(ranges[s]) for s in SPLITS},
        "sources": summary,
    }
    atomic_json(interim / "index_plan.json", plan)
    return plan


def build_tasks(raw, interim, files, plan):
    legacy = runpy.run_path("scripts/prepare_multidomain_corpus.py")
    tasks = []
    for key, info in plan["sources"].items():
        index = int(key)
        spec = files[index]
        if not info["replay"]:
            continue
        arr = np.load(interim / "indices" / f"{index:04d}.npy", mmap_mode="r")
        stop = info["max_row"] + 1
        if spec["adapter"] == "parquet_text":
            tasks.append(dict(file=index, start=0, stop=stop, bytes=spec["size"]))
        else:
            path = legacy["raw_path"](raw, spec)
            with path.open("rb") as handle:
                row = 0
                start = 0
                byte_start = 0
                while row < stop:
                    line = handle.readline()
                    if not line:
                        raise EOFError("Parent source ordinal exceeds raw file")
                    if not line.strip():
                        continue
                    row += 1
                    if (
                        row - start >= 2048
                        or handle.tell() - byte_start >= 8 * 2**20
                        or row == stop
                    ):
                        if np.searchsorted(arr["row"], row) > np.searchsorted(arr["row"], start):
                            tasks.append(
                                dict(
                                    file=index,
                                    start=start,
                                    stop=row,
                                    byte_start=byte_start,
                                    bytes=handle.tell() - byte_start,
                                )
                            )
                        start = row
                        byte_start = handle.tell()
    for t in tasks:
        t["id"] = f"{t['file']:04d}_{t['start']:09d}_{t['stop']:09d}"
    tasks.sort(key=lambda t: t["bytes"], reverse=True)
    atomic_json(interim / "tasks.json", tasks)
    return tasks


def parent_table(parent, split, start, stop):
    pf = pq.ParquetFile(parent / f"{split}.parquet")
    sizes = np.array([pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)])
    offsets = np.r_[0, np.cumsum(sizes)]
    first = int(np.searchsorted(offsets, start, side="right") - 1)
    last = int(np.searchsorted(offsets, stop - 1, side="right") - 1)
    return pf.read_row_groups(list(range(first, last + 1))).slice(
        start - int(offsets[first]), stop - start
    )


def shard_schema(schema):
    return (
        schema.append(pa.field("parent_start", pa.int64()))
        .append(pa.field("parent_count", pa.int32()))
        .append(pa.field("first_raw_line", pa.int32()))
        .append(pa.field("last_raw_line", pa.int32()))
        .append(pa.field("parent_domains", pa.list_(pa.string())))
        .append(pa.field("continuity_policy", pa.string()))
    )


def export_task(task):
    parent, interim, raw, files, legacy, converter = _WORKER
    split, start, stop = task["split"], task["start"], task["stop"]
    directory = interim / "shards" / task["id"]
    receipt = directory / "complete.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    table = parent_table(parent, split, start, stop)
    schema = shard_schema(table.schema)
    records = table.to_pylist()
    flags = np.load(interim / f"{split}.joins.npy", mmap_mode="r")
    lines = np.load(interim / f"{split}.lines.npy", mmap_mode="r")
    old_offsets = np.load(parent / f"{split}.offsets.npy", mmap_mode="r")
    old_tokens = np.memmap(parent / f"{split}.uint16", dtype="<u2", mode="r")
    offsets = [0]
    rows = []
    family = Counter()
    count = 0
    joined = 0

    def flush_group(a, b, binary, textfile):
        nonlocal count, joined
        selected = records[a:b]
        first = selected[0]
        global_start = start + a
        if any(
            r["source_page_id"] != first["source_page_id"]
            or r["source_file"] != first["source_file"]
            for r in selected
        ):
            raise ValueError("Group crossed a document/source boundary")
        coordinates = [r["sample_id"].split(":") for r in selected]
        if any(c[:2] != coordinates[0][:2] for c in coordinates):
            raise ValueError("Group crossed a raw row")
        values = [np.array([1], dtype="<u2")]
        for offset in range(a, b):
            index = start + offset
            body = old_tokens[int(old_offsets[index]) + 1 : int(old_offsets[index + 1]) - 1]
            if not np.all(body >= 4):
                raise ValueError("Original paragraph body contains controls")
            if offset > a:
                values.append(np.array([3], dtype="<u2"))
            values.append(body)
        values.append(np.array([2], dtype="<u2"))
        ids = np.concatenate(values)
        binary.write(ids.tobytes())
        offsets.append(offsets[-1] + len(ids))
        text = "\n".join(r["text"] for r in selected)
        textfile.write(text + "\n\n")
        domains = sorted({r["domain"] for r in selected})
        record = dict(first)
        record.update(
            sample_id=f"run:{split}:{global_start}:{b - a}",
            text=text,
            text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            han_count=sum(r["han_count"] for r in selected),
            parent_start=global_start,
            parent_count=b - a,
            first_raw_line=int(lines[global_start]),
            last_raw_line=int(lines[start + b - 1]),
            parent_domains=domains,
            continuity_policy="raw_consecutive_verified"
            if b - a > 1
            else "isolated_verified_parent",
            domain=domains[0] if len(domains) == 1 else "多领域文档片段",
        )
        if len(domains) > 1:
            record["domain_method"] = "inherited mixed paragraph labels"
        rows.append(record)
        family[first["source_family"]] += record["han_count"]
        count += 1
        joined += b - a - 1

    with (
        (directory / "data.uint16").open("wb", buffering=8 * 2**20) as binary,
        (directory / "data.txt").open("w", encoding="utf-8", buffering=8 * 2**20) as textfile,
        pq.ParquetWriter(directory / "data.parquet", schema, compression="zstd") as writer,
    ):
        a = 0
        for b in range(1, len(records) + 1):
            if b < len(records) and flags[start + b]:
                continue
            flush_group(a, b, binary, textfile)
            a = b
            if len(rows) >= 4096:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                rows = []
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    np.save(directory / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    result = {
        **task,
        "paragraphs": count,
        "parent_paragraphs": stop - start,
        "joins": joined,
        "tokens": offsets[-1],
        "effective_targets": offsets[-1] - count,
        "family_han": dict(family),
    }
    atomic_json(receipt, result)
    return result


def verify_export_task(task):
    parent, interim, raw, files, legacy, converter = _WORKER
    split, start, stop = task["split"], task["start"], task["stop"]
    directory = interim / "shards" / task["id"]
    original = parent_table(parent, split, start, stop).to_pylist()
    new = pq.read_table(directory / "data.parquet").to_pylist()
    parent_offsets = np.load(parent / f"{split}.offsets.npy", mmap_mode="r")
    parent_tokens = np.memmap(parent / f"{split}.uint16", dtype="<u2", mode="r")
    offsets = np.load(directory / "offsets.npy")
    tokens = np.memmap(directory / "data.uint16", dtype="<u2", mode="r")
    flags = np.load(interim / f"{split}.joins.npy", mmap_mode="r")
    cursor = start
    if len(offsets) != len(new) + 1 or offsets[0] != 0 or offsets[-1] != len(tokens):
        raise ValueError("Shard offset mismatch")
    for i, row in enumerate(new):
        n = row["parent_count"]
        if row["parent_start"] != cursor or n < 1:
            raise ValueError("Parent coverage/order mismatch")
        parents = original[cursor - start : cursor - start + n]
        expected = "\n".join(v["text"] for v in parents)
        if (
            row["text"] != expected
            or row["text_sha256"] != hashlib.sha256(expected.encode()).hexdigest()
        ):
            raise ValueError("Text changed")
        if row["han_count"] != sum(v["han_count"] for v in parents):
            raise ValueError("Han count changed")
        if any(
            v["source_page_id"] != row["source_page_id"] or v["split"] != split for v in parents
        ):
            raise ValueError("Source/split changed")
        changed = {"sample_id", "text", "text_sha256", "han_count", "domain", "domain_method"}
        for key, value in parents[0].items():
            if key not in changed and row[key] != value:
                raise ValueError("Inherited metadata changed")
        if flags[cursor] or (n > 1 and not flags[cursor + 1 : cursor + n].all()):
            raise ValueError("Group boundary proof mismatch")
        sequence = tokens[offsets[i] : offsets[i + 1]]
        if sequence[0] != 1 or sequence[-1] != 2:
            raise ValueError("Missing BOS/EOS")
        position = 1
        for j in range(cursor, cursor + n):
            body = parent_tokens[parent_offsets[j] + 1 : parent_offsets[j + 1] - 1]
            if not np.array_equal(sequence[position : position + len(body)], body):
                raise ValueError("Glyph content changed")
            position += len(body)
            if j < cursor + n - 1:
                if sequence[position] != 3:
                    raise ValueError("Paragraph newline control missing")
                position += 1
        if position != len(sequence) - 1:
            raise ValueError("Unexpected extra glyphs")
        cursor += n
    if cursor != stop:
        raise ValueError("Dropped or repeated parents")
    return {"id": task["id"], "verified_parents": stop - start, "verified_runs": len(new)}


def verify_combined_task(task):
    parent, interim, raw, files, legacy, converter = _WORKER
    root = Path(task["output"])
    split = task["split"]
    shard = interim / "shards" / task["id"]
    original = pq.read_table(shard / "data.parquet")
    combined = parent_table(root, split, task["new_start"], task["new_start"] + task["paragraphs"])
    if not original.equals(combined):
        raise ValueError("Combined Parquet differs from verified shard")
    offsets = np.load(root / f"{split}.offsets.npy", mmap_mode="r")
    local = np.load(shard / "offsets.npy")
    begin = int(offsets[task["new_start"]])
    if not np.array_equal(
        offsets[task["new_start"] : task["new_start"] + len(local)], local + begin
    ):
        raise ValueError("Combined offsets differ from verified shard")
    tokens = np.memmap(root / f"{split}.uint16", dtype="<u2", mode="r")
    expected = np.memmap(shard / "data.uint16", dtype="<u2", mode="r")
    if not np.array_equal(tokens[begin : begin + len(expected)], expected):
        raise ValueError("Combined glyph stream mismatch")
    return {"id": task["id"], "verified_combined_runs": len(original)}


def pool_map(function, tasks, args, folder=None):
    output = []
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
        initializer=init_worker,
        initargs=(str(args.parent), str(args.interim), str(args.raw), args.files),
    ) as pool:
        futures = {
            pool.submit(function, t, folder) if folder else pool.submit(function, t): t
            for t in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            output.append(result)
            if len(output) % 32 == 0 or len(output) == len(tasks):
                atomic_json(
                    args.interim / "status.json",
                    {
                        "phase": function.__name__,
                        "completed": len(output),
                        "tasks": len(tasks),
                        "workers": args.workers,
                        "seconds": time.monotonic() - started,
                        "time": datetime.now(UTC).isoformat(),
                        "pid": os.getpid(),
                    },
                )
    return output


def verify_replay(task):
    return replay_task(task, verify=True)


def combine_split(split, args, shards):
    selected = sorted([r for r in shards if r["split"] == split], key=lambda r: r["start"])
    schema = pq.ParquetFile(
        args.interim / "shards" / selected[0]["id"] / "data.parquet"
    ).schema_arrow
    offsets = [np.zeros(1, dtype=np.int64)]
    total = 0
    family = Counter()
    runs = parents = joins = 0
    with (
        (args.output / f"{split}.uint16").open("wb", buffering=8 * 2**20) as binary,
        (args.output / f"{split}.txt").open("wb", buffering=8 * 2**20) as text,
        pq.ParquetWriter(args.output / f"{split}.parquet", schema, compression="zstd") as writer,
    ):
        for r in selected:
            d = args.interim / "shards" / r["id"]
            with (d / "data.uint16").open("rb") as src:
                shutil.copyfileobj(src, binary, 8 * 2**20)
            with (d / "data.txt").open("rb") as src:
                shutil.copyfileobj(src, text, 8 * 2**20)
            for batch in pq.ParquetFile(d / "data.parquet").iter_batches(batch_size=65536):
                writer.write_batch(batch)
            local = np.load(d / "offsets.npy")
            offsets.append(local[1:] + total)
            total += int(local[-1])
            family.update(r["family_han"])
            runs += r["paragraphs"]
            parents += r["parent_paragraphs"]
            joins += r["joins"]
    offsets = np.concatenate(offsets)
    np.save(args.output / f"{split}.offsets.npy", offsets)
    targets = np.diff(offsets) - 1
    statistics = {}
    for ctx in [256, 512, 1024, 2048]:
        counts = (targets + ctx - 1) // ctx
        full = targets // ctx
        statistics[str(ctx)] = {
            "chunks": int(counts.sum()),
            "mean_effective_targets": float(targets.sum() / counts.sum()),
            "full_chunks": int(full.sum()),
            "full_chunk_fraction": float(full.sum() / counts.sum()),
        }
        if ctx == 1024:
            starts = np.r_[0, np.cumsum(counts)[:-1]]
            full_ids = (
                np.concatenate(
                    [
                        np.arange(a, a + b, dtype=np.int64)
                        for a, b in zip(starts[full > 0], full[full > 0], strict=True)
                    ]
                )
                if full.any()
                else np.empty(0, dtype=np.int64)
            )
            np.save(args.output / f"{split}.ctx1024_full_chunks.npy", full_ids)
    for minimum in [512, 1024]:
        np.save(args.output / f"{split}.runs_ge{minimum}.npy", np.flatnonzero(targets >= minimum))
    return {
        "split": split,
        "paragraphs": runs,
        "parent_paragraphs": parents,
        "joins": joins,
        "han_characters": sum(family.values()),
        "effective_targets": int(targets.sum()),
        "family_han": dict(family),
        "lengths": {
            "mean_content_grids": float((targets - 1).mean()),
            "p50": float(np.quantile(targets - 1, 0.5)),
            "p95": float(np.quantile(targets - 1, 0.95)),
            "max": int(targets.max() - 1),
        },
        "contexts": statistics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent", type=Path, default=Path("data/processed/chinese_multidomain_v1")
    )
    parser.add_argument("--raw", type=Path, default=Path("data/raw/chinese_multidomain_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interim", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 96:
        raise ValueError("Invalid worker count")
    if args.output.resolve() == args.parent.resolve():
        raise ValueError("Never overwrite parent")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Clean committed source required")
    if (args.output / "verification.json").exists():
        raise FileExistsError("Verified output is immutable")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Use --resume for an existing attempt")
    args.interim.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = (args.interim / "prepare.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_path = Path("configs/datasets/chinese_multidomain_v1.json")
    args.files = json.loads(source_path.read_text())["files"]
    parent_manifest = json.loads((args.parent / "manifest.json").read_text())
    receipt = json.loads((args.parent / "verification.json").read_text())
    if not receipt["passed"] or receipt["manifest_sha256"] != digest(args.parent / "manifest.json"):
        raise ValueError("Parent verification mismatch")
    if parent_manifest["source_config_sha256"] != digest(source_path):
        raise ValueError("Parent source list differs")
    identity = {
        "parent_manifest_sha256": digest(args.parent / "manifest.json"),
        "source_config_sha256": digest(source_path),
        "script_sha256": digest(Path(__file__)),
        "cleaner_sha256": digest("scripts/prepare_multidomain_corpus.py"),
        "base_cleaner_sha256": digest("src/hansgpt_research/prepare_corpus.py"),
        "smoke": args.smoke,
        "parent": str(args.parent.resolve()),
        "raw": str(args.raw.resolve()),
        "output": str(args.output.resolve()),
    }
    path = args.interim / "identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Resume identity mismatch")
    atomic_json(path, identity)
    started = time.monotonic()
    timings = {}

    def phase(name):
        space = os.statvfs(args.output)
        if space.f_bavail * space.f_frsize < 40 * 2**30:
            raise OSError("Free space below 40 GiB reserve")
        atomic_json(
            args.interim / "status.json",
            {"phase": name, "pid": os.getpid(), "time": datetime.now(UTC).isoformat()},
        )

    try:
        phase("verify_parent")
        tick = time.monotonic()
        needed = {
            **receipt["model_consumed_sha256"],
            **{f"{s}.parquet": parent_manifest["output_sha256"][f"{s}.parquet"] for s in SPLITS},
        }
        with ThreadPoolExecutor(max_workers=8) as pool:
            actual = list(
                pool.map(
                    lambda item: (item[0], digest(args.parent / item[0]) == item[1]), needed.items()
                )
            )
        if not all(ok for _, ok in actual):
            raise ValueError("Parent files changed")
        timings["verify_parent"] = time.monotonic() - tick
        phase("index")
        tick = time.monotonic()
        plan_path = args.interim / "index_plan.json"
        plan = (
            json.loads(plan_path.read_text())
            if plan_path.exists()
            else build_indices(args.parent, args.interim, args.files, args.smoke)
        )
        task_path = args.interim / "tasks.json"
        tasks = (
            json.loads(task_path.read_text())
            if task_path.exists()
            else build_tasks(args.raw, args.interim, args.files, plan)
        )
        timings["index"] = time.monotonic() - tick
        phase("verify_raw")
        tick = time.monotonic()
        legacy = runpy.run_path("scripts/prepare_multidomain_corpus.py")
        used = sorted({t["file"] for t in tasks})

        def check_raw(i):
            spec = args.files[i]
            p = legacy["raw_path"](args.raw, spec)
            return p.stat().st_size == spec["size"] and digest(p) == spec["sha256"]

        with ThreadPoolExecutor(max_workers=8) as pool:
            if not all(pool.map(check_raw, used)):
                raise ValueError("Raw source changed")
        timings["verify_raw"] = time.monotonic() - tick
        benchmark_path = args.interim / "benchmark.json"
        if args.benchmark and benchmark_path.exists():
            args.workers = json.loads(benchmark_path.read_text())["selected_workers"]
        elif args.benchmark:
            samples = tasks[:: max(1, len(tasks) // 96)][:96]
            bench = []
            benchmark_id = str(time.time_ns())
            for workers in [8, 24, 48]:
                args.workers = workers
                tick = time.monotonic()
                results = pool_map(
                    replay_task, samples, args, f"benchmark_{benchmark_id}_{workers}"
                )
                bench.append(
                    {"workers": workers, "seconds": time.monotonic() - tick, "results": results}
                )
            signatures = [{r["id"]: r["content_sha256"] for r in b["results"]} for b in bench]
            if not all(s == signatures[0] for s in signatures):
                raise ValueError("Parallelism changed replay results")
            args.workers = min(bench, key=lambda b: b["seconds"])["workers"]
            atomic_json(
                args.interim / "benchmark.json", {"runs": bench, "selected_workers": args.workers}
            )
        phase("replay")
        tick = time.monotonic()
        results = pool_map(replay_task, tasks, args)
        timings["replay"] = time.monotonic() - tick
        flags = [np.zeros(plan["counts"][s], dtype=np.bool_) for s in SPLITS]
        lines = [np.full(plan["counts"][s], -1, dtype=np.int32) for s in SPLITS]
        seen = [np.zeros(plan["counts"][s], dtype=np.uint8) for s in SPLITS]
        for key, info in plan["sources"].items():
            if info["replay"]:
                continue
            arr = np.load(args.interim / "indices" / f"{int(key):04d}.npy", mmap_mode="r")
            for sid in range(3):
                seen[sid][arr["parent"][arr["split"] == sid]] = 1
        by_id = {t["id"]: t for t in tasks}
        for result in results:
            task = by_id[result["id"]]
            arr = np.load(args.interim / "indices" / f"{task['file']:04d}.npy", mmap_mode="r")
            with np.load(result["path"]) as values:
                arr = arr[int(values["lo"]) : int(values["hi"])]
                for sid in range(3):
                    mask = arr["split"] == sid
                    ids = arr["parent"][mask]
                    if seen[sid][ids].any():
                        raise ValueError("Repeated parent replay")
                    seen[sid][ids] = 1
                    flags[sid][ids] = values["joins"][mask]
                    lines[sid][ids] = values["lines"][mask]
        export_tasks = []
        for sid, split in enumerate(SPLITS):
            expected = sum(b - a for a, b in plan["ranges"][split])
            if int(seen[sid].sum()) != expected:
                raise ValueError("Missing parent replay coverage")
            np.save(args.interim / f"{split}.joins.npy", flags[sid])
            np.save(args.interim / f"{split}.lines.npy", lines[sid])
            for start, stop in plan["ranges"][split]:
                if flags[sid][start]:
                    raise ValueError("Selected range begins inside a join")
                while start < stop:
                    end = min(start + 100000, stop)
                    while end < stop and flags[sid][end]:
                        end += 1
                    export_tasks.append(
                        dict(
                            split=split,
                            start=start,
                            stop=end,
                            id=f"{split}_{start:010d}_{end:010d}",
                        )
                    )
                    start = end
        phase("export")
        tick = time.monotonic()
        shards = pool_map(export_task, export_tasks, args)
        timings["export"] = time.monotonic() - tick
        phase("verify_source_edges")
        tick = time.monotonic()
        edges = pool_map(verify_replay, tasks, args)
        timings["verify_edges"] = time.monotonic() - tick
        if sum(r["verified_edges"] for r in edges) != sum(int(x.sum()) for x in flags):
            raise ValueError("Unverified edges")
        phase("verify_shards")
        tick = time.monotonic()
        verified = pool_map(verify_export_task, export_tasks, args)
        timings["verify_shards"] = time.monotonic() - tick
        phase("combine")
        tick = time.monotonic()
        with ThreadPoolExecutor(max_workers=3) as pool:
            split_reports = list(pool.map(lambda s: combine_split(s, args, shards), SPLITS))
        for name in ["glyph_bank.npz", "glyph_inventory.json"]:
            shutil.copyfile(args.parent / name, args.output / name)
        timings["combine"] = time.monotonic() - tick
        phase("verify_combined")
        tick = time.monotonic()
        combine_tasks = []
        for split in SPLITS:
            position = 0
            for r in sorted([v for v in shards if v["split"] == split], key=lambda v: v["start"]):
                combine_tasks.append({**r, "output": str(args.output), "new_start": position})
                position += r["paragraphs"]
            if pq.ParquetFile(args.output / f"{split}.parquet").metadata.num_rows != position:
                raise ValueError("Combined row count mismatch")
        pool_map(verify_combined_task, combine_tasks, args)
        for split in SPLITS:
            expected_hash = hashlib.sha256()
            for r in sorted([v for v in shards if v["split"] == split], key=lambda v: v["start"]):
                with (args.interim / "shards" / r["id"] / "data.txt").open("rb") as f:
                    for chunk in iter(lambda: f.read(8 * 2**20), b""):
                        expected_hash.update(chunk)
            if expected_hash.hexdigest() != digest(args.output / f"{split}.txt"):
                raise ValueError("Combined plaintext mismatch")
        for name in ["glyph_bank.npz", "glyph_inventory.json"]:
            if digest(args.output / name) != receipt["model_consumed_sha256"][name]:
                raise ValueError("Glyph assets changed")
        timings["verify_combined"] = time.monotonic() - tick
        family = Counter()
        for r in split_reports:
            family.update(r["family_han"])
            if not args.smoke:
                p = parent_manifest["splits"][r["split"]]
                if (
                    r["han_characters"] != p["han_characters"]
                    or r["effective_targets"] != p["effective_targets"]
                ):
                    raise ValueError("Full corpus counts changed")
        hashes = {
            p.name: digest(p)
            for p in args.output.iterdir()
            if p.is_file()
            and p.name not in {"manifest.json", "verification.json"}
            and p.suffix != ".tmp"
        }
        manifest = {
            "status": "verified_contiguous_excerpts_corpus",
            "identity": identity,
            "mode": "smoke" if args.smoke else "full",
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "splits": {r["split"]: r for r in split_reports},
            "family_han": dict(family),
            "timings": timings,
            "workers": args.workers,
            "seconds": time.monotonic() - started,
            "recommended_context": 1024,
            "controls": "BOS/EOS per run; NEWLINE between original paragraphs",
            "deduplication": "Parent paragraph dedup inherited; no new document-level dedup",
            "continuity_policy": {
                "copy_only_source_indices": sorted(COPY_FILES),
                "oversized_raw_row_limit": 200000,
                "section_regex": SECTION.pattern,
            },
            "limitations": [
                "Contiguous retained excerpts, not complete original documents.",
                "QA pairs, poetry collections and topic-prefixed article excerpts remain isolated.",
                "Titles/filtered lines break runs. Blank formatting lines do not.",
                "Long views are indices, not a change to domain sampling weights.",
            ],
            "parent_manifest": parent_manifest,
            "output_sha256": hashes,
        }
        atomic_json(args.output / "manifest.json", manifest)
        verification = {
            "passed": True,
            "manifest_sha256": digest(args.output / "manifest.json"),
            "scope": (
                "All parents covered once; text/glyph streams match parent slices; "
                "all new edges independently checked against raw lines"
            ),
            "verified_parent_paragraphs": sum(r["verified_parents"] for r in verified),
            "verified_edges": sum(r["verified_edges"] for r in edges),
            "model_consumed_sha256": {
                k: v
                for k, v in hashes.items()
                if k.endswith((".uint16", ".offsets.npy"))
                or k in ["glyph_bank.npz", "glyph_inventory.json"]
            },
        }
        atomic_json(args.output / "verification.json", verification)
        atomic_json(
            args.interim / "status.json",
            {
                "phase": "complete",
                "output": str(args.output),
                "verification_sha256": digest(args.output / "verification.json"),
                "seconds": time.monotonic() - started,
                "splits": manifest["splits"],
            },
        )
        print(
            json.dumps(
                {"seconds": manifest["seconds"], "splits": manifest["splits"]}, ensure_ascii=False
            ),
            flush=True,
        )
    except BaseException as error:
        atomic_json(
            args.interim / "status.json",
            {
                "phase": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "pid": os.getpid(),
            },
        )
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
