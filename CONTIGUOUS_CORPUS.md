# Contiguous retained corpus, version 2

This is a lossless reorganization of verified `chinese_multidomain_v1`, not a new
download or a replacement for its paragraph-level deduplication. Original Unicode
content, glyph assets, source identity and train/validation/test membership remain
fixed. Only boundaries between verified adjacent retained paragraphs can change.

## Continuity rules

Raw source hashes are checked against the pinned source list. Retained sample IDs
and text hashes are indexed in compact read-only NumPy arrays, keyed by source
file, raw row and the original filtered-unit ordinal. Workers replay the original
normalizer, recover raw line coordinates and match every retained paragraph.
An edge is allowed only between consecutive nonblank source lines, retained in
the same original row and split, with adjacent parent indices. Rejected lines and
deduplicated/unrenderable/over-budget paragraphs break continuity. Blank formatting
lines do not. Explicit essay/chapter/question headings begin a new segment.

Poetry anthologies, complete QA records and topic-prefixed encyclopedia excerpts
remain isolated. Raw rows over 200K characters also retain paragraph isolation;
they are not assumed to be single documents. Source excerpts may still contain
quoted or heterogeneous material; continuity is not a claim of complete articles.

## Parallel pipeline

1. Verify the parent and create approximately gigabyte-scale compact source indexes.
2. Index byte ranges of large JSONL files, avoiding a single giant-file worker.
3. Benchmark 8/24/48 spawned workers on identical stratified tasks; compare exact
   recovered edge/coordinate arrays before selecting throughput-oriented parallelism.
4. Replay source ranges without any global SQLite query, dedup rewrite or font render.
5. Export independently bounded parent-row shards in parallel, respecting run edges.
6. Independently recheck every new join against physical raw lines and the original
   full-row cleaner. Verify every exported paragraph and glyph slice against the
   parent, then verify the combined Parquet, offsets, token stream and plaintext.

The parent keeps only bounded task metadata; workers exchange numeric provenance
files rather than huge text payloads. Read-only indexes and token memory maps share
the OS page cache. This uses available memory for useful cached data rather than
allocating memory merely to raise utilization. Progress and stage timings are saved.
Completed replay and shard tasks can resume with exactly matching input/code identity.

## Output and training use

Runs are stored as `[BOS, paragraph, NEWLINE, paragraph, ..., EOS]`. Replacing an
intermediate EOS/BOS pair with NEWLINE preserves the total number of valid next-grid
targets. Every original paragraph is retained once. The same glyph bank contains
the NEWLINE control; no new Unicode glyphs are rendered.

The output includes normal split Parquet/text/uint16/offset files, parent row-range
provenance, context-length distributions, run indices above 512/1024 targets, and
indices of full 1024-target chunks. Short samples are preserved; sampling weights
are not silently changed. Training must explicitly configure the new verified
corpus identity/type rather than passing old Wikipedia-only snapshot guards.

## Run

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 uv run --frozen python scripts/reorganize_corpus.py --smoke --workers 8 --output data/processed/chinese_contiguous_smoke_v2 --interim data/interim/chinese_contiguous_smoke_v2
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 uv run --frozen python scripts/reorganize_corpus.py --benchmark --workers 48 --output data/processed/chinese_contiguous_v2 --interim data/interim/chinese_contiguous_v2
```

Use tmux for the full run. Append `--resume` only for the same unverified version.
Verified versions are immutable. Temporary shards/indexes remain in the interim
directory for review/resume; their paths are never confused with the parent corpus.
