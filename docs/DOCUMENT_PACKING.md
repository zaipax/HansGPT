# Document recovery and context-1024 packing

Version 3 addresses two separate issues: destroyed document continuity and wasted
training positions. Real document length and packed-window utilization are reported
separately; packing short texts does not create long-range semantic supervision.

## Long documents

`scripts/prepare_document_corpus.py` replays pinned raw sources with configurable
parallelism; the final server run uses 64 processes on 104 available CPU cores.
Read-only held-out indexes are shared through Linux fork/page caching. Large JSONL
files use the existing byte-range task index; Parquet files are independent tasks.

Short Chinese headings and transition lines remain attached to adjacent prose.
Sentence endings and paired delimiters are checked on the complete segment, so
quotes can cross paragraph boundaries. Non-Chinese content, missing glyphs, spam
and extraction artifacts still create hard breaks. Explicit chapter boundaries,
ambiguous anthology/QA sources and rows over 200K characters remain conservative.

Only training rows already anchored in the verified parent are eligible. At least
one recovered segment must reach 1024 positions, and every retained parent paragraph
must remain present before a row is replaced. Otherwise all its v2 runs remain.
Validation/test files are copied unchanged. Restored lines, old-cleaner units and
segments are checked against original and legacy held-out canonical text using
exact matches and the original anchor/five-gram Jaccard >=0.9 criterion, without
the old candidate cap. This is approximate decontamination, not an exhaustive
substring or semantic leakage guarantee.

Candidate documents are deduplicated in deterministic source order. Paragraph
duplicates within distinct training documents may reappear intentionally: deleting
them would again damage context. The corpus is no longer globally paragraph-unique.
Every candidate interval is independently read from the raw source; every exported
text is checked against its glyph stream. Parent datasets are never overwritten.

## Packing and training

`PackedGlyphSequenceDataset` uses the original `[BOS, text, EOS]` stream and windows
of 1024 next-grid positions, with one input overlap at window boundaries. It masks
EOS-to-BOS prediction loss, preserves every other target exactly once, and pads only
the final window. Attention is ordinary causal attention across EOS, with continuous
positions within each window; it is **not** block-diagonal document isolation.

For CVAE training set `training.packing` to `eos_causal` in the configuration's
training section and its `sequence_length` to 1024. Validation/test still use
document-isolated chunks. Set `data_requirements.corpus_type` to
`chinese_document_packed_v3` and pin `data_requirements.manifest_sha256` to the
verified full export. The original Wikipedia snapshot guard remains the default.
No training budget, long-document oversampling or model shape is silently changed.

The exported `*.ctx1024_full_chunks.npy` selectors refer to the ordinary
`GlyphSequenceDataset`, **not** packed-window indices. They support a separate
genuine long-context evaluation or explicitly configured sampling experiment.

## Execution

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --frozen python scripts/prepare_document_corpus.py --smoke --workers 8 --output data/processed/chinese_document_smoke_v3 --interim data/interim/chinese_document_smoke_v3
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --frozen python scripts/prepare_document_corpus.py --workers 64 --output data/processed/chinese_document_v3 --interim data/interim/chinese_document_v3
```

Run in tmux. Each attempt requires new directories; finalized receipts bind all
model-consumed files to their hashes. The smoke export includes only one retained
v2 training row groups plus restored sample tasks; it must never be used as the full run.

The optional `--reuse-candidates` path accepts only the explicitly compatible
producer revision and a matching parent/source/exclusion-database identity.
Candidates are copied into the new attempt and independently rechecked against
raw intervals. Partial Parquet writes are replayed. Cleaning counters then cover
recomputed tasks only; final export counts and verification still cover everything.

## Verified full export

The completed dataset is `data/processed/chinese_document_v3`, produced from
commit `789db3c797ff0518c3ac1effaa68b82abddc1af1`. Its manifest SHA-256 is
`9962a55afc778caf74ef12a24fe017e594c2a683729738e68bce364ab11298a9`.

| Training measure | v2 | v3 |
| --- | ---: | ---: |
| Mean effective isolated ctx-1024 chunk | 142.12 | 143.88 |
| Genuine complete 1024-target document windows | 51,828 | 74,212 |
| Targets in documents at least 1024 long | 66,949,161 | 95,711,660 |
| Han characters | 967,780,009 | 975,335,307 |

The new packed reader gives 1,070,933 training windows, averaging 1017.00 valid
targets (99.316% utilization), and preserves all 1,089,139,385 effective targets.
V2 would also fill windows efficiently with this reader; packing is distinct from
recovering document context. Most original documents remain short. These numbers
do not claim a measured GPU training speedup or a 1024-character mean document.

All 125,050 candidate intervals were independently checked. Export retained
115,527 restored segments from 44,175 source rows; duplicate candidates fell back
to v2. The final 64-worker run reused 1576 pinned candidate tasks and took 2206
seconds, excluding earlier preparation and interrupted runs. A live sample used
63.8 CPU cores. Full stream/text verification, runtime identity validation, actual
packed-reader accounting and a small real-data ctx-1024 CVAE forward/backward check
passed. No new model training was launched.

Detailed server receipts: `artifacts/reports/chinese_document_v3/comparison.json`
and `artifacts/reports/chinese_document_v3/runtime_readiness.json`.
