# Document recovery and context-1024 packing

Version 3 addresses two separate issues: destroyed document continuity and wasted
training positions. Real document length and packed-window utilization are reported
separately; packing short texts does not create long-range semantic supervision.

## Long documents

`scripts/prepare_document_corpus.py` replays pinned raw sources with 24 processes.
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
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --frozen python scripts/prepare_document_corpus.py --workers 24 --output data/processed/chinese_document_v3 --interim data/interim/chinese_document_v3
```

Run in tmux. Each attempt requires new directories; finalized receipts bind all
model-consumed files to their hashes. The smoke export includes only one retained
v2 training batch plus restored sample tasks; it must never be used as the full run.
