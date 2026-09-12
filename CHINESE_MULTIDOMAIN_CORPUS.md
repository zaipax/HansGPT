# Pure-Chinese multi-domain corpus expansion

This collection is downloaded and prepared on the training server. It is a new
corpus version, never a replacement for the dataset of the currently running model.
The requested `/dev/nvidia0` is a character device, not a data directory. After the
authorized checkpoint cleanup, the root filesystem has sufficient disk space.

## Selected sources

The pinned manifest is `configs/datasets/chinese_multidomain_v1.json`: 1,014 files,
approximately 12.32 GiB of source data, all downloaded directly from ModelScope.

| Provider dataset | Selection | Coverage |
|---|---|---|
| YeungNLP/firefly-pretrain-dataset | prose, composition, classical Chinese, poems, CSL abstracts, THUCNews, webText2019zh | Narrative, traditional culture, academic/STEM, news and daily-life Q&A |
| AI-ModelScope/COIG-CQIA | Named finance, medicine, symptoms, agriculture, wikiHow and high-score Zhihu subsets | Finance/economics, health, agriculture and practical questions |
| opencsg/Fineweb-Edu-Chinese-V2.1 | 1,000 files spread across its 4_5 score partition | Education and industry webpages, including law, economics and technology |

All selected files have publisher-provided SHA-256 and byte sizes. File revisions
are pinned where available. Fineweb's API supplies no commit revision for these
objects: download via its branch with a mandatory fixed SHA-256 check, failing if
the bytes change. Do not silently substitute mirrors or new hashes. The selection
is bounded, not a claim to have downloaded the full multi-terabyte datasets.

License metadata and caveats remain in the manifest. In particular, CQIA's mirror
license label and upstream card differ in specificity; Fineweb also cites the
OpenCSG Community License and commercial-use permission requirements. Preserve
per-record copyright metadata. Preparing research data is not a clearance to
redistribute all underlying texts. No raw corpus is committed to Git.

## Pure Chinese and semantic boundaries

- Keep Han characters and the project's explicit Chinese punctuation set only.
  Normalize ASCII punctuation, NFC and simplified Chinese with OpenCC. Digits,
  Latin letters, other scripts, emojis and unsupported internal symbols cause
  rejection; they are not removed from the middle of a sentence.
- Prose is retained as complete original paragraphs, generally at least 20 Han
  characters and at most 4096 characters. Classical text has an 8-Han minimum.
  Require sentence endings and balanced delimiters; reject leading fragments,
  known extraction holes, excessive repetitions and obvious promotional boilerplate.
- Instruction/input/answer fields are accepted or rejected together. Passing pairs
  use explicit Chinese `问：…补充：…答：…` markers. Original adjacent lines may lose
  formatting, but no rejected answer/question fragment is spliced into another.
  Markdown heading/bold markers and punctuation-adjacent whitespace are formatting
  only; their Chinese content is retained. Digits or Latin content still reject
  the entire pair. The current smoke version is `chinese_multidomain_smoke_v4`.
- Template-prompted encyclopedia articles (MBA, medical entries and agriculture)
  use a separate article adapter. Each passing complete paragraph is exported
  with its original Chinese topic and explicit `资料摘录：` marker. Rejected
  paragraphs are never concatenated; these excerpts are not presented as complete
  answers. Genuine Q&A sources retain the all-or-nothing pair policy above.
- Already-simplified text skips OpenCC only if no changed dictionary mapping can
  occur. A conservative trigger set is derived from the installed dictionaries;
  all other text uses the original converter. Equivalence tests cover every key.
- Only font-supported, nonempty, unclipped glyphs from the existing pinned Noto
  font are used. The 32x32 binary rendering and four control glyphs stay compatible.

This structural filtering does not independently fact-check every webpage or
guarantee that every all-Han quotation is Chinese. Source subsets supply domain
labels where available. Fineweb/news use transparent keyword heuristics, not
claims of expert annotation. Samples are stored for manual quality inspection.

## Deduplication, budgets and splits

Exclude exact/punctuation-variant copies of all existing Wikipedia splits. Add
existing validation/test paragraphs to the approximate near-duplicate index.
New paragraphs are globally exact- and near-deduplicated using the existing
eight-anchor, confirmed five-gram Jaccard method. Its candidate search is approximate;
neither semantic leakage nor every duplicate work is guaranteed eliminated.

All paragraphs from an identical original document share a deterministic 98/1/1
split. Store source file, row/document identity, domain method and attribution.
Per-family retained-Han budgets prevent one source from consuming the whole run:
education/web 800M, news 250M, life Q&A 150M, literature 120M, academic 100M,
classical 60M, finance 30M, medicine 20M, agriculture 5M. These are maximum budgets,
not achieved counts; actual retention must be read from the final manifest.

## Server operation

Use clean committed main, `uv sync --frozen` and CPU tests first. The small smoke
downloads only four representative sources, processes up to 200 records per file,
and must pass independent verification before the full job starts.

```bash
bash scripts/run_multidomain_corpus.sh smoke
nice -n 10 ionice -c 2 -n 7 bash scripts/run_multidomain_corpus.sh full
```

Use tmux for the full job. It has four download threads, 24 cleaning processes
(one Arrow thread per worker), no visible GPUs and a 40GiB free-space reserve.
Downloads resume owned `.part` files. Worker tasks are bounded to two queued batches
per worker, with 32 source rows per batch. Outputs are consumed in original source
order, so process scheduling does not change the retained corpus or its splits.
Workers perform normalization, filtering, glyph checks, labels, hashes and shingle
anchors; one parent owns the global deduplication database. SQLite uses an on-demand
8GiB cache, WAL/NORMAL transactions and an equivalent grouped-anchor query that
avoids fetching record bodies before requiring two matching anchors. Differential
tests compare decisions and counters with the original store. NORMAL preserves
database consistency, though a host crash can lose recent commits; cursor and
records roll back together and can be replayed from verified sources.
Processing saves its source cursor and dedup/statistics state in one SQLite
transaction. For interrupted preparation, rerun the preparation command with
the same paths/configuration and `--resume`, then run the verifier separately.
Worker/cache settings can change without changing data semantics. A code upgrade
can also change `--commit-rows` (default 10000, formerly 1000). Larger transactions
reduce repeated index writes; file boundaries still commit. Progress reports show
committed counts, so they update less often. Interruption replays at most the
pending transaction; records and the source cursor always commit together.
A code upgrade
requires `--resume --upgrade-from-script-sha256 <exact previous script hash>`;
all data identity fields must remain equal and the old/new identities are recorded
in `code_history.json`. An exclusive lock prevents concurrent preparers using the
same intermediate store. Already committed data is retained during the parallel
upgrade; only uncommitted rows are replayed.

| Server path | Contents |
|---|---|
| `data/raw/chinese_multidomain_v1/` | Verified original JSONL/Parquet files |
| `data/interim/chinese_multidomain_v1/` | Resume database, identity and progress |
| `data/processed/chinese_multidomain_v1/` | Clean split Parquet/text, glyph bank, uint16 streams, offsets, manifests |

The verifier independently checks every output row's allowed characters, hashes,
exact deduplication, document split, glyph-stream correspondence, rendering and
counts. Only then is `verification.json` published. Each family gets 30 reproducible
reservoir audit samples. Final outputs are marked as a bounded multi-domain corpus;
future training must explicitly configure this new corpus type and identity rather
than bypassing the old Wikipedia-specific full-snapshot checks.
