# 中文语料与数据处理管线 (Corpus & Data Pipeline)

本专题整合了从原始维基百科/多领域文本下载、严格清洗过滤、32×32 确定性矢量点阵渲染，到长文档连续性恢复与 EOS 因果打包流水线的全流程规范与技术实现。

---

## 目录

1. [第一部分：中文维基下载、过滤与 32x32 点阵渲染初探](#第一部分中文维基下载过滤与-32x32-点阵渲染初探)
2. [第二部分：多领域（文学/文言/网络）纯中文语料构建规范](#第二部分多领域（文学文言网络）纯中文语料构建规范)
3. [第三部分：连续中文语料重构与上下文恢复机制](#第三部分连续中文语料重构与上下文恢复机制)
4. [第四部分：长文档因果流打包（GPT-style Packing）与目标掩码规范](#第四部分长文档因果流打包（gpt-style-packing）与目标掩码规范)

---

## 第一部分：中文维基下载、过滤与 32x32 点阵渲染初探
> 原文档来源：`CHINESE_CORPUS_PILOT.md`

# 中文维基二值字形语料

此 CPU 管线从用户指定的 ModelScope 下载中文维基数据，输出纯汉字和允许标点的段落，以及每个 Token 对应的 32×32 二值格子。支持小规模烟测和完整中文快照预处理；训练输入必须还原为像素，预测目标也是下一格的 1024 个二值像素。

## 固定来源

使用 ModelScope 的 `AI-ModelScope/wikipedia`，中文子集 `20231101.zh`，六个 Parquet 文件合计约 1.72 GB。下载地址、文件大小和 SHA-256 固定在 [来源配置](configs/datasets/modelscope_wikipedia_zh.json)，文件修订为 `af6f00069c4a886b60b7b08d80882efde42c5432`。这是上游 `wikimedia/wikipedia` 的镜像；用户因服务器外网连接问题明确选择 ModelScope，管线不会自动回退到 Hugging Face 或其他镜像。

数据已提取为 `id,url,title,text` 格式，不能再次当作 wikitext 解析。页面 ID 和页面 URL 来自原行；上游没有提供文章修订 ID，因此 `source_revision_id` 保留为空，并单独保存快照日期、来源文件和文件修订。来源配置记录上游仓库修订与授权说明。ModelScope 的 Apache-2.0 标签与其上游说明中的 Wikipedia CC BY-SA/GFDL 不一致，不能据此把百科正文声明为 Apache 授权；导出保留页面归属信息，公开再分发前仍需检查页面适用权利。

字体直接来自 [Noto 官方仓库 Sans2.004 标签的 Regular OTF](https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf)，遵循 SIL OFL 1.1，记录下载后的 SHA-256。

## 服务器运行

全部下载、Python 执行和预处理都在服务器 `/root/HansGPT` 的干净 main 分支完成。按 AGENTS.md 拉取代码并执行 `uv sync --frozen` 后，先运行：

```bash
uv run python scripts/check_environment.py --cpu
uv run pytest -q tests/test_prepare_corpus.py
uv run python -m hansgpt_research.prepare_corpus \
  --shards 2 --max-pages 10000 --max-han 500000 \
  --output data/processed/modelscope_zhwiki_smoke_v1
uv run python scripts/verify_corpus.py data/processed/modelscope_zhwiki_smoke_v1
```

烟测通过后，完整扫描全部六个中文分片，不设置字符或页面截断：

```bash
uv run python -m hansgpt_research.prepare_corpus \
  --shards all --max-pages 0 --max-han 0 \
  --output data/processed/modelscope_zhwiki_full_v1
uv run python scripts/verify_corpus.py data/processed/modelscope_zhwiki_full_v1
```

默认原始文件目录为 `data/raw/hansgpt_modelscope_wikipedia/20231101/`，烟测下载的第三个分片会在完整运行时复用并重新校验。下载器支持 `.part` 断点续传，完成后核对固定大小和 SHA-256 再发布文件；已存在的成品也必须通过校验。ModelScope 下载失败会保留 partial 并报错，不更换来源。完整下载完成后开始正文扫描。manifest 的 `source_scan` 逐文件记录 Parquet 总行数、实际扫描行数与完成状态；完整快照状态必须由全部分片的实际扫描完成证明，不能仅根据命令参数设置。

长任务用 tmux，日志写入 `artifacts/logs/`。已存在的输出目录不会覆盖，失败后重跑应使用新的输出版本；经过校验的 raw 文件仍能复用。源码修改在本地 worktree 完成并推送，服务器只拉取和运行。旧版官方 XML 分片入口仍可通过 `--source wikimedia --snapshot YYYYMMDD --raw ...` 显式使用，不属于默认路径。

## 清洗、去重和拆分

- NFC、显式标点映射后，先拒绝明显不支持的字符，再对可接受段落执行固定 OpenCC `t2s` 并重新检查。避免在最终必然丢弃的大量混合文字段落上运行繁简转换。规则版本与转换前拒绝量单独记录，只保留汉字与允许中文标点。
- 夹杂数字、外文、内部空格或其他符号的段落整段拒绝，不删除中间片段后拼接残句，也不进行未经审核的数字或术语改写。段落首尾空白会去除。
- 默认保留至少 40 个汉字、至多 4096 个字符的段落，拒绝连续十次重复字符、字体不支持、空白或裁切的字形。
- 针对烟测抽检发现的上游提取残缺，`strict_han_v3_artifact_rejection` 整段拒绝空括号、仅标点的括号、显式外语名或学名标签后的空值（如“（学名：）”），以及“面积，位于法国”等紧邻明确地理位置的面积空值和“人口为，”等缺失数值的句式。地理前缀及标签采用代码中的显式名单，避免把所有括号或所有“面积，”一概删除。三类拒绝量分别记录；不猜测缺失数字、恢复外文或拼接剩余句子。
- 全局 SHA-256 精确去重；去标点后汉字序列相同的变体也仅保留第一次出现。
- 近重复候选使用去标点汉字文本的五字片段：选八个最小 CRC32 哈希，至少命中两个且长度比不低于约 0.9 才比较；最多检查按共享锚点数排序的前一百条候选，确认真实字符串片段集合 Jaccard ≥ 0.9 后删除后出现的段落。CRC32 仅生成候选，不能单独决定重复。候选超限次数写入统计。
- 近重复检查在全局进行，随后来源页面固定在一个拆分中，以页面 ID 和种子哈希约 98%/1%/1% 分配 train/validation/test；实际数量写入 manifest。
- 固定字号 26、基线 27、32×32 画布、阈值 128。检测空白和裁切，记录像素碰撞；输出仅为 0/1。

近重复候选召回是近似的，不能承诺清除全部转载、改写或语义重复。字符检查也不能代替语言识别：来源为中文维基，仍可能保留全汉字的日语引文等少量情况。烟测还显示，上游提取过程可能已经删去外文、公式或数值，留下仍通过纯中文字符检查的残句；新增规则只处理明确识别的空值模式，不能证明所有句子完整。实验报告应披露各类残缺段落拒绝量，并在完整训练前重新人工查看固定种子的抽检正文。严格过滤会系统性损失数字密集、科技外文术语多的段落。快照较旧且领域集中，不代表广域现代中文。没有完成外部评测集去污染或全面人工质量审核。

正文按 Parquet batch 流式读取，保留记录和近重复索引暂存 SQLite，不把完整原始语料装进内存。时间成本以总字符数、每段锚点的数据库索引查询和候选相似度比较为主；相似模板极多的部分会慢一些。SQLite 缓存设为约 64 MiB，另需字符缓存、源 batch、输出偏移和页面集合内存。磁盘需同时容纳原始 Parquet、临时记录索引和最终导出；预处理完成后删除 SQLite 暂存文件。

## 产物与验证

| 文件 | 用途 |
|---|---|
| `train/validation/test.parquet` | 中文正文、页面归属、文件修订、哈希和拆分元数据 |
| `train/validation/test.txt` | 可阅读的正文；空行仅作导出分隔 |
| `glyph_bank.npz` | `uint8[N,32,32]`，元素仅为 0 或 1 |
| `glyph_inventory.json` | 字形资产索引、控制格子、像素碰撞分组 |
| `*.uint16`、`*.offsets.npy` | 每段 BOS、正文格子、EOS 序列，little-endian 索引 |
| `manifest.json` | Git 提交、来源及字体校验和、规则、版本、统计和输出校验和 |
| `verification.json` | 独立读取产物后的文本、二值像素、索引、来源、拆分和去重验证 |
| `audit_samples.json` | 固定种子选取的验证和测试段落正文、来源与样本 ID，供人工抽检 |

验证器按 batch 读 Parquet，以 memmap 读序列，校验输出 SHA-256、纯中文字符、二值像素、逐段索引到文本的一致性、页面不跨集合、精确及标点变体不重复、文件修订与 manifest 一致。它独立复算每段汉字数、各集合页面数、段落数、Token 数与全部过滤计数，并对照原始 Parquet metadata 检查来源扫描证明。

字形检查要求库存索引唯一连续、控制地址及控制二值格完全匹配；根据记录的字体 SHA-256 和渲染参数，重新渲染全部库存字符，逐像素比对字形库，重新计算碰撞分组。原始字体因此必须仍保留在服务器 raw 目录。

独立近重复抽检默认以固定种子各选取 32 个 validation/test 段落，将每个样本与全部训练段落做真实五字片段集合 Jaccard 比较。实现使用样本片段倒排索引，保留任何存在共享片段的候选，不复用预处理的八锚点或一百候选限制；因此对这批样本的比较是完整的，但不能扩展成对全部留出集或语义污染的保证。`verification.json` 保存最大相似度、达到 0.9 的匹配数和最接近训练样本的来源；出现匹配必须在启动完整训练前检查。`--audit-per-split N` 可增加抽检数量。

验证开始先撤销旧的成功凭据，全部检查结束后原子写入 `verification.json`，绑定当前 manifest SHA-256 和八个模型消费文件的 SHA-256。中断或失败不能留下有效的旧成功凭据。训练入口必须核对该凭据与当前数据，完整实验还必须核对六分片的实际扫描完成证明；不得使用未完成的导出目录。

索引只用于读取图像资产，不作为可学习的字符 embedding 或字符分类目标。PAD 和 NEWLINE 控制图虽未出现在正文序列中，也保留在字形库，统计时应与内容字符分开。页面不重叠的语言建模拆分允许相同汉字在训练和测试中出现，不能据此声称未见汉字的组合泛化；字形碰撞也不能作为可区分身份处理。

---

## 第二部分：多领域（文学/文言/网络）纯中文语料构建规范
> 原文档来源：`CHINESE_MULTIDOMAIN_CORPUS.md`

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

Worker batches stop at 32 rows or 65536 input characters. A larger document runs
alone as one task, preserving its identity and paragraphs. This avoids assigning
dozens of multi-million-character poetry collections to a single worker.

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

---

## 第三部分：连续中文语料重构与上下文恢复机制
> 原文档来源：`CONTIGUOUS_CORPUS.md`

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

---

## 第四部分：长文档因果流打包（GPT-style Packing）与目标掩码规范
> 原文档来源：`DOCUMENT_PACKING.md`

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

---

