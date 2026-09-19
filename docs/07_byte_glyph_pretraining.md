# Transformer C 字节预训练与 1.5B 扩展 (Transformer C Byte LM Pretraining)

本专题整合了当前项目的正式预训练主线：基于 Patch 注意力编码器、Llama/Qwen3 语言主干与 128 步字节解码器的端到端预训练。包含单卡基准、空白/逗号循环死锁归因排查、多卡分布式调度、全局有效位置余弦学习率协议、全语料训练与 15 亿参数扩展方案。

---

## 目录

1. [第一部分：Context-1024 默认加速单卡训练基准](#第一部分context-1024-默认加速单卡训练基准)
2. [第二部分：字节模型空白输出排查与差分验证](#第二部分字节模型空白输出排查与差分验证)
3. [第三部分：逗号循环生成归因：优化步数饥饿与条件弱化证实](#第三部分逗号循环生成归因优化步数饥饿与条件弱化证实)
4. [第四部分：Batch-4 稳定预训练显存曲线与吞吐确认](#第四部分batch-4-稳定预训练显存曲线与吞吐确认)
5. [第五部分：4 卡 1000 万/1 亿位置字节模型训练与吞吐基准](#第五部分4-卡-1000-万1-亿位置字节模型训练与吞吐基准)
6. [第六部分：全局有效位置数余弦学习率调度协议](#第六部分全局有效位置数余弦学习率调度协议)
7. [第七部分：4 卡全语料 10.8 亿位置训练记录与断点管理策略](#第七部分4-卡全语料-10.8-亿位置训练记录与断点管理策略)
8. [第八部分：15 亿参数 Qwen3 结构 C 模型规模化扩展方案](#第八部分15-亿参数-qwen3-结构-c-模型规模化扩展方案)

---

## 第一部分：Context-1024 默认加速单卡训练基准
> 原文档来源：`BYTE_C_CTX1024.md`

# C byte Transformer: context 1024, batch 32

Restore the ABC round-2 pure-Transformer architecture (275,349,504 parameters):
64 input patches per glyph, four width-128 input Transformer layers, a
24-layer width-1024 GQA backbone (FFN 2816, 16Q/4KV heads), and a four-layer
width-256 causal byte decoder (8 heads, FFN 768). Inference predicts 128 bytes
per glyph; training uses shifted target bytes with causal attention.

The new default runner is `scripts/train_byte_glyph.py`, with default config
`configs/experiments/hansgpt_attention_c_ctx1024.json`. Acceleration defaults are
defined in `hansgpt_research.byte_training.DEFAULT_ACCELERATION`: xFormers
CUTLASS attention, full-graph compiled byte-head loss, and PyTorch fused AdamW.
The runner requires these settings and has no silent fallback. Compilation
does not include the outer backbone or the autoregressive inference loop.

The current throughput run uses physical GPU7, batch32, context1024, head
chunk256, accumulation1, FP16 and a constant LR of 0.0003. Backbone gradient
checkpointing bounds the fourfold increase in positional batch versus the old
context256 experiment. Glyph encoder activations are shared using CPU pixel
deduplication within the current batch only; gradients aggregate normally.
Target byte conversion on CPU uses row-major, MSB-first packing without glyph
IDs in model inputs. Padded input suffixes have zero loss.

Dataset is the same checksum-pinned packed Chinese document v3 corpus used by
the recent CVAE runs, not the old short-document ABC corpus. All parameters
start randomly initialized. A three-update smoke must pass before the 1000
successful-update run. The benchmark excludes the first ten updates and final
checkpoint I/O, and includes prefetch waits, transfers, backward and optimizer
steps. The final checkpoint, metadata, per-step NLL and throughput are retained.
This is a throughput run, not a completed language-quality evaluation.

The checkpointed three-update smoke passed without AMP overflow, with a final
step time of 4.824 seconds for 32,736 valid targets (about 6,786 targets/s).
Peak allocated/reserved memory was approximately 9.31/10.88 GiB. Its aggregate
smoke rate includes compilation and must not be used as the steady-state rate.
Disabling backbone checkpointing exhausted the 31.74 GiB device before the first
update, so the formal run keeps checkpointing enabled. Eager accelerated byte
loss and gradients match the original teacher-forced joint byte likelihood in
unit tests; the full model smoke exercised the compiled xFormers path.

Server paths use experiment name `hansgpt_byte_c_ctx1024_bsz32_v1_full` under
`artifacts/logs/`, `artifacts/reports/`, and `artifacts/checkpoints/`.
The report is `result.json`; current progress is `status.json`; final weights
are `final.pt`. A separate `_smoke` name prevents mixing smoke and formal weights.
Run under tmux with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=7`,
the existing optional xFormers PYTHONPATH, and the project uv environment.

## Batch 8 without recomputation

At the user's request, a separate GPU7 configuration
`hansgpt_attention_c_ctx1024_bsz8.json` disables backbone checkpointing and
uses batch8 with the same context1024, chunk256 and acceleration defaults.
The three-update smoke passed. The subsequent 70-update run (first ten
excluded from timing) completed with zero AMP overflow:

- 60 measured updates in 70.177 seconds, **6,954.72 effective positions/s**.
- Mean timed update including logging/prefetch overhead: **1.170 seconds**.
- Peak allocated/reserved GPU memory: **17.055 / 17.869 GiB**.
- All 70 updates consumed 569,450 valid positions, including 509,559 Han.

Artifacts use `hansgpt_byte_c_ctx1024_bsz8_no_gc_v1_full`. Final weights are
retained. The original batch32 1000-update run had not started when the user
requested this smaller-batch probe. Its smoke-only throughput is not a matched
long-run baseline for claiming a precise speedup.

## Batch 10, head chunk 2048, without recomputation

GPU7 configuration `hansgpt_attention_c_ctx1024_bsz10_h2048.json` keeps
context1024 and the same acceleration stack, with batch10 and chunk2048.
Both the three-update smoke and 70-update measurement passed with zero AMP
overflow. Training implementation/config commit: `14cba37`.

| Metric | Batch8 / chunk256 | Batch10 / chunk2048 |
| --- | ---: | ---: |
| Measured updates after ten warmups | 60 | 60 |
| Measured seconds | 70.177 | 81.576 |
| Effective prediction positions/s | 6,954.72 | 7,479.61 |
| Mean update seconds | 1.170 | 1.360 |
| Peak allocated GiB | 17.055 | 28.926 |
| Peak reserved GiB | 17.869 | 30.439 |

The combined batch/chunk change improves measured throughput by 7.55%, at
substantially higher memory usage. It is not an isolated chunk-size ablation.
GPU7 device memory sampled during the run was 31,564 MiB with 100% utilization.
The short run does not establish worst-case memory safety across the full corpus.
All 70 updates consumed 711,629 valid positions including 636,807 Han.
Artifacts and final checkpoint use experiment name
`hansgpt_byte_c_ctx1024_bsz10_h2048_no_gc_v1_full`.

---

## 第二部分：字节模型空白输出排查与差分验证
> 原文档来源：`BYTE_BLANK_DIAGNOSIS.md`

# Blank byte-generation diagnosis

The new four-GPU 10M-position checkpoint greedily generates 32 blank grids for
the saved validation prompt. `scripts/diagnose_byte_blank.py` reproduces the
first two outputs exactly, then tests inference and gradients on that checkpoint.
The old model weights were deleted by user request; historical logs, raw arrays
and evaluation reports are still available.

## Findings

Training opportunities differ substantially:

| Run | Successful positions | Updates | Mean valid positions/update |
| --- | ---: | ---: | ---: |
| Old C, final | 100,003,725 | 30,494 | 3,279.5 |
| Old C, first ~10M | 10,008,162 | 3,028 | 3,305.2 |
| New C, final | 10,000,000 | 308 | 32,467.5 |

Old short documents left much of context256 unused. New packed context1024
windows provide nearly full global positional batches. At approximately 300
updates, old training NLL was 0.198973 and new NLL 0.190394 nats/pixel. Old
training NLL subsequently reached 0.029469 at 1000 updates and 0.011769 at 1500.
Data, context and LR schedule differ, so this is evidence consistent with early
optimization, not a controlled proof that additional training alone fixes it.

Four fixed prompts produce entirely blank next glyphs with both native SDPA
and xFormers. Cached and uncached native byte decoding agree exactly. The
native/xFormers hidden-state relative L2 difference is 0.000298. Thus the saved
blank output is reproduced independently of the accelerated attention backend
and cache path on these probes.

Given an all-zero byte prefix, all 128 positions choose zero by argmax for each
of the four prompts. Mean zero-byte probabilities range from 0.922 to 0.939.
Categorical sampling produces 191/141/56/81 black pixels in the four next
glyphs, showing the model is not numerically forced to output zero. This does
not establish that sampled glyphs are valid or readable.

The original forward/NLL and accelerated full compiled backward agree on a
real masked, right-padded mini-batch of the trained full model:

- Original NLL 0.21641992; accelerated NLL 0.21642301.
- Gradient relative L2: encoder 0.000897, backbone 0.000781, byte decoder 0.000463.
- Each module's gradient cosine exceeds 0.9999996.

An audit of the exact entire ordered 10M successful-target prefix found **zero
blank target glyphs**, 7,354 used glyph assets, and 50.98% zero bytes. Large
background regions explain frequent zero bytes, but the target data does not
ask the model to emit entirely blank glyphs.

## Interpretation and next experiment

The evidence favors an early-training greedy zero-prefix fixed point over a
backend/cache defect. Restoring architecture did not restore the old training
state, update budget, short-document data or warmup/cosine schedule. The new
constant-LR run was a throughput probe, not an equivalent quality reproduction.

Before a large quality claim, compare checkpoints at fixed successful update
counts (e.g. 1000, 1500, 3000), retaining raw greedy and sampled generations,
validation NLL and blank-output rates on multiple prompts. Roughly 3000 updates
at the present packing/global batch require about 98M positions. A recovery
entry point is needed to continue from the retained checkpoint. Do not claim
the blank-generation symptom has been fixed: it remains reproducible.

Raw diagnosis: `artifacts/reports/byte_blank_diagnosis/result.json` on the server.

---

## 第三部分：逗号循环生成归因：优化步数饥饿与条件弱化证实
> 原文档来源：`BYTE_COMMA_DIAGNOSIS.md`

# Diagnosis of comma-only byte generation

## Main evidence

The current eight-GPU model is the same C architecture, but equal 100M-position
budgets did not provide equal optimizer update counts:

| | Old C final | Current C final |
| --- | ---: | ---: |
| Successful positions | 100,003,725 | 100,000,000 |
| Updates | 30,494 | 1,546 |
| Mean valid positions/update | 3,279.5 | 64,683.1 |
| Training context | 256 | 1024 |
| Global sequence batch | 32 | 64 |
| Data | Short Wikipedia documents | Packed multidomain Chinese v3 |
| LR horizon | 100M positions | Full 1,089,139,385-position pass |

Most old document batches had much less than 256 valid positions per sequence;
new packing fills windows. World size and context also increased. Thus each
current update consumes about 19.7 times as many targets. GPU parallelism itself
does not reduce updates when global batch is fixed; the global batch changed.

Crucially, the retained old `generation_step00001505.npz` also contains one
repeated bitmap, **pixel-identical to the current comma**. Its validation NLL
was 0.0103249 at 1505 updates, versus current 0.0103424 at 1546 updates. The
validation sets differ, so NLL equality is supporting historical evidence,
not a controlled generalization comparison. Old final NLL was 0.0035446.

## Controlled tests on the current checkpoint

`scripts/diagnose_byte_comma.py` replays saved outputs and runs fixed probes.
Eight original prompts produce the same next glyph with native SDPA and
xFormers; cached and uncached native decoding agree on the checked prompts.

For 32 held-out packed-window targets, vary only how much left context is
provided, ending at the same target:

| Context length | Greedy next output | Paired target NLL/pixel |
| --- | --- | ---: |
| 16 | 32/32 commas | 0.0081423 |
| 64 | 32/32 commas | 0.0081800 |
| 256 | 32/32 commas | 0.0081830 |
| 1024 | 32/32 commas | 0.0081688 |

Longer inference context does not remove the symptom. This does **not** isolate
the causal effect of training at context1024; that requires matched retraining.

At context1024, shuffling real prefix states changes mean target NLL from
0.0081688 to 0.0083760 (+2.54%). Only 19/32 targets worsen. Distinguishing
different real contexts is weak in this probe. Zeroing the state gives 0.436623,
but zero is out of distribution and cannot by itself prove semantic usage.

For eight contexts, the comma also has lowest joint glyph NLL among ten common
diagnostic candidates. This is not solely a local-greedy implementation artifact;
the learned conditional distribution currently favors the comma. Candidate
scoring is diagnostic only and never replaces model feedback with a font lookup.

## Sampling reveals partial glyph learning

Sampling one next glyph for each of 32 contexts yields 25 exact content glyphs
at temperature0.7, 14 at temperature1, and one at temperature1.2. These tiny
samples are exploratory, not an optimized decoding protocol or OCR accuracy.

Free-running sampling at temperature1 on eight fixed prompts produces 256
raw grids: 122 exact content glyphs (47.66%), one other control, and adjacent
repeat rate 0.40%. Original greedy outputs were all commas. Visual inspection
shows many recognizable glyphs alongside malformed glyphs and incoherent text.
Sampling exposes other modes; it does not establish fluent language generation.

## Next experiment

Prioritize optimization/sample efficiency, rather than assuming context1024 is
broken or raising temperature as a cure. Keep context1024 and the full-corpus
LR schedule; compare global batch8 (one sequence per GPU) against global batch64
on the same successful-position prefix. A 10M-position comparison would provide
approximately 1220 versus 154 updates. Track wall time, NLL, raw glyph validity,
blank/comma rates, context-shuffle sensitivity, and both greedy/sampled text.
For a direct context ablation, subsequently hold global valid targets/update
approximately fixed while changing context and adjusting sequence batch.

No training or default decoding settings were changed during this diagnosis.
The greedy symptom remains reproducible. Raw reports and samples are under
`artifacts/reports/byte_comma_diagnosis/`; this evidence does not guarantee that
additional training alone restores the old final quality.

---

## 第四部分：Batch-4 稳定预训练显存曲线与吞吐确认
> 原文档来源：`BYTE_C_B4_MEMORY.md`

# Single-GPU batch4 memory test

GPU7 completed 110 successful updates of the 275,349,504-parameter C byte
Transformer, with batch4, context1024, head chunk2048 and no backbone
recomputation. xFormers CUTLASS, compiled byte head and fused PyTorch AdamW
remain enabled. The full-corpus global LR schedule is used, with the same
1,089,139,385-position horizon and 10,891,394-position warmup as formal training.
This is a memory/throughput probe, not a language-quality experiment.

The three-update smoke passed before the continuous test. Excluding the first
ten updates and final checkpoint I/O, the next 100 updates took 59.648 seconds:

| Metric | Result |
| --- | ---: |
| Valid prediction positions/s | 6,817.22 |
| Mean timed update | 0.5965 seconds |
| Peak PyTorch allocated | 20.818 GiB |
| Peak PyTorch reserved | 21.980 GiB |
| Maximum sampled CUDA device use | 22.366 GiB |
| AMP overflow | 0 |

Device-use samples are collected after synchronized optimizer updates and
include allocator reservations/runtime overhead. Allocated/reserved peaks
cover the full measured training interval. The post-update live allocation
alone is much smaller and must not be interpreted as the training memory need.

All 110 steps consumed 447,279 valid positions including 400,002 Han. The final
weights are retained. Experiment name: `hansgpt_byte_c_single_b4_h2048_memory_v1_full`;
config: `configs/experiments/hansgpt_byte_c_single_b4_memory.json`.
Implementation: `fed3c7a`. Reports/logs/checkpoints use standard artifact paths.

The user's next intended topology is GPU4–7 with batch4 per rank (global16).
This single-GPU test leaves substantial headroom on a 31.74-GiB card, but does
not measure the additional NCCL communication allocation or guarantee the
worst-case batch across the entire corpus. No four-GPU training was launched
as part of this memory-only request.

## Matched batch8 follow-up

GPU7 subsequently ran the same global-schedule memory probe at batch8, with
context1024/head2048/no recomputation unchanged. Configuration:
`configs/experiments/hansgpt_byte_c_single_b8_memory.json`; implementation/config
commit `5df2cba`. Both the three-update smoke and 110-update run passed.

| Metric, after ten warmups | Batch4 | Batch8 |
| --- | ---: | ---: |
| Measured updates | 100 | 100 |
| Measured seconds | 59.648 | 111.816 |
| Effective positions/s | 6,817.22 | 7,274.83 |
| Peak allocated GiB | 20.818 | 26.181 |
| Peak reserved GiB | 21.980 | 27.818 |
| Sampled device-use range GiB | 22.350–22.366 | 28.192–28.204 |
| AMP overflow | 0 | 0 |

Batch8 improves measured position throughput by 6.71%, while device occupancy
increases by about 5.84 GiB. Each update processes approximately twice as many
positions, so equal-position training makes about half as many updates; this
memory benchmark does not determine which batch learns better. Neither row
includes multi-GPU communication buffers.

The batch8 run consumed 894,826 valid positions including 800,850 Han. Final
weights, status and results use `hansgpt_byte_c_single_b8_h2048_memory_v1_full`.

---

## 第五部分：4 卡 1000 万/1 亿位置字节模型训练与吞吐基准
> 原文档来源：`BYTE_C_FOUR_GPU.md`

# Four-GPU C byte-Transformer training

## Completed result

Implementation commit `c5cdab5`. Completed exactly 10,000,000 valid positions
(8,956,249 Han) in 308 successful updates, zero AMP overflow. Total wall time,
including initialization, checkpoint and diagnostics, was 547.54 seconds
(9.13 minutes). All four replicas matched elementwise before checkpointing.

After the first ten updates, 9,674,727 positions took 468.747 seconds:
**20,639.55 effective prediction positions/s aggregate**. Peak allocated memory
was 26.30–26.40 GiB per rank; peak reserved memory was 27.83–28.34 GiB.
This is not directly a scaling ratio against the previous single-GPU probes,
which used either batch8/chunk256 or batch10/chunk2048.

Final fixed-subset validation NLL was 0.186943 nats/pixel (4,261 targets).
The single-prompt greedy diagnostic repeated the same bitmap for all 32
generated grids, with zero exact content-glyph matches. This small budget
establishes training/throughput functionality, not successful text generation.
The final weights and raw diagnostic remain available at the paths below.

This run trains the original pure-Transformer C architecture from scratch on
exactly 10,000,000 successful valid prediction positions using GPUs 4–7.
The 275,349,504 parameters, context1024, head chunk2048 and default acceleration
remain fixed. Per-device batch8 gives global batch32, accumulation1. Backbone
recomputation is disabled. LR is constant at 0.0003, matching the throughput
probes; there is no VAE or KL objective in this architecture.

Config: `configs/experiments/hansgpt_byte_c_four_gpu_10m.json`.
Runner: `scripts/train_byte_multigpu.py` via `uv run torchrun --standalone
--nproc_per_node=4`. Set `CUDA_VISIBLE_DEVICES=4,5,6,7`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, the established xFormers PYTHONPATH,
`NCCL_P2P_DISABLE=1`, and `NCCL_CUMEM_HOST_ENABLE=0`.

Acceleration is xFormers CUTLASS, full-graph compiled teacher-forced byte loss,
and PyTorch fused AdamW. CPU pixel dedup shares glyph encoder work inside each
rank's batch. FP32 gradient buckets synchronize after backward, weighted by
the local fraction of global valid positions. This is synchronous data
parallelism without communication/backward overlap. The corpus is the pinned
packed Chinese document v3 dataset, with disjoint rank shards of one seeded
sortish sequence. Padding and masked document transitions do not count.

The last batch is trimmed to the exact global target budget. All ranks must
have elementwise-identical weights before checkpoint publication. Final
checkpoint includes optimizer, scaler, global cursor and per-rank RNG states.
The initial runner has no resume CLI. Validation NLL is recorded initially and
at the end; final raw byte generation is a single fixed validation prompt
diagnostic, not a broad language-quality assessment.

Throughput excludes initialization and the first ten optimizer updates, final
checkpoint saving, validation and generation. It includes ordinary data
prefetch waits, transfers, backward and optimizer/communication work. Per-rank
peak allocated and reserved memory are recorded. The maximum rank duration
defines aggregate throughput. Small smoke runs do not report a steady-state
rate when fewer than ten updates occur.

Reports, logs and checkpoints use the experiment name
`hansgpt_byte_c_four_gpu_b8_h2048_10m_v1_full` under `artifacts/`.
The checkpoint is `positions_010000000.pt`; final metrics are `complete.json`.
The separate smoke run checks exact milestones and generation, including
unaligned short byte-cache masks required by xFormers CUTLASS.

## 100M-position follow-up

`configs/experiments/hansgpt_byte_c_four_gpu_100m.json` changes only the
experiment name and target budget from the completed 10M run. It starts from
the same random seed on GPUs 4–7, with batch8 per rank, context1024, chunk2048,
no backbone recomputation and constant LR 0.0003. It does not resume the
10M checkpoint. The already-completed 10M run validates this unchanged code
and numerical configuration before the longer run.

Retain checkpoints at exactly 10M, 20M, ..., 100M positions, requiring about
31 GiB in total. Validate at each milestone; final generation remains the
single fixed-prompt diagnostic. The run should perform approximately 3,080
successful updates, versus 308 in the 10M probe. Actual counts are recorded.

Artifacts use `hansgpt_byte_c_four_gpu_b8_h2048_100m_v1_full` under the standard
logs/reports/checkpoints directories. The server tmux session is `byte-four-100m`
on socket `hansgpt-lr-v1`. The current timing field excludes startup/first ten
updates and final saving/evaluation, but includes intermediate milestone I/O
and validation; interpret it accordingly for this multi-checkpoint run.

---

## 第六部分：全局有效位置数余弦学习率调度协议
> 原文档来源：`BYTE_C_EIGHT_GPU_GLOBAL_LR.md`

# Eight-GPU byte Transformer with full-corpus LR scheduling

This supersedes the constant-LR four-GPU run at the user's request. Start from
the original random seed; do not resume weights trained under the superseded
schedule. GPUs 0–7 use per-rank batch8, global batch64, context1024, head2048,
accumulation1, no backbone recomputation. Pure C architecture and default
xFormers/compiled byte head/PyTorch fused AdamW remain unchanged.

## Separate schedule horizon from pilot budget

The checksum-pinned `chinese_document_v3` training split contains exactly
**1,089,139,385 valid prediction positions**. The schedule horizon is one full
training pass, excluding validation/test, padding and masked transitions.
The pilot stops at **100,000,000 global successful positions** (9.18% of that
pass); it uses exactly the LR values a full-pass run would use at those positions.

Chosen full-pass protocol: linear warmup over 1% of training positions,
rounded to 10,891,394; peak LR 0.0003; then cosine decay to 0.00003 at the
end of the full training pass. The peak is not multiplied by GPU count.

| Global successful position | LR |
| --- | ---: |
| 1,000,000 | 0.00002754468 |
| 10,000,000 | 0.00027544683 |
| 10,891,394 | 0.00030000000 |
| 50,000,000 | 0.00029912453 |
| 100,000,000 (pilot stops) | 0.00029547556 |
| 1,089,139,385 (full pass ends) | 0.00003000000 |

`schedule_total_positions` and `warmup_positions` control the schedule;
`target_tokens` controls stopping only. The runner verifies the full horizon
against the actual training dataset. Tests assert that pilot/full-run LR
prefixes agree and that changing world size does not change LR at a fixed
global position count. Legacy constant-LR configs retain their old semantics.

## Execution and artifacts

Config: `configs/experiments/hansgpt_byte_c_eight_gpu_global_lr_100m.json`.
Use `uv run torchrun --standalone --nproc_per_node=8 scripts/train_byte_multigpu.py
--config configs/experiments/hansgpt_byte_c_eight_gpu_global_lr_100m.json`, with
the established xFormers PYTHONPATH, `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `NCCL_P2P_DISABLE=1`, `NCCL_CUMEM_HOST_ENABLE=0`.

Retain ten checkpoints at 10M increments. Sharding, exact position budgets,
weighted post-backward gradient synchronization and checkpoint replica checks
remain as in the four-GPU runner. Eight-rank smoke precedes the formal launch.
The checkpoints retain per-rank RNG and global progress for future recovery;
the current runner still has no resume CLI. A future resume must preserve the
global schedule horizon and restore progress rather than restart warmup.

Artifacts use `hansgpt_byte_c_eight_gpu_b8_h2048_global_lr_100m_v1_full` under
`artifacts/logs`, `artifacts/reports` and `artifacts/checkpoints`. Server tmux
session: `byte-eight-global-100m`, socket `hansgpt-lr-v1`.

---

## 第七部分：4 卡全语料 10.8 亿位置训练记录与断点管理策略
> 原文档来源：`BYTE_C_FULL_CORPUS.md`

# Full-corpus four-GPU byte pretraining

User-authorized full-data run, from random initialization on physical GPUs
4–7. One epoch means exactly **1,089,139,385 valid prediction positions** of
the checksum-pinned Chinese document v3 training split. Validation/test data
are excluded. This is a fresh run, not a continuation of a prior pilot.

## Configuration

- Pure C Transformer, 275,349,504 parameters: 24-layer/1024-width backbone,
  patch Transformer encoder, four-layer autoregressive byte decoder.
- Per-rank batch8, global batch32, accumulation1, context1024, head chunk2048.
- No backbone recomputation; FP16; xFormers CUTLASS, compiled byte head,
  PyTorch fused AdamW. No silent acceleration fallback.
- Global schedule over the full 1,089,139,385 positions: 10,891,394-position
  warmup to 0.0003, followed by cosine decay to 0.00003 at the end.
- Seed20260911; rank-major disjoint shards from the shared sortish order;
  global loss weighting uses actual valid targets. Padding is excluded.
- NCCL P2P disabled and CUMEM host allocation disabled, matching successful
  distributed runs. Gradients synchronize after the chunked backward.

Config: `configs/experiments/hansgpt_byte_c_four_gpu_full_corpus.json`.
Training implementation: `31ae8c3`. Previous successful four/eight-GPU runs
validate the numerical paths; 11 budget/schedule/retention tests passed before
launch, and the full dataset count was checked against both schedule and stop.

## Checkpoint storage

Save and validate every 10M positions. At launch, the filesystem had about
109 GiB free; keeping every full optimizer checkpoint would need over 330 GiB.
Therefore keep the latest three checkpoints plus every 100M-position archive
and the final checkpoint, roughly 42 GiB total, plus transient save space.
Pruning happens only after the new file is atomically saved and hashed, within
this newly created run directory. Existing experiment checkpoints are untouched.
All milestone metrics and retention deletion records remain in training.jsonl.
Checkpoints include model, optimizer, scaler, per-rank RNG and global cursor.
The current runner still has no resume CLI; recovery requires restoring that
state and must retain the global LR horizon and progress.

## Runtime and artifacts

Tmux socket/session: `hansgpt-lr-v1` / `byte-four-full-corpus`.
Run via `uv run torchrun --standalone --nproc_per_node=4
scripts/train_byte_multigpu.py --config
configs/experiments/hansgpt_byte_c_four_gpu_full_corpus.json`, with the established
CUDA visibility, xFormers PYTHONPATH and NCCL environment.

Artifact name: `hansgpt_byte_c_four_gpu_b8_h2048_full_corpus_v1_full`, under
the standard `artifacts/logs`, `artifacts/reports`, `artifacts/checkpoints` roots.
Console: `artifacts/logs/byte_four_full_corpus.console.log`.
Final automatic generation is a one-prompt diagnostic; assess wider raw
generation separately before claiming fluent text. Based on the earlier
20.6k-position/s four-GPU measurement, allow approximately 15–16 hours including
periodic checkpoint and validation overhead; observed throughput takes precedence.

---

## 第八部分：15 亿参数 Qwen3 结构 C 模型规模化扩展方案
> 原文档来源：`QWEN3_DENSE_C_1P5B.md`

# HansGPT C-Qwen1.5B

This experiment scales the current pure Transformer C task model to the dense
Qwen3 1.7B shape while keeping the task objective and data unchanged.

## Outer language transformer

| setting | value |
| --- | ---: |
| hidden size | 2048 |
| decoder layers | 30 |
| query heads / KV heads | 16 / 8 |
| head dimension | 128 |
| SwiGLU intermediate size | 6144 |
| RMSNorm epsilon | 1e-6 |
| RoPE theta | 1,000,000 |
| query/key head RMSNorm | enabled, before RoPE |
| attention and MLP bias | disabled |
| attention pattern | full causal |

The Q/K normalization uses the Hugging Face Qwen3 attention implementation,
but the surrounding model still accepts only `inputs_embeds`. The vocabulary
embedding is removed, and no text token IDs or character IDs enter the model.

## Task-specific modules

The 4x4 patch `AttentionGlyphEncoder` remains width 128, four layers, and four
heads. Each outer position is still one 32x32 binary glyph. The byte decoder
remains four causal layers with inner width 256, eight heads, and FFN width 768.
It emits 128 bytes with 256 classes per byte, which are unpacked MSB-first into
the 32x32 target. Generated binary grids are fed back to the same glyph
encoder; there is no glyph-bank lookup, OCR, or candidate projection.

The model has 1,515,243,008 trainable parameters with the default C decoder.
The training target remains byte categorical cross-entropy, reported as
nats/pixel after division by 1024, over `chinese_document_v3` packed windows.

## GPU benchmark

The four-rank benchmark uses one real context-1024 window per rank, head chunk
128, FP16, gradient checkpointing, xFormers attention, and fused AdamW. The
timed region includes the outer forward/backward, byte-head loss, NCCL gradient
all-reduce, and optimizer update. Input preparation is done before timing.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
NCCL_P2P_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0 \
torchrun --standalone --nproc_per_node=4 \
  scripts/benchmark_qwen3_c_multigpu.py \
  --output artifacts/logs/qwen3_c_1p5b_gpu0_3.json
```

The output records the source commit, verified dataset hashes, model parameter
count, per-rank peak memory, and global valid-targets/second. GPUs 4-7 are not
selected by this command.

---

