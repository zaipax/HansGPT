# 模型架构探索：A/B/C 对比与双解码器 (Architecture Explorations & Dual Decoders)

本专题整合了对不同视觉编码器（Patch 注意力 vs CNN）与解码器（像素 vs 字节）的 A/B/C 系统对比，以及语义与空间双解码器联合一次预测、字形 Codec 修复等探索性工作。

---

## 目录

1. [第一部分：A/B/C 三类架构设计与首轮对比实验](#第一部分abc-三类架构设计与首轮对比实验)
2. [第二部分：A/B/C 第二轮训练与自回归字节解码方案确立](#第二部分abc-第二轮训练与自回归字节解码方案确立)
3. [第三部分：语义与空间双 Transformer 解码器联合预测方案](#第三部分语义与空间双-transformer-解码器联合预测方案)
4. [第四部分：双解码器生成输出诊断与语义退化分析](#第四部分双解码器生成输出诊断与语义退化分析)
5. [第五部分：双解码器参数消融与采样有效性测试](#第五部分双解码器参数消融与采样有效性测试)
6. [第六部分：解码器接口漂移与因果破坏诊断](#第六部分解码器接口漂移与因果破坏诊断)
7. [第七部分：空间查询编码器修复与不可变编解码器接口规范](#第七部分空间查询编码器修复与不可变编解码器接口规范)

---

## 第一部分：A/B/C 三类架构设计与首轮对比实验
> 原文档来源：`ATTENTION_ABC_EXPERIMENT.md`

# HansGPT attention A/B/C experiment

This is a matched, from-scratch pilot, not a completed experiment report. All three
models train jointly on the existing full Chinese Wikipedia binary-glyph dataset.
The corpus manifest is pinned in each configuration; data readiness verification
checks all eight consumed files. Page splits allow repeated characters across
splits, so this experiment does not establish unseen-character generalization.

| Variant | Physical GPU | Glyph encoder | Next-grid distribution |
|---|---:|---|---|
| A | 4 | Patch Transformer | Independent Bernoulli pixels |
| B | 5 | Existing CNN | Conditional autoregressive byte Transformer |
| C | 6 | Patch Transformer | Conditional autoregressive byte Transformer |

One outer language position remains one 32×32 binary image. The attention encoder
uses 64 non-overlapping 4×4 patches, linear projection, learned row/column positions,
a shared glyph aggregation token, and four bidirectional blocks of width 128.
There are no convolution modules in A or C. The outer 12-layer, width-768 GQA
Transformer is unchanged. B/C generate 128 row-major, MSB-first bytes using four
causal blocks of width 256, conditioned on the outer prefix state. The byte classes
enumerate all eight-bit patterns, never Unicode IDs. Target bytes are shifted;
only preceding bytes are visible. Generated raw 0/1 images return to the encoder.

## Matched training protocol

- Same seed `20260911`, initial outer weights, corpus, document chunk order,
  256-grid context, batch 4 and accumulation 8. B/C byte heads initialize identically.
- Each run budgets 5,000,000 successful-update grid targets. Padding and AMP skips
  do not count; update boundaries may overshoot. This is a pilot, not v1's 1.5B run.
- AdamW, peak LR 3e-4, 100K-target warmup, cosine decay, FP16/GradScaler.
- Entire models train from random initialization; no v1 weights or optimizer state.
- Small head chunks bound byte-decoder memory while propagating the accumulated
  hidden-state gradient through the backbone once per microbatch.
- Every 1M targets, score exact joint NLL on the same seeded 128 validation chunks.
  This is a fixed subset, not full validation. Save best/latest/final checkpoints.
- Save one fixed validation prompt's 32-grid raw continuation at each validation.
  This is only a diagnostic; final F1/IoU/retrieval and broader generation evaluation
  remain separate work. Byte teacher-forced argmax is never reported as generation.

Equal target budgets do not imply equal compute. Record parameters, GPU memory,
throughput, elapsed time, AMP skips, data hashes, and source commit. The 128-step
inner decoder is expected to be slower; measure before increasing the budget.
The historical v1 is contextual evidence, not a matched fourth control.

## Server execution and monitoring

After the canonical clean-main pull, `uv sync --frozen`, environment check and
tests, run `bash scripts/run_attention_abc.sh smoke hansgpt_abc_r1`. Inspect all
three receipts and logs, then run `bash scripts/run_attention_abc.sh full hansgpt_abc_r1`.
The launcher checks GPU occupancy and current-source smoke receipts, and starts
three independent tmux sessions. Existing runs are never overwritten.

Use `--resume artifacts/checkpoints/<run>/latest.pt` with the same configuration,
run name, device selection and mode to resume a safe update boundary. Source/data
identity mismatches abort. Long-run logs and arrays live under `artifacts/logs/`;
weights live under `artifacts/checkpoints/`.

After confirming full training has started, install an hourly cron invocation of
`uv run --frozen python scripts/check_attention_abc.py --round hansgpt_abc_r1`.
The checker records progress, loss, estimated remaining time, GPU utilization,
missing processes and stale status in `artifacts/logs/hansgpt_abc_r1_monitor/`.
It does not modify training, restart jobs, or send messages. Installation and actual
launch status must be reported separately; this document is not a launch receipt.

---

## 第二部分：A/B/C 第二轮训练与自回归字节解码方案确立
> 原文档来源：`ATTENTION_ABC_R2.md`

# HansGPT ABC round 2: larger models and GPU throughput

Round 1 was interrupted at the user's request to improve GPU utilization and
increase model capacity. Its logs, metadata and last safe checkpoints are retained;
it must not be reported as completed training. Round 2 starts fresh on the same
pinned dataset and keeps the A/B/C input/output ablation unchanged.

## Measured bottleneck

The original batch of 4 with eight accumulation steps, 32-grid output chunks and
gradient checkpointing produced many small GPU operations and unnecessary
recomputation. The original models had 77,200,896 / 81,102,304 / 80,184,832 parameters.

`scripts/benchmark_attention_abc.py` measured identical 32 full-length real training
chunks on each assigned V100S. Each case performs two warmup updates followed by
three measured updates, including transfers, backward and optimizer steps. Loading,
checkpoint writes and validation are excluded, so these are not end-to-end run rates.
All measured updates succeeded. Benchmark source: `b06d609`; raw reports are retained
on the server under `artifacts/logs/abc_perf_{A,B,C}.json`.

| Configuration, cumulative changes | A grids/s | B grids/s | C grids/s |
|---|---:|---:|---:|
| Original small model configuration | 5,413 | 2,608 | 2,207 |
| Actual batch 32, accumulation 1 | 12,246 | 3,106 | 3,083 |
| Output chunk 256 | 18,655 | 5,853 | 5,371 |
| Disable gradient checkpointing | 27,200 | 6,081 | 5,839 |
| Encoder chunk 256 | 32,619 | 6,139 | 6,035 |
| Enlarge shared outer Transformer | 14,796 | 4,814 | 4,906 |

At unchanged model sizes, the combined changes improve this benchmark by roughly
6.03x / 2.35x / 2.73x. Increasing model size consumes more compute but remains faster
than the original configurations. This is a short deterministic capacity/throughput
measurement, not evidence of better language quality or universal speedups.

## Round 2 configuration

| Setting | Value |
|---|---|
| Outer Transformer | 24 layers, hidden 1024, intermediate 2816, 16 query / 4 KV heads |
| A / B / C total parameters | 272,562,176 / 276,758,496 / 275,349,504 |
| Physical GPU assignment | A=4, B=5, C=6 |
| Peak allocated memory in full-length benchmark | A=19.08 GiB, B=20.97 GiB, C=20.88 GiB |
| Actual batch / accumulation / context | 32 / 1 / 256 binary grids |
| Encoder / output chunks | 256 / 256 |
| Gradient checkpointing | Disabled; bounded output chunk backward is retained |
| Budget per model | 100,000,000 successful-update targets, approximately 1.045 corpus passes |
| Initialization / seed | From scratch / 20260911, identical shared backbone weights |
| LR / warmup | 3e-4 cosine schedule / 1,000,000 targets |
| Validation / checkpoint / training log | Every 5M targets / 1,000 updates / 10 updates |

The glyph encoder and byte decoder sizes remain the same as round 1; only the
shared language backbone grows. The three total parameter counts differ because
the ablation modules differ. All round 2 branches share the new optimization,
sampling, budget and validation settings. This approximately one-pass run is an
initial larger-model experiment, not a claim of compute-optimal or converged training.

Original round 1 configuration files remain unchanged. Round 2 uses
`configs/experiments/hansgpt_attention_{a,b,c}_r2.json`. After a clean server update,
locked uv sync, checks and occupancy inspection:

```bash
bash scripts/run_attention_abc.sh smoke hansgpt_abc_r2 r2
# Inspect all three successful smoke receipts before starting full training.
bash scripts/run_attention_abc.sh full hansgpt_abc_r2 r2
```

The hourly cron entry is switched to `--round hansgpt_abc_r2` after all three full
runs start. It records progress and anomalies under
`artifacts/logs/hansgpt_abc_r2_monitor/`. Actual launch and completion are established
by server status/receipts, not by this protocol document.

---

## 第三部分：语义与空间双 Transformer 解码器联合预测方案
> 原文档来源：`DUAL_DECODER_EXPERIMENT.md`

# HansGPT semantic and glyph Transformer decoders

This experiment implements one-pass next-glyph prediction. A shared per-glyph ViT
encodes 64 nonoverlapping 4x4 patches into a glyph embedding. The existing 24-layer,
width-1024 causal GPT models the glyph sequence. All parameters train jointly from
scratch on the same pinned full Chinese Wikipedia corpus as ABC round 2.

## Two decoder stages

The semantic decoder projects GPT states to width 256 and applies two causal
Transformer layers across sequence positions. Its output is combined with the
current GPT state and projected to four width-256 semantic condition vectors.
This retains contextual information without introducing a character vocabulary.

The glyph decoder uses sixteen learned part slots and sixty-four spatial queries
with row/column positions. Three width-256 Transformer decoder layers perform
bidirectional self-attention within the next glyph and cross-attention to the four
semantic conditions. Each spatial query outputs sixteen pixel logits, assembled
by a fixed reshape/permutation into one 32x32 bitmap. Part slots are not assumed to
be identified radicals without a separate interpretability experiment.

Both stages execute once per generated glyph, with fixed network depth. Only the
outer character sequence is autoregressive. The GPT and semantic decoder have
separate KV caches; the spatial decoder has no generation loop or recurrent state.
No CNN, byte decoder, GAN, diffusion, candidate glyph selection, OCR feedback or
target-image features are used. The target image only enters the pixel loss.

## Training and measurement

- Physical GPU 5; FP16 autocast and GradScaler, seed 20260914.
- Same corpus manifest and all eight consumed-file hashes as ABC round 2.
- Context 256 grids; budget 100,000,000 successful-update targets.
- Unweighted pixel BCE, fixed generation threshold 0.5. Spatial self-attention
  improves the architectural bias but does not remove the output distribution's
  conditional pixel independence. Good glyph/semantic results are not guaranteed.
- Three-stage bounded backward: encode/GPT/semantic states, spatial-head chunks,
  accumulated gradients back through the context graph. Chunking batches different
  glyphs; it never serializes the pixels of one output glyph.
- Benchmark 64 fixed full-length real training chunks, two warmups and three timed
  updates per case, including transfer, backward and optimizer updates. Compare
  batch 16/32/48, spatial chunks 128/256/512, fused AdamW and batch64 with recomputation.
  Benchmark memory is a capacity check, not a promise to maximize utilization at
  every instant; initialization, validation and checkpoint I/O remain separate.
- Periodic fixed-subset validation and raw-grid generation diagnostics. Cycle
  detection includes periods up to 16; legal glyphs alone are not language success.

## Selected GPU5 configuration

The implemented model has **277,572,880 parameters**. All timed updates below
completed successfully. Values include GPU transfers/backward/optimizer updates,
but exclude dataset loading, validation and checkpoint I/O. The fixed 64-chunk
update is split into microbatches as indicated, so batch48 also has a trailing
16-chunk microbatch. Measurements establish a practical configuration, not a
universal throughput maximum.

| Actual batch | Spatial chunk | Recompute | Optimizer | Grids/s | Peak allocated GiB |
|---:|---:|---|---|---:|---:|
| 16 | 128 | no | AdamW | 5,195 | 13.62 |
| 32 | 128 | no | AdamW | 5,410 | 21.13 |
| 32 | 256 | no | AdamW | 6,410 | 21.48 |
| 32 | 512 | no | AdamW | 6,774 | 22.29 |
| 32 | 256 | no | fused AdamW | 6,415 | 21.48 |
| 48 | 256 | no | fused AdamW | 6,421 | 27.48 |
| 64 | 256 | yes | fused AdamW | 5,852 | 6.02 |
| **32** | **1024** | **no** | **AdamW** | **7,165** | **23.74** |
| 32 | 2048 | no | AdamW | 7,256 | 26.72 |

Select batch32, accumulation1, spatial chunk1024, encoder chunk256, no recompute
and ordinary AdamW. Chunk2048 is only about 1.3% faster while using roughly 3GiB
more memory; the selected configuration preserves headroom for varying real glyph
diversity. The first seven cases use three timed updates; the last two use five,
all after two warmups. Raw results and exact source/config identities:
[initial](reports/dual_decoder_tuning/initial.json) and
[extended](reports/dual_decoder_tuning/extended.json).

Configuration: `configs/experiments/hansgpt_dual_decoder.json`. Use the canonical
clean-main server pull and `uv sync --frozen` before GPU checks. Run tests and the
benchmark before choosing the final committed configuration, then a fresh smoke
run. Only a successful current-source smoke receipt permits the full launch:

```bash
bash scripts/run_dual_decoder.sh smoke hansgpt_dual_r1
bash scripts/run_dual_decoder.sh full hansgpt_dual_r1
```

Logs/arrays: `artifacts/logs/<run>/`. Checkpoints: `artifacts/checkpoints/<run>/`.
The hourly checker `scripts/check_glyph_run.py --run-name <run> --gpu 5` records
progress, throughput, losses and failed/missing processes, without restarting jobs
or sending messages. Training completion is established by the server receipt,
not by this protocol. After successful full training, the same worker runs full-test
likelihood scoring and the fixed 32-page raw-generation evaluation on GPU5, writing
an independent evaluation receipt under `artifacts/reports/<run>_evaluation/`.

---

## 第四部分：双解码器生成输出诊断与语义退化分析
> 原文档来源：`DUAL_DECODER_DIAGNOSIS.md`

# Dual decoder output diagnosis

## Run and reproducibility

Diagnosed `hansgpt_dual_r1`, best checkpoint SHA-256
`67f91aee5aecf54237cdbfb9159f8fa2dc30b22bfd0b3c916f58781791edadb2`.
The formal weights were never modified. Diagnostic optimization ran on an
in-memory copy on GPU 5; no diagnostic weights were saved.

```bash
CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/diagnose_dual_outputs.py \
  --checkpoint artifacts/checkpoints/hansgpt_dual_r1/best.pt \
  --output artifacts/reports/hansgpt_dual_r1_diagnosis
```

The report uses 256 seeded paragraphs from each of train and test, each with
17 input grids and one next-grid target. These are diagnostic subsets, not
full-test estimates. Sampled paragraphs are not guaranteed to be distinct pages.
Threshold comparisons use identical logits. Gallery metrics include control
tiles: a blank PAD can count as a legal gallery bitmap, so this metric alone
must never be called valid Chinese generation. Constrained gallery scoring is
diagnostic only, not raw image generation or an implemented decoding change.

## Findings

| Control | Result |
|---|---|
| FP32 cached versus uncached, four test prefixes | Maximum probability difference 4.47e-7; no binary differences |
| FP16 versus FP32, same prefixes | Maximum probability difference 0.000947; no binary differences |
| 45 existing data/model tests on GPU | All passed, including patch coordinates, causal behavior, cache behavior and chunked gradient parity |
| Teacher-forced train / test NLL | 0.31544 / 0.32214 nats per pixel |
| Test NLL with wrong paragraph context | 0.34663 |
| Training-frequency per-pixel prior, evaluated on test targets | 0.33581 |
| Test predicted mean black probability / actual black fraction | 17.82% / 18.27% |
| Test black fraction after threshold 0.5 | 4.30% |
| Test fraction of probabilities between 0.1 and 0.9 | 50.12% |

Correct context helps, but its NLL advantage over the context-free pixel prior
is only about 4.1%. This relative NLL reduction is not language accuracy.
The deficit already occurs with true prefixes, before generated-image feedback.

| Threshold | Test foreground F1 | Exact next-target bitmap |
|---|---:|---:|
| 0.25 | 0.5471 | 0.39% |
| 0.40 | 0.4445 | 1.56% |
| 0.50 | 0.2298 | 0.78% |

Lowering the threshold adds strokes but does not recover complete correct glyphs.
These test diagnostics must not be used to tune the production threshold.

## Capacity control

Fine-tuning the complete model on 32 fixed train prefixes and their next-grid
targets, with AdamW at 3e-4 and no weight decay, achieved 100% exact match at
step 75. It temporarily fell to 46.875% at step 100, recovered by step 125,
and remained at 100% through the final step 300 (NLL 0.000349).
This demonstrates memorization capacity and working gradients; it does not
demonstrate generalization, fluent generation, or successful glyph-only pretraining.

## Interpretation and next experiments

No tested cache, precision, pixel ordering, label boundary or gradient defect
explains the failure. The output head has one product-Bernoulli component.
Its spatial attention shares deterministic features, but the modeled pixels
remain conditionally independent. With an uncertain next character, pixelwise
likelihood can favor marginal stroke probabilities rather than a complete glyph.
The well-calibrated mean foreground mass but sparse thresholded images is
consistent with that mechanism; calibration at individual pixels was not measured.

The evidence does not isolate insufficient training from distributional limitations.
Do not claim that more data, a lower threshold, or a larger model necessarily fixes
the problem. First compare additional training and glyph reconstruction warm-up
under controlled budgets, tracking raw exact glyph validity and held-out language
generation alongside NLL. Any stochastic joint-glyph output change is a separate
architecture experiment requiring explicit design, not a threshold patch.

Raw six-prompt generation and the original 32-page evaluation remain under
`artifacts/reports/hansgpt_dual_r1_speaking/` and
`artifacts/reports/hansgpt_dual_r1_evaluation/` respectively. No fluent continuation
was observed in the six custom prompts. The 32-page evaluation produced 29 exact
content glyphs among 2970 generated body grids (0.98%).

---

## 第五部分：双解码器参数消融与采样有效性测试
> 原文档来源：`DUAL_ABLATION_PILOT.md`

# Dual decoder pilot: continuation versus glyph reconstruction

All arms initialize from the same completed r1 best checkpoint and use the pinned
Wikipedia corpus. This is the initial 10-million-target pilot, not the proposed
100-million-target extension. The expanded corpus is not used.

| Arm | GPU | Treatment |
|---|---|---|
| A | 5 | 10M additional valid language targets |
| B | 6 | 2000 glyph reconstruction updates, then the same 10M language targets |
| C | 7 | Language training until its measured update time matches B's two stages |

Language training keeps ctx 256, batch 32, accumulation 1, FP16, the same seed,
sortish ordering, and the existing chunked gradient path. All arms start a fresh
AdamW optimizer: this is a controlled continuation experiment, not an exact resume
of r1 optimizer momentum. LR warms up over 100K targets to 1e-4, then decays to
3e-5 at 10M targets; C holds that floor if it runs beyond 10M. Validation and raw
generation diagnostics run every 1M targets. Initial and final evaluations are
also saved. Best snapshots store weights only; optimizer checkpoints are written
every 5M language targets and at completion to reduce shared-disk contention.
Final comparisons use final weights at the budget boundary, avoiding
different best-checkpoint training budgets.

B uniformly samples training-corpus glyphs, batch 256, LR 3e-4, AdamW with no
weight decay. Only the Patch Transformer, spatial decoder and a temporary
1024-to-1024 mapping train during reconstruction. The main GPT and semantic
decoder are not in the reconstruction path. The temporary mapping is discarded
before language training. Identical glyph aliases are grouped before a seeded
90/10 reconstruction split. Held-out reconstruction glyphs may have been seen
by the original r1 language model; this is not character-disjoint LM generalization.
Reconstruction training/validation accuracy and overflow events are recorded.

C follows B's published cumulative synchronized optimizer-update wall time.
This includes transfers and forward/backward/optimizer work, but excludes data
loading, validation, checkpoint writing and waits. It is a time-matched control,
not an exact FLOP match. C waits when it catches B's current budget, and stops
after B publishes its final training budget, with at most one C update overshoot.
If B fails, C fails explicitly. Different hardware throughput must be considered
when interpreting comparisons. Actual GPU model and software versions are recorded.

## Run

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/run_dual_ablation.py --arm A
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 uv run --frozen python scripts/run_dual_ablation.py --arm B
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 uv run --frozen python scripts/run_dual_ablation.py --arm C
```

Run all three with `--smoke` first. Formal runs require new output directories;
the runner intentionally rejects existing directories rather than silently
restarting or overwriting them. Checkpoints contain optimizer, scaler, RNG and
language cursor state, but automatic phase-aware resume is not implemented.

Outputs use `dual_ablation_pilot_v1_{A,B,C}` under ignored logs, checkpoints and
reports directories. Final reports include full-test NLL and 32 fixed independent
test pages with eight teacher-forced predictions each (F1, exact bitmap match,
retrieval and true-versus-wrong-context NLL), plus the same 32 independent
test-page continuations, with exact raw bitmap transcription, EOS and repetition
metrics and PNGs. The 0.5 threshold stays fixed and gallery corrections never
enter generation. Periodic validation generations are only four fixed chunks;
their legality must not be presented as an exhaustive generation assessment.

`uv run --frozen python scripts/check_dual_ablation.py` writes the current
three-arm comparison for hourly monitoring without changing any training process.

---

## 第六部分：解码器接口漂移与因果破坏诊断
> 原文档来源：`DUAL_INTERFACE_DIAGNOSIS.md`

# Glyph reconstruction interface diagnosis

## Scope

GPU5 replayed the B arm's reconstruction stage from the unchanged r1 best
checkpoint, then extended reconstruction from 2000 to 10000 updates at the same
3e-4 learning rate and batch 256. No language optimization was performed.
The 10707 training and 1189 held-out reconstruction bitmaps use the same seeded,
alias-grouped split as the pilot. Held-out here means reconstruction-held-out,
not unseen by the original language model.

The one-step smoke reproduced the original validation NLL. The full replay
reproduced B's 2000-step reconstruction losses and post-warm-up validation NLL.
Intermediate module checkpoints now preserve the temporary adapter, unlike the
original pilot, allowing future inspection without another replay.

## Counterfactual module swaps

The original GPT and semantic decoder stay fixed. Only the glyph encoder and
spatial decoder are exchanged. All cases use the same 128 validation chunks.

| Components entering the original language path | After 2000 reconstruction updates | After 10000 |
|---|---:|---:|
| Original encoder, original spatial decoder | 0.31773 | 0.31773 |
| Changed encoder, original spatial decoder | 0.39164 | 0.38638 |
| Original encoder, changed spatial decoder | 0.49706 | 2.47526 |
| Changed encoder, changed spatial decoder | 0.61239 | 2.61787 |

Both interfaces cause degradation; the decoder-side mismatch dominates after
longer reconstruction. The decoder learns conditions from a temporary mapping
of the glyph embedding, while language inference provides a different semantic
state. The encoder also changes while its downstream GPT remains fixed. Simply
discarding the adapter and reconnecting the old language path is not a valid
transfer procedure. Retaining an adapter alone would not prove alignment either:
the glyph and semantic condition distributions must explicitly be reconciled.

## Reconstruction capacity and generalization

| Updates | Training exact match | Training mean pixel errors | Held-out exact match | Held-out mean pixel errors | Held-out NLL |
|---|---:|---:|---:|---:|---:|
| 2000 | 7 / 10707 (0.065%) | See raw report | 0 / 1189 | See raw report | 0.17247 |
| 6000 | 691 / 10707 (6.45%) | 5.72 | 0 / 1189 | 65.21 | 0.37883 |
| 10000 | 2235 / 10707 (20.87%) | 2.35 | 0 / 1189 | 62.76 | 0.48882 |

At 10000 updates, foreground F1 is 0.99482 on training glyphs and 0.86088 on
held-out glyphs. Exact match is strict, so this is not evidence that every held-out
image is visually unrecognizable. Nevertheless, the generalization gap is large;
held-out NLL deteriorates as reconstruction becomes more confident. More updates
alone do not establish a successful compositional glyph codec.

The encoder currently compresses 64 patch tokens into one width-128 summary,
projects it to 1024, then derives four conditioning slots. This is an architectural
observation, not proof that 128 dimensions cannot represent the glyphs. Whether
retaining spatial information improves held-out reconstruction requires a control.

## Recommended next experiments

1. Keep the original model intact. Diagnose the glyph codec separately before
   another large language run, using identical glyph splits and meaningful
   held-out F1, Hamming error, and exact-match metrics.
2. Frozen-encoder control: retain the original encoder and train a permanent
   adapter plus spatial decoder. Test whether the original representation contains
   enough stroke information without damaging the GPT input interface.
3. Spatial-information control: compare the single summary against a small set
   of learned latent queries attending to patch tokens. Do not use raw-pixel skip
   connections, character IDs, or gallery projection to make reconstruction appear
   successful. Compare compute budgets and parameter counts explicitly.
4. Once a codec works, freeze it and define one shared latent interface. Train the
   semantic side to supply that interface, then consider limited joint tuning.
   Do not swap newly learned glyph modules into an unchanged old GPT and assume
   compatibility. Changing the encoder requires input alignment or retraining too.
5. Treat next-character ambiguity as a separate problem. Latent MSE can also
   average alternatives; codec reconstruction alone does not solve conditional
   language generation or the factorized pixel-output limitation.

Artifacts: `artifacts/reports/dual_interface_diagnosis/report.json` and the
2000/10000-step module snapshots in that directory. The formal r1 and pilot
checkpoints were not overwritten.

---

## 第七部分：空间查询编码器修复与不可变编解码器接口规范
> 原文档来源：`GLYPH_CODEC_REPAIR.md`

# Persistent glyph codec repair

The repaired glyph module has one explicit `[...,4,256]` latent contract.
`forward(image)` calls `decode(encode(image))`; a future semantic predictor must
supply exactly that latent interface. The fixed arm's mapping is permanent and
included in checkpoints. It is never discarded or bypassed during inference.
This does not imply the old GPT semantic states are already aligned to the codec.

## Two candidates

- **fixed, GPU5:** the original r1 glyph encoder is frozen and stays in evaluation
  mode. A permanent linear/RMSNorm mapping and the spatial decoder learn reconstruction.
  Frozen encoder features can be cached without changing inference semantics.
- **spatial, GPU6:** four learned queries read the 64 patch features through a
  Transformer pooling layer, then project to four width-256 vectors. The patch
  encoder initializes from r1 and trains. The single-summary projection is removed.

Both retain the same spatial decoder architecture, binary loss, batch 256, glyph
split and 1024-dimensional output condition. Their parameter counts and timing
are recorded. Freezing differs, so this is an engineering comparison, not a
single-variable proof about spatial pooling. Neither path uses a raw-pixel skip,
character-ID predictor, gallery projection, CNN, or iterative image generation.

## Stages and gates

1. Memorize 256 reconstruction-training glyphs with at most 4000 updates. Require
   99% exact bitmap reconstruction before expanding; a failed gate ends the run.
2. Expand to the same 10707 training bitmaps as the pilot, at most 16000 updates.
   Use cosine LR from 3e-4 to 3e-5. Stop after five evaluations without validation
   NLL improvement and select the best validation-NLL checkpoint.
3. Split the previous 1189 reconstruction-held-out glyphs into 595 validation and
   594 audit glyphs. Audit is measured only after checkpoint selection. These are
   not globally unseen glyphs: r1 and earlier research inspected the old corpus.
4. Before language alignment, require 95% training exact match, at least 50%
   validation and audit exact match, and mean Hamming error at most four pixels
   on both held-out partitions. These are declared engineering readiness gates,
   not claims of fluent language generation or statistically established thresholds.

The original GPT input encoder and formal checkpoints are never replaced. A
codec that passes still needs explicit semantic-side alignment; latent MSE alone
can average possible next characters and is not presumed to solve generation.

## Execution and artifacts

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/train_glyph_codec.py --arm fixed
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 uv run --frozen python scripts/train_glyph_codec.py --arm spatial
```

Use `--smoke` first; smoke exercises both phases and inference but cannot pass
the language-readiness gate. Outputs use `glyph_codec_repair_v1[_smoke]_<arm>`
under ignored reports, logs and checkpoints directories. Existing directories
are rejected rather than overwritten. Checkpoints include model, optimizer, AMP,
RNG, phase, step and interface metadata; automatic resume is not implemented.

Tests check permanent mapping checkpoint round-trips, reconstruction versus
external-latent decoding equivalence, frozen-encoder immutability, and gradients
through spatial queries. Each formal run also verifies cached training and
uncached inference produce identical binary outputs on audit examples.

## Adaptive spatial-capacity follow-up

`glyph_codec_spatial16.json` separately tests 16 learned query slots on GPU7 after
the four-slot path reduced pixel errors without achieving exact held-out glyphs.
Its interface is explicitly versioned as `glyph_latents_16x256_v1`; it cannot be
substituted for a four-slot semantic output. This increases latent capacity from
1024 to 4096 scalars and attention cost, so results are not an equal-latent-budget
comparison. Decoder weights transfer except for the new slot-position embeddings.
The splits, gates, optimizer and maximum update budgets stay the same.

Interface version denotes a shape/format, not interchangeable learned coordinates.
Any future semantic alignment must pin the exact selected codec checkpoint hash.

## Completed reconstruction results

The four-slot candidates failed readiness: selected fixed-encoder audit mean
Hamming error was 193.38 pixels; four-query audit error was 34.13, both with zero
exact audit glyphs. The sixteen-query candidate passed: at its selected step
15000, train exact match was 10707/10707, validation 594/595, audit 591/594.

Reserved control tiles were excluded from that reconstruction corpus, and EOS
initially had 138 wrong pixels. `adapt_codec_controls.py` freezes the successful
encoder and trains only its decoder with 224 replayed training glyphs plus eight
copies of each of four control tiles per batch, LR 3e-5. After 500 updates all four
controls are exact, train remains 100%, validation is 589/595, audit is 592/594.
The resulting immutable codec SHA-256 is
`5882dfa7b73304b9f6cfb2fdae64eae2cce1209925d7884370a823cec1c76606`.

## Semantic alignment pilot

`fit_codec_semantics.py` starts from r1 and that exact control-ready codec. It
freezes the original GPT input encoder, original GPT backbone, and entire codec.
Only the existing semantic decoder, permanent 1024-to-4096 mapping and per-slot
normalization train. No new encoder silently replaces the old GPT input encoder.

The one-million-target pilot uses the same Wikipedia corpus, ctx 256, batch 32,
FP16, a 50K-target warmup to LR 1e-4 followed by cosine decay to 3e-5. Its loss is
pixel NLL plus 0.05 times in-batch latent contrastive loss at temperature 0.1.
Detached target-codec features are supervision only; bitmap aliases/repetitions
are multiple positives. No vocabulary lookup or gallery correction enters
inference. This adds an alignment objective, so it is not a loss-matched causal
comparison to r1. It is also not a guarantee that next-character ambiguity is solved.

Run a smoke first, then the pilot on GPU5:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/fit_codec_semantics.py --smoke
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/fit_codec_semantics.py
```

Full-test likelihood, paired glyph metrics, wrong-context controls and 32 raw
continuations are saved automatically. The glyph repair is distinct from achieving
fluent language generation; only those downstream measurements assess the latter.

## Semantic pilot outcome and limits

The pilot completed 1,001,324 valid targets (299 updates, zero AMP overflows).
Full-test NLL was 0.33429 versus r1's 0.31883. Paired foreground F1 was 0.15385
and exact next-target bitmap match was zero. Across 32 independent prompts,
4096 body grids contained no exact content glyphs and no EOS termination.
The semantic pilot therefore did not restore readable generation.

A post-training tensor-by-tensor check verified all 274 original input-encoder/GPT
tensors and all 139 codec tensors were unchanged. The failed language result is
not another accidental replacement or update of the frozen glyph interface.
The permanent codec/mapping contracts, readiness gates and reconstruction repair
are implemented and tested; fluent speaking remains unresolved. More alignment
training versus a different conditional output distribution needs a separate
controlled experiment. A successful deterministic glyph codec does not itself
model multiple plausible next characters.

Final artifacts are under `artifacts/reports/codec_semantic_alignment_v1/`;
`frozen_weights_verified.json` records the real-checkpoint immutability check.

---

