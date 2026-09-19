# 条件字形变分自编码器体系 (Conditional Glyph VAE Research)

本专题整合了条件字形变分自编码器（CVAE）的全部研究档案，包括潜在空间构建、24 层主干可读性评估、单卡/多卡显存与吞吐 Smoke、8 卡并发网格搜索、以及 1 亿位置训练关于后验重构与先验生成差距的最终结论。

---

## 目录

1. [第一部分：条件字形变分自编码器 (CVAE) 架构设计提案](#第一部分条件字形变分自编码器-cvae-架构设计提案)
2. [第二部分：24 层 CVAE 主干训练与多级字形可读性评估](#第二部分24-层-cvae-主干训练与多级字形可读性评估)
3. [第三部分：CVAE 全模型显存压力与吞吐 Smoke 验证](#第三部分cvae-全模型显存压力与吞吐-smoke-验证)
4. [第四部分：GPU 利用率低与梯度反传瓶颈诊断](#第四部分gpu-利用率低与梯度反传瓶颈诊断)
5. [第五部分：样本顺序匹配下的 Batch 规模对比分析](#第五部分样本顺序匹配下的-batch-规模对比分析)
6. [第六部分：CVAE 头部优化与局部图编译加速验证](#第六部分cvae-头部优化与局部图编译加速验证)
7. [第七部分：10M 位置 CVAE 留出集重构与先验生成差距评估](#第七部分10m-位置-cvae-留出集重构与先验生成差距评估)
8. [第八部分：8 卡并发 CVAE 学习率网格搜索报告](#第八部分8-卡并发-cvae-学习率网格搜索报告)
9. [第九部分：4 卡同步 CVAE 吞吐基准与扩展极限测试](#第九部分4-卡同步-cvae-吞吐基准与扩展极限测试)
10. [第十部分：4 卡 1 亿位置 CVAE 训练结论：后验重构与先验生成的本质差异](#第十部分4-卡-1-亿位置-cvae-训练结论后验重构与先验生成的本质差异)

---

## 第一部分：条件字形变分自编码器 (CVAE) 架构设计提案
> 原文档来源：`CONDITIONAL_VAE_EXPERIMENT.md`

# Conditional VAE, one-million-Han experiment

All weights initialize randomly, including glyph encoders and the spatial decoder.
No previous language, codec, or toy checkpoint is loaded. The pinned original
ModelScope Wikipedia corpus supplies 32x32 binary targets and font assets only.

## Architecture and probability model

Each glyph uses a patch Transformer and 16 learned pooling queries. A six-layer
width-512 causal GPT and two-layer semantic Transformer encode its prefix.
A prior Transformer produces the mean/log-variance of one 64-dimensional diagonal
Gaussian per next glyph. A separate posterior Transformer sees prefix features
and the actual target glyph only during training and likelihood evaluation.
Its reparameterized sample, together with prefix features, conditions the
three-layer width-256 spatial Transformer, which outputs all 1024 pixel logits
in one pass. All core networks are Transformers plus ordinary linear/norm heads.

Training minimizes `(sum pixel BCE + beta * KL(q||p)) / 1024`. KL is computed
analytically in nats per whole glyph; beta grows to one over 500K successful Han
targets. Log-variance is bounded to [-6,2] for numerical stability. No fixed glyph
vocabulary, image retrieval, pixelwise random sampling, CNN or refinement loop
is used for inference. Shared-z marginalization permits pixel dependence; a
diagonal Gaussian prior and posterior collapse can still limit performance.

## Budget and optimizer

Train exactly 1,000,000 successful Han next-glyph targets, not one million distinct
characters. Punctuation and EOS also receive loss but are counted separately.
The last batch loss mask ends at the millionth Han target. AMP-skipped updates
do not consume the successful-target budget; attempted targets are recorded.

Context 256, batch 16, FP16, AdamW beta=(0.9,0.95), weight decay 0.1, gradient norm
cap 1, peak LR 3e-4, 50K-Han warmup, cosine decay to 3e-5, seed 20260915. All model
parameters train. The actual parameter count is saved in run metadata.

## Tests and evaluation

1. Unit tests cover Gaussian KL, reparameterized gradients, absence of posterior
   target access in generation, one decoder call per glyph, causal/cache behavior,
   exact Han budget masking, and chunked/shared-encoder gradient equivalence.
2. An independent tiny randomly initialized CVAE trains on one identical prefix
   with two next glyphs, tea/water. It reports 512 prior draws versus an analytic
   pixel-mean baseline. Toy weights are discarded, never transferred.
3. A formal-architecture smoke exercises training, validation, checkpoint and
   prior-only generation before the million-Han experiment.
4. Validation separately reports posterior reconstruction, actual KL, and negative
   ELBO at beta=1. None is mislabeled as exact marginal likelihood or fluent speech.
5. Final evaluation uses 32 fixed independent test-page prompts and up to 128
   prior-generated glyphs. Gallery lookup is exact-match scoring only. A 64-sample
   IWAE likelihood estimate on 256 fixed positions and shuffled-posterior-z
   diagnostics measure latent behavior. This is not an exact full-test NLL.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/train_conditional_vae.py --mode toy
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/train_conditional_vae.py --mode smoke
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/train_conditional_vae.py --mode full
```

Outputs use `conditional_vae_1m_v1_<mode>` under ignored reports, logs and checkpoints
directories. Existing directories are rejected; automatic resume is not implemented.
The small budget tests the mechanism and does not imply a from-scratch model will
already acquire fluent Chinese.

## Observed results

The independent 452,976-parameter toy generated exact tea/water bitmaps in
145/512 and 214/512 prior draws, respectively (70.12% combined). The remaining
153 draws were not exact matches to either target. Posterior sample reconstruction
was 100%. This demonstrates both modes but falls short of the declared 80% toy
criterion; the corpus run proceeded as the user-requested exploratory experiment,
not as a claim that the toy readiness criterion passed. Each toy image after the
prefix is an independent next-glyph draw, not an autoregressive tea/water sentence.

The formal model has 29,411,984 parameters, all trainable and randomly initialized.
It completed exactly 1,000,000 successful Han targets, 1,121,219 total targets,
677 optimizer updates, and zero AMP overflows. Training plus its initial automatic
evaluation took about 242 seconds after setup on GPU5. Final checkpoint SHA-256:
`56f257365e393cf8915975f0facee19af4e18f3efcc50df9c062e1560332aa93`.

| Measurement | Result |
|---|---:|
| Validation posterior reconstruction BCE/pixel, 6019 targets | 0.27734 |
| Validation posterior-sample foreground F1 | 0.48401 |
| Validation posterior-sample exact bitmap rate | 0.40% |
| Validation actual KL, nats/glyph | 7.20336 |
| Validation negative ELBO/pixel | 0.28438 |
| Test IWAE-64 estimate/pixel, 256 fixed positions | 0.28320 |
| Test single-prior-draw foreground F1 / IoU | 0.28902 / 0.16892 |
| Test single-prior-draw exact target / nearest-glyph top1 | 0% / 0% |
| Raw prior continuations: exact content glyphs | 0 / 4096 |
| Raw prior continuations: EOS termination | 0 / 32 |

For the same test context, 64 prior draws produced 64 different bitmaps, none an
exact content glyph. Shuffling posterior-mean latents raised paired reconstruction
BCE from 0.27847 to 0.45330. Latents affect the output, but variation alone is not
valid glyph generation. Even answer-conditioned reconstruction is still weak on
the corpus. This experiment has not solved coherent glyph generation or fluent
Chinese, and does not establish that longer training necessarily would.

The final evaluator was rerun read-only on the same verified checkpoint to add
Dice, IoU and retrieval fields; training weights were not changed. Reports remain
under `artifacts/reports/conditional_vae_1m_v1_full/`, with raw PNGs and NPZs.

---

## 第二部分：24 层 CVAE 主干训练与多级字形可读性评估
> 原文档来源：`CVAE_24L_EXPERIMENT.md`

# Original-scale CVAE: 24 layers, width 1024

This experiment restores the original GPT backbone size: 24 layers, hidden width
1024, 16 query heads, four KV heads, FFN width 2816. It initializes all weights
randomly and trains exactly one million successful Han next-glyph targets.
Punctuation and EOS receive loss and are counted separately. No earlier language,
codec or toy weights are transferred.

The remaining CVAE settings match the previous small run: two-layer width-128
patch encoder with 16 queries, two-layer width-256 semantic decoder, one-layer
prior/posterior Transformers, one 64-dimensional shared Gaussian latent per glyph,
and a three-layer width-256 spatial decoder. Context 256, batch 16, seed 20260915,
AdamW peak LR 3e-4, 50K-Han LR warmup and 500K-Han KL warmup are unchanged.
Parameter counts and peak allocation are recorded from the actual model.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/train_conditional_vae.py --config configs/experiments/conditional_vae_24l_1m.json --mode smoke
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/train_conditional_vae.py --config configs/experiments/conditional_vae_24l_1m.json --mode full
```

## Evaluation and readability

The model generates only raw bitmaps. Evaluation never replaces a generated grid
with a gallery glyph before feeding it back into the model.

- Exact bitmap match remains a fidelity metric, not the sole readability judgment.
- Hamming proximity rates at 1/2/4/8/16/32 pixels show near matches separately.
- Best foreground F1/IoU against distinct content bitmaps and the runner-up margin
  describe similarity and ambiguity. Alias-identical references are grouped.
- Blank and control outputs are counted explicitly and never treated as readable
  content merely because they are close to sparse punctuation.
- Nearest-font transcriptions are labeled diagnostic guesses, not raw model text
  or ground-truth OCR. Threshold rates are similarity rates, not human readability.
- Output-only review sheets show the first 32 body grids of all 32 test prompts.
  Review raw pixels before using prompts, then inspect full 128-grid continuations
  for semantic coherence, repetition and ending behavior. Visual review records
  distinguish mostly identifiable, partly identifiable, mostly unclear and empty.
- Posterior reconstruction and prior-only generation remain separate. ELBO and
  IWAE estimates retain their approximate-likelihood labels.

Outputs use `conditional_vae_24l_1m_v1_<mode>` under ignored logs, reports and
checkpoints. The previous six-layer run is retained for comparison with the same
similarity evaluator. This is one seed per architecture, not a statistical claim
that scale alone causes any observed difference.

## Completed run

The instantiated model has 285,309,584 parameters, all trainable. The actual run
completed exactly 1,000,000 Han targets, 1,121,219 total targets and 677 successful
updates, with zero AMP overflows. Training and automatic evaluation took about
348 seconds after setup on GPU5. A discarded random-init stress model separately
passed a full 16x256 update with 4096 distinct input glyphs, peak allocation
13.25 GiB. Neither stress nor smoke weights transferred to the formal model.

Training hyperparameters, non-backbone architecture and all 32 evaluation prompts
were checked to match the small CVAE experiment. Final checkpoint SHA-256:
`e293a1bad6c47fbcf482bab1c6129f0885eb917f06a4c56f1642038cadbd1c1c`.

| Generated-body metric, denominator 4096 | 6-layer / width-512 CVAE | 24-layer / width-1024 CVAE |
|---|---:|---:|
| Exact content glyph match | 0 | 0 |
| Within 8 pixels of a content glyph | 0 | 0 |
| Within 16 pixels | 12 (0.29%) | 31 (0.76%) |
| Best foreground F1 >=0.7 | 650 (15.87%) | 644 (15.72%) |
| Best foreground F1 >=0.8 | 46 (1.12%) | 44 (1.07%) |
| Best foreground F1 >=0.9 | 0 | 0 |
| Blank/PAD body grids | 23 | 104 |
| EOS termination within 128 steps | 0/32 | 0/32 |

Hamming proximity improved slightly while foreground similarity did not show a
clear improvement. These are font-similarity statistics, not measured human OCR
accuracy. The larger model's test-position IWAE-64 estimate was 0.28537 nats/pixel;
prior single-draw target F1 was 0.26822 and IoU 0.15488. These likelihood and target
metrics do not replace raw-output readability assessment.

## Visual audit

The assistant inspected output-only first-32-grid sheets for all 32 prompts, then
all 32 full continuations with their prompts. This is assistant visual inspection,
not independent human annotation, and no numerical human character-accuracy rate
is asserted. Isolated simple shapes are recognizable or plausibly guessable, but
most outputs remain fragmented horizontal/vertical strokes and boxes. No reliably
transcribable coherent continuation was observed. Where glyphs cannot be read
reliably, detailed semantic quality is marked not assessable rather than guessed.

This distinction matters: zero exact matches does not mean every individual
glyph is visually unrecognizable. Nevertheless, this one-million-Han trial did
not produce readable continuous Chinese after restoring backbone scale.

The report directory includes `glyph_similarity.json`, explicitly uncertain
`nearest_font_diagnostic.txt`, raw review sheets, and `assistant_visual_review.json`.
`nearest_font_examples.png` shows the selected top 16 similarities with raw,
reference and difference rows; these selected examples are not average output.
The combined report is `artifacts/reports/cvae_1m_scale_comparison.json`.

---

## 第三部分：CVAE 全模型显存压力与吞吐 Smoke 验证
> 原文档来源：`CVAE_GPU7_SMOKE.md`

# GPU 7 context-1024 memory smoke

All seven probes passed on a Tesla V100S 32GB, using the verified
`chinese_document_v3` corpus with EOS causal packing. The model has 24 layers,
width 1024 and 285,309,584 trainable parameters. Each process starts from random
weights and performs full forward/backward and AdamW updates with FP16 AMP,
beta=1 and no gradient checkpointing. No long training run or checkpoint is created.

| Batch | Head chunk | Peak allocated GiB | Peak reserved GiB | Targets/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 7.74 | 7.98 | 2220 |
| 2 | 128 | 10.82 | 11.28 | 2505 |
| 4 | 128 | 16.86 | 17.45 | 2612 |
| 8 | 128 | 29.10 | 30.34 | 2329 |
| 4 | 256 | 17.36 | 18.33 | 3683 |
| 6 | 256 | 23.38 | 24.28 | 3835 |
| 8 | 256 | 29.32 | 30.44 | 3892 |

Batch 8 with head chunk 256 is a viable full-training candidate after the follow-up
probe. Batch 6 / chunk 256 leaves more memory headroom. Their measured throughput
difference is only about 1.5%, too small to claim a reliable speed advantage from
these short runs. Comparing batch 8 / chunk 128 against batch 6 / chunk 256
confounds batch size with decoder chunk size and does not establish that batch 8
is slower. There is no mandatory fixed memory-reserve percentage.

The first four cases ran four successful steps each; batch 4/6 with chunk 256 ran
eight, and the follow-up batch 8 / chunk 256 ran twelve. All 44 updates succeeded
with finite gradient norms and no AMP skips. The first
successful step in each process is excluded from throughput; CPU sample loading,
validation and checkpoint I/O are not timed. Memory includes optimizer state and
steady training steps. PyTorch reserved memory includes its cache but excludes
some driver/context allocations. These short measurements are not a full-epoch
throughput or long-run stability guarantee. GPU 7 was released after testing.

## Reproduce

Run from clean committed source on the training server, in tmux, using a new
output directory:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 uv run --frozen python scripts/smoke_conditional_vae_memory.py --batch-size 8 --head-chunk-size 256 --steps 12 --output artifacts/reports/cvae_gpu7_ctx1024_repeat/b8_h256
```

Probe source commit: `131ac2e`; follow-up commit: `d91ab40` (documentation only;
same probe/model code). Full configuration, corpus identity, per-step
losses, gradient norms, memory and versions are recorded under
`artifacts/reports/cvae_gpu7_ctx1024_v1/`; `summary.json` combines the seven cases.

---

## 第四部分：GPU 利用率低与梯度反传瓶颈诊断
> 原文档来源：`GPU7_UTILIZATION_DIAGNOSIS.md`

# GPU 7 utilization investigation

Investigated the running 10-million-Han CVAE experiment on 2026-09-13 (Asia/Shanghai),
without stopping training or changing model/optimizer settings. Source commit:
`e447cea`; batch 8, context 1024, head chunk 256, FP16 AMP.

## Observation and hypotheses

A 45-second NVIDIA dmon trace reproduced alternating utilization: 76–100%, mean
92.11%. The corresponding process sample remained in training and advanced 21
steps / 152,668 Han in 44.05 seconds. This confirms visible fluctuation, not an
OOM or stalled training loop.

Ranked explanations were host/kernel scheduling and synchronization in the chunked
head, DataLoader starvation, periodic validation/checkpointing, and GPU memory
migration. The live trace and a 20-second py-spy sample distinguish these without
introducing another GPU workload.

## Evidence

- Framebuffer use stayed at 31,622 MiB across all 45 dmon samples.
- Training-process swap remained zero; major page faults and host swap-in/out
  counters did not increase. RSS stayed near 3.8 GiB rather than growing.
- Mean PCIe receive/transmit rates were 68.33 / 9.69 MB/s. The one 1018 MB/s receive
  burst coincided with 99% GPU utilization, not the utilization troughs.
- py-spy collected 1066 samples, including 570 with training Python stacks.
  No training stack was waiting in DataLoader. Samples included 72 in attention
  mask checks, 16 in scalar finite-loss checks, and 15 in loss-statistic reads.
  These are sampled Python stacks, not CUDA kernel-duration measurements.
- The head processes approximately 32 chunks per batch. Its finite-loss test and
  two `float(cuda_tensor)` statistics reads introduce repeated host/device
  synchronization, alongside many smaller Transformer operations. The backbone
  and head therefore need not have the same GPU utilization.
- A later interval advanced 9 steps at 3818 effective targets/s and 3408 Han/s,
  close to the 3892 targets/s short smoke. Its final status had just entered
  validation; this is not a pure GPU-kernel benchmark.
- No CUDA OOM appeared; AMP skipped-update count remained zero at 2,010,650 Han.

## Conclusion and limits

The evidence does not support GPU tensors spilling into system RAM. This code
uses ordinary PyTorch CUDA allocation and has no CPU/offload or managed-memory
configuration; exhausted ordinary CUDA allocations normally raise OOM. CPU RAM
also holds memory-mapped corpus pages and DataLoader buffers, independently of GPU
memory. Unified-memory page migrations were not directly traced with CUPTI.

The observed behavior is consistent with alternating backbone/head computation
and CPU/GPU synchronization. Average utilization and measured throughput remain
healthy. Periodic validation and checkpoint writes can additionally interrupt
training activity. No training change or regression test is warranted solely to
make the utilization graph flat; no allocation failure was reproduced.

A future controlled optimization can aggregate detached loss statistics on the
GPU and read them once per step, then compare identical batches and gradients.
It should preserve finite-gradient safeguards. No such change was hot-patched into
this experiment, and no temporary debug logging was added to its training loop.

Server evidence is under `artifacts/logs/gpu7_util_diagnosis.*`: dmon, process
samples, raw Python stacks and throughput measurements. No extra GPU job or
background diagnostic sampler remains running.

---

## 第五部分：样本顺序匹配下的 Batch 规模对比分析
> 原文档来源：`CVAE_BATCH_COMPARISON.md`

# Batch 6 versus batch 8 at ten million Han

GPU 7 continues its existing `conditional_vae_24l_10m_ctx1024_v1_full` run
without restart or parameter changes. The independent GPU 6 control is
`conditional_vae_24l_10m_ctx1024_bsz6_v1_full`, starting from random weights.

Both use the same 285,309,584-parameter CVAE, initialization seed 20260915,
verified document-v3 corpus, context 1024, head chunk 256, FP16, AdamW,
learning-rate schedule and KL warmup. Validation batch size remains 8 in both
runs; only the training batch size changes from 8 to 6. Each run stops at exactly
10,000,000 successful Han predictions and runs identical final evaluations.

## Matching training content

The ordinary sortish sampler changes its flat window order when batch size
changes. The control therefore sets `sampler_reference_batch_size=8`, retaining
the original GPU 7 order and regrouping it into batches of 6 in DataLoader.
Defaults preserve the behavior of existing configurations. The existing GPU 7
process keeps its already-loaded code and original sampling behavior.

All 1,070,933 epoch indices were compared and matched exactly. Their int64 stream
SHA-256 is `8dfb95fbfe3aab6380a85b0b847054772a1dbde4228acc5d1d02106f9fe89e5a`.
Without skipped AMP batches, both runs see the same ordered Han prefix and the
same effective target count at the final budget. Different microbatch grouping
still changes optimizer update counts and stochastic latent draws; this is not
an attempt to make the optimization trajectories identical.

The GPU 6 production-path smoke passed 32,768 Han / 36,404 effective targets in
6 updates, with zero AMP skips and peak allocated memory 23.37 GiB. It also passed
validation, checkpoint writing and final evaluation. Nineteen tests passed,
including reference-order and comparison-identity tests. Smoke weights are not
loaded into formal training.

## Outputs and comparison

Both runs use their own `artifacts/logs/`, `artifacts/checkpoints/` and
`artifacts/reports/` experiment directories. After GPU 6 training and evaluation
finish, its tmux command runs:

```bash
CUDA_VISIBLE_DEVICES='' uv run --frozen python scripts/compare_cvae_batch_runs.py
```

The comparison is written to
`artifacts/reports/cvae_10m_batch_comparison/comparison.json`; the verified
sampling order is recorded alongside it in `sampler_order_verified.json`.
It checks completed budgets and compatible configuration, and compares peak
memory, validation ELBO, glyph metrics, repetition/EOS results and latent
ablations. Both per-run evaluations remain included for inspection.

Wall-clock times include validation/evaluation and are affected by different
GPUs and overlapping server workloads. They are not a controlled hardware-speed
benchmark. Soft font matches are evaluation aids, not substituted generation
outputs or proof of fluent text.

---

## 第六部分：CVAE 头部优化与局部图编译加速验证
> 原文档来源：`CVAE_GPU5_OPTIMIZATION.md`

# GPU 5 optimized CVAE training

The optimized run retains GPU 7's model, initialization seed, corpus identity,
sample order, batch 8, context 1024, head chunk 256, FP16, learning-rate/KL
schedules and exact 10-million-Han budget. It starts from random weights, not
smoke weights. GPU 6 and GPU 7 experiments are not restarted or reconfigured.

Configuration: `configs/experiments/conditional_vae_24l_10m_optimized.json`.
Formal run: `conditional_vae_24l_10m_ctx1024_optimized_v1_full`.

## Execution changes

- Accumulate detached loss statistics on the GPU and transfer once per batch.
  Reject nonfinite statistics before the backbone backward/optimizer update;
  retain AMP and finite-gradient safeguards.
- Enable PyTorch fused AdamW with unchanged learning rate, betas, epsilon and
  weight decay. Floating-point operation order can differ.
- Compile one pure head region using `torch.compile(fullgraph=True,
  dynamic=False)`: prior, posterior after glyph encoding, spatial decoder and
  BCE/KL reductions. Uniqueness, scalar decisions and Gaussian draws stay outside.
- Pad the final head group to 256 and give padding zero loss weight. Draw noise
  only for real targets, preserving the baseline's RNG consumption pattern.
- Disable Inductor CUDA graphs to avoid introducing a separate CUDA graph memory
  pool. The backbone and dynamic glyph encoder stay eager.

The original model and parameter names are retained, so saving/loading ordinary
CVAE state dictionaries and eager final evaluation remain compatible. Existing
configs follow the original backward path unless explicitly enabled.

## GPU 5 measurements

Each fresh process used identical source windows, seed and eight successful
updates. Throughput excludes its first step and CPU data loading.

| Variant | Targets/s | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: |
| Original | 3711 | 29.32 | 30.44 |
| Aggregated statistics + fused AdamW, eager | 3801 | 29.32 | 30.42 |
| Same optimizations + compiled head | 4539 | 29.25 | 30.32 |

The compiled case improved measured steady training-step throughput by about
22.3%; memory use barely changed. Its first step took 35.49 seconds including
compilation, versus about 1.8 seconds subsequently. These are short, sequential
same-GPU probes on a shared server, not a guaranteed full-training speedup.

Ten correctness tests passed, including eager loss/all-gradient equivalence with
tail groups, nonfinite-statistic rejection, and CUDA FP16 compiled-head loss,
input-gradient and parameter-gradient comparisons. Checks use explicit numerical
tolerances; compilation/fused updates do not promise bitwise-identical trajectories.
The production-path smoke completed exactly 32,768 Han / 36,404 targets in five
updates, without AMP skips, and completed validation, checkpoint writing and
prior-generation evaluation.

Evidence: `artifacts/reports/cvae_gpu5_optimization_v1/summary.json`, per-case
metadata/results, and the experiment's usual log/checkpoint/report directories.
Probe implementation commit: `a276abe`; CUDA correctness test: `0637d28`;
formal configuration: `09cbe1d`.

## Run

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 uv run --frozen python scripts/train_conditional_vae.py --mode full --config configs/experiments/conditional_vae_24l_10m_optimized.json
```

Use tmux. The fixed-budget run automatically performs the same final glyph,
generation, repetition/EOS and latent-ablation evaluation as the baseline.

---

## 第七部分：10M 位置 CVAE 留出集重构与先验生成差距评估
> 原文档来源：`CVAE_10M_GPU7_EVALUATION.md`

# GPU 7 evaluation after ten million Han

The run completed exactly 10,000,000 Han / 11,164,221 effective targets in 1372
successful updates, with no AMP skips. Numerical training completed normally;
usable autonomous glyph/text generation was not achieved.

Run: `conditional_vae_24l_10m_ctx1024_v1_full`.
Final checkpoint SHA-256:
`322302ad3e6af32c2a4eac026a0119a9eb81831a3d4cd89ea2fa9e6b5f73f172`.

## Reconstruction versus generation

On 37,286 fixed validation targets, posterior reconstruction BCE/pixel improved
from 0.67977 to 0.13093; final posterior-sample foreground F1 was 0.82271 and exact
bitmap match 0.16894. The posterior sees the target, so these are reconstruction
results, not next-character generation accuracy.

Prior-only generation used 32 held-out pages, 16-tile prompts and at most 128 new
tiles per prompt, with raw predicted bitmaps fed back. Across 4096 output tiles:

- Eleven exact content matches, all commas; zero exact Han font matches.
- 7.62% had best font F1 >=0.8; 1.61% reached >=0.9. These are similarity scores,
  not human readability labels or OCR accuracy.
- No adjacent exact repeats or detected short cycles. Varied malformed glyphs
  can satisfy these statistics, so this does not establish good language quality.
- None of the 32 sequences generated exact EOS within the 128-tile limit.

Assistant visual inspection covered the first 32 outputs of all 32 sequences
(1024 tiles), using four output-only review sheets. There are recognizable
individual shapes and CJK-like components, but frequent missing strokes and
mixed structures. No coherent readable continuation was observed in those
prefixes. This was not an independent human readability study; unreviewed suffixes
are covered by automatic metrics only. Nearest-font guesses were not substituted
for the raw generated outputs.

## Latent-variable evidence

On the same 256 test positions, reconstruction BCE/pixel was:

| Decoder latent | BCE/pixel |
| --- | ---: |
| Correct posterior mean | 0.13577 |
| Shuffled posterior mean | 0.88412 |
| All-zero latent | 0.41765 |
| Prior mean | 0.35552 |

Mean KL was 16.79 nats/glyph. Six of 64 posterior-mean dimensions exceeded sample
variance 0.01; this threshold does not prove the remaining dimensions unused.
For one fixed context, 64 prior draws produced 64 different bitmaps, but none
matched a font glyph exactly. The latent affects decoding, while useful prior
generation remains poor. This does not isolate prior expressiveness, prior fit,
decoder support away from posterior samples, or insufficient language training
as the unique cause.

## Targeted termination check

A supplementary check used the actual last 16 content tiles of the same 32
held-out documents, where the next stored target is EOS. Of 256 prior next-tile
draws, zero matched EOS; zero of 32 sequences terminated within 16 generated tiles.
Prior draws averaged 330.47 differing pixels from EOS.

Even posterior-mean reconstruction given the EOS target had zero exact matches:
18–21 differing pixels, mean 19.84, foreground F1 0.96016. The current exact-bitmap
stop condition therefore rejects even these close EOS reconstructions. The prior
also needs to learn when/how to produce EOS; relaxing a pixel test alone would not
address all observed failures.

Priority follow-up is to separate prior/decoder generalization from target-assisted
reconstruction, and make termination robust to glyph reconstruction error. This
run has not passed the readable, coherent-generation gate for scaling the budget.
Existing GPU 5/6 controls were not stopped or altered by this assessment.

Evidence lives under `artifacts/reports/conditional_vae_24l_10m_ctx1024_v1_full/`:
`prior_evaluation.json`, `glyph_similarity.json`, `generation.json`,
`assistant_visual_review.json`, and `terminal_context_evaluation.json`. The
supplementary check verified the final checkpoint hash and recorded its seed,
dataset identity and evaluation source revision.

---

## 第八部分：8 卡并发 CVAE 学习率网格搜索报告
> 原文档来源：`CVAE_LR_SEARCH.md`

# Conditional VAE learning-rate search

## Protocol

Eight fresh-initialization trials each train on exactly 10,000,000 successful
Han targets. Training implementation: commit `33614b5`. Peak learning rate is
the only intentional experimental difference besides GPU and artifact name.

| GPU | Peak learning rate |
| --- | --- |
| 0 | 0.00005 |
| 1 | 0.0001 |
| 2 | 0.00015 |
| 3 | 0.0002 |
| 4 | 0.0003 |
| 5 | 0.0004 |
| 6 | 0.0005 |
| 7 | 0.0008 |

Configs live in `configs/experiments/lr_search_v1/gpu{0..7}.json`.
All trials use the 285,309,584-parameter Transformer CVAE: 24 backbone layers,
width 1024, 16 query/4 KV heads, FFN 2816. Batch size is 10, context 1024,
head chunk 2048, accumulation 1, seed 20260915, precision FP16.
Acceleration uses xFormers CUTLASS, compiled head and PyTorch fused AdamW.
CUDA Graphs and Apex are disabled.

AdamW uses betas (0.9, 0.95), weight decay 0.1 and gradient clipping 1.
LR warms up over 500,000 Han, then follows cosine decay to 10% of its peak.
KL beta warms up to 1 over 1,000,000 Han. Successful Han counts drive both
schedules. AMP overflow retries preserve the batch and sampled noise.

## Data and verification

Dataset: `data/processed/chinese_document_v3`. Its pinned manifest checksum
is recorded in every config. The shared ordered prefix contains 10,000,000
Han and 11,164,221 effective targets across 1098 successful updates.
`scripts/plan_cvae_lr_search.py` records order hashes and checks controls.

Server tests passed: 30 passed, one optional CUDA Graph test skipped.
The highest-LR smoke started at the most diverse planned batch (cursor 6950)
and completed 65,536 Han, eight updates, zero overflows, checkpoint saving
and generation evaluation. Peak allocated GPU memory was 24.67 GiB.
Smoke weights are not reused for formal training.

## Monitoring and selection

Server tmux socket: `hansgpt-lr-v1`; sessions: `lr-search-gpu0` through
`lr-search-gpu7`, plus `lr-search-monitor`.
The monitor runs `uv run --frozen python scripts/summarize_cvae_lr_search.py
--watch` and updates `artifacts/reports/cvae_lr_search_v1/summary.json`.
Per-trial status and metrics are under `artifacts/logs/<experiment>_full/`.

Validation runs every million Han. Full resumable latest checkpoints are
saved every two million Han using a global serialization lock and atomic
replacement; final checkpoints hardlink the latest file to avoid duplication.
Final reports are under `artifacts/reports/<experiment>_full/complete.json`.

Select using validation loss together with prior glyph readability,
continuous generation, repetition, EOS and latent ablations. ELBO ranking
alone does not establish fluent generation. Test data remains excluded from
LR selection. This single-seed 10M-Han search estimates a useful learning
rate for this budget, not a universal optimum for longer training.

---

## 第九部分：4 卡同步 CVAE 吞吐基准与扩展极限测试
> 原文档来源：`CVAE_FOUR_GPU_THROUGHPUT.md`

# Four-GPU synchronous CVAE throughput

## Results

Both 100-update runs passed with zero invalid updates. Four-GPU replicas had
exactly zero maximum elementwise parameter difference after training.

| Metric | Single GPU 7 | GPUs 4,5,6,7 |
| --- | ---: | ---: |
| Global batch | 10 | 40 |
| Measured seconds, 100 updates | 118.316 | 164.964 |
| Effective prediction positions/s | 8,595.66 | 24,661.56 |
| Han targets/s | 7,698.29 | 22,079.85 |
| Mean update seconds | 1.183 | 1.650 |
| Peak allocated GiB per rank | 24.97 | 24.88–25.01 |
| Peak reserved GiB per rank | 27.13 | 26.77–27.16 |

Aggregate effective-target throughput increases **2.87x**, giving **71.7%**
weak-scaling efficiency relative to ideal 4x scaling. Rank 0 spends 0.480 seconds
per step in the explicitly timed gradient synchronization (about 29.1% of its
mean step). This interval can include waiting for slower ranks as well as
transport and bucket copies; it is not pure network bandwidth. Other ranks
measure 0.467–0.476 seconds.

Four cards process four times the batch per update, but an update takes longer
because of communication. This result supports faster processing of a fixed
corpus budget, not equal optimization behavior: batch 40 makes one quarter as
many optimizer updates as batch 10 for approximately the same target budget.
Formal training must deliberately choose the global batch and LR schedule.

## Protocol

Run `scripts/benchmark_cvae_multigpu.py` using `uv run torchrun --standalone
--nproc_per_node=4` with `CUDA_VISIBLE_DEVICES=4,5,6,7`. The single-GPU control
uses the same script via `uv run python`, restricted to GPU 7, after the four-GPU
job exits. Both exclude ten warmup updates and measure 100 successful updates.

Training implementation commit: `58d4ebd`. Model/config is the LR search GPU4
configuration: 285,309,584 parameters, per-device batch 10, context 1024,
head chunk 2048, FP16, xFormers CUTLASS, compiled head, fused PyTorch AdamW.
For this throughput probe LR is fixed at 0.0003 and KL beta at 1. Fresh weights
are used. No research checkpoints are overwritten or saved.

Each rank receives disjoint batches from the same seeded sortish corpus order.
The four-GPU global batch is 40, versus 10 for the single-GPU control. These
are weak-scaling throughput measurements, not equal-global-batch convergence
experiments. End-to-end timing includes DataLoader prefetch waits, transfers,
forward/backward, communication, gradient clipping and optimizer updates;
it excludes setup, compilation warmup, evaluation and checkpoint I/O.

## Synchronization and transport

The existing chunked backward path uses several detached backward passes.
This benchmark explicitly sums FP32 gradient buckets after the whole backward,
instead of using overlapping DDP autograd hooks. Each rank's locally normalized
loss is weighted by its fraction of global valid targets before synchronization.
Consequently unequal padding/loss masks still implement a global mean objective.
Scratch buckets are bounded at 32 MiB. All ranks apply the same synchronized
gradient before AMP unscale, clipping and AdamW. Every parameter is compared
elementwise across ranks after timing; divergence fails the benchmark.

DataLoader workers use spawn to avoid forking after NCCL initialization.
The default transport stalled before the first update in this container, even
after switching workers to spawn. The successful runs set `NCCL_P2P_DISABLE=1`
and `NCCL_CUMEM_HOST_ENABLE=0`. This establishes a working fallback, not which
individual setting fixes the underlying host/container issue. No host driver
or CUDA installation was changed. GPUs 4/5 and 6/7 are PCIe pairs; no NVLink is
reported. The measured path does not represent optimized overlapping DDP or
working direct GPU peer access.

## Artifacts

Server report directories under `artifacts/reports/`:

- `cvae_four_gpu_smoke_v3/`: completed five-update smoke, identical replicas.
- `cvae_four_gpu_benchmark/`: four-GPU metadata and per-rank results.
- `cvae_single_gpu_benchmark/`: sequential single-GPU control.

Matching console logs are under `artifacts/logs/`. Results distinguish effective
prediction positions (including punctuation/controls) from Han targets.

---

## 第十部分：4 卡 1 亿位置 CVAE 训练结论：后验重构与先验生成的本质差异
> 原文档来源：`CVAE_FOUR_GPU_100M.md`

# Four-GPU CVAE: 100 million prediction positions

## Fixed configuration

From-scratch Transformer CVAE, 285,309,584 trainable parameters. GPUs 4–7
synchronously update one model with per-rank batch 10, global batch 40,
context 1024, head chunk 2048 and accumulation 1. Acceleration and transport
match the throughput probe: xFormers CUTLASS, compiled head, fused PyTorch
AdamW, FP16, post-backward FP32 NCCL gradient buckets,
`NCCL_P2P_DISABLE=1`, `NCCL_CUMEM_HOST_ENABLE=0`.

Config: `configs/experiments/cvae_four_gpu_100m_positions_v1.json`.
Training implementation: `b7a50a8`. Data remains the checksum-pinned
`data/processed/chinese_document_v3`, seed 20260915. Ranks consume disjoint
batches from the shared seeded sortish order. Loss normalization weights
each rank by its actual valid-target count before global gradient summation.

## Budget and scheduling

Stop after exactly **100,000,000 successful valid prediction positions**,
including Han, punctuation and control targets; padding and masked EOS-to-BOS
transitions do not count. Han targets are reported separately.

Peak LR remains 0.0003. Warm up over 500,000 valid positions, then cosine
decay to 0.00003 at the final budget. KL beta warms up to 1 over 1,000,000
valid positions. These schedules use global successful position counts,
not local counts or the previous Han-budget unit. AdamW betas are (0.9, 0.95),
weight decay 0.1, gradient clipping 1.

## Checkpoints and evaluation

Retain one checkpoint at each exact 10,000,000-position milestone, ten in
total. A batch crossing a milestone is split across updates; remaining
targets are consumed after saving, rather than dropped or repeated.
AMP overflow retries retain the data and posterior noise without advancing
the successful budget. Each checkpoint stores model, optimizer, scaler,
per-rank RNG states, configuration and the mid-batch cursor/target offset.
Replica parameters must match elementwise before a checkpoint is published.
This initial runner has no resume CLI; checkpoint state is retained for a
future explicit recovery implementation.

Checkpoint directory: `artifacts/checkpoints/cvae_four_gpu_100m_positions_v1_full/`.
Filenames run from `positions_010000000.pt` to `positions_100000000.pt`.
Allow approximately 32 GiB for all ten files.

Evaluate the fixed validation subset initially and at each checkpoint.
At completion, automatically evaluate 32 validation prompts for raw prior
glyph generation, font similarity, repetition, EOS and latent ablations.
Posterior reconstruction must not be interpreted as fluent generation.
Logs/status: `artifacts/logs/cvae_four_gpu_100m_positions_v1_full/`.
Final metrics/images: `artifacts/reports/cvae_four_gpu_100m_positions_v1_full/`.

## Execution

Run from the committed server checkout via `uv run torchrun --standalone
--nproc_per_node=4 scripts/train_cvae_multigpu.py --config
configs/experiments/cvae_four_gpu_100m_positions_v1.json`, with the physical
GPU and transport environment above and the established xFormers PYTHONPATH.
The server tmux session is `four-train-100m`, socket `hansgpt-lr-v1`.
The independent `--smoke` run uses 65,536 positions and two retained milestones
to exercise partial batches, synchronized updates, saving and evaluation.

---

