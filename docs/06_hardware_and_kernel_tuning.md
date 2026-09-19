# 底层加速与硬件调优 (Hardware Acceleration & Kernel Tuning)

本专题整合了在 Tesla V100S PCIe 服务器上的算子级底层加速探索，包括 Apex 融合优化器、CUDA 图、xFormers CUTLASS 注意力边界以及 Head Chunk / Batch 极限标定测试。

---

## 目录

1. [第一部分：Tesla V100 优化技巧与性能对比测试](#第一部分tesla-v100-优化技巧与性能对比测试)
2. [第二部分：xFormers 显存扩展边界与 OOM 临界点探测](#第二部分xformers-显存扩展边界与-oom-临界点探测)
3. [第三部分：Head Chunk 尺寸对吞吐与显存利用率的消融测试](#第三部分head-chunk-尺寸对吞吐与显存利用率的消融测试)
4. [第四部分：固定 Head Chunk 2048 下的极限 Batch 标定记录](#第四部分固定-head-chunk-2048-下的极限-batch-标定记录)

---

## 第一部分：Tesla V100 优化技巧与性能对比测试
> 原文档来源：`V100_TRICKS_RESULTS.md`

# V100 CUDA optimization results

Completed on the existing PyTorch 2.8.0+cu128 environment without changing the
host driver or system CUDA installation. All runs use 285,309,584 parameters,
ctx=1024, batch=8, head chunk=256, FP16, identical prepared corpus batches and
the same initialization/noise seeds. Every card ran its own baseline first.

## Final measurements

Each row has 12 measured successful updates after two warmup updates. All eight
baseline/treatment runs passed: 96 measured updates without AMP skips.

| GPU | Treatment | Baseline targets/s | Treatment targets/s | Change | Steady device memory GiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 5 | xFormers CUTLASS | 5688 | 7604 | +33.7% | 16.91 |
| 6 | CUDA Graph | 5696 | 7167 | +25.8% | 23.75 |
| 7 | NVIDIA Apex FusedAdam | 5731 | 5708 | -0.4% | 31.30 |
| 4 | xFormers + CUDA Graph + Apex | 5742 | 7086 | +23.4% | 25.64 |

Baselines used about 31.30 GiB device memory. Device memory is sampled after
updates/capture and includes graph pools and driver allocations; it is not the
misleading live-tensor-only count after graph capture. CUDA Graph initialization
still reached 29.30 GiB allocated / 30.91 GiB reserved during warmup, even though
its steady footprint was lower. The combined case had 23.67 GiB peak reserved
during setup. These are short operator/step benchmarks, not full training runs.

**Best measured choice: xFormers plus the existing compiled head and PyTorch
fused AdamW.** Apex alone has no meaningful speed advantage here. Stacking all
three is slower and uses more memory than xFormers alone in this experiment.

## Common fixed-shape preparation

All rows use the same new fixed path, including the baseline. CPU pixel-based
deduplication builds the union of input/target glyphs, pads a checked unique-glyph
bucket, and supplies index maps. The shared glyph encoder runs once per step.
Detached feature leaves accumulate both backbone and posterior gradients before
a single encoder backward. No learned vocabulary or parameter reduction is added.

The main positional shape remains 8192 slots and 32 head groups. Original loss
masks are preserved: masked EOS-to-BOS transitions are not trained merely to fill
the shape. Uniqueness, data preparation and Gaussian noise generation are outside
capture. The benchmark requires full input windows and rejects bucket overflow;
general production scheduling must bucket/recapture or use an eager fallback for
other shapes, never silently truncate inputs.

CUDA Graph captures full model forward/backward. Dynamic AMP finite-gradient
checks, clipping and optimizer updates remain outside. A persistent device scale
buffer is refreshed from GradScaler before replay. Tests verify new noise produces
new gradients and replay overwrites rather than accumulates the prior step.

## Attention and optimizer integration

xFormers 0.0.32.post2 replaces SDPA calls throughout the process: HF Llama backbone,
semantic Transformer, glyph encoder, prior/posterior and spatial decoder. GQA
keys/values are explicitly expanded with correct gradient accumulation; causal
attention uses LowerTriangularMask. CUTLASS forward/backward is explicitly selected;
there is no hidden fallback. Outputs/gradients were checked for ordinary and GQA
attention, with and without causality.

Official FlashAttention-1 **1.0.9** source checks `sm75/sm8x/sm90`, excluding V100's
sm70. It is therefore not a runnable V100 treatment; xFormers is the tested
attention alternative. The earlier missing-build-dependency messages were not
the definitive hardware check.

NVIDIA Apex source revision `a1d527a857e8da64c4e7237ca89ec699fb4d9eaf` was compiled
with the existing CUDA 12.2.140 toolkit and CPP/CUDA extensions enabled. Its exact
minor-version check was relaxed in the experimental source; no major-version
PyTorch check or host driver was bypassed. FusedAdam CUDA updates and the full
training probes passed on cu128. This validates the tested operator, not every
Apex extension. The unrelated PyPI `apex` package was removed from the optional
environment. An API adapter supports `zero_grad(set_to_none=...)`.

## Scope and evidence

GPU step timing includes input copies, forward/backward, finite checks and updates,
but excludes CPU batch construction/deduplication (about 0.31–0.32 s per batch in
this run). These CPU costs should be prefetched/overlapped before claiming an
end-to-end production speedup. The ~5700 baseline also includes shared-encoder
preparation, so it must not be confused with the earlier ~4540 compiled-head probe.

Four correctness tests passed. On the full runs, maximum per-step reconstruction
BCE differences versus the same-card baseline were below 4.1e-7; all parameter
counts and target counts matched. Separate GPU hardware and short runs limit the
precision of small speed differences. No long-horizon model-quality equivalence
is claimed from these smoke tests.

Source: `1ba6de7`. Server results:
`artifacts/reports/cvae_tricks_final/summary.json` and each case's `metadata.json`
and `result.json`. Implementation: `scripts/benchmark_cvae_tricks.py` and
`src/hansgpt_research/cvae_fixed_step.py`. Earlier failed attempts remain separate
under `cvae_tricks_v2`; they are not counted as successful final results.

All GPU benchmark processes and obsolete local source-transfer servers were
stopped/finished after measurement.

## GPU 7 follow-up without Apex

Both variants used PyTorch fused AdamW, the same fixed preparation, model/data,
compiled head and initialization/noise seed. Each ran 24 measured updates after
two warmups, sequentially on GPU 7.

| Variant | Targets/s | Steady device GiB | Reserved GiB |
| --- | ---: | ---: | ---: |
| xFormers | 7647 | 17.11 | 16.73 |
| xFormers + CUDA Graph | 7026 | 25.94 | 23.97 |

The graph combination was 8.12% slower and used 8.82 GiB more steady device memory.
All 48 measured updates succeeded. Maximum paired reconstruction-BCE difference
was 9.01e-8 and gradient-norm difference 3.35e-6. CPU preparation cost approximately
0.31 s/batch and is excluded from the throughput, as in the initial experiment.

Removing Apex does not remove the observed graph-combination penalty. This finding
applies to this capture implementation and workload, not CUDA Graphs in general;
kernel-level profiling would be needed to identify the precise cause. The preferred
configuration remains xFormers + compiled head + PyTorch fused AdamW, without Apex.

Source: `15e5c4b`. Server/local artifact directory:
`artifacts/reports/xformers_graph_gpu7/`, including `summary.json` and both per-run
metadata/results. GPU 7 was released when both probes completed.

---

## 第二部分：xFormers 显存扩展边界与 OOM 临界点探测
> 原文档来源：`XFORMERS_BATCH_SCALING.md`

# xFormers batch-size scaling on V100

Model and non-batch settings remain fixed: 285,309,584 parameters, context 1024,
head chunk 256, FP16, xFormers CUTLASS, compiled head, shared glyph encoder and
PyTorch fused AdamW. Neither Apex nor CUDA Graphs is enabled. Probes ran in
independent processes on GPUs 4/5/6/7 with the same corpus and initialization seed.

## Capacity search

The first round tested batch 8/12/16/24, followed by 17/18/19/20. Twelve measured
updates passed at 8, 12, 16, 17, 18 and 19. Batch 20 and 24 failed with CUDA OOM
during warmup. Batch 19 reached about 31.64 GiB device usage in that short run.

Longer probes use 32 measured updates plus two warmups and a same-GPU batch-8
control. Each probe prepares multiple distinct batches and pads the unique-glyph
buffer to accommodate its complete sample set.

| Batch | GPU | Targets/s | Change vs same-card batch 8 | Device GiB | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 4–7 | 7566–7639 | baseline | 17.11 | passed |
| 12 | 4 | 7801 | +2.1% | 22.79 | passed |
| 16 | 5 | 7814 | +3.3% | 28.23 | passed |
| 17 | 7 | 7944 | +4.1% | 29.27 | passed |
| 18 | 6 | 7851 | +3.5% | 30.69 | passed |
| 19 | 7 | — | — | OOM | failed in warmup |

Batch 19 is not a reliable operating choice: its unique-glyph bucket grew from
2048 in the short run to 2304 in the larger sample set and allocation failed.
Batch **18 is the largest that passed this 32-step test**, not a guarantee that
every full-corpus batch will fit. Rare glyph diversity and allocator behavior
can change the required memory. All successful runs had successful AMP updates;
there were 328 measured successful updates across the entire sweep.

## Throughput interpretation

Reported targets/s measures the GPU training step including input copies,
forward/backward and optimizer updates. CPU batch construction/deduplication is
excluded and cost about 0.50 s/batch at batch 12, 0.72–0.73 s at 16–17, and 0.82 s
at 18. Summing separately measured preparation and step costs gives only a serial
estimate: approximately 5800–5950 targets/s for larger batches versus 5840–5880
for their batch-8 controls. This is not a measured overlapped production pipeline.

Larger batches therefore consume much more memory for only a few percent more
GPU throughput. With head chunk fixed at 256, the decoder still processes more
groups sequentially as batch size increases; doubling batch size does not imply
doubling throughput. Small speed differences are subject to timing/thermal noise,
and training quality was not assessed by these probes.

Recommendation: retain batch 8 for headroom, or choose **batch 12** if increasing
batch size. Batch 16 is an optional compromise; pushing to 18 is not justified
by a clear throughput advantage here. Preserve the original model and focus on
overlapping CPU preparation before claiming end-to-end training gains.

## Evidence

Source commit: `036813f`. Script: `scripts/benchmark_cvae_tricks.py`, now accepting
`--batch-size` and separately recording OOM, preparation costs and measured-step
throughput. Default batch remains 8. Server/local results:
`artifacts/reports/xformers_batch_sweep_v1/summary.json`; per-case metadata and
results remain on the server. All four GPUs were released after the probes.

---

## 第三部分：Head Chunk 尺寸对吞吐与显存利用率的消融测试
> 原文档来源：`XFORMERS_HEAD_SCALING.md`

# Head-chunk scaling at batch 8, context 1024

Model size remains 285,309,584 parameters. All cases use xFormers CUTLASS,
compiled head, shared glyph encoding and PyTorch fused AdamW, without Apex or
CUDA Graphs. Each of GPUs 4/5/6/7 tested head chunk 256/512/1024/2048 respectively,
with identical prepared batches and exogenous Gaussian noise.

## Measurements

Each initial case performed 24 measured successful updates after two warmups.

| Head chunk | GPU | Groups/batch | Targets/s | Device GiB |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 4 | 32 | 7616 | 17.13 |
| 512 | 5 | 16 | 8007 | 18.02 |
| 1024 | 6 | 8 | 8247 | 20.26 |
| 2048 | 7 | 4 | 8449 | 23.34 |

A same-card confirmation on GPU 7, 32 measured updates per case, gave 7591
targets/s at 256 versus 8376 at 2048: **10.3% higher throughput**, with 23.34 GiB
device usage. All 160 measured updates across the successful runs completed
without AMP skips. Maximum paired reconstruction-BCE difference in the initial
sweep was below 4.5e-7 and gradient-norm difference below 1.2e-5.

Recommendation among these four choices: **batch 8, context 1024, head chunk
2048**. Chunk 1024 is an alternative with lower memory use and slightly less
throughput. Increasing head chunk directly reduces sequential decoder groups;
in this workload it helped more than increasing batch size while leaving chunk
256 fixed. It still does not provide linear scaling.

Throughput includes GPU input copies, forward/backward, finite checks and optimizer
updates, but excludes CPU batch preparation. First-shape compilation/warmup is
excluded. Device memory includes allocator reservations and CUDA overhead, sampled
after setup and steps. These are representative short probes, not guarantees for
full-corpus memory usage or long-run model quality.

## Compilation compatibility fix

The initial new-shape attempts failed in compilation, not CUDA memory allocation.
xFormers 0.0.32.post2 supplies empty CPU Philox placeholders in dropout-free
CUTLASS backward, which triggers mixed-device FakeTensor propagation in Torch 2.8.
The adapter now keeps those unused placeholders on CUDA. It passes the backward
operator's `out` *input* positionally so Dynamo does not confuse it with an output
destination, and canonicalizes Q/K/V to contiguous CUTLASS layouts to match
compiled backward stride metadata.

The attention algorithm, dropout setting and trained parameter set are unchanged.
Fullgraph compilation remains enabled; no silent eager fallback was introduced.
Five regression tests passed, covering ordinary/GQA causal and noncausal gradients,
compiled attention backward, shared-encoder gradients and graph replay. The adapter
is specific to the pinned Torch/xFormers API and should be retested on upgrades.
All four final cases were rerun with the same corrected adapter, including 256.

Source commit: `5ca5374`. Successful server/local results:
`artifacts/reports/xformers_head_sweep_v2/summary.json`, with per-case metadata and
results on the server. Initial compile failures remain separately in
`xformers_head_sweep_v1`. GPUs were released after the tests.

## Larger-chunk follow-up: 4096 and 8192

GPU 6 tested 4096 and GPU 7 tested 8192. Both first ran a same-card 2048 control,
with unchanged model, batch 8, context 1024, source data and optimization flags.
Each passing run completed 24 measured updates after warmup.

| GPU | Head chunk | Targets/s | Device GiB | Result |
| ---: | ---: | ---: | ---: | --- |
| 6 | 2048 | 8397 | 23.34 | passed |
| 6 | 4096 | 8343 | 30.66 | passed |
| 7 | 2048 | 8462 | 23.34 | passed |
| 7 | 8192 | — | OOM | failed during warmup |

4096 changed throughput by -0.64%, effectively flat at this measurement precision,
while increasing observed device memory by 7.32 GiB. It does not offer a measured
speed benefit. 8192 could not allocate a further 960 MiB with only about 534 MiB
free; this was actual CUDA OOM, not a compiler failure. All 72 measured updates
in the passing runs succeeded, with matching per-step target counts in the GPU 6
comparison. No long-run quality conclusion follows from these probes.

Recommendation remains **head chunk 2048**. Throughput excludes CPU preparation.
Results: `artifacts/reports/xformers_head_large_v1/summary.json` and per-case
metadata/results on the server. Implementation is unchanged from the previous
sweep; source revision for this follow-up is `2fed9e6`. GPUs were released.

---

## 第四部分：固定 Head Chunk 2048 下的极限 Batch 标定记录
> 原文档来源：`XFORMERS_H2048_BATCH_TUNING.md`

# Batch tuning with head chunk 2048

All probes retain 285,309,584 parameters, context 1024, xFormers CUTLASS, compiled
head, shared glyph encoding and PyTorch fused AdamW. Head chunk is fixed at 2048.
Even batch sizes divide this fixed head grouping; this search is not a proof of
the global optimum over every possible implementation or training configuration.

## Capacity and first sweep

GPU 4/5/6/7 tested batches 8/10/12/16 in parallel. At 24 measured steps, throughput
was 8496/8583/8643 targets/s for 8/10/12; batch 16 ran out of memory. Device usage
was approximately 23.34/27.38/28.87 GiB for the passing cases.

GPU 4 then passed batch 14 at 32 measured steps (8788 targets/s, 31.49 GiB), but
failed at warmup with the larger 64-step sample set. Its unique-glyph buffer grew
from 2048 to 2304 slots. The short result is therefore not a reliable capacity
guarantee. No batch-14 throughput is reported for the failed larger sample set.

## Same-card confirmation

Each pair below used 64 measured steps per case. GPU 5/6 ran batch 8 first;
GPU 7 reversed the order to check an order/temperature confound.

| GPU | Candidate batch | Batch-8 targets/s | Candidate targets/s | Change | Device GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 10 | 8310 | 8482 | +2.1% | 27.34 |
| 6 | 12 | 8316 | 8565 | +3.0% | 29.06 |
| 7 | 10 | 8336 | 8602 | +3.2% | 27.34 |

Direct 10-versus-12 checks, 32 measured steps each, gave:

- GPU 4, 10 then 12: 8680 versus 8672 targets/s, effectively tied.
- GPU 5, 12 then 10: 8510 versus 8614 targets/s, batch 12 about 1.2% faster.

Batch 12 has no clear practical advantage over 10 in these short comparisons,
despite using roughly 1.5–1.7 GiB more device memory. Small speed differences
remain subject to hardware, thermal and sampling noise.

## Recommendation and limits

Among these tested settings, prefer **batch 10 / context 1024 / head chunk 2048**
as a balance of throughput and headroom. Batch 8 remains reasonable if memory
margin is more important: its measured throughput is only about 2–3% lower.
Batch 12 is usable but offers little additional gain; batch 14 and 16 are not
recommended given observed OOMs.

GPU step timing includes transfers, forward/backward and optimizer updates, but
not CPU batch construction/deduplication. Serial CPU-plus-GPU estimates are saved
separately and do not represent a prefetched production pipeline. Finding the
best end-to-end training configuration also requires overlapping this preparation.
Increasing batch size changes update counts and training dynamics; these probes
do not assess long-run model quality or guarantee full-corpus memory safety.

All passing runs had successful AMP updates. Evidence is in
`artifacts/reports/xformers_h2048_batch_v1/summary.json` and per-case metadata/results;
the summary is also saved locally. Source commit: `8eef3dd` (same benchmark code
as the previous head-chunk sweep). GPUs 4/5/6/7 were released after measurement.

---

