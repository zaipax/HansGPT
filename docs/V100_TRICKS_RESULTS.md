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
