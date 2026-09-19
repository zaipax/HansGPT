# GPU2 single-GPU training speed ablations

## Protocol

Run date: 2026-09-18 local time (2026-09-17 UTC). Physical GPU2 only,
Tesla V100S PCIe 32 GB, SM70; other training jobs remained running.
PyTorch 2.8.0+cu128, Transformers 5.16.1, xFormers 0.0.32.post2.
Baseline source: `ba7ae69`; compiler boundary: `8006bfe`; block compilation:
`442379a`; distinct-window validation: `e4b1b74`.

Architecture/config: `configs/experiments/hansgpt_qwen3_parallel_byte_gpu3_100m_recovery.json`.
The 30-layer, width-2048 backbone and one-layer parallel 128-byte head are
jointly trained. Batch 4, context 1024, FP16 autocast, FP32 parameters/AdamW
state, fused AdamW, xFormers CUTLASS, compiled head, backbone-only nonreentrant
checkpointing. Initial AMP scale 1024 for all cases. Same seed and random
initialization, no checkpoint loading or saving.

Each case runs in a fresh process. Ten warmup updates precede 20 measured
updates on the same four real full-length corpus windows, 4096 valid targets
per update. Timing includes DataLoader wait, pinned H2D copies, checked forward,
backward, gradient clipping and optimizer update. Two spawned data workers
preprocess asynchronously. Setup, corpus verification, compilation/warmup,
validation and checkpoint I/O are excluded. No distributed process group or
gradient-reduction scratch bucket is created. Thus these are matched ablations,
not a direct comparison against the existing world-size-one distributed runner.

## Fixed-window results

| Case | Valid positions/s | Change vs baseline | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| Baseline, chunk 1024 | 3336.04 | — | 24.279 | 24.896 |
| 1. Chunk 2048 | 3349.79 | +0.41% | 24.278 | 25.318 |
| 2. Disable backbone checkpointing | OOM | unavailable | 31.22 at failure | — |
| 3. Compile each backbone block, compatibility boundary enabled | 4076.40 | +22.19% | 24.016 | 24.645 |
| 4. Packed-byte CPU preprocessing | 3335.35 | -0.02% | 24.279 | 24.896 |
| Repeat baseline | 3314.48 | -0.65% | 24.279 | 24.896 |
| Compatibility boundary only, eager backbone | 3331.20 | -0.15% | 24.279 | 24.896 |

The chunk-2048 gain is smaller than the observed baseline repeat variation;
it is not convincing evidence of a useful speedup. It also reserves an extra
0.42 GiB. Disabling checkpointing fails during the first forward at batch 4;
changing batch size would no longer isolate the requested optimization.

The packed-byte path checks the glyph bank once, gathers prepacked byte rows
from corpus asset addresses, deduplicates by pixel bytes, and unpacks only unique
input tiles. The model still receives pixels, never learned character IDs.
Input tensors match the existing path on full windows, document-boundary cases
and the final partial window. Baseline loader wait averages 0.115 ms/update;
the existing workers already hide preprocessing behind roughly 1.23 seconds
of GPU work. CPU savings therefore need not increase end-to-end throughput.
An alternating-order CPU-only retrieval+collation comparison over 20 measured
batches (four warmup batches excluded) reduced median batch preparation from
92.27 ms to 39.16 ms, a 57.6% reduction / 2.36x speedup. All 24 batches matched
tensor-for-tensor. These CPU timings are separate from end-to-end training.

## Compiler compatibility and numerical checks

Compiling the entire backbone first failed because Torch 2.8's
`aten._efficient_attention_backward` operator schema contains `out` and
`window_size`, but its registered Python meta implementation has a mismatched
signature. The observed error was `multiple values for argument 'scale'`.

The opt-in `hansgpt::cutlass_backward` custom operator keeps the existing CUTLASS
CUDA implementation and supplies a correct fake-tensor shape boundary. It is
limited to first-order, fixed-length, bias-free, dropout-free attention. It is
not a handwritten CUDA kernel. Existing training's default path is unchanged.
`scripts/check_compiled_cutlass.py` passed on GPU2 for noncausal/default-scale
and causal/explicit-scale attention: identical forward outputs and Q/K/V
gradients within `rtol=atol=1e-3` versus eager CUTLASS.
The focused server regression suite passed 18 tests, with the explicitly
opt-in CUDA-graph test skipped. This included the enabled xFormers CUDA tests,
byte-training and parallel-decoder checks, and nonfinite recovery checks.
Local Ruff checks passed for all four new/modified Python files.

The full-backbone wrapper then encountered a separate Transformers graph-capture
exception. Compiling each layer's `forward` instead leaves HF's outer
configuration/output wrapper eager while compiling all 30 compute blocks.
Nonreentrant checkpointing remains enabled outside those compiled forwards.
This path completed all updates with finite loss and gradient norms, with no
AMP skips. The eager compatibility-only control shows the speed gain comes
from block compilation, rather than the operator boundary itself.

## Distinct-window validation

These use 10 warmup and 40 measured updates,
with seeded distinct full windows sampled across the training corpus. All cases
use identical sample order and initialization; no checkpoint is written.

| Case | Valid positions/s | Change vs paired baseline | Peak allocated/reserved GiB |
| --- | ---: | ---: | ---: |
| Baseline | 3349.51 | — | 24.371 / 25.160 |
| Compiled blocks + backward boundary | 4066.19 | +21.40% | 24.111 / 24.969 |
| Above + chunk 2048 + packed preprocessing | 4085.89 | +21.98% | 24.110 / 24.963 |

Every measured update succeeded, with finite losses and gradient norms. Maximum
absolute NLL-per-pixel differences from baseline across the 40 measured updates
were `1.0781e-5` for compiled blocks and `1.4529e-5` for the combined case.
This is a short trajectory check, not bitwise-identical training or proof of
long-run convergence equivalence. Fixed-window compiled NLL differences were
at most `3.5763e-6`.

Recommendation: retain batch 4/context 1024/checkpointing/chunk 1024 and use
compiled backbone blocks with the compatibility boundary for the next isolated
single-GPU training trial. The approximately 21–22% throughput gain reproduces
on distinct windows. The combined case adds only 0.48% over compilation alone,
below the 0.65% baseline repeat spread, so it is not established as a better
default. No running trainer/configuration/checkpoint was changed by this study.

The recorded total warmup times are 17.00 s for the fixed-window baseline and
20.59 s for block compilation, with the existing server compiler caches.
These numbers include ten updates and must not be presented as isolated cold
compilation times. Separate fresh-machine compilation costs were not measured.

## Profiling and next optimization targets

A separate post-timing baseline profiler step recorded approximately 48.1% of
self CUDA time under matrix multiplication, 11.2% under `aten::copy_`, 10.3%
under `aten::mul`, 7.2% under efficient-attention backward, and 3.6% under the
optimizer step. Operator and kernel rows overlap and must not be added together.
`copy_` includes device casts/copies; it is not a measurement of PCIe H2D time.
The profile is diagnostic, not a throughput result.

The next plausible targets are selective activation recomputation and fused
RMSNorm/QK normalization/RoPE/SwiGLU/casting paths supported on SM70. Measure
remaining costs after compilation before writing custom kernels; compiler
fusion already removes part of this work. Replacing CUTLASS/cuBLAS GEMMs is
not the first choice. More invasive changes such as reduced model size,
shorter context or lower-precision optimizer state need separate convergence
and glyph-quality comparisons and are not equivalent performance settings.

## Reproduction and artifacts

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
uv run --no-sync python scripts/benchmark_parallel_single_gpu.py \
  --opaque-backward --compile-layers --vary-data --steps 40 \
  --output artifacts/reports/single_gpu2_speed_v1/reproduction.json
```

Omit optimization flags for baseline. Individual flags are `--chunk 2048`,
`--no-checkpoint`, `--compile-backbone` (original failing full-wrapper case),
and `--packed-data`. Final working compilation requires both
`--opaque-backward --compile-layers`. Output files must not already exist.

Server artifacts: `artifacts/reports/single_gpu2_speed_v1/`; logs:
`artifacts/logs/gpu2_speed_*.log`, with compatibility regression log
`artifacts/logs/gpu2_compile_check.log`. Successful reports include config,
Git revision, dataset hashes, seed, runtime versions, sample indices and every
measured update. Failure tracebacks are preserved separately. These short
random-initialization performance tests do not establish long-run stability,
checkpoint-resume behavior or glyph generation quality.
