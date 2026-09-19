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
