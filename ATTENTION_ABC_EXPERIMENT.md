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
