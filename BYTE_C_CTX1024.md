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
