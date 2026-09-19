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
