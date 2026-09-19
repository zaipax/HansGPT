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
