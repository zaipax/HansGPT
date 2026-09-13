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
