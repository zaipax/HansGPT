# Four-GPU C byte-Transformer training

## Completed result

Implementation commit `c5cdab5`. Completed exactly 10,000,000 valid positions
(8,956,249 Han) in 308 successful updates, zero AMP overflow. Total wall time,
including initialization, checkpoint and diagnostics, was 547.54 seconds
(9.13 minutes). All four replicas matched elementwise before checkpointing.

After the first ten updates, 9,674,727 positions took 468.747 seconds:
**20,639.55 effective prediction positions/s aggregate**. Peak allocated memory
was 26.30–26.40 GiB per rank; peak reserved memory was 27.83–28.34 GiB.
This is not directly a scaling ratio against the previous single-GPU probes,
which used either batch8/chunk256 or batch10/chunk2048.

Final fixed-subset validation NLL was 0.186943 nats/pixel (4,261 targets).
The single-prompt greedy diagnostic repeated the same bitmap for all 32
generated grids, with zero exact content-glyph matches. This small budget
establishes training/throughput functionality, not successful text generation.
The final weights and raw diagnostic remain available at the paths below.

This run trains the original pure-Transformer C architecture from scratch on
exactly 10,000,000 successful valid prediction positions using GPUs 4–7.
The 275,349,504 parameters, context1024, head chunk2048 and default acceleration
remain fixed. Per-device batch8 gives global batch32, accumulation1. Backbone
recomputation is disabled. LR is constant at 0.0003, matching the throughput
probes; there is no VAE or KL objective in this architecture.

Config: `configs/experiments/hansgpt_byte_c_four_gpu_10m.json`.
Runner: `scripts/train_byte_multigpu.py` via `uv run torchrun --standalone
--nproc_per_node=4`. Set `CUDA_VISIBLE_DEVICES=4,5,6,7`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, the established xFormers PYTHONPATH,
`NCCL_P2P_DISABLE=1`, and `NCCL_CUMEM_HOST_ENABLE=0`.

Acceleration is xFormers CUTLASS, full-graph compiled teacher-forced byte loss,
and PyTorch fused AdamW. CPU pixel dedup shares glyph encoder work inside each
rank's batch. FP32 gradient buckets synchronize after backward, weighted by
the local fraction of global valid positions. This is synchronous data
parallelism without communication/backward overlap. The corpus is the pinned
packed Chinese document v3 dataset, with disjoint rank shards of one seeded
sortish sequence. Padding and masked document transitions do not count.

The last batch is trimmed to the exact global target budget. All ranks must
have elementwise-identical weights before checkpoint publication. Final
checkpoint includes optimizer, scaler, global cursor and per-rank RNG states.
The initial runner has no resume CLI. Validation NLL is recorded initially and
at the end; final raw byte generation is a single fixed validation prompt
diagnostic, not a broad language-quality assessment.

Throughput excludes initialization and the first ten optimizer updates, final
checkpoint saving, validation and generation. It includes ordinary data
prefetch waits, transfers, backward and optimizer/communication work. Per-rank
peak allocated and reserved memory are recorded. The maximum rank duration
defines aggregate throughput. Small smoke runs do not report a steady-state
rate when fewer than ten updates occur.

Reports, logs and checkpoints use the experiment name
`hansgpt_byte_c_four_gpu_b8_h2048_10m_v1_full` under `artifacts/`.
The checkpoint is `positions_010000000.pt`; final metrics are `complete.json`.
The separate smoke run checks exact milestones and generation, including
unaligned short byte-cache masks required by xFormers CUTLASS.

## 100M-position follow-up

`configs/experiments/hansgpt_byte_c_four_gpu_100m.json` changes only the
experiment name and target budget from the completed 10M run. It starts from
the same random seed on GPUs 4–7, with batch8 per rank, context1024, chunk2048,
no backbone recomputation and constant LR 0.0003. It does not resume the
10M checkpoint. The already-completed 10M run validates this unchanged code
and numerical configuration before the longer run.

Retain checkpoints at exactly 10M, 20M, ..., 100M positions, requiring about
31 GiB in total. Validate at each milestone; final generation remains the
single fixed-prompt diagnostic. The run should perform approximately 3,080
successful updates, versus 308 in the 10M probe. Actual counts are recorded.

Artifacts use `hansgpt_byte_c_four_gpu_b8_h2048_100m_v1_full` under the standard
logs/reports/checkpoints directories. The server tmux session is `byte-four-100m`
on socket `hansgpt-lr-v1`. The current timing field excludes startup/first ten
updates and final saving/evaluation, but includes intermediate milestone I/O
and validation; interpret it accordingly for this multi-checkpoint run.
