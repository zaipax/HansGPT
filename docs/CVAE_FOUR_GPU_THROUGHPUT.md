# Four-GPU synchronous CVAE throughput

## Results

Both 100-update runs passed with zero invalid updates. Four-GPU replicas had
exactly zero maximum elementwise parameter difference after training.

| Metric | Single GPU 7 | GPUs 4,5,6,7 |
| --- | ---: | ---: |
| Global batch | 10 | 40 |
| Measured seconds, 100 updates | 118.316 | 164.964 |
| Effective prediction positions/s | 8,595.66 | 24,661.56 |
| Han targets/s | 7,698.29 | 22,079.85 |
| Mean update seconds | 1.183 | 1.650 |
| Peak allocated GiB per rank | 24.97 | 24.88–25.01 |
| Peak reserved GiB per rank | 27.13 | 26.77–27.16 |

Aggregate effective-target throughput increases **2.87x**, giving **71.7%**
weak-scaling efficiency relative to ideal 4x scaling. Rank 0 spends 0.480 seconds
per step in the explicitly timed gradient synchronization (about 29.1% of its
mean step). This interval can include waiting for slower ranks as well as
transport and bucket copies; it is not pure network bandwidth. Other ranks
measure 0.467–0.476 seconds.

Four cards process four times the batch per update, but an update takes longer
because of communication. This result supports faster processing of a fixed
corpus budget, not equal optimization behavior: batch 40 makes one quarter as
many optimizer updates as batch 10 for approximately the same target budget.
Formal training must deliberately choose the global batch and LR schedule.

## Protocol

Run `scripts/benchmark_cvae_multigpu.py` using `uv run torchrun --standalone
--nproc_per_node=4` with `CUDA_VISIBLE_DEVICES=4,5,6,7`. The single-GPU control
uses the same script via `uv run python`, restricted to GPU 7, after the four-GPU
job exits. Both exclude ten warmup updates and measure 100 successful updates.

Training implementation commit: `58d4ebd`. Model/config is the LR search GPU4
configuration: 285,309,584 parameters, per-device batch 10, context 1024,
head chunk 2048, FP16, xFormers CUTLASS, compiled head, fused PyTorch AdamW.
For this throughput probe LR is fixed at 0.0003 and KL beta at 1. Fresh weights
are used. No research checkpoints are overwritten or saved.

Each rank receives disjoint batches from the same seeded sortish corpus order.
The four-GPU global batch is 40, versus 10 for the single-GPU control. These
are weak-scaling throughput measurements, not equal-global-batch convergence
experiments. End-to-end timing includes DataLoader prefetch waits, transfers,
forward/backward, communication, gradient clipping and optimizer updates;
it excludes setup, compilation warmup, evaluation and checkpoint I/O.

## Synchronization and transport

The existing chunked backward path uses several detached backward passes.
This benchmark explicitly sums FP32 gradient buckets after the whole backward,
instead of using overlapping DDP autograd hooks. Each rank's locally normalized
loss is weighted by its fraction of global valid targets before synchronization.
Consequently unequal padding/loss masks still implement a global mean objective.
Scratch buckets are bounded at 32 MiB. All ranks apply the same synchronized
gradient before AMP unscale, clipping and AdamW. Every parameter is compared
elementwise across ranks after timing; divergence fails the benchmark.

DataLoader workers use spawn to avoid forking after NCCL initialization.
The default transport stalled before the first update in this container, even
after switching workers to spawn. The successful runs set `NCCL_P2P_DISABLE=1`
and `NCCL_CUMEM_HOST_ENABLE=0`. This establishes a working fallback, not which
individual setting fixes the underlying host/container issue. No host driver
or CUDA installation was changed. GPUs 4/5 and 6/7 are PCIe pairs; no NVLink is
reported. The measured path does not represent optimized overlapping DDP or
working direct GPU peer access.

## Artifacts

Server report directories under `artifacts/reports/`:

- `cvae_four_gpu_smoke_v3/`: completed five-update smoke, identical replicas.
- `cvae_four_gpu_benchmark/`: four-GPU metadata and per-rank results.
- `cvae_single_gpu_benchmark/`: sequential single-GPU control.

Matching console logs are under `artifacts/logs/`. Results distinguish effective
prediction positions (including punctuation/controls) from Han targets.
