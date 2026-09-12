# GPU 7 utilization investigation

Investigated the running 10-million-Han CVAE experiment on 2026-09-13 (Asia/Shanghai),
without stopping training or changing model/optimizer settings. Source commit:
`e447cea`; batch 8, context 1024, head chunk 256, FP16 AMP.

## Observation and hypotheses

A 45-second NVIDIA dmon trace reproduced alternating utilization: 76–100%, mean
92.11%. The corresponding process sample remained in training and advanced 21
steps / 152,668 Han in 44.05 seconds. This confirms visible fluctuation, not an
OOM or stalled training loop.

Ranked explanations were host/kernel scheduling and synchronization in the chunked
head, DataLoader starvation, periodic validation/checkpointing, and GPU memory
migration. The live trace and a 20-second py-spy sample distinguish these without
introducing another GPU workload.

## Evidence

- Framebuffer use stayed at 31,622 MiB across all 45 dmon samples.
- Training-process swap remained zero; major page faults and host swap-in/out
  counters did not increase. RSS stayed near 3.8 GiB rather than growing.
- Mean PCIe receive/transmit rates were 68.33 / 9.69 MB/s. The one 1018 MB/s receive
  burst coincided with 99% GPU utilization, not the utilization troughs.
- py-spy collected 1066 samples, including 570 with training Python stacks.
  No training stack was waiting in DataLoader. Samples included 72 in attention
  mask checks, 16 in scalar finite-loss checks, and 15 in loss-statistic reads.
  These are sampled Python stacks, not CUDA kernel-duration measurements.
- The head processes approximately 32 chunks per batch. Its finite-loss test and
  two `float(cuda_tensor)` statistics reads introduce repeated host/device
  synchronization, alongside many smaller Transformer operations. The backbone
  and head therefore need not have the same GPU utilization.
- A later interval advanced 9 steps at 3818 effective targets/s and 3408 Han/s,
  close to the 3892 targets/s short smoke. Its final status had just entered
  validation; this is not a pure GPU-kernel benchmark.
- No CUDA OOM appeared; AMP skipped-update count remained zero at 2,010,650 Han.

## Conclusion and limits

The evidence does not support GPU tensors spilling into system RAM. This code
uses ordinary PyTorch CUDA allocation and has no CPU/offload or managed-memory
configuration; exhausted ordinary CUDA allocations normally raise OOM. CPU RAM
also holds memory-mapped corpus pages and DataLoader buffers, independently of GPU
memory. Unified-memory page migrations were not directly traced with CUPTI.

The observed behavior is consistent with alternating backbone/head computation
and CPU/GPU synchronization. Average utilization and measured throughput remain
healthy. Periodic validation and checkpoint writes can additionally interrupt
training activity. No training change or regression test is warranted solely to
make the utilization graph flat; no allocation failure was reproduced.

A future controlled optimization can aggregate detached loss statistics on the
GPU and read them once per step, then compare identical batches and gradients.
It should preserve finite-gradient safeguards. No such change was hot-patched into
this experiment, and no temporary debug logging was added to its training loop.

Server evidence is under `artifacts/logs/gpu7_util_diagnosis.*`: dmon, process
samples, raw Python stacks and throughput measurements. No extra GPU job or
background diagnostic sampler remains running.
