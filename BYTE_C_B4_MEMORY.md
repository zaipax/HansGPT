# Single-GPU batch4 memory test

GPU7 completed 110 successful updates of the 275,349,504-parameter C byte
Transformer, with batch4, context1024, head chunk2048 and no backbone
recomputation. xFormers CUTLASS, compiled byte head and fused PyTorch AdamW
remain enabled. The full-corpus global LR schedule is used, with the same
1,089,139,385-position horizon and 10,891,394-position warmup as formal training.
This is a memory/throughput probe, not a language-quality experiment.

The three-update smoke passed before the continuous test. Excluding the first
ten updates and final checkpoint I/O, the next 100 updates took 59.648 seconds:

| Metric | Result |
| --- | ---: |
| Valid prediction positions/s | 6,817.22 |
| Mean timed update | 0.5965 seconds |
| Peak PyTorch allocated | 20.818 GiB |
| Peak PyTorch reserved | 21.980 GiB |
| Maximum sampled CUDA device use | 22.366 GiB |
| AMP overflow | 0 |

Device-use samples are collected after synchronized optimizer updates and
include allocator reservations/runtime overhead. Allocated/reserved peaks
cover the full measured training interval. The post-update live allocation
alone is much smaller and must not be interpreted as the training memory need.

All 110 steps consumed 447,279 valid positions including 400,002 Han. The final
weights are retained. Experiment name: `hansgpt_byte_c_single_b4_h2048_memory_v1_full`;
config: `configs/experiments/hansgpt_byte_c_single_b4_memory.json`.
Implementation: `fed3c7a`. Reports/logs/checkpoints use standard artifact paths.

The user's next intended topology is GPU4–7 with batch4 per rank (global16).
This single-GPU test leaves substantial headroom on a 31.74-GiB card, but does
not measure the additional NCCL communication allocation or guarantee the
worst-case batch across the entire corpus. No four-GPU training was launched
as part of this memory-only request.

## Matched batch8 follow-up

GPU7 subsequently ran the same global-schedule memory probe at batch8, with
context1024/head2048/no recomputation unchanged. Configuration:
`configs/experiments/hansgpt_byte_c_single_b8_memory.json`; implementation/config
commit `5df2cba`. Both the three-update smoke and 110-update run passed.

| Metric, after ten warmups | Batch4 | Batch8 |
| --- | ---: | ---: |
| Measured updates | 100 | 100 |
| Measured seconds | 59.648 | 111.816 |
| Effective positions/s | 6,817.22 | 7,274.83 |
| Peak allocated GiB | 20.818 | 26.181 |
| Peak reserved GiB | 21.980 | 27.818 |
| Sampled device-use range GiB | 22.350–22.366 | 28.192–28.204 |
| AMP overflow | 0 | 0 |

Batch8 improves measured position throughput by 6.71%, while device occupancy
increases by about 5.84 GiB. Each update processes approximately twice as many
positions, so equal-position training makes about half as many updates; this
memory benchmark does not determine which batch learns better. Neither row
includes multi-GPU communication buffers.

The batch8 run consumed 894,826 valid positions including 800,850 Han. Final
weights, status and results use `hansgpt_byte_c_single_b8_h2048_memory_v1_full`.
