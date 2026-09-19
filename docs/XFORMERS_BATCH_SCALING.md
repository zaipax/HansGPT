# xFormers batch-size scaling on V100

Model and non-batch settings remain fixed: 285,309,584 parameters, context 1024,
head chunk 256, FP16, xFormers CUTLASS, compiled head, shared glyph encoder and
PyTorch fused AdamW. Neither Apex nor CUDA Graphs is enabled. Probes ran in
independent processes on GPUs 4/5/6/7 with the same corpus and initialization seed.

## Capacity search

The first round tested batch 8/12/16/24, followed by 17/18/19/20. Twelve measured
updates passed at 8, 12, 16, 17, 18 and 19. Batch 20 and 24 failed with CUDA OOM
during warmup. Batch 19 reached about 31.64 GiB device usage in that short run.

Longer probes use 32 measured updates plus two warmups and a same-GPU batch-8
control. Each probe prepares multiple distinct batches and pads the unique-glyph
buffer to accommodate its complete sample set.

| Batch | GPU | Targets/s | Change vs same-card batch 8 | Device GiB | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 4–7 | 7566–7639 | baseline | 17.11 | passed |
| 12 | 4 | 7801 | +2.1% | 22.79 | passed |
| 16 | 5 | 7814 | +3.3% | 28.23 | passed |
| 17 | 7 | 7944 | +4.1% | 29.27 | passed |
| 18 | 6 | 7851 | +3.5% | 30.69 | passed |
| 19 | 7 | — | — | OOM | failed in warmup |

Batch 19 is not a reliable operating choice: its unique-glyph bucket grew from
2048 in the short run to 2304 in the larger sample set and allocation failed.
Batch **18 is the largest that passed this 32-step test**, not a guarantee that
every full-corpus batch will fit. Rare glyph diversity and allocator behavior
can change the required memory. All successful runs had successful AMP updates;
there were 328 measured successful updates across the entire sweep.

## Throughput interpretation

Reported targets/s measures the GPU training step including input copies,
forward/backward and optimizer updates. CPU batch construction/deduplication is
excluded and cost about 0.50 s/batch at batch 12, 0.72–0.73 s at 16–17, and 0.82 s
at 18. Summing separately measured preparation and step costs gives only a serial
estimate: approximately 5800–5950 targets/s for larger batches versus 5840–5880
for their batch-8 controls. This is not a measured overlapped production pipeline.

Larger batches therefore consume much more memory for only a few percent more
GPU throughput. With head chunk fixed at 256, the decoder still processes more
groups sequentially as batch size increases; doubling batch size does not imply
doubling throughput. Small speed differences are subject to timing/thermal noise,
and training quality was not assessed by these probes.

Recommendation: retain batch 8 for headroom, or choose **batch 12** if increasing
batch size. Batch 16 is an optional compromise; pushing to 18 is not justified
by a clear throughput advantage here. Preserve the original model and focus on
overlapping CPU preparation before claiming end-to-end training gains.

## Evidence

Source commit: `036813f`. Script: `scripts/benchmark_cvae_tricks.py`, now accepting
`--batch-size` and separately recording OOM, preparation costs and measured-step
throughput. Default batch remains 8. Server/local results:
`artifacts/reports/xformers_batch_sweep_v1/summary.json`; per-case metadata and
results remain on the server. All four GPUs were released after the probes.
