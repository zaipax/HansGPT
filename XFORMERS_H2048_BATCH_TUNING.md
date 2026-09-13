# Batch tuning with head chunk 2048

All probes retain 285,309,584 parameters, context 1024, xFormers CUTLASS, compiled
head, shared glyph encoding and PyTorch fused AdamW. Head chunk is fixed at 2048.
Even batch sizes divide this fixed head grouping; this search is not a proof of
the global optimum over every possible implementation or training configuration.

## Capacity and first sweep

GPU 4/5/6/7 tested batches 8/10/12/16 in parallel. At 24 measured steps, throughput
was 8496/8583/8643 targets/s for 8/10/12; batch 16 ran out of memory. Device usage
was approximately 23.34/27.38/28.87 GiB for the passing cases.

GPU 4 then passed batch 14 at 32 measured steps (8788 targets/s, 31.49 GiB), but
failed at warmup with the larger 64-step sample set. Its unique-glyph buffer grew
from 2048 to 2304 slots. The short result is therefore not a reliable capacity
guarantee. No batch-14 throughput is reported for the failed larger sample set.

## Same-card confirmation

Each pair below used 64 measured steps per case. GPU 5/6 ran batch 8 first;
GPU 7 reversed the order to check an order/temperature confound.

| GPU | Candidate batch | Batch-8 targets/s | Candidate targets/s | Change | Device GiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 10 | 8310 | 8482 | +2.1% | 27.34 |
| 6 | 12 | 8316 | 8565 | +3.0% | 29.06 |
| 7 | 10 | 8336 | 8602 | +3.2% | 27.34 |

Direct 10-versus-12 checks, 32 measured steps each, gave:

- GPU 4, 10 then 12: 8680 versus 8672 targets/s, effectively tied.
- GPU 5, 12 then 10: 8510 versus 8614 targets/s, batch 12 about 1.2% faster.

Batch 12 has no clear practical advantage over 10 in these short comparisons,
despite using roughly 1.5–1.7 GiB more device memory. Small speed differences
remain subject to hardware, thermal and sampling noise.

## Recommendation and limits

Among these tested settings, prefer **batch 10 / context 1024 / head chunk 2048**
as a balance of throughput and headroom. Batch 8 remains reasonable if memory
margin is more important: its measured throughput is only about 2–3% lower.
Batch 12 is usable but offers little additional gain; batch 14 and 16 are not
recommended given observed OOMs.

GPU step timing includes transfers, forward/backward and optimizer updates, but
not CPU batch construction/deduplication. Serial CPU-plus-GPU estimates are saved
separately and do not represent a prefetched production pipeline. Finding the
best end-to-end training configuration also requires overlapping this preparation.
Increasing batch size changes update counts and training dynamics; these probes
do not assess long-run model quality or guarantee full-corpus memory safety.

All passing runs had successful AMP updates. Evidence is in
`artifacts/reports/xformers_h2048_batch_v1/summary.json` and per-case metadata/results;
the summary is also saved locally. Source commit: `8eef3dd` (same benchmark code
as the previous head-chunk sweep). GPUs 4/5/6/7 were released after measurement.
