# Head-chunk scaling at batch 8, context 1024

Model size remains 285,309,584 parameters. All cases use xFormers CUTLASS,
compiled head, shared glyph encoding and PyTorch fused AdamW, without Apex or
CUDA Graphs. Each of GPUs 4/5/6/7 tested head chunk 256/512/1024/2048 respectively,
with identical prepared batches and exogenous Gaussian noise.

## Measurements

Each initial case performed 24 measured successful updates after two warmups.

| Head chunk | GPU | Groups/batch | Targets/s | Device GiB |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 4 | 32 | 7616 | 17.13 |
| 512 | 5 | 16 | 8007 | 18.02 |
| 1024 | 6 | 8 | 8247 | 20.26 |
| 2048 | 7 | 4 | 8449 | 23.34 |

A same-card confirmation on GPU 7, 32 measured updates per case, gave 7591
targets/s at 256 versus 8376 at 2048: **10.3% higher throughput**, with 23.34 GiB
device usage. All 160 measured updates across the successful runs completed
without AMP skips. Maximum paired reconstruction-BCE difference in the initial
sweep was below 4.5e-7 and gradient-norm difference below 1.2e-5.

Recommendation among these four choices: **batch 8, context 1024, head chunk
2048**. Chunk 1024 is an alternative with lower memory use and slightly less
throughput. Increasing head chunk directly reduces sequential decoder groups;
in this workload it helped more than increasing batch size while leaving chunk
256 fixed. It still does not provide linear scaling.

Throughput includes GPU input copies, forward/backward, finite checks and optimizer
updates, but excludes CPU batch preparation. First-shape compilation/warmup is
excluded. Device memory includes allocator reservations and CUDA overhead, sampled
after setup and steps. These are representative short probes, not guarantees for
full-corpus memory usage or long-run model quality.

## Compilation compatibility fix

The initial new-shape attempts failed in compilation, not CUDA memory allocation.
xFormers 0.0.32.post2 supplies empty CPU Philox placeholders in dropout-free
CUTLASS backward, which triggers mixed-device FakeTensor propagation in Torch 2.8.
The adapter now keeps those unused placeholders on CUDA. It passes the backward
operator's `out` *input* positionally so Dynamo does not confuse it with an output
destination, and canonicalizes Q/K/V to contiguous CUTLASS layouts to match
compiled backward stride metadata.

The attention algorithm, dropout setting and trained parameter set are unchanged.
Fullgraph compilation remains enabled; no silent eager fallback was introduced.
Five regression tests passed, covering ordinary/GQA causal and noncausal gradients,
compiled attention backward, shared-encoder gradients and graph replay. The adapter
is specific to the pinned Torch/xFormers API and should be retested on upgrades.
All four final cases were rerun with the same corrected adapter, including 256.

Source commit: `5ca5374`. Successful server/local results:
`artifacts/reports/xformers_head_sweep_v2/summary.json`, with per-case metadata and
results on the server. Initial compile failures remain separately in
`xformers_head_sweep_v1`. GPUs were released after the tests.

## Larger-chunk follow-up: 4096 and 8192

GPU 6 tested 4096 and GPU 7 tested 8192. Both first ran a same-card 2048 control,
with unchanged model, batch 8, context 1024, source data and optimization flags.
Each passing run completed 24 measured updates after warmup.

| GPU | Head chunk | Targets/s | Device GiB | Result |
| ---: | ---: | ---: | ---: | --- |
| 6 | 2048 | 8397 | 23.34 | passed |
| 6 | 4096 | 8343 | 30.66 | passed |
| 7 | 2048 | 8462 | 23.34 | passed |
| 7 | 8192 | — | OOM | failed during warmup |

4096 changed throughput by -0.64%, effectively flat at this measurement precision,
while increasing observed device memory by 7.32 GiB. It does not offer a measured
speed benefit. 8192 could not allocate a further 960 MiB with only about 534 MiB
free; this was actual CUDA OOM, not a compiler failure. All 72 measured updates
in the passing runs succeeded, with matching per-step target counts in the GPU 6
comparison. No long-run quality conclusion follows from these probes.

Recommendation remains **head chunk 2048**. Throughput excludes CPU preparation.
Results: `artifacts/reports/xformers_head_large_v1/summary.json` and per-case
metadata/results on the server. Implementation is unchanged from the previous
sweep; source revision for this follow-up is `2fed9e6`. GPUs were released.
