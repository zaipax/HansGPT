# GPU 7 context-1024 memory smoke

All seven probes passed on a Tesla V100S 32GB, using the verified
`chinese_document_v3` corpus with EOS causal packing. The model has 24 layers,
width 1024 and 285,309,584 trainable parameters. Each process starts from random
weights and performs full forward/backward and AdamW updates with FP16 AMP,
beta=1 and no gradient checkpointing. No long training run or checkpoint is created.

| Batch | Head chunk | Peak allocated GiB | Peak reserved GiB | Targets/s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 7.74 | 7.98 | 2220 |
| 2 | 128 | 10.82 | 11.28 | 2505 |
| 4 | 128 | 16.86 | 17.45 | 2612 |
| 8 | 128 | 29.10 | 30.34 | 2329 |
| 4 | 256 | 17.36 | 18.33 | 3683 |
| 6 | 256 | 23.38 | 24.28 | 3835 |
| 8 | 256 | 29.32 | 30.44 | 3892 |

Batch 8 with head chunk 256 is a viable full-training candidate after the follow-up
probe. Batch 6 / chunk 256 leaves more memory headroom. Their measured throughput
difference is only about 1.5%, too small to claim a reliable speed advantage from
these short runs. Comparing batch 8 / chunk 128 against batch 6 / chunk 256
confounds batch size with decoder chunk size and does not establish that batch 8
is slower. There is no mandatory fixed memory-reserve percentage.

The first four cases ran four successful steps each; batch 4/6 with chunk 256 ran
eight, and the follow-up batch 8 / chunk 256 ran twelve. All 44 updates succeeded
with finite gradient norms and no AMP skips. The first
successful step in each process is excluded from throughput; CPU sample loading,
validation and checkpoint I/O are not timed. Memory includes optimizer state and
steady training steps. PyTorch reserved memory includes its cache but excludes
some driver/context allocations. These short measurements are not a full-epoch
throughput or long-run stability guarantee. GPU 7 was released after testing.

## Reproduce

Run from clean committed source on the training server, in tmux, using a new
output directory:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 uv run --frozen python scripts/smoke_conditional_vae_memory.py --batch-size 8 --head-chunk-size 256 --steps 12 --output artifacts/reports/cvae_gpu7_ctx1024_repeat/b8_h256
```

Probe source commit: `131ac2e`; follow-up commit: `d91ab40` (documentation only;
same probe/model code). Full configuration, corpus identity, per-step
losses, gradient norms, memory and versions are recorded under
`artifacts/reports/cvae_gpu7_ctx1024_v1/`; `summary.json` combines the seven cases.
