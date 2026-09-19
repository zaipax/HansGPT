# HansGPT ABC round 2: larger models and GPU throughput

Round 1 was interrupted at the user's request to improve GPU utilization and
increase model capacity. Its logs, metadata and last safe checkpoints are retained;
it must not be reported as completed training. Round 2 starts fresh on the same
pinned dataset and keeps the A/B/C input/output ablation unchanged.

## Measured bottleneck

The original batch of 4 with eight accumulation steps, 32-grid output chunks and
gradient checkpointing produced many small GPU operations and unnecessary
recomputation. The original models had 77,200,896 / 81,102,304 / 80,184,832 parameters.

`scripts/benchmark_attention_abc.py` measured identical 32 full-length real training
chunks on each assigned V100S. Each case performs two warmup updates followed by
three measured updates, including transfers, backward and optimizer steps. Loading,
checkpoint writes and validation are excluded, so these are not end-to-end run rates.
All measured updates succeeded. Benchmark source: `b06d609`; raw reports are retained
on the server under `artifacts/logs/abc_perf_{A,B,C}.json`.

| Configuration, cumulative changes | A grids/s | B grids/s | C grids/s |
|---|---:|---:|---:|
| Original small model configuration | 5,413 | 2,608 | 2,207 |
| Actual batch 32, accumulation 1 | 12,246 | 3,106 | 3,083 |
| Output chunk 256 | 18,655 | 5,853 | 5,371 |
| Disable gradient checkpointing | 27,200 | 6,081 | 5,839 |
| Encoder chunk 256 | 32,619 | 6,139 | 6,035 |
| Enlarge shared outer Transformer | 14,796 | 4,814 | 4,906 |

At unchanged model sizes, the combined changes improve this benchmark by roughly
6.03x / 2.35x / 2.73x. Increasing model size consumes more compute but remains faster
than the original configurations. This is a short deterministic capacity/throughput
measurement, not evidence of better language quality or universal speedups.

## Round 2 configuration

| Setting | Value |
|---|---|
| Outer Transformer | 24 layers, hidden 1024, intermediate 2816, 16 query / 4 KV heads |
| A / B / C total parameters | 272,562,176 / 276,758,496 / 275,349,504 |
| Physical GPU assignment | A=4, B=5, C=6 |
| Peak allocated memory in full-length benchmark | A=19.08 GiB, B=20.97 GiB, C=20.88 GiB |
| Actual batch / accumulation / context | 32 / 1 / 256 binary grids |
| Encoder / output chunks | 256 / 256 |
| Gradient checkpointing | Disabled; bounded output chunk backward is retained |
| Budget per model | 100,000,000 successful-update targets, approximately 1.045 corpus passes |
| Initialization / seed | From scratch / 20260911, identical shared backbone weights |
| LR / warmup | 3e-4 cosine schedule / 1,000,000 targets |
| Validation / checkpoint / training log | Every 5M targets / 1,000 updates / 10 updates |

The glyph encoder and byte decoder sizes remain the same as round 1; only the
shared language backbone grows. The three total parameter counts differ because
the ablation modules differ. All round 2 branches share the new optimization,
sampling, budget and validation settings. This approximately one-pass run is an
initial larger-model experiment, not a claim of compute-optimal or converged training.

Original round 1 configuration files remain unchanged. Round 2 uses
`configs/experiments/hansgpt_attention_{a,b,c}_r2.json`. After a clean server update,
locked uv sync, checks and occupancy inspection:

```bash
bash scripts/run_attention_abc.sh smoke hansgpt_abc_r2 r2
# Inspect all three successful smoke receipts before starting full training.
bash scripts/run_attention_abc.sh full hansgpt_abc_r2 r2
```

The hourly cron entry is switched to `--round hansgpt_abc_r2` after all three full
runs start. It records progress and anomalies under
`artifacts/logs/hansgpt_abc_r2_monitor/`. Actual launch and completion are established
by server status/receipts, not by this protocol document.
