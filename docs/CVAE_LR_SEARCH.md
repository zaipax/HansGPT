# Conditional VAE learning-rate search

## Protocol

Eight fresh-initialization trials each train on exactly 10,000,000 successful
Han targets. Training implementation: commit `33614b5`. Peak learning rate is
the only intentional experimental difference besides GPU and artifact name.

| GPU | Peak learning rate |
| --- | --- |
| 0 | 0.00005 |
| 1 | 0.0001 |
| 2 | 0.00015 |
| 3 | 0.0002 |
| 4 | 0.0003 |
| 5 | 0.0004 |
| 6 | 0.0005 |
| 7 | 0.0008 |

Configs live in `configs/experiments/lr_search_v1/gpu{0..7}.json`.
All trials use the 285,309,584-parameter Transformer CVAE: 24 backbone layers,
width 1024, 16 query/4 KV heads, FFN 2816. Batch size is 10, context 1024,
head chunk 2048, accumulation 1, seed 20260915, precision FP16.
Acceleration uses xFormers CUTLASS, compiled head and PyTorch fused AdamW.
CUDA Graphs and Apex are disabled.

AdamW uses betas (0.9, 0.95), weight decay 0.1 and gradient clipping 1.
LR warms up over 500,000 Han, then follows cosine decay to 10% of its peak.
KL beta warms up to 1 over 1,000,000 Han. Successful Han counts drive both
schedules. AMP overflow retries preserve the batch and sampled noise.

## Data and verification

Dataset: `data/processed/chinese_document_v3`. Its pinned manifest checksum
is recorded in every config. The shared ordered prefix contains 10,000,000
Han and 11,164,221 effective targets across 1098 successful updates.
`scripts/plan_cvae_lr_search.py` records order hashes and checks controls.

Server tests passed: 30 passed, one optional CUDA Graph test skipped.
The highest-LR smoke started at the most diverse planned batch (cursor 6950)
and completed 65,536 Han, eight updates, zero overflows, checkpoint saving
and generation evaluation. Peak allocated GPU memory was 24.67 GiB.
Smoke weights are not reused for formal training.

## Monitoring and selection

Server tmux socket: `hansgpt-lr-v1`; sessions: `lr-search-gpu0` through
`lr-search-gpu7`, plus `lr-search-monitor`.
The monitor runs `uv run --frozen python scripts/summarize_cvae_lr_search.py
--watch` and updates `artifacts/reports/cvae_lr_search_v1/summary.json`.
Per-trial status and metrics are under `artifacts/logs/<experiment>_full/`.

Validation runs every million Han. Full resumable latest checkpoints are
saved every two million Han using a global serialization lock and atomic
replacement; final checkpoints hardlink the latest file to avoid duplication.
Final reports are under `artifacts/reports/<experiment>_full/complete.json`.

Select using validation loss together with prior glyph readability,
continuous generation, repetition, EOS and latent ablations. ELBO ranking
alone does not establish fluent generation. Test data remains excluded from
LR selection. This single-seed 10M-Han search estimates a useful learning
rate for this budget, not a universal optimum for longer training.
