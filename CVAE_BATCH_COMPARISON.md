# Batch 6 versus batch 8 at ten million Han

GPU 7 continues its existing `conditional_vae_24l_10m_ctx1024_v1_full` run
without restart or parameter changes. The independent GPU 6 control is
`conditional_vae_24l_10m_ctx1024_bsz6_v1_full`, starting from random weights.

Both use the same 285,309,584-parameter CVAE, initialization seed 20260915,
verified document-v3 corpus, context 1024, head chunk 256, FP16, AdamW,
learning-rate schedule and KL warmup. Validation batch size remains 8 in both
runs; only the training batch size changes from 8 to 6. Each run stops at exactly
10,000,000 successful Han predictions and runs identical final evaluations.

## Matching training content

The ordinary sortish sampler changes its flat window order when batch size
changes. The control therefore sets `sampler_reference_batch_size=8`, retaining
the original GPU 7 order and regrouping it into batches of 6 in DataLoader.
Defaults preserve the behavior of existing configurations. The existing GPU 7
process keeps its already-loaded code and original sampling behavior.

All 1,070,933 epoch indices were compared and matched exactly. Their int64 stream
SHA-256 is `8dfb95fbfe3aab6380a85b0b847054772a1dbde4228acc5d1d02106f9fe89e5a`.
Without skipped AMP batches, both runs see the same ordered Han prefix and the
same effective target count at the final budget. Different microbatch grouping
still changes optimizer update counts and stochastic latent draws; this is not
an attempt to make the optimization trajectories identical.

The GPU 6 production-path smoke passed 32,768 Han / 36,404 effective targets in
6 updates, with zero AMP skips and peak allocated memory 23.37 GiB. It also passed
validation, checkpoint writing and final evaluation. Nineteen tests passed,
including reference-order and comparison-identity tests. Smoke weights are not
loaded into formal training.

## Outputs and comparison

Both runs use their own `artifacts/logs/`, `artifacts/checkpoints/` and
`artifacts/reports/` experiment directories. After GPU 6 training and evaluation
finish, its tmux command runs:

```bash
CUDA_VISIBLE_DEVICES='' uv run --frozen python scripts/compare_cvae_batch_runs.py
```

The comparison is written to
`artifacts/reports/cvae_10m_batch_comparison/comparison.json`; the verified
sampling order is recorded alongside it in `sampler_order_verified.json`.
It checks completed budgets and compatible configuration, and compares peak
memory, validation ELBO, glyph metrics, repetition/EOS results and latent
ablations. Both per-run evaluations remain included for inspection.

Wall-clock times include validation/evaluation and are affected by different
GPUs and overlapping server workloads. They are not a controlled hardware-speed
benchmark. Soft font matches are evaluation aids, not substituted generation
outputs or proof of fluent text.
