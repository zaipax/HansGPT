# GPU 5 optimized CVAE training

The optimized run retains GPU 7's model, initialization seed, corpus identity,
sample order, batch 8, context 1024, head chunk 256, FP16, learning-rate/KL
schedules and exact 10-million-Han budget. It starts from random weights, not
smoke weights. GPU 6 and GPU 7 experiments are not restarted or reconfigured.

Configuration: `configs/experiments/conditional_vae_24l_10m_optimized.json`.
Formal run: `conditional_vae_24l_10m_ctx1024_optimized_v1_full`.

## Execution changes

- Accumulate detached loss statistics on the GPU and transfer once per batch.
  Reject nonfinite statistics before the backbone backward/optimizer update;
  retain AMP and finite-gradient safeguards.
- Enable PyTorch fused AdamW with unchanged learning rate, betas, epsilon and
  weight decay. Floating-point operation order can differ.
- Compile one pure head region using `torch.compile(fullgraph=True,
  dynamic=False)`: prior, posterior after glyph encoding, spatial decoder and
  BCE/KL reductions. Uniqueness, scalar decisions and Gaussian draws stay outside.
- Pad the final head group to 256 and give padding zero loss weight. Draw noise
  only for real targets, preserving the baseline's RNG consumption pattern.
- Disable Inductor CUDA graphs to avoid introducing a separate CUDA graph memory
  pool. The backbone and dynamic glyph encoder stay eager.

The original model and parameter names are retained, so saving/loading ordinary
CVAE state dictionaries and eager final evaluation remain compatible. Existing
configs follow the original backward path unless explicitly enabled.

## GPU 5 measurements

Each fresh process used identical source windows, seed and eight successful
updates. Throughput excludes its first step and CPU data loading.

| Variant | Targets/s | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: |
| Original | 3711 | 29.32 | 30.44 |
| Aggregated statistics + fused AdamW, eager | 3801 | 29.32 | 30.42 |
| Same optimizations + compiled head | 4539 | 29.25 | 30.32 |

The compiled case improved measured steady training-step throughput by about
22.3%; memory use barely changed. Its first step took 35.49 seconds including
compilation, versus about 1.8 seconds subsequently. These are short, sequential
same-GPU probes on a shared server, not a guaranteed full-training speedup.

Ten correctness tests passed, including eager loss/all-gradient equivalence with
tail groups, nonfinite-statistic rejection, and CUDA FP16 compiled-head loss,
input-gradient and parameter-gradient comparisons. Checks use explicit numerical
tolerances; compilation/fused updates do not promise bitwise-identical trajectories.
The production-path smoke completed exactly 32,768 Han / 36,404 targets in five
updates, without AMP skips, and completed validation, checkpoint writing and
prior-generation evaluation.

Evidence: `artifacts/reports/cvae_gpu5_optimization_v1/summary.json`, per-case
metadata/results, and the experiment's usual log/checkpoint/report directories.
Probe implementation commit: `a276abe`; CUDA correctness test: `0637d28`;
formal configuration: `09cbe1d`.

## Run

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 uv run --frozen python scripts/train_conditional_vae.py --mode full --config configs/experiments/conditional_vae_24l_10m_optimized.json
```

Use tmux. The fixed-budget run automatically performs the same final glyph,
generation, repetition/EOS and latent-ablation evaluation as the baseline.
