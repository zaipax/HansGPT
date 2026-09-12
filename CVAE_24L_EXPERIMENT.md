# Original-scale CVAE: 24 layers, width 1024

This experiment restores the original GPT backbone size: 24 layers, hidden width
1024, 16 query heads, four KV heads, FFN width 2816. It initializes all weights
randomly and trains exactly one million successful Han next-glyph targets.
Punctuation and EOS receive loss and are counted separately. No earlier language,
codec or toy weights are transferred.

The remaining CVAE settings match the previous small run: two-layer width-128
patch encoder with 16 queries, two-layer width-256 semantic decoder, one-layer
prior/posterior Transformers, one 64-dimensional shared Gaussian latent per glyph,
and a three-layer width-256 spatial decoder. Context 256, batch 16, seed 20260915,
AdamW peak LR 3e-4, 50K-Han LR warmup and 500K-Han KL warmup are unchanged.
Parameter counts and peak allocation are recorded from the actual model.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/train_conditional_vae.py --config configs/experiments/conditional_vae_24l_1m.json --mode smoke
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/train_conditional_vae.py --config configs/experiments/conditional_vae_24l_1m.json --mode full
```

## Evaluation and readability

The model generates only raw bitmaps. Evaluation never replaces a generated grid
with a gallery glyph before feeding it back into the model.

- Exact bitmap match remains a fidelity metric, not the sole readability judgment.
- Hamming proximity rates at 1/2/4/8/16/32 pixels show near matches separately.
- Best foreground F1/IoU against distinct content bitmaps and the runner-up margin
  describe similarity and ambiguity. Alias-identical references are grouped.
- Blank and control outputs are counted explicitly and never treated as readable
  content merely because they are close to sparse punctuation.
- Nearest-font transcriptions are labeled diagnostic guesses, not raw model text
  or ground-truth OCR. Threshold rates are similarity rates, not human readability.
- Output-only review sheets show the first 32 body grids of all 32 test prompts.
  Review raw pixels before using prompts, then inspect full 128-grid continuations
  for semantic coherence, repetition and ending behavior. Human review records
  distinguish mostly identifiable, partly identifiable, mostly unclear and empty.
- Posterior reconstruction and prior-only generation remain separate. ELBO and
  IWAE estimates retain their approximate-likelihood labels.

Outputs use `conditional_vae_24l_1m_v1_<mode>` under ignored logs, reports and
checkpoints. The previous six-layer run is retained for comparison with the same
similarity evaluator. This is one seed per architecture, not a statistical claim
that scale alone causes any observed difference.
