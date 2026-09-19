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
  for semantic coherence, repetition and ending behavior. Visual review records
  distinguish mostly identifiable, partly identifiable, mostly unclear and empty.
- Posterior reconstruction and prior-only generation remain separate. ELBO and
  IWAE estimates retain their approximate-likelihood labels.

Outputs use `conditional_vae_24l_1m_v1_<mode>` under ignored logs, reports and
checkpoints. The previous six-layer run is retained for comparison with the same
similarity evaluator. This is one seed per architecture, not a statistical claim
that scale alone causes any observed difference.

## Completed run

The instantiated model has 285,309,584 parameters, all trainable. The actual run
completed exactly 1,000,000 Han targets, 1,121,219 total targets and 677 successful
updates, with zero AMP overflows. Training and automatic evaluation took about
348 seconds after setup on GPU5. A discarded random-init stress model separately
passed a full 16x256 update with 4096 distinct input glyphs, peak allocation
13.25 GiB. Neither stress nor smoke weights transferred to the formal model.

Training hyperparameters, non-backbone architecture and all 32 evaluation prompts
were checked to match the small CVAE experiment. Final checkpoint SHA-256:
`e293a1bad6c47fbcf482bab1c6129f0885eb917f06a4c56f1642038cadbd1c1c`.

| Generated-body metric, denominator 4096 | 6-layer / width-512 CVAE | 24-layer / width-1024 CVAE |
|---|---:|---:|
| Exact content glyph match | 0 | 0 |
| Within 8 pixels of a content glyph | 0 | 0 |
| Within 16 pixels | 12 (0.29%) | 31 (0.76%) |
| Best foreground F1 >=0.7 | 650 (15.87%) | 644 (15.72%) |
| Best foreground F1 >=0.8 | 46 (1.12%) | 44 (1.07%) |
| Best foreground F1 >=0.9 | 0 | 0 |
| Blank/PAD body grids | 23 | 104 |
| EOS termination within 128 steps | 0/32 | 0/32 |

Hamming proximity improved slightly while foreground similarity did not show a
clear improvement. These are font-similarity statistics, not measured human OCR
accuracy. The larger model's test-position IWAE-64 estimate was 0.28537 nats/pixel;
prior single-draw target F1 was 0.26822 and IoU 0.15488. These likelihood and target
metrics do not replace raw-output readability assessment.

## Visual audit

The assistant inspected output-only first-32-grid sheets for all 32 prompts, then
all 32 full continuations with their prompts. This is assistant visual inspection,
not independent human annotation, and no numerical human character-accuracy rate
is asserted. Isolated simple shapes are recognizable or plausibly guessable, but
most outputs remain fragmented horizontal/vertical strokes and boxes. No reliably
transcribable coherent continuation was observed. Where glyphs cannot be read
reliably, detailed semantic quality is marked not assessable rather than guessed.

This distinction matters: zero exact matches does not mean every individual
glyph is visually unrecognizable. Nevertheless, this one-million-Han trial did
not produce readable continuous Chinese after restoring backbone scale.

The report directory includes `glyph_similarity.json`, explicitly uncertain
`nearest_font_diagnostic.txt`, raw review sheets, and `assistant_visual_review.json`.
`nearest_font_examples.png` shows the selected top 16 similarities with raw,
reference and difference rows; these selected examples are not average output.
The combined report is `artifacts/reports/cvae_1m_scale_comparison.json`.
