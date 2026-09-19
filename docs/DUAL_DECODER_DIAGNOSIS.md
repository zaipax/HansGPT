# Dual decoder output diagnosis

## Run and reproducibility

Diagnosed `hansgpt_dual_r1`, best checkpoint SHA-256
`67f91aee5aecf54237cdbfb9159f8fa2dc30b22bfd0b3c916f58781791edadb2`.
The formal weights were never modified. Diagnostic optimization ran on an
in-memory copy on GPU 5; no diagnostic weights were saved.

```bash
CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/diagnose_dual_outputs.py \
  --checkpoint artifacts/checkpoints/hansgpt_dual_r1/best.pt \
  --output artifacts/reports/hansgpt_dual_r1_diagnosis
```

The report uses 256 seeded paragraphs from each of train and test, each with
17 input grids and one next-grid target. These are diagnostic subsets, not
full-test estimates. Sampled paragraphs are not guaranteed to be distinct pages.
Threshold comparisons use identical logits. Gallery metrics include control
tiles: a blank PAD can count as a legal gallery bitmap, so this metric alone
must never be called valid Chinese generation. Constrained gallery scoring is
diagnostic only, not raw image generation or an implemented decoding change.

## Findings

| Control | Result |
|---|---|
| FP32 cached versus uncached, four test prefixes | Maximum probability difference 4.47e-7; no binary differences |
| FP16 versus FP32, same prefixes | Maximum probability difference 0.000947; no binary differences |
| 45 existing data/model tests on GPU | All passed, including patch coordinates, causal behavior, cache behavior and chunked gradient parity |
| Teacher-forced train / test NLL | 0.31544 / 0.32214 nats per pixel |
| Test NLL with wrong paragraph context | 0.34663 |
| Training-frequency per-pixel prior, evaluated on test targets | 0.33581 |
| Test predicted mean black probability / actual black fraction | 17.82% / 18.27% |
| Test black fraction after threshold 0.5 | 4.30% |
| Test fraction of probabilities between 0.1 and 0.9 | 50.12% |

Correct context helps, but its NLL advantage over the context-free pixel prior
is only about 4.1%. This relative NLL reduction is not language accuracy.
The deficit already occurs with true prefixes, before generated-image feedback.

| Threshold | Test foreground F1 | Exact next-target bitmap |
|---|---:|---:|
| 0.25 | 0.5471 | 0.39% |
| 0.40 | 0.4445 | 1.56% |
| 0.50 | 0.2298 | 0.78% |

Lowering the threshold adds strokes but does not recover complete correct glyphs.
These test diagnostics must not be used to tune the production threshold.

## Capacity control

Fine-tuning the complete model on 32 fixed train prefixes and their next-grid
targets, with AdamW at 3e-4 and no weight decay, achieved 100% exact match at
step 75. It temporarily fell to 46.875% at step 100, recovered by step 125,
and remained at 100% through the final step 300 (NLL 0.000349).
This demonstrates memorization capacity and working gradients; it does not
demonstrate generalization, fluent generation, or successful glyph-only pretraining.

## Interpretation and next experiments

No tested cache, precision, pixel ordering, label boundary or gradient defect
explains the failure. The output head has one product-Bernoulli component.
Its spatial attention shares deterministic features, but the modeled pixels
remain conditionally independent. With an uncertain next character, pixelwise
likelihood can favor marginal stroke probabilities rather than a complete glyph.
The well-calibrated mean foreground mass but sparse thresholded images is
consistent with that mechanism; calibration at individual pixels was not measured.

The evidence does not isolate insufficient training from distributional limitations.
Do not claim that more data, a lower threshold, or a larger model necessarily fixes
the problem. First compare additional training and glyph reconstruction warm-up
under controlled budgets, tracking raw exact glyph validity and held-out language
generation alongside NLL. Any stochastic joint-glyph output change is a separate
architecture experiment requiring explicit design, not a threshold patch.

Raw six-prompt generation and the original 32-page evaluation remain under
`artifacts/reports/hansgpt_dual_r1_speaking/` and
`artifacts/reports/hansgpt_dual_r1_evaluation/` respectively. No fluent continuation
was observed in the six custom prompts. The 32-page evaluation produced 29 exact
content glyphs among 2970 generated body grids (0.98%).
