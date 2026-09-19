# GPU 7 evaluation after ten million Han

The run completed exactly 10,000,000 Han / 11,164,221 effective targets in 1372
successful updates, with no AMP skips. Numerical training completed normally;
usable autonomous glyph/text generation was not achieved.

Run: `conditional_vae_24l_10m_ctx1024_v1_full`.
Final checkpoint SHA-256:
`322302ad3e6af32c2a4eac026a0119a9eb81831a3d4cd89ea2fa9e6b5f73f172`.

## Reconstruction versus generation

On 37,286 fixed validation targets, posterior reconstruction BCE/pixel improved
from 0.67977 to 0.13093; final posterior-sample foreground F1 was 0.82271 and exact
bitmap match 0.16894. The posterior sees the target, so these are reconstruction
results, not next-character generation accuracy.

Prior-only generation used 32 held-out pages, 16-tile prompts and at most 128 new
tiles per prompt, with raw predicted bitmaps fed back. Across 4096 output tiles:

- Eleven exact content matches, all commas; zero exact Han font matches.
- 7.62% had best font F1 >=0.8; 1.61% reached >=0.9. These are similarity scores,
  not human readability labels or OCR accuracy.
- No adjacent exact repeats or detected short cycles. Varied malformed glyphs
  can satisfy these statistics, so this does not establish good language quality.
- None of the 32 sequences generated exact EOS within the 128-tile limit.

Assistant visual inspection covered the first 32 outputs of all 32 sequences
(1024 tiles), using four output-only review sheets. There are recognizable
individual shapes and CJK-like components, but frequent missing strokes and
mixed structures. No coherent readable continuation was observed in those
prefixes. This was not an independent human readability study; unreviewed suffixes
are covered by automatic metrics only. Nearest-font guesses were not substituted
for the raw generated outputs.

## Latent-variable evidence

On the same 256 test positions, reconstruction BCE/pixel was:

| Decoder latent | BCE/pixel |
| --- | ---: |
| Correct posterior mean | 0.13577 |
| Shuffled posterior mean | 0.88412 |
| All-zero latent | 0.41765 |
| Prior mean | 0.35552 |

Mean KL was 16.79 nats/glyph. Six of 64 posterior-mean dimensions exceeded sample
variance 0.01; this threshold does not prove the remaining dimensions unused.
For one fixed context, 64 prior draws produced 64 different bitmaps, but none
matched a font glyph exactly. The latent affects decoding, while useful prior
generation remains poor. This does not isolate prior expressiveness, prior fit,
decoder support away from posterior samples, or insufficient language training
as the unique cause.

## Targeted termination check

A supplementary check used the actual last 16 content tiles of the same 32
held-out documents, where the next stored target is EOS. Of 256 prior next-tile
draws, zero matched EOS; zero of 32 sequences terminated within 16 generated tiles.
Prior draws averaged 330.47 differing pixels from EOS.

Even posterior-mean reconstruction given the EOS target had zero exact matches:
18–21 differing pixels, mean 19.84, foreground F1 0.96016. The current exact-bitmap
stop condition therefore rejects even these close EOS reconstructions. The prior
also needs to learn when/how to produce EOS; relaxing a pixel test alone would not
address all observed failures.

Priority follow-up is to separate prior/decoder generalization from target-assisted
reconstruction, and make termination robust to glyph reconstruction error. This
run has not passed the readable, coherent-generation gate for scaling the budget.
Existing GPU 5/6 controls were not stopped or altered by this assessment.

Evidence lives under `artifacts/reports/conditional_vae_24l_10m_ctx1024_v1_full/`:
`prior_evaluation.json`, `glyph_similarity.json`, `generation.json`,
`assistant_visual_review.json`, and `terminal_context_evaluation.json`. The
supplementary check verified the final checkpoint hash and recorded its seed,
dataset identity and evaluation source revision.
