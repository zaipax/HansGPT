# Conditional VAE, one-million-Han experiment

All weights initialize randomly, including glyph encoders and the spatial decoder.
No previous language, codec, or toy checkpoint is loaded. The pinned original
ModelScope Wikipedia corpus supplies 32x32 binary targets and font assets only.

## Architecture and probability model

Each glyph uses a patch Transformer and 16 learned pooling queries. A six-layer
width-512 causal GPT and two-layer semantic Transformer encode its prefix.
A prior Transformer produces the mean/log-variance of one 64-dimensional diagonal
Gaussian per next glyph. A separate posterior Transformer sees prefix features
and the actual target glyph only during training and likelihood evaluation.
Its reparameterized sample, together with prefix features, conditions the
three-layer width-256 spatial Transformer, which outputs all 1024 pixel logits
in one pass. All core networks are Transformers plus ordinary linear/norm heads.

Training minimizes `(sum pixel BCE + beta * KL(q||p)) / 1024`. KL is computed
analytically in nats per whole glyph; beta grows to one over 500K successful Han
targets. Log-variance is bounded to [-6,2] for numerical stability. No fixed glyph
vocabulary, image retrieval, pixelwise random sampling, CNN or refinement loop
is used for inference. Shared-z marginalization permits pixel dependence; a
diagonal Gaussian prior and posterior collapse can still limit performance.

## Budget and optimizer

Train exactly 1,000,000 successful Han next-glyph targets, not one million distinct
characters. Punctuation and EOS also receive loss but are counted separately.
The last batch loss mask ends at the millionth Han target. AMP-skipped updates
do not consume the successful-target budget; attempted targets are recorded.

Context 256, batch 16, FP16, AdamW beta=(0.9,0.95), weight decay 0.1, gradient norm
cap 1, peak LR 3e-4, 50K-Han warmup, cosine decay to 3e-5, seed 20260915. All model
parameters train. The actual parameter count is saved in run metadata.

## Tests and evaluation

1. Unit tests cover Gaussian KL, reparameterized gradients, absence of posterior
   target access in generation, one decoder call per glyph, causal/cache behavior,
   exact Han budget masking, and chunked/shared-encoder gradient equivalence.
2. An independent tiny randomly initialized CVAE trains on one identical prefix
   with two next glyphs, tea/water. It reports 512 prior draws versus an analytic
   pixel-mean baseline. Toy weights are discarded, never transferred.
3. A formal-architecture smoke exercises training, validation, checkpoint and
   prior-only generation before the million-Han experiment.
4. Validation separately reports posterior reconstruction, actual KL, and negative
   ELBO at beta=1. None is mislabeled as exact marginal likelihood or fluent speech.
5. Final evaluation uses 32 fixed independent test-page prompts and up to 128
   prior-generated glyphs. Gallery lookup is exact-match scoring only. A 64-sample
   IWAE likelihood estimate on 256 fixed positions and shuffled-posterior-z
   diagnostics measure latent behavior. This is not an exact full-test NLL.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/train_conditional_vae.py --mode toy
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/train_conditional_vae.py --mode smoke
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/train_conditional_vae.py --mode full
```

Outputs use `conditional_vae_1m_v1_<mode>` under ignored reports, logs and checkpoints
directories. Existing directories are rejected; automatic resume is not implemented.
The small budget tests the mechanism and does not imply a from-scratch model will
already acquire fluent Chinese.
