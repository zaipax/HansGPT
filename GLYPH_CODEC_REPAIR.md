# Persistent glyph codec repair

The repaired glyph module has one explicit `[...,4,256]` latent contract.
`forward(image)` calls `decode(encode(image))`; a future semantic predictor must
supply exactly that latent interface. The fixed arm's mapping is permanent and
included in checkpoints. It is never discarded or bypassed during inference.
This does not imply the old GPT semantic states are already aligned to the codec.

## Two candidates

- **fixed, GPU5:** the original r1 glyph encoder is frozen and stays in evaluation
  mode. A permanent linear/RMSNorm mapping and the spatial decoder learn reconstruction.
  Frozen encoder features can be cached without changing inference semantics.
- **spatial, GPU6:** four learned queries read the 64 patch features through a
  Transformer pooling layer, then project to four width-256 vectors. The patch
  encoder initializes from r1 and trains. The single-summary projection is removed.

Both retain the same spatial decoder architecture, binary loss, batch 256, glyph
split and 1024-dimensional output condition. Their parameter counts and timing
are recorded. Freezing differs, so this is an engineering comparison, not a
single-variable proof about spatial pooling. Neither path uses a raw-pixel skip,
character-ID predictor, gallery projection, CNN, or iterative image generation.

## Stages and gates

1. Memorize 256 reconstruction-training glyphs with at most 4000 updates. Require
   99% exact bitmap reconstruction before expanding; a failed gate ends the run.
2. Expand to the same 10707 training bitmaps as the pilot, at most 16000 updates.
   Use cosine LR from 3e-4 to 3e-5. Stop after five evaluations without validation
   NLL improvement and select the best validation-NLL checkpoint.
3. Split the previous 1189 reconstruction-held-out glyphs into 595 validation and
   594 audit glyphs. Audit is measured only after checkpoint selection. These are
   not globally unseen glyphs: r1 and earlier research inspected the old corpus.
4. Before language alignment, require 95% training exact match, at least 50%
   validation and audit exact match, and mean Hamming error at most four pixels
   on both held-out partitions. These are declared engineering readiness gates,
   not claims of fluent language generation or statistically established thresholds.

The original GPT input encoder and formal checkpoints are never replaced. A
codec that passes still needs explicit semantic-side alignment; latent MSE alone
can average possible next characters and is not presumed to solve generation.

## Execution and artifacts

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/train_glyph_codec.py --arm fixed
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 uv run --frozen python scripts/train_glyph_codec.py --arm spatial
```

Use `--smoke` first; smoke exercises both phases and inference but cannot pass
the language-readiness gate. Outputs use `glyph_codec_repair_v1[_smoke]_<arm>`
under ignored reports, logs and checkpoints directories. Existing directories
are rejected rather than overwritten. Checkpoints include model, optimizer, AMP,
RNG, phase, step and interface metadata; automatic resume is not implemented.

Tests check permanent mapping checkpoint round-trips, reconstruction versus
external-latent decoding equivalence, frozen-encoder immutability, and gradients
through spatial queries. Each formal run also verifies cached training and
uncached inference produce identical binary outputs on audit examples.
