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

## Adaptive spatial-capacity follow-up

`glyph_codec_spatial16.json` separately tests 16 learned query slots on GPU7 after
the four-slot path reduced pixel errors without achieving exact held-out glyphs.
Its interface is explicitly versioned as `glyph_latents_16x256_v1`; it cannot be
substituted for a four-slot semantic output. This increases latent capacity from
1024 to 4096 scalars and attention cost, so results are not an equal-latent-budget
comparison. Decoder weights transfer except for the new slot-position embeddings.
The splits, gates, optimizer and maximum update budgets stay the same.

Interface version denotes a shape/format, not interchangeable learned coordinates.
Any future semantic alignment must pin the exact selected codec checkpoint hash.

## Completed reconstruction results

The four-slot candidates failed readiness: selected fixed-encoder audit mean
Hamming error was 193.38 pixels; four-query audit error was 34.13, both with zero
exact audit glyphs. The sixteen-query candidate passed: at its selected step
15000, train exact match was 10707/10707, validation 594/595, audit 591/594.

Reserved control tiles were excluded from that reconstruction corpus, and EOS
initially had 138 wrong pixels. `adapt_codec_controls.py` freezes the successful
encoder and trains only its decoder with 224 replayed training glyphs plus eight
copies of each of four control tiles per batch, LR 3e-5. After 500 updates all four
controls are exact, train remains 100%, validation is 589/595, audit is 592/594.
The resulting immutable codec SHA-256 is
`5882dfa7b73304b9f6cfb2fdae64eae2cce1209925d7884370a823cec1c76606`.

## Semantic alignment pilot

`fit_codec_semantics.py` starts from r1 and that exact control-ready codec. It
freezes the original GPT input encoder, original GPT backbone, and entire codec.
Only the existing semantic decoder, permanent 1024-to-4096 mapping and per-slot
normalization train. No new encoder silently replaces the old GPT input encoder.

The one-million-target pilot uses the same Wikipedia corpus, ctx 256, batch 32,
FP16, a 50K-target warmup to LR 1e-4 followed by cosine decay to 3e-5. Its loss is
pixel NLL plus 0.05 times in-batch latent contrastive loss at temperature 0.1.
Detached target-codec features are supervision only; bitmap aliases/repetitions
are multiple positives. No vocabulary lookup or gallery correction enters
inference. This adds an alignment objective, so it is not a loss-matched causal
comparison to r1. It is also not a guarantee that next-character ambiguity is solved.

Run a smoke first, then the pilot on GPU5:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/fit_codec_semantics.py --smoke
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 uv run --frozen python scripts/fit_codec_semantics.py
```

Full-test likelihood, paired glyph metrics, wrong-context controls and 32 raw
continuations are saved automatically. The glyph repair is distinct from achieving
fluent language generation; only those downstream measurements assess the latter.

## Semantic pilot outcome and limits

The pilot completed 1,001,324 valid targets (299 updates, zero AMP overflows).
Full-test NLL was 0.33429 versus r1's 0.31883. Paired foreground F1 was 0.15385
and exact next-target bitmap match was zero. Across 32 independent prompts,
4096 body grids contained no exact content glyphs and no EOS termination.
The semantic pilot therefore did not restore readable generation.

A post-training tensor-by-tensor check verified all 274 original input-encoder/GPT
tensors and all 139 codec tensors were unchanged. The failed language result is
not another accidental replacement or update of the frozen glyph interface.
The permanent codec/mapping contracts, readiness gates and reconstruction repair
are implemented and tested; fluent speaking remains unresolved. More alignment
training versus a different conditional output distribution needs a separate
controlled experiment. A successful deterministic glyph codec does not itself
model multiple plausible next characters.

Final artifacts are under `artifacts/reports/codec_semantic_alignment_v1/`;
`frozen_weights_verified.json` records the real-checkpoint immutability check.
