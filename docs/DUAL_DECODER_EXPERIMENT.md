# HansGPT semantic and glyph Transformer decoders

This experiment implements one-pass next-glyph prediction. A shared per-glyph ViT
encodes 64 nonoverlapping 4x4 patches into a glyph embedding. The existing 24-layer,
width-1024 causal GPT models the glyph sequence. All parameters train jointly from
scratch on the same pinned full Chinese Wikipedia corpus as ABC round 2.

## Two decoder stages

The semantic decoder projects GPT states to width 256 and applies two causal
Transformer layers across sequence positions. Its output is combined with the
current GPT state and projected to four width-256 semantic condition vectors.
This retains contextual information without introducing a character vocabulary.

The glyph decoder uses sixteen learned part slots and sixty-four spatial queries
with row/column positions. Three width-256 Transformer decoder layers perform
bidirectional self-attention within the next glyph and cross-attention to the four
semantic conditions. Each spatial query outputs sixteen pixel logits, assembled
by a fixed reshape/permutation into one 32x32 bitmap. Part slots are not assumed to
be identified radicals without a separate interpretability experiment.

Both stages execute once per generated glyph, with fixed network depth. Only the
outer character sequence is autoregressive. The GPT and semantic decoder have
separate KV caches; the spatial decoder has no generation loop or recurrent state.
No CNN, byte decoder, GAN, diffusion, candidate glyph selection, OCR feedback or
target-image features are used. The target image only enters the pixel loss.

## Training and measurement

- Physical GPU 5; FP16 autocast and GradScaler, seed 20260914.
- Same corpus manifest and all eight consumed-file hashes as ABC round 2.
- Context 256 grids; budget 100,000,000 successful-update targets.
- Unweighted pixel BCE, fixed generation threshold 0.5. Spatial self-attention
  improves the architectural bias but does not remove the output distribution's
  conditional pixel independence. Good glyph/semantic results are not guaranteed.
- Three-stage bounded backward: encode/GPT/semantic states, spatial-head chunks,
  accumulated gradients back through the context graph. Chunking batches different
  glyphs; it never serializes the pixels of one output glyph.
- Benchmark 64 fixed full-length real training chunks, two warmups and three timed
  updates per case, including transfer, backward and optimizer updates. Compare
  batch 16/32/48, spatial chunks 128/256/512, fused AdamW and batch64 with recomputation.
  Benchmark memory is a capacity check, not a promise to maximize utilization at
  every instant; initialization, validation and checkpoint I/O remain separate.
- Periodic fixed-subset validation and raw-grid generation diagnostics. Cycle
  detection includes periods up to 16; legal glyphs alone are not language success.

## Selected GPU5 configuration

The implemented model has **277,572,880 parameters**. All timed updates below
completed successfully. Values include GPU transfers/backward/optimizer updates,
but exclude dataset loading, validation and checkpoint I/O. The fixed 64-chunk
update is split into microbatches as indicated, so batch48 also has a trailing
16-chunk microbatch. Measurements establish a practical configuration, not a
universal throughput maximum.

| Actual batch | Spatial chunk | Recompute | Optimizer | Grids/s | Peak allocated GiB |
|---:|---:|---|---|---:|---:|
| 16 | 128 | no | AdamW | 5,195 | 13.62 |
| 32 | 128 | no | AdamW | 5,410 | 21.13 |
| 32 | 256 | no | AdamW | 6,410 | 21.48 |
| 32 | 512 | no | AdamW | 6,774 | 22.29 |
| 32 | 256 | no | fused AdamW | 6,415 | 21.48 |
| 48 | 256 | no | fused AdamW | 6,421 | 27.48 |
| 64 | 256 | yes | fused AdamW | 5,852 | 6.02 |
| **32** | **1024** | **no** | **AdamW** | **7,165** | **23.74** |
| 32 | 2048 | no | AdamW | 7,256 | 26.72 |

Select batch32, accumulation1, spatial chunk1024, encoder chunk256, no recompute
and ordinary AdamW. Chunk2048 is only about 1.3% faster while using roughly 3GiB
more memory; the selected configuration preserves headroom for varying real glyph
diversity. The first seven cases use three timed updates; the last two use five,
all after two warmups. Raw results and exact source/config identities:
[initial](reports/dual_decoder_tuning/initial.json) and
[extended](reports/dual_decoder_tuning/extended.json).

Configuration: `configs/experiments/hansgpt_dual_decoder.json`. Use the canonical
clean-main server pull and `uv sync --frozen` before GPU checks. Run tests and the
benchmark before choosing the final committed configuration, then a fresh smoke
run. Only a successful current-source smoke receipt permits the full launch:

```bash
bash scripts/run_dual_decoder.sh smoke hansgpt_dual_r1
bash scripts/run_dual_decoder.sh full hansgpt_dual_r1
```

Logs/arrays: `artifacts/logs/<run>/`. Checkpoints: `artifacts/checkpoints/<run>/`.
The hourly checker `scripts/check_glyph_run.py --run-name <run> --gpu 5` records
progress, throughput, losses and failed/missing processes, without restarting jobs
or sending messages. Training completion is established by the server receipt,
not by this protocol. After successful full training, the same worker runs full-test
likelihood scoring and the fixed 32-page raw-generation evaluation on GPU5, writing
an independent evaluation receipt under `artifacts/reports/<run>_evaluation/`.
