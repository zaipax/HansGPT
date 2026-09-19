# Parallel byte decoder pilot

This pilot tests raw glyph integrity after replacing the 128-step causal byte
decoder with 128 learned position queries, additive conditioning from the outer
prefix hidden state, one bidirectional self-attention/SwiGLU block, and a shared
256-class byte head. All byte logits are computed simultaneously. Targets enter
the loss only; generated bitmaps are fed back without font/inventory projection.

The Qwen3-style 1.5B backbone and patch encoder retain their existing dimensions.
All weights start randomly and are jointly trained. This is a generative pilot,
not the original frozen-backbone, character-disjoint probe experiment.

Physical GPU3 only; batch 4, context 1024, FP16, backbone recomputation, xFormers,
compiled head, fused AdamW. Stop at exactly 10,000,000 successful valid prediction
positions (including punctuation/controls). Use a 1M-position warmup to 2e-4,
then the existing full-corpus cosine horizon of 1,089,139,385 positions. The shorter
warmup avoids spending the entire pilot below the peak learning rate. This is
not a matched quality comparison to the existing 20M-warmup serial run.

Run the decoder tests and a 65,536-position smoke test before the full pilot.
The single-rank distributed runner preserves exact stopping, validation and
checkpoint semantics. Evaluate the final checkpoint on eight held-out validation
documents at prompt lengths 8/128/512, 256 generated glyphs, greedy decoding and
categorical sampling at temperatures 0.7/1.0. Retain raw NPZ arrays, PNG contact
sheets, exact inventory membership, blank/repetition rates, EOS, and reference
F1/IoU/Dice/exact match. These are document-held-out diagnostics, not evidence of
character-disjoint generalization. Ten million positions may be insufficient
for language fluency; valid-looking punctuation alone is not success.

Training/evaluation logs, configuration, source commit, dataset/font provenance,
checkpoint hash and environment metadata live in ignored artifacts directories.
Review raw contact sheets before concluding that glyph output is normal.

Reproduction after the canonical clean checkout/pull/environment verification:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3
export NCCL_P2P_DISABLE=1 NCCL_CUMEM_HOST_ENABLE=0
uv run --no-sync pytest -q tests/test_parallel_byte_decoder.py tests/test_byte_glyph_decoder.py tests/test_byte_training.py tests/test_attention_glyph_lm.py
uv run --no-sync torchrun --standalone --nproc_per_node=1 scripts/train_byte_multigpu.py --config configs/experiments/hansgpt_qwen3_parallel_byte_gpu3_10m.json --smoke
uv run --no-sync torchrun --standalone --nproc_per_node=1 scripts/train_byte_multigpu.py --config configs/experiments/hansgpt_qwen3_parallel_byte_gpu3_10m.json
uv run --no-sync python scripts/infer_qwen3_checkpoint.py --physical-gpu 3 --checkpoint artifacts/checkpoints/hansgpt_qwen3_parallel_byte_gpu3_10m_v1_full/positions_010000000.pt --output artifacts/reports/parallel_byte_gpu3_10m_inference_v1 --prompt-lengths 8,128,512
```

Use tmux and redirect console output to `artifacts/logs/` for the long commands.
All output directories must be fresh. The installed environment includes
`xformers==0.0.32.post2`; retain it with `uv sync --frozen --inexact` as documented
in the handoff. `--no-sync` avoids repeated environment mutations during a run.

## 10M result and 100M continuation

The 10M run completed on GPU3 at commit `0c58011`: 2,458 updates, no AMP
overflows, 3,122.43 valid positions/s, peak allocated memory 25.15 GiB, and
validation NLL 0.22115571 nats/pixel on 5,517 targets. The checkpoint SHA256 is
`b13594f068d912a0a4897dc0a4d9fd467ab2fcd221bff24aa4af26c26efada99`.
All nine broad inference conditions had zero exact content inventory membership
and zero EOS. Visual inspection of the greedy contact sheet showed fragmented
horizontal strokes and blank glyphs. These results establish failure at 10M,
but do not distinguish insufficient training from an architectural limitation.

The requested continuation uses
`configs/experiments/hansgpt_qwen3_parallel_byte_gpu3_100m.json` and the 10M
checkpoint via `--resume`. It restores model, AdamW moments/steps, AMP scaler,
per-rank RNG, epoch, data cursor and partially consumed batch. Only cumulative
budget and checkpoint retention change; the LR schedule does not restart.
The destination directories are new. Train 90M additional positions, stopping
at 100M cumulative, with checkpoints at 50M and 100M. Keep the latest checkpoint
and the final 100M milestone; the original 10M checkpoint is preserved.

Before full continuation, `--smoke --resume ...` trains 65,536 additional
positions in a separate smoke directory. Smoke weights are not used by the
formal continuation. The post-training inference command is the same as above,
with the 100M checkpoint/run name and a fresh
`artifacts/reports/parallel_byte_gpu3_100m_inference_v1` output directory.
