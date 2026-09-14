# HansGPT Handoff

## Current state

The active research direction is the pure Transformer C architecture: 4×4 patch self-attention glyph encoder, 24-layer width-1024 GQA GPT backbone (16Q/4KV, FFN 2816), and a 4-layer width-256 causal byte Transformer (8 heads, FFN 768) that autoregressively emits 128 bytes and reconstructs one 32×32 binary glyph. Actual baseline size is 275,349,504 parameters. Full design and prior ABC results are in `ATTENTION_ABC_R2.md` and `reports/attention_abc_r2/REPORT.md`.

## Experiments completed

- CVAE learning-rate search: 8 GPUs, bsz10, ctx1024, chunk2048, 10M Han each. Best validation convergence was peak LR `3e-4`; high LR `8e-4` was poor. This was posterior reconstruction, not fluent generation.
- Four-GPU CVAE 100M positions: global bsz40, peak LR `3e-4`; reconstruction improved but prior generation remained weak.
- C byte model, single GPU: ctx1024, bsz10, chunk2048, no recomputation: about 7,480 positions/s, ~28.9 GiB allocated.
- C byte model, four GPUs 4–7: bsz8/rank, chunk2048, 10M positions: 20,640 positions/s; raw greedy output initially collapsed to blank/comma glyphs.
- C byte model, eight GPUs 0–7: bsz8/rank, chunk2048, full-corpus LR schedule, 100M positions: 37,630 positions/s. Validation NLL reached 0.01034 nats/pixel, but greedy generation produced repeated commas. Sampling produced varied but incoherent text.
- Four-GPU full-corpus run on GPUs 4–7 used bsz8/rank, ctx1024, chunk2048, no recomputation, one complete dataset pass target (1,089,139,385 valid positions), global cosine schedule, peak LR3e-4, warmup10,891,394, and final LR3e-5. It stopped near completion after reaching 1.08B positions; inspect `artifacts/logs/hansgpt_byte_c_four_gpu_b8_h2048_full_corpus_v1_full/status.json` before acting.
- Single-GPU memory probes with chunk128 and recomputation showed safe scaling through 1.42B parameters (~23.7 GiB device use); 1.79B used ~29.1 GiB and is an aggressive ceiling. Chunk128 is useful for capacity probing; chunk2048 is faster but near the limit for large models.
- Qwen3-style C-Qwen1.5B benchmark completed on physical GPUs 0-3 at commit `08210da`: hidden 2048, 30 layers, 16Q/8KV, head dim 128, FFN 6144, QK-Norm before RoPE, RoPE theta 1e6, and 1,515,243,008 parameters. With FP16, gradient checkpointing, xFormers, fused AdamW, context 1024, head chunk 128, and batch 1 per rank, measured throughput was 1,377 valid glyph targets/s globally; per-rank peak allocated memory was 22.81 GiB (reserved 23.38 GiB). The benchmark used real `chinese_document_v3` windows and selected only GPUs 0-3; the active 4-7 run remained at roughly 29 GiB per card.
- Qwen3-style C-Qwen1.5B single-GPU diagnostic on GPU0 at commit `5a6e540`: the same context, batch, head chunk, xFormers, FP16, fused AdamW, and 10-step warmup measured 2,041.6 valid glyph targets/s with backbone recomputation enabled (four measured steps, 0.494-0.498 s/step, peak allocated 23.19 GiB). With recomputation disabled, the measured steps were 0.427-0.429 s (about 2.34-2.39k targets/s) and peak device use was 27.36 GiB; its final checkpoint could not be written because the server filesystem was full, but the throughput measurements completed.
- The four-GPU result is communication-bound: 4,096 global targets took 2.970-2.978 s/update versus about 0.495 s for one rank. Relative to ideal replication of the recomputation-on single-GPU result (about 8,166 targets/s), 1,377 targets/s is 16.9% scaling efficiency. The model has 1,515,243,008 FP32 gradients (about 5.64 GiB); the benchmark's 8,388,608-element FP32 scratch buffer is 32 MiB, so each update performs roughly 181 sequential NCCL all-reduces. GPUs 0-3 have PCIe PIX/NODE links with no NVLink, making synchronization the dominant cost.
- The GPU4-7 full-corpus C run reached 1,080,000,000 of 1,089,139,385 target positions (33,293 updates) and then stopped at `2026-09-14T06:54:12Z` with `Insufficient space for retained checkpoint`; all GPUs are now idle. Preserve its existing checkpoints and logs before further cleanup.
- Evaluation of the latest retained checkpoint `positions_1070000000.pt` (1.07B targets, 32 fixed validation chunks, 4,261 targets) completed on GPU0. Teacher-forced greedy byte decoding reached NLL `0.002565` nats/pixel (`2.6269` nats/grid), foreground F1 `0.6481`, IoU `0.4794`, Dice `0.6481`, exact bitmap match `0.4605`, and pixel accuracy `0.8828`. A fixed 32-glyph free-running diagnostic had 100% exact content membership, zero adjacent repeats, and 22 unique glyph bitmaps; eight independent 16-glyph prompts produced 242 body glyphs with 100% exact content membership and 2.56% adjacent repeats. The model no longer falls into the 100M checkpoint's all-comma cycle, but samples still contain phrase-level repetition and only 1/8 prompts emitted EOS within 32 glyphs. Persistent report: `artifacts/reports/hansgpt_byte_c_four_gpu_b8_h2048_full_corpus_v1_eval_1070m/result.json`.

## Important diagnosis

The apparent comma/blank failure is primarily optimizer-step starvation and weak conditioning, not proven xFormers corruption. Old C at 100M positions performed 30,494 updates because short ctx256 windows contained ~3,280 valid positions/update. New packed ctx1024/global batch64 performed only 1,546 updates at 100M. Historical old checkpoint at step1505 also produced the same comma loop. Native SDPA, xFormers, cached and uncached decoding matched. Sampling exposes learned modes; greedy currently selects a dominant punctuation mode.

## Immediate goals

1. Treat `positions_1070000000.pt` as the current final usable checkpoint; compare its raw generation against the 100M all-comma baseline and retain the evaluation report.
2. Compare small global batches (per-rank bsz2 or4) against bsz8 at fixed successful-position prefixes to increase optimizer updates while preserving ctx1024 and the full-data LR horizon.
3. For the chosen batch, continue to at least the update regime where old C became readable (~3,000 updates), then evaluate multiple prompts with greedy and categorical sampling, EOS, repetition, raw glyph validity and context-shuffle sensitivity.
4. Do not select a model from posterior reconstruction alone. Require raw prior generation and semantic/repetition review; keep test data out of LR selection.

## Reproduction and artifacts

Use project `uv` only (`uv sync --frozen --inexact`, `uv run ...`). Source changes follow the local worktree → commit/rebase/merge/push → server pull workflow. Reports, checkpoints and images live under ignored `artifacts/`; checkpoints may be pruned only after checking active processes and preserving the active run.

The current server checkout is `/root/HansGPT`. Connection details and the supported proxy workflow are kept only in the local ignored `SERVER_CONNECTIONS.md`; never copy its secrets into this document or public messages. The project-level operational rules in `AGENTS.md` remain authoritative.

## Suggested skills

- `diagnose` for blank/comma collapse, update-count comparisons and decoding regressions.
- `handoff` when transferring this state to another session.
- `visualize` for loss curves and side-by-side raw glyph generation.
- `improve-codebase-architecture` before adding resume or distributed-training abstractions.
