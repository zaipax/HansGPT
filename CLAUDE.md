# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Where to look first

- [`AGENTS.md`](AGENTS.md) is the operational rulebook: local/server split, uv, git/worktree, downloads, and the public-repo security boundary.
- [`handoff.md`](handoff.md) is the live experiment state (architecture, completed runs, active jobs, diagnoses, next goals). Read it before continuing training work and update it when a major experiment finishes.
- [`README.md`](README.md) names the current pretraining entry and indexes historical design/report docs. Do not treat those older reports as the active training protocol.

## Two products

1. **Web typesetter** (`index.html`, `styles.css`, `app.js`): zero-dependency 32×32 Chinese-character dot-matrix board. Open `index.html` in a browser; no build step.
2. **Research package** `hansgpt_research` (`src/hansgpt_research`, Hatchling, Python 3.12): native glyph language modeling and older frozen-LLM probes. Configs live under `configs/`; launchers, corpus builders, and GPU diagnostics live under `scripts/`; unit tests under `tests/`.

## Local vs server

Local Windows is source, docs, git, and static checks only. Do not download model weights, fonts, or research datasets locally. Do not `uv run` here if that would pull PyTorch CUDA wheels; the committed environment is CUDA 12.8 (`torch==2.8.0` from the explicit `pytorch-cu128` index in `pyproject.toml`).

The training server checkout is `/root/HansGPT`. Datasets, caches, and checkpoints stay out of git (`data/`, `artifacts/`, `/root/.cache/`). Never edit source on the server: local commit → `git push origin/main` → server `git pull --ff-only` → `uv sync --frozen` → run.

Connection details stay in the gitignored local file `SERVER_CONNECTIONS.md`. Refer to the host symbolically; never print or commit secrets.

## Common commands

### Local static checks

```powershell
uvx ruff check .
uvx ruff check --fix .
uvx ruff format --check .
uvx ruff format .
```

Ruff is configured in `pyproject.toml` (line length 100, `E,F,I,UP,B,SIM`).

### Server environment

```bash
uv sync --frozen
uv run python scripts/check_environment.py
uv run python scripts/check_environment.py --cpu
```

`xformers` is required by the current training path but is **not** declared in `pyproject.toml`. A strict `uv sync --frozen` can remove it. Restore `xformers==0.0.32.post2` (or `uv sync --frozen --inexact` when it is already present) before training. New C-byte entrypoints refuse to run without xFormers + compiled byte-head loss + PyTorch fused AdamW; they do not silently fall back.

This host needs PCIe NCCL fallbacks on multi-GPU jobs:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_HOST_ENABLE=0
```

### Tests (server `uv` env)

```bash
uv run pytest
uv run pytest tests/test_byte_training.py
uv run pytest tests/test_byte_training.py -k test_accelerated_loss_matches_joint_autoregressive_likelihood_and_gradients
```

### Current pretraining (server)

Formal C-byte training is `scripts/train_byte_multigpu.py`, not `python -m hansgpt_research.train_glyph_lm`. Learning rate is scheduled on the **full training-split valid-position horizon**; `target_tokens` is only a stop budget.

```bash
uv run torchrun --standalone --nproc_per_node=8 \
  scripts/train_byte_multigpu.py \
  --config configs/experiments/hansgpt_byte_c_eight_gpu_global_lr_100m.json

uv run torchrun --standalone --nproc_per_node=8 \
  scripts/train_byte_multigpu.py \
  --config configs/experiments/hansgpt_byte_c_eight_gpu_global_lr_100m.json \
  --smoke
```

`scripts/train_byte_glyph.py` and old constant-LR configs are historical throughput comparisons, not the formal schedule. Protocol: [`BYTE_C_EIGHT_GPU_GLOBAL_LR.md`](BYTE_C_EIGHT_GPU_GLOBAL_LR.md). Qwen3-shaped 1.5B C scaling: [`QWEN3_DENSE_C_1P5B.md`](QWEN3_DENSE_C_1P5B.md).

Raw glyph rollout (no gallery/OCR/font projection of generated grids):

```bash
uv run python scripts/infer_qwen3_checkpoint.py \
  --checkpoint artifacts/checkpoints/<run>/positions_<n>.pt \
  --output artifacts/reports/<eval_name> \
  --split validation
```

Corpus prep (server only; current packed document set is `data/processed/chinese_document_v3`):

```bash
uv run python -m hansgpt_research.prepare_corpus --config configs/datasets/modelscope_wikipedia_zh.json
uv run python scripts/prepare_document_corpus.py
uv run python scripts/verify_corpus.py --corpus data/processed/modelscope_zhwiki_full_v1
```

## Architecture

One outer language position is one 32×32 binary glyph, not a BPE token.

**Data contract.** Parquet corpora store uint16 *asset addresses*. `GlyphSequenceDataset` / `PackedGlyphSequenceDataset` resolve those IDs to pixels before a batch leaves the dataset. Model tensors are `[batch, seq, 1, 32, 32]` with uint8 values in `{0,1}`. Unicode, BPE IDs, and character-class labels never enter `inputs_embeds`. Control tokens (BOS/EOS/PAD) are also rendered glyphs. Generated grids are re-encoded by the same visual encoder; there is no glyph-bank lookup on the generation path.

**Packed language data.** Current training uses EOS-delimited GPT-style packing (`packed_glyph_data.py`) over `chinese_document_v3`. Windows are 1024 next-grid positions with one-input overlap; EOS→BOS loss is masked; attention is ordinary causal attention across EOS (not document-isolated). Packing short documents does not create long-range semantic supervision. Asset IDs retrieve bitmaps only.

**Current C stack** (`attention_glyph_lm.py` variant `C`):

- `AttentionGlyphEncoder`: bidirectional 4×4 patch Transformer (64 patches + a glyph token), no convolution and no character IDs.
- Causal GQA backbone: Hugging Face Llama (or Qwen3 attention when `qk_norm` is on) fed `inputs_embeds` only; vocab embeddings are unused.
- Glyph head: either `ConditionalByteDecoder` (128-step causal bytes, 256 classes per byte, unpacked MSB-first to 32×32) or `ParallelByteDecoder` (128 independent byte categoricals from learned queries). Byte classes are bit patterns, not Unicode.

Baseline C size in handoff is ~275M (hidden 1024, 24 layers, 16Q/4KV, FFN 2816, byte decoder width 256). Qwen3-style C is ~1.515B with the same encoder/decoder and a denser backbone.

**Training path.** `byte_training.py` is the accelerated C-byte loop: CPU `ByteCollator` unique-ifies input glyphs and packbits targets; `ByteBackward` encodes unique tiles, runs the eager backbone, then a `torch.compile`d byte-head loss in `head_chunk_size` slices. Compile covers the byte head, not the outer backbone (whole-backbone compile has hit Torch 2.8 CUTLASS / HF wrapper errors). `position_schedule.py` implements `global_cosine` over `schedule_total_positions` / `warmup_positions`. `distributed_sync.py` does weighted post-backward gradient reduction. `install_xformers()` in `cvae_fixed_step.py` routes SDPA through xFormers CUTLASS with no fallback.

**Loss and metrics.** Byte training reports joint byte CE as nats/pixel (`sum of 128 byte NLL / 1024`). Pixel accuracy is uninformative on sparse 32×32 grids. Record foreground F1, IoU, Dice, exact bitmap match, nearest-glyph / exact content-inventory membership, EOS, and repetition. Teacher-forced argmax is not generation; CVAE posterior reconstruction is not fluent prior generation.

**Older modules still in tree** (repro and tests; not the current pretraining entry): frozen Qwen hidden-state probes (`glyph_probe.py`, `Linear(hidden, 1024)`); CNN `GlyphGPT` pixel BCE (`glyph_lm.py`); structured mixture/GAN (`structured_glyph_lm.py`); ABC variants A/B; dual decoder; glyph codec; conditional VAE. `GPU_BY_VARIANT` in `attention_glyph_lm.py` is the original matched ABC GPU assignment, not the current multi-GPU launcher.

## Invariants

- Do not ingest Unicode/BPE/character IDs as model features.
- Keep character-disjoint splits when claiming compositional glyph reconstruction (probe-era eval). Packed C training is language modeling over glyph sequences; do not relabel it as character-disjoint reconstruction.
- Do not interpret same-character reconstruction on overlapping splits as compositional generalization.
- Do not choose a generation model from reconstruction metrics alone.
- Record git commit, config hashes, dataset checksums, seed, dtype, GPU, and library versions in experiment outputs.
- Smoke-test before long jobs; run long jobs in tmux; write logs/checkpoints under gitignored `artifacts/`.

## Git workflow

Feature, fix, and dependency work happens in a `codex/` (or named) worktree under `C:\Users\<username>\Desktop\worktree\HansGPT`, then rebase onto `main`, fast-forward merge, push `origin/main`, and remove the worktree. Full steps are in `AGENTS.md`. Documentation-only edits may stay in the current checkout.

On the server, if `git status --short` is not empty, stop. Do not discard remote changes as a shortcut, and do not train from uncommitted or detached code.
