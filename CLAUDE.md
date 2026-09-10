# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview & Architecture

HansGPT consists of two complementary components:

1. **Web Interface (Zero-Dependency Typesetter)**:
   - `index.html`, `styles.css`, `app.js`: Real-time browser-based 32×32 Chinese character dot-matrix typesetting board.
   - Converts input text dynamically onto individual 32×32 HTML5 Canvas tiles (20 characters per line, responsive wrap, incremental redraw).
   - Operates with zero build steps or external dependencies.

2. **Research Pipeline (`hansgpt_research` Python Package)**:
   - **Frozen LLM Glyph Probes** (`glyph_probe.py`, `diagnostics.py`):
     - Probes whether frozen causal language models (Qwen3.5-2B/4B, Qwen3-4B) preserve linearly decodable Chinese character glyphs in intermediate hidden states via `Linear(hidden_size, 1024)`.
   - **Native Autoregressive Binary Glyph LM** (`glyph_lm.py`, `structured_glyph_lm.py`, `byte_glyph_decoder.py`):
     - Predicts next 32×32 binary pixel tiles (`[batch, seq, 1, 32, 32]`, uint8 0/1) without BPE tokens, Unicode embeddings, or discrete character classification.
     - Architecture: Shared visual CNN (`GlyphEncoder`) encodes 32×32 bitmaps → Causal Transformer backbone (Llama-style with RoPE) → Next-tile pixel predictor (pixel BCE, structured Bernoulli/Categorical distributions, conditional GAN discriminator, or autoregressive byte-level decoder).
   - **Corpus & Bitmap Preparation** (`prepare_corpus.py`, `verify_corpus.py`):
     - Downloads and cleans Chinese text (ModelScope zhwiki), normalizes with OpenCC `t2s`, filters characters against a strict Chinese punctuation and character whitelist (filtering out Latin, digits, and emojis), and deterministically renders 32×32 binary bitmaps using Noto Sans CJK SC Regular into deduplicated Parquet dataset splits.
   - **Evaluation & Diagnostics** (`evaluate_glyph_lm.py`, `evaluate_structured_glyph_lm.py`, `diagnose_glyph_generation.py`, `diagnose_glyph_reconstruction.py`):
     - Enforces character-disjoint train/val/test splits.
     - Evaluates with foreground F1, IoU, Dice, exact bitmap match, and nearest-glyph character inventory retrieval (raw pixel accuracy is misleading due to background sparsity).

## Local vs. Server Execution Boundary

- **Local Machine (Windows 11)**:
  - Source code, documentation, configuration, and git operations only.
  - Run static analysis via `uvx ruff check .` and `uvx ruff format --check .`.
  - **Do NOT run `uv run ...` locally if it triggers downloading heavy CUDA dependencies (e.g. PyTorch cu128 wheels).** Local machine does not have a system Python or CUDA runtime.
  - **Never download model weights, fonts, or raw research datasets locally.**
- **Training Server (Linux + GPU, e.g. CUDA 12.8 / RTX 5060 Ti / V100S)**:
  - Canonical repository location: `/root/HansGPT`.
  - All dataset downloads, preprocessing, hidden-state extraction, training, GPU evaluations, and integration tests run here.
  - Datasets and checkpoints are kept outside git in `data/`, `artifacts/`, and `/root/.cache/`.
  - Never edit source files directly on the server; code changes flow via `local git push -> server git pull --ff-only`.

## Common Commands

### Local Static Checks (Windows)
```powershell
# Linting
uvx ruff check .
uvx ruff check --fix .

# Formatting
uvx ruff format --check .
uvx ruff format .
```

### Server Setup & Verification
```bash
# Sync committed uv environment
uv sync --frozen

# Verify environment, CUDA availability, and precision support
uv run python scripts/check_environment.py
# Or check CPU-only preprocessing mode
uv run python scripts/check_environment.py --cpu
```

### Running Tests (Server uv Environment)
```bash
# Run all unit and integration tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_glyph_lm.py

# Run a single test function
uv run pytest tests/test_glyph_lm.py -k test_glyph_gpt_forward_shape
```

### Research Pipeline Execution (Server)
```bash
# Prepare Chinese corpus and binary glyph datasets
uv run python -m hansgpt_research.prepare_corpus --config configs/datasets/modelscope_wikipedia_zh.json

# Verify corpus integrity and character rendering
uv run python scripts/verify_corpus.py --corpus data/processed/modelscope_zhwiki_full_v1

# Train base GlyphGPT (v1)
uv run python -m hansgpt_research.train_glyph_lm --config configs/experiments/hansgpt_binary_v1.json

# Train structured GlyphGPT (v2: BCE / mixture / GAN)
uv run python -m hansgpt_research.train_structured_glyph_lm --config configs/experiments/hansgpt_binary_v2_mixture4.json

# Run generation diagnostics on a checkpoint
uv run python -m hansgpt_research.diagnose_glyph_generation \
    --checkpoint artifacts/checkpoints/hansgpt_binary_v1/best.pt \
    --data data/processed/modelscope_zhwiki_full_v1 \
    --output artifacts/reports/d0_diagnostics \
    --split validation --prompt-count 4 --prompt-lengths 16 --thresholds 0.45 --max-new 8

# Execute end-to-end v2 experimental pipeline (smoke / diagnose / train / evaluate)
bash scripts/run_binary_v2_round.sh round1 smoke
```

### Web Interface
- Open `index.html` directly in any web browser. No server or compilation required.

## Key Research Invariants & Code Conventions

- **Input/Output Purity**: In `GlyphGPT` and `StructuredGlyphGPT`, input tensors must be shape `[batch, seq, 1, 32, 32]` with uint8 values restricted to 0 or 1. Model inputs must never ingest Unicode codepoints, BPE token IDs, or character categories.
- **Evaluation Standards**: Because background pixels dominate 32×32 grids, standard accuracy is uninformative. Always record foreground F1, IoU, Dice, exact bitmap match rate, and nearest-glyph retrieval rank.
- **Generalization Claims**: Any evaluation measuring compositional generalization or character reconstruction must maintain strictly character-disjoint splits.
- **Security Boundary**: This repository is public. Never commit server hostnames, IP addresses, SSH ports, credentials, keys, private connection commands, or `.env` files. `SERVER_CONNECTIONS.md` is strictly local and gitignored.

## Worktree & Git Workflow

All feature work, bug fixes, dependency updates, and documentation changes must be developed in a worktree off `main`:

1. Create a worktree under `C:\Users\Faker\Desktop\worktree\<project-name>` on a branch prefixed with `codex/` or feature name:
   ```bash
   git worktree add -b codex/<branch-name> C:\Users\Faker\Desktop\worktree\HansGPT main
   ```
2. Enter the worktree and verify clean state:
   ```bash
   git rev-parse --show-toplevel
   git branch --show-current
   git status --short
   ```
3. Run local static checks (`uvx ruff check .`, `uvx ruff format --check .`).
4. Commit changes in the worktree branch.
5. Rebase onto `main`, fast-forward merge into `main`, and push to `origin/main`.
6. Clean up: remove the worktree and delete the feature branch:
   ```bash
   git worktree remove C:\Users\Faker\Desktop\worktree\HansGPT
   git branch -d codex/<branch-name>
   ```
