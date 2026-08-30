# HansGPT agent workflow

## Security boundary

This repository is public. Never commit or print server passwords, tokens, private keys, SSH connection strings, private endpoints, or secret-bearing `.env` files.

- Remote host, port, and user are supplied out of band. Refer to them as `HANSGPT_REMOTE_HOST`, `HANSGPT_REMOTE_PORT`, and `HANSGPT_REMOTE_USER`.
- Do not put passwords on a command line, in a script, in Git configuration, or in shell history. Use an interactive password prompt or an approved secret manager.
- Do not install SSH keys or create persistent access unless the user explicitly authorizes it.
- Before adding a file, assume anything committed will be visible publicly.

## Local and remote responsibilities

The local Windows machine is for source-code and documentation development only.

Local machine responsibilities:

- Edit source code and documentation.
- Run lightweight static checks that do not require the research Python environment, model weights, datasets, or GPU training.
- Commit, rebase, merge, and push changes to the remote Git repository.
- Never download model weights or research datasets locally.
- Never create the training environment locally and then copy it to the server.
- Never transfer models or datasets from local to remote with SCP or another file-copy workflow.

Training server responsibilities:

- Pull committed code from the remote Git repository.
- Create and synchronize the project `uv` environment.
- Download all model weights, fonts, Unicode data, and Chinese corpora directly from their original sources.
- Run preprocessing, hidden-state extraction, training, evaluation, and GPU tests.
- Store datasets, caches, checkpoints, and experiment artifacts outside Git tracking.
- Do not edit project source code on the training server. Fix code locally, push it, and pull it on the server.

The canonical flow is always:

```text
local worktree edit
  → local verification
  → commit/rebase/merge main
  → push origin/main
  → server git pull --ff-only
  → server uv sync --frozen
  → server run or train
```

## Python and uv rules

All Python work uses `uv`. Do not use system `pip`, `pip install`, Conda, Poetry, Pipenv, or an ad-hoc virtual environment.

- The project Python version is declared in `.python-version` and `pyproject.toml`.
- Resolve dependencies with `uv lock` after changing `pyproject.toml`.
- Install the committed environment on the server with `uv sync --frozen`.
- Run Python programs with `uv run python ...`.
- Run Python tools with `uv run ...`.
- For one-off non-project Python tools, use `uvx`.
- PyTorch comes from the explicit CUDA 12.8 index declared in `pyproject.toml`.
- Do not install a system CUDA Toolkit merely because `nvcc` is absent. PyTorch wheels carry their CUDA runtime; the NVIDIA driver is the required host dependency.

## Local worktree workflow

Any feature, fix, documentation change, dependency change, or other repository mutation must be developed outside `main`.

If the current directory is not a Git repository, initialize it and add an appropriate `.gitignore` before development.

1. Create a branch with the `codex/` prefix and a worktree under:

   ```text
   C:\Users\Faker\Desktop\worktree\<project-name>
   ```

2. Enter the worktree and confirm:

   ```bash
   git rev-parse --show-toplevel
   git branch --show-current
   git status --short
   ```

3. Make changes only in the worktree branch.
4. Preserve unrelated user changes and never use destructive reset/checkout commands.
5. Run checks appropriate to the change. Python/GPU integration checks run on the server, not locally.
6. Commit the worktree branch.
7. Rebase the branch onto local `main`.
8. Fast-forward merge into `main`, commit if required, and push `origin/main`.
9. After a successful push, verify the exact worktree path, remove the worktree, and delete the merged branch.

## Server clone and update workflow

The canonical server checkout is:

```text
/root/HansGPT
```

First clone on the training server:

```bash
git clone https://github.com/zaipax/HansGPT.git /root/HansGPT
cd /root/HansGPT
git branch --show-current
git status --short
uv sync --frozen
uv run python scripts/check_environment.py
```

Before every subsequent server run:

```bash
cd /root/HansGPT
git status --short
git fetch origin
git switch main
git pull --ff-only origin main
uv sync --frozen
uv run python scripts/check_environment.py
```

If `git status --short` is not empty on the server, stop and inspect it. Do not discard, overwrite, or commit remote changes as a shortcut.

## Remote execution rules

- Use short, reproducible commands and record the Git commit in every experiment result.
- Run a smoke test before a long preprocessing or training job.
- Use `tmux` or another server-side session manager for long jobs after confirming it is installed.
- Save logs under `artifacts/logs/` and checkpoints under `artifacts/checkpoints/`; both are ignored by Git.
- Record model revision, tokenizer revision, dataset checksums, font checksum, random seed, dtype, GPU, PyTorch version, Transformers version, and Git commit.
- Do not run training from an uncommitted or detached code state.
- Do not modify or upgrade the NVIDIA driver, WSL kernel, OS packages, or system CUDA installation without explicit user authorization.

## Model and dataset download rules

All downloads happen on the training server after the repository is current.

Recommended server paths:

```text
/root/HansGPT/data/raw/            research source data
/root/HansGPT/data/interim/        extracted and cleaned intermediates
/root/HansGPT/data/processed/      generated Parquet/bitmap datasets
/root/HansGPT/artifacts/           features, logs, checkpoints, reports
/root/.cache/huggingface/          Hugging Face model cache
/root/.cache/uv/                   uv package cache
```

- Use resumable downloads and verify official checksums when available.
- Download Hugging Face models with `uv run hf download ...` or project Python code using `huggingface_hub`.
- Pin model revisions before formal experiments.
- Download Unihan, Noto Sans CJK SC, and Chinese Wikipedia directly from the official URLs in `RESEARCH_PLAN.md`.
- Never commit raw data, model weights, hidden-state caches, checkpoints, `.part` files, or extracted dumps.
- Do not silently switch to an unofficial mirror. If an official source is unavailable, report the failure before changing the source.

## Research invariants

- The pretrained model backbone, embeddings, and intermediate layers remain frozen for the primary experiment.
- The primary trainable module is `Linear(hidden_size, 1024)`.
- Qwen3.5-4B-Base is the primary model; Qwen3-4B-Base is the text-only control; Qwen3.5-2B-Base is the development model.
- The 32×32 binary target is generated deterministically from the pinned Noto Sans CJK SC Regular font.
- Character-disjoint train/validation/test splits are mandatory for the main claim.
- Report foreground F1, IoU, Dice, exact bitmap match, and nearest-glyph retrieval; plain pixel accuracy is not sufficient.
- Keep native-tokenizer and shared single-token intersection results separate.
- Never interpret same-character reconstruction on a character-overlapping split as evidence of compositional generalization.

