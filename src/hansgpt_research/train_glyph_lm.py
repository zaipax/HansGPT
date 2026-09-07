"""Reproducible pixel-input/pixel-output GPT training, with separate smoke runs."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from hansgpt_research.glyph_lm import (
    GlyphGPT,
    GlyphSequenceDataset,
    ModelConfig,
    collate_glyph_sequences,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_data_readiness(
    data_dir: Path,
    *,
    require_full_snapshot: bool = False,
    expected_provider: str | None = None,
    expected_shards: int | None = None,
) -> dict:
    """Require a completed independent verification of the exact model-consumed files."""
    manifest_path, receipt_path = data_dir / "manifest.json", data_dir / "verification.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise ValueError("Dataset needs finalized manifest.json and successful verification.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest_hash = sha256(manifest_path)
    if receipt.get("passed") is not True or receipt.get("manifest_sha256") != manifest_hash:
        raise ValueError("Verification failed or belongs to a different dataset manifest")
    names = ["glyph_bank.npz", "glyph_inventory.json"]
    names.extend(
        f"{split}.{suffix}"
        for split in ("train", "validation", "test")
        for suffix in ("uint16", "offsets.npy")
    )
    if any(not (data_dir / name).is_file() for name in names):
        raise ValueError("Dataset export is incomplete: a model-consumed file is missing")
    hashes = {name: sha256(data_dir / name) for name in sorted(names)}
    if receipt.get("model_consumed_sha256") != hashes:
        raise ValueError("Model-consumed files differ from the successful verification receipt")
    if any(
        manifest.get("output_sha256", {}).get(name) != digest for name, digest in hashes.items()
    ):
        raise ValueError("Model-consumed files differ from the finalized export manifest")
    if require_full_snapshot:
        source, scan = manifest.get("source", {}), manifest.get("source_scan", {})
        files = scan.get("files", [])
        expected_rows, scanned_rows = scan.get("expected_rows"), scan.get("scanned_rows")
        complete = (
            manifest.get("status") == "full_snapshot_experiment_corpus"
            and scan.get("all_snapshot_shards_selected") is True
            and scan.get("all_selected_rows_scanned") is True
            and isinstance(expected_rows, int)
            and expected_rows > 0
            and scanned_rows == expected_rows
            and bool(files)
            and all(
                isinstance(file.get("expected_rows"), int)
                and file["expected_rows"] > 0
                and file.get("scanned_rows") == file["expected_rows"]
                and file.get("completed") is True
                for file in files
            )
        )
        if not complete:
            raise ValueError("Full training requires proved completion of the entire source scan")
        names_scanned = [file["file"] for file in files]
        source_files = source.get("files", [])
        if (
            len(set(names_scanned)) != len(names_scanned)
            or set(names_scanned) != {Path(file["path"]).name for file in source_files}
            or sum(file["expected_rows"] for file in files) != expected_rows
            or sum(file["scanned_rows"] for file in files) != scanned_rows
        ):
            raise ValueError("Full-source scan proof has inconsistent files or totals")
        if expected_provider is not None and source.get("provider") != expected_provider:
            raise ValueError("Full experiment source provider differs from the declared source")
        if expected_shards is not None and len(files) != expected_shards:
            raise ValueError("Full experiment did not scan the declared source shard count")
        if (
            expected_provider == "ModelScope"
            and expected_shards == 6
            and sorted(source.get("selected_shards", [])) != list(range(6))
        ):
            raise ValueError("Full experiment requires all six canonical ModelScope shards")
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_hash,
        "verification": receipt,
        "verification_sha256": sha256(receipt_path),
        "model_consumed_sha256": hashes,
    }


def runtime_metadata(
    config: dict, data_dir: Path, device: torch.device, *, mode: str = "full"
) -> dict:
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Training/evaluation requires a clean, committed checkout")
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    if not branch:
        raise RuntimeError("Do not run experiments from detached HEAD")
    requirements = config.get("data_requirements", {})
    readiness = verify_data_readiness(
        data_dir,
        require_full_snapshot=mode == "full",
        expected_provider=requirements.get("full_source_provider", "ModelScope"),
        expected_shards=requirements.get("full_source_shards", 6),
    )
    manifests = {
        "manifest.json": {"sha256": readiness["manifest_sha256"], "content": readiness["manifest"]},
        "verification.json": {
            "sha256": readiness["verification_sha256"],
            "content": readiness["verification"],
        },
    }
    for name in ("data_card.json", "glyph_inventory.json"):
        path = data_dir / name
        if path.exists():
            manifests[name] = {"sha256": sha256(path), "content": json.loads(path.read_text())}
    versions = {}
    for name in ("torch", "numpy", "pillow", "transformers"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_branch": branch,
        "config": config,
        "config_sha256": canonical_hash(config),
        "data_dir": str(data_dir),
        "data_sha256": readiness["model_consumed_sha256"],
        "data_verification_sha256": readiness["verification_sha256"],
        "data_manifest_sha256": readiness["manifest_sha256"],
        "data_manifests": manifests,
        "model_revision": "random initialization; architecture pinned by git_commit",
        "tokenizer_revision": "none; one binary 32x32 glyph is one sequence position",
        "versions": versions,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": config["training"]["seed"],
        "precision": config["training"]["precision"],
    }


def autocast_context(device: torch.device, precision: str):
    if device.type == "cuda" and precision in {"fp16", "bf16"}:
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def loss_sum(logits: torch.Tensor, batch: dict) -> tuple[torch.Tensor, int]:
    losses = F.binary_cross_entropy_with_logits(
        logits.float(), batch["targets"].float(), reduction="none"
    )
    per_tile = losses.flatten(2).mean(-1)
    mask = batch["loss_mask"].bool()
    return per_tile.masked_select(mask).sum(), int(mask.sum().item())


class EpochSampler(Sampler[int]):
    """Rebuild an epoch permutation; cursor refers to consumed, never prefetched samples."""

    def __init__(self, length: int, seed: int, epoch: int, cursor: int = 0):
        if not 0 <= cursor <= length:
            raise ValueError("Sampler cursor lies outside the epoch")
        self.length, self.seed, self.epoch, self.cursor = length, seed, epoch, cursor

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.length, generator=generator).tolist()
        return iter(indices[self.cursor :])

    def __len__(self):
        return self.length - self.cursor


def sequence_lengths(dataset: GlyphSequenceDataset) -> np.ndarray:
    """Derive every chunk's valid length without reading or rendering any pixels."""
    lengths = np.full(len(dataset), dataset.sequence_length, dtype=np.int32)
    final_indices = dataset.chunk_offsets[1:] - 1
    lengths[final_indices] = (np.diff(dataset.offsets) - 2) % dataset.sequence_length + 1
    if int(lengths.sum()) != dataset.target_count:
        raise ValueError("Chunk lengths do not cover the dataset's valid targets")
    return lengths


class SortishEpochSampler(EpochSampler):
    """Randomized length pools and shuffled full batches, with an exact consumed cursor.

    Every epoch uses every chunk once. The one partial batch stays last so a flat
    DataLoader sampler cannot regroup a partial batch with the next full batch.
    No source document is packed or joined with another document.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        seed: int,
        epoch: int,
        batch_size: int,
        cursor: int = 0,
        pool_batches: int = 64,
    ):
        super().__init__(len(lengths), seed, epoch, cursor)
        if batch_size <= 0 or pool_batches <= 0 or np.any(lengths <= 0):
            raise ValueError("Sortish sampler requires positive lengths and pool/batch sizes")
        self.lengths, self.batch_size, self.pool_batches = lengths, batch_size, pool_batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        random_order = torch.randperm(self.length, generator=generator).numpy()
        pool_size = self.batch_size * self.pool_batches
        batches, partial = [], np.empty(0, dtype=np.int64)
        for start in range(0, self.length, pool_size):
            pool = random_order[start : start + pool_size]
            ordered = pool[np.argsort(-self.lengths[pool], kind="stable")]
            full_end = len(ordered) // self.batch_size * self.batch_size
            batches.extend(ordered[:full_end].reshape(-1, self.batch_size))
            if full_end != len(ordered):
                partial = ordered[full_end:]
        batch_order = torch.randperm(len(batches), generator=generator).tolist()
        indices = np.concatenate([*(batches[index] for index in batch_order), partial])
        return iter(indices[self.cursor :].tolist())


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, model, optimizer, scaler, progress, metadata) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "progress": progress,
            "rng": rng_state(),
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(path)


def learning_rate(tokens: int, cfg: dict) -> float:
    warmup = cfg["warmup_tokens"]
    if tokens < warmup:
        return cfg["learning_rate"] * max(tokens, 1) / max(warmup, 1)
    fraction = min(1.0, (tokens - warmup) / max(cfg["target_tokens"] - warmup, 1))
    floor = cfg["minimum_learning_rate_ratio"]
    return cfg["learning_rate"] * (floor + (1 - floor) * (1 + math.cos(math.pi * fraction)) / 2)


def train(args: argparse.Namespace) -> None:
    # Import here so evaluation can reuse the runtime helpers without a cycle.
    from hansgpt_research.evaluate_glyph_lm import validate

    config = json.loads(Path(args.config).read_text())
    cfg = config["training"]
    for name in (
        "target_tokens",
        "max_epochs",
        "batch_size",
        "sequence_length",
        "gradient_accumulation_steps",
        "num_workers",
        "sampler",
    ):
        value = getattr(args, name, None)
        if value is not None:
            cfg[name] = value
    if args.mode != "full":
        cfg["target_tokens"] = args.smoke_tokens
        cfg["warmup_tokens"] = min(cfg["warmup_tokens"], max(args.smoke_tokens // 10, 1))
        cfg["validate_every_tokens"] = args.smoke_tokens
    if cfg["target_tokens"] <= 0 or cfg["max_epochs"] <= 0:
        raise ValueError("Positive declared token and epoch budgets are required")
    if args.mode == "full" and "smoke" in args.run_name.lower():
        raise ValueError("Name full training independently of a smoke run")
    if args.mode != "full" and args.mode not in args.run_name.lower():
        raise ValueError("Smoke/benchmark run names must include their mode")
    device = torch.device(args.device)
    if device.type == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("This experiment is authorized on GPU0: set CUDA_VISIBLE_DEVICES=0")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    torch.set_float32_matmul_precision("highest")
    data_dir = Path(args.data)
    log_dir = Path("artifacts/logs") / args.run_name
    checkpoint_dir = Path("artifacts/checkpoints") / args.run_name
    if not args.resume and (log_dir.exists() or checkpoint_dir.exists()):
        raise FileExistsError("Use a new run name, or explicitly resume its checkpoint")
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = runtime_metadata(config, data_dir, device, mode=args.mode)
    metadata["mode"] = args.mode
    metadata["run_name"] = args.run_name
    dataset = GlyphSequenceDataset(data_dir, "train", cfg["sequence_length"])
    validation = GlyphSequenceDataset(data_dir, "validation", cfg["sequence_length"])
    lengths = sequence_lengths(dataset)
    sampler_kind = cfg.get("sampler", "random")
    if sampler_kind not in {"random", "sortish"}:
        raise ValueError("Sampler must be random or sortish")
    metadata["sampler"] = {
        "algorithm": sampler_kind,
        "version": 1,
        "pool_batches": cfg.get("sortish_pool_batches", 64),
    }
    model = GlyphGPT(ModelConfig.from_dict(config["model"])).to(device)
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    metadata["parameters"] = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["learning_rate"],
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and cfg["precision"] == "fp16"
    )
    progress = {
        "epoch": 0,
        "cursor": 0,
        "optimizer_steps": 0,
        "valid_tokens": 0,
        "attempted_tokens": 0,
        "overflow_steps": 0,
        "best_validation_bce": None,
        "best_step": 0,
        "best_tokens": 0,
        "best_threshold": 0.5,
        "validations_without_improvement": 0,
        "last_validation_tokens": 0,
        "training_seconds": 0.0,
    }
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in (
            "config_sha256",
            "data_sha256",
            "data_manifest_sha256",
            "sampler",
            "run_name",
            "mode",
        ):
            if saved["metadata"][key] != metadata[key]:
                raise ValueError(f"Resume identity mismatch: {key}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        progress = saved["progress"]
        restore_rng(saved["rng"])
        metadata["resume_from"] = {
            "checkpoint_sha256": sha256(Path(args.resume)),
            "original_git_commit": saved["metadata"]["git_commit"],
        }
    write_json(log_dir / "metadata.json", metadata)
    write_json(log_dir / "status.json", {"status": "running", "progress": progress})
    started = time.monotonic()
    previous_seconds = progress["training_seconds"]

    def log(record: dict) -> None:
        record["time"] = datetime.now(UTC).isoformat()
        with (log_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def checkpoint(name: str) -> None:
        progress["training_seconds"] = previous_seconds + time.monotonic() - started
        save_checkpoint(checkpoint_dir / name, model, optimizer, scaler, progress.copy(), metadata)

    def validate_and_save() -> bool:
        metrics = validate(
            model,
            validation,
            config,
            device,
            max_batches=args.smoke_validation_batches if args.mode != "full" else None,
        )
        value = metrics["bce_per_pixel"]
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite validation loss")
        previous = progress["best_validation_bce"]
        improved = previous is None or value < previous - cfg["early_stopping_min_delta"]
        if improved:
            progress.update(
                best_validation_bce=value,
                best_step=progress["optimizer_steps"],
                best_tokens=progress["valid_tokens"],
                best_threshold=metrics["selected_threshold"],
                validations_without_improvement=0,
            )
            checkpoint("best.pt")
        else:
            progress["validations_without_improvement"] += 1
        progress["last_validation_tokens"] = progress["valid_tokens"]
        log({"kind": "validation", **progress, "metrics": metrics})
        checkpoint("latest.pt")
        model.train()
        patience = cfg["early_stopping_patience"]
        return bool(
            patience
            and progress["valid_tokens"] >= cfg["early_stopping_min_tokens"]
            and progress["validations_without_improvement"] >= patience
        )

    stop_reason = "epoch_budget"
    overflow_streak = 0
    try:
        if progress["best_validation_bce"] is None:
            validate_and_save()
        if progress["valid_tokens"] >= cfg["target_tokens"]:
            stop_reason = "token_budget"
        while progress["epoch"] < cfg["max_epochs"] and stop_reason == "epoch_budget":
            sampler = (
                SortishEpochSampler(
                    lengths,
                    cfg["seed"],
                    progress["epoch"],
                    cfg["batch_size"],
                    progress["cursor"],
                    cfg.get("sortish_pool_batches", 64),
                )
                if sampler_kind == "sortish"
                else EpochSampler(len(dataset), cfg["seed"], progress["epoch"], progress["cursor"])
            )
            # Its dedicated generator keeps worker startup from perturbing dropout RNG after resume.
            loader = DataLoader(
                dataset,
                batch_size=cfg["batch_size"],
                sampler=sampler,
                num_workers=cfg["num_workers"],
                pin_memory=device.type == "cuda",
                generator=torch.Generator().manual_seed(cfg["seed"]),
                collate_fn=collate_glyph_sequences,
            )
            iterator = iter(loader)
            exhausted = False
            while not exhausted:
                group = []
                for _ in range(cfg["gradient_accumulation_steps"]):
                    try:
                        group.append(next(iterator))
                    except StopIteration:
                        exhausted = True
                        break
                if not group:
                    break
                tokens = sum(int(batch["loss_mask"].sum()) for batch in group)
                if not tokens:
                    raise ValueError("A training update contains no supervised targets")
                lr = learning_rate(progress["valid_tokens"] + tokens, cfg)
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = 0.0
                update_started = time.monotonic()
                for cpu_batch in group:
                    batch = move_batch(cpu_batch, device)
                    with autocast_context(device, cfg["precision"]):
                        logits = model(batch["glyphs"], attention_mask=batch["attention_mask"])
                        total_loss, _ = loss_sum(logits, batch)
                    if not torch.isfinite(total_loss):
                        raise FloatingPointError("Nonfinite training loss; checkpoint preserved")
                    scaler.scale(total_loss / tokens).backward()
                    accumulated_loss += total_loss.detach().item()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                overflow = scaler.get_scale() < old_scale
                if not scaler.is_enabled() and not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradients with no enabled GradScaler")
                progress["cursor"] += sum(batch["glyphs"].shape[0] for batch in group)
                progress["attempted_tokens"] += tokens
                if overflow:
                    progress["overflow_steps"] += 1
                    overflow_streak += 1
                    log({"kind": "overflow", "scale": scaler.get_scale(), **progress})
                    if overflow_streak >= cfg["max_consecutive_overflows"]:
                        raise FloatingPointError("Repeated FP16 overflows; aborting failed run")
                    continue
                overflow_streak = 0
                progress["optimizer_steps"] += 1
                progress["valid_tokens"] += tokens
                if (
                    progress["optimizer_steps"] == 1
                    or progress["optimizer_steps"] % cfg["log_every_steps"] == 0
                    or progress["valid_tokens"] >= cfg["target_tokens"]
                ):
                    log(
                        {
                            "kind": "train",
                            "bce_per_pixel": accumulated_loss / tokens,
                            "learning_rate": lr,
                            "grad_norm": float(norm),
                            "tokens_per_second": tokens / (time.monotonic() - update_started),
                            "valid_position_fraction": tokens
                            / sum(batch["loss_mask"].numel() for batch in group),
                            "grad_scaler_scale": scaler.get_scale(),
                            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
                            if device.type == "cuda"
                            else None,
                            **progress,
                        }
                    )
                if progress["valid_tokens"] >= cfg["target_tokens"]:
                    stop_reason = "token_budget"
                    break
                if (
                    progress["valid_tokens"] - progress["last_validation_tokens"]
                    >= cfg["validate_every_tokens"]
                    and validate_and_save()
                ):
                    stop_reason = "validation_early_stopping"
                    break
                if progress["optimizer_steps"] % cfg["checkpoint_every_steps"] == 0:
                    checkpoint("latest.pt")
                    write_json(log_dir / "status.json", {"status": "running", "progress": progress})
            if stop_reason != "epoch_budget":
                break
            progress["epoch"] += 1
            progress["cursor"] = 0
            checkpoint("latest.pt")
        if progress["last_validation_tokens"] != progress["valid_tokens"]:
            validate_and_save()
        checkpoint("final.pt")
        completion = {
            "status": "complete",
            "mode": args.mode,
            "stop_reason": stop_reason,
            "progress": progress,
            "metadata_sha256": sha256(log_dir / "metadata.json"),
            "best_checkpoint_sha256": sha256(checkpoint_dir / "best.pt"),
            "final_checkpoint_sha256": sha256(checkpoint_dir / "final.pt"),
            "evaluation_complete": False,
        }
        write_json(log_dir / "training_complete.json", completion)
        write_json(log_dir / "status.json", completion)
        log({"kind": "training_complete", **completion})
    except BaseException as error:
        # Last completed update remains resumable. Never save half-accumulated optimizer state.
        write_json(
            log_dir / "status.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "progress": progress,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/hansgpt_binary_v1.json")
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=("full", "smoke", "benchmark"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--sampler", choices=("random", "sortish"))
    parser.add_argument("--smoke-tokens", type=int, default=32768)
    parser.add_argument("--smoke-validation-batches", type=int, default=2)
    for name in (
        "target-tokens",
        "max-epochs",
        "batch-size",
        "sequence-length",
        "gradient-accumulation-steps",
        "num-workers",
    ):
        parser.add_argument("--" + name, type=int)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
