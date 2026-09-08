"""Train binary glyph mixture/GAN heads with explicit update and resume accounting."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from hansgpt_research.glyph_lm import (
    GlyphSequenceDataset,
    ModelConfig,
    collate_glyph_sequences,
)
from hansgpt_research.structured_glyph_lm import (
    ConditionalGlyphDiscriminator,
    StructuredGlyphGPT,
    hard_binary_st,
)
from hansgpt_research.train_glyph_lm import (
    EpochSampler,
    SortishEpochSampler,
    autocast_context,
    learning_rate,
    move_batch,
    restore_rng,
    rng_state,
    runtime_metadata,
    sequence_lengths,
    sha256,
    write_json,
)

CHECKPOINT_FORMAT = "structured_glyph_lm_v2"
IDENTITY_KEYS = (
    "config_sha256",
    "data_sha256",
    "data_manifest_sha256",
    "data_verification_sha256",
    "init_checkpoint_sha256",
    "git_commit",
    "source_sha256",
    "sampler",
    "validation_selection",
    "run_name",
    "mode",
    "model_family",
    "config_file_sha256",
)


def effective_config(config: dict, *, mode: str, smoke_tokens: int) -> dict:
    config = json.loads(json.dumps(config))
    cfg = config["training"]
    defaults = {
        "seed": 20260907,
        "precision": "fp16",
        "max_epochs": 1000,
        "gradient_accumulation_steps": 1,
        "num_workers": 2,
        "sampler": "sortish",
        "sortish_pool_batches": 64,
        "warmup_tokens": 100000,
        "minimum_learning_rate_ratio": 0.1,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "head_chunk_size": 256,
        "freeze_backbone": True,
        "unfreeze_after_tokens": None,
        "adversarial_weight": 0.0,
        "discriminator_lr": 0.0001,
        "discriminator_weight_decay": 0.0,
        "discriminator_channels": 32,
        "threshold": 0.5,
        "gan_component_strategy": "sample_threshold",
        "validation_samples": 2048,
        "validation_batch_size": cfg["batch_size"],
        "validate_every_tokens": 1000000,
        "checkpoint_every_steps": 200,
        "log_every_steps": 10,
        "gradient_checkpointing": True,
        "max_consecutive_overflows": 20,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    config.setdefault("head", {})
    config["head"].setdefault("components", 1)
    config["head"].setdefault("init_noise", 0.001)
    if mode == "smoke":
        cfg["target_tokens"] = smoke_tokens
        cfg["warmup_tokens"] = min(cfg["warmup_tokens"], max(1, smoke_tokens // 10))
        cfg["validate_every_tokens"] = smoke_tokens
    for key in (
        "target_tokens",
        "max_epochs",
        "batch_size",
        "sequence_length",
        "gradient_accumulation_steps",
        "head_chunk_size",
        "validation_samples",
        "validation_batch_size",
        "sortish_pool_batches",
        "validate_every_tokens",
        "checkpoint_every_steps",
        "log_every_steps",
        "max_consecutive_overflows",
    ):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["num_workers"] < 0 or cfg["warmup_tokens"] < 0:
        raise ValueError("Worker count and warmup tokens cannot be negative")
    if cfg["precision"] not in {"fp16", "bf16", "fp32"}:
        raise ValueError("Unsupported training precision")
    if cfg["sampler"] not in {"sortish", "random"}:
        raise ValueError("Sampler must be sortish or random")
    if not isinstance(cfg["freeze_backbone"], bool):
        raise ValueError("freeze_backbone must be boolean")
    unfreeze = cfg["unfreeze_after_tokens"]
    if unfreeze is not None and (not isinstance(unfreeze, int) or unfreeze < 0):
        raise ValueError("unfreeze_after_tokens must be null or a nonnegative integer")
    if not 0 < cfg["threshold"] < 1 or cfg["adversarial_weight"] < 0:
        raise ValueError("Invalid threshold or adversarial weight")
    if cfg["learning_rate"] <= 0 or cfg["discriminator_lr"] <= 0 or cfg["max_grad_norm"] <= 0:
        raise ValueError("Learning rates and gradient clip must be positive")
    if cfg["gan_component_strategy"] not in {"mode_threshold", "sample_threshold"}:
        raise ValueError("GAN component selection must preserve hard threshold pixels")
    if cfg["sequence_length"] > config["model"]["max_position_embeddings"]:
        raise ValueError("Training context exceeds the configured model capacity")
    return config


def frozen_at(cfg: dict, valid_tokens: int) -> bool:
    transition = cfg["unfreeze_after_tokens"]
    return cfg["freeze_backbone"] and (transition is None or valid_tokens < transition)


def configure_backbone(model, *, frozen: bool) -> None:
    for module in (model.glyph_encoder, model.backbone):
        module.requires_grad_(not frozen)
        module.train(model.training and not frozen)


@contextlib.contextmanager
def fixed_discriminator(discriminator):
    """Freeze both parameters and training buffers while retaining image-input gradients."""
    flags = [parameter.requires_grad for parameter in discriminator.parameters()]
    was_training = discriminator.training
    discriminator.requires_grad_(False)
    discriminator.eval()
    try:
        yield
    finally:
        for parameter, flag in zip(discriminator.parameters(), flags, strict=True):
            parameter.requires_grad_(flag)
        discriminator.train(was_training)


def valid_hidden(model, batch: dict, *, frozen: bool):
    with torch.no_grad() if frozen else contextlib.nullcontext():
        hidden = model.forward_hidden(batch["glyphs"], attention_mask=batch["attention_mask"])
    mask = batch["loss_mask"].bool()
    return hidden[mask], batch["targets"][mask]


def complete_optimizer_step(optimizer, scaler, parameters, max_grad_norm: float) -> dict:
    """An AMP skip is a distinct outcome; unscaled nonfinite gradients never update weights."""
    parameters = list(parameters)
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    finite = bool(torch.isfinite(norm))
    if not scaler.is_enabled() and not finite:
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Nonfinite gradients without an enabled AMP scaler")
    if not finite and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    ):
        # AMP checks individual entries before clipping. A finite vector can still
        # overflow its norm; clipping may then zero every entry without setting
        # AMP's found-inf flag. Abort before momentum/decay can mutate any weights.
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Nonfinite aggregate gradient norm with finite entries")
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    succeeded = scaler.get_scale() >= previous_scale
    return {
        "succeeded": succeeded,
        "grad_norm": float(norm) if finite else None,
        "scale": scaler.get_scale(),
    }


def discriminator_backward(model, discriminator, group, cfg, device, scaler, tokens, *, frozen):
    """Accumulate D only; both real conditions and generated hard pixels are detached."""
    discriminator.train()
    total_loss = 0.0
    cache = [] if frozen else None
    for cpu_batch in group:
        batch = move_batch(cpu_batch, device)
        with torch.no_grad(), autocast_context(device, cfg["precision"]):
            hidden, targets = valid_hidden(model, batch, frozen=True)
        if cache is not None:
            cache.append((hidden.detach(), targets.detach()))
        for start in range(0, len(hidden), cfg["head_chunk_size"]):
            stop = start + cfg["head_chunk_size"]
            condition, real = hidden[start:stop].detach(), targets[start:stop].detach()
            with torch.no_grad(), autocast_context(device, cfg["precision"]):
                distribution = model.distribution(condition)
                fake = distribution.decode(
                    threshold=cfg["threshold"], strategy=cfg["gan_component_strategy"]
                ).detach()
            with autocast_context(device, cfg["precision"]):
                real_score = discriminator(real, condition).float()
                fake_score = discriminator(fake, condition).float()
                loss = (F.softplus(-real_score) + F.softplus(fake_score)).sum() / 2
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite discriminator loss")
            scaler.scale(loss / tokens).backward()
            total_loss += float(loss.detach())
    return total_loss, cache


def generator_backward(
    model, discriminator, group, cfg, device, scaler, tokens, *, frozen, hidden_cache=None
) -> dict:
    """Backpropagate small head chunks, then send their accumulated gradient through LM once."""
    nll_sum = adversarial_sum = 0.0
    guard = (
        fixed_discriminator(discriminator)
        if discriminator is not None
        else contextlib.nullcontext()
    )
    with guard:
        for batch_index, cpu_batch in enumerate(group):
            if hidden_cache is None:
                batch = move_batch(cpu_batch, device)
                with autocast_context(device, cfg["precision"]):
                    hidden, targets = valid_hidden(model, batch, frozen=frozen)
            else:
                hidden, targets = hidden_cache[batch_index]
            # A separate hidden leaf bounds the lifetime of K*1024 head activations.
            # Its scaled gradient is propagated through the LM once after all chunks.
            leaf = hidden.detach().requires_grad_(hidden.requires_grad)
            for start in range(0, len(hidden), cfg["head_chunk_size"]):
                stop = start + cfg["head_chunk_size"]
                condition, real = leaf[start:stop], targets[start:stop]
                with autocast_context(device, cfg["precision"]):
                    distribution = model.distribution(condition)
                    nll = distribution.nll(real).sum()
                    adversarial = nll.new_zeros(())
                    if discriminator is not None:
                        logits = distribution.selected_pixel_logits(
                            strategy=cfg["gan_component_strategy"]
                        )
                        fake = hard_binary_st(logits, threshold=cfg["threshold"])
                        score = discriminator(fake, condition.detach()).float()
                        adversarial = F.softplus(-score).sum()
                    loss = (nll + cfg["adversarial_weight"] * adversarial) / tokens
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite generator loss")
                scaler.scale(loss).backward()
                nll_sum += float(nll.detach())
                adversarial_sum += float(adversarial.detach())
            if hidden.requires_grad:
                if leaf.grad is None:
                    raise RuntimeError("Missing structured-head gradient for the language backbone")
                hidden.backward(leaf.grad)
    return {"nll_sum": nll_sum, "adversarial_sum": adversarial_sum}


def initial_progress() -> dict:
    return {
        "epoch": 0,
        "cursor": 0,
        "optimizer_steps": 0,
        "valid_tokens": 0,
        "attempted_tokens": 0,
        "overflow_steps": 0,
        "overflow_streak": 0,
        "discriminator_steps": 0,
        "discriminator_valid_tokens": 0,
        "discriminator_attempted_tokens": 0,
        "discriminator_overflow_steps": 0,
        "discriminator_overflow_streak": 0,
        "best_validation_nll": None,
        "best_step": 0,
        "best_tokens": 0,
        "last_validation_tokens": 0,
        "training_seconds": 0.0,
    }


def advance_progress(
    progress: dict, *, tokens: int, samples: int, generator_ok: bool, discriminator_ok
):
    if tokens <= 0 or samples <= 0:
        raise ValueError("An update attempt must consume positive samples and targets")
    progress["cursor"] += samples
    progress["attempted_tokens"] += tokens
    if generator_ok:
        progress["optimizer_steps"] += 1
        progress["valid_tokens"] += tokens
        progress["overflow_streak"] = 0
    else:
        progress["overflow_steps"] += 1
        progress["overflow_streak"] += 1
    if discriminator_ok is not None:
        progress["discriminator_attempted_tokens"] += tokens
        if discriminator_ok:
            progress["discriminator_steps"] += 1
            progress["discriminator_valid_tokens"] += tokens
            progress["discriminator_overflow_streak"] = 0
        else:
            progress["discriminator_overflow_steps"] += 1
            progress["discriminator_overflow_streak"] += 1


def validate_resume_identity(saved: dict, metadata: dict) -> None:
    if saved.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Resume requires a structured v2 checkpoint")
    for key in IDENTITY_KEYS:
        if saved["metadata"].get(key) != metadata.get(key):
            raise ValueError(f"Resume identity mismatch: {key}")


def validate_initialization(saved: dict, model_config: ModelConfig, data_hashes: dict) -> dict:
    metadata = saved.get("metadata", {})
    if metadata.get("data_sha256") != data_hashes:
        raise ValueError("Initialization checkpoint and requested corpus identities differ")
    source_config = metadata.get("config", {}).get("model")
    if not isinstance(source_config, dict):
        raise ValueError("Initialization checkpoint lacks its architecture configuration")
    if ModelConfig.from_dict(source_config) != model_config:
        raise ValueError(
            "Initialization architecture differs after normalizing ModelConfig defaults"
        )
    return metadata


def save_checkpoint(
    path, model, optimizer, scaler, discriminator, d_optimizer, d_scaler, progress, metadata
):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "discriminator": discriminator.state_dict() if discriminator is not None else None,
            "discriminator_optimizer": d_optimizer.state_dict()
            if d_optimizer is not None
            else None,
            "discriminator_scaler": d_scaler.state_dict() if d_scaler is not None else None,
            "progress": dict(progress),
            "metadata": metadata,
            "rng": rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def restore_checkpoint(
    saved, model, optimizer, scaler, discriminator, d_optimizer, d_scaler, metadata
):
    validate_resume_identity(saved, metadata)
    if (saved["discriminator"] is None) != (discriminator is None):
        raise ValueError("Resume discriminator presence mismatch")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    if discriminator is not None:
        discriminator.load_state_dict(saved["discriminator"], strict=True)
        d_optimizer.load_state_dict(saved["discriminator_optimizer"])
        d_scaler.load_state_dict(saved["discriminator_scaler"])
    progress = dict(saved["progress"])
    if set(initial_progress()) - progress.keys():
        raise ValueError("Resume checkpoint lacks complete update/overflow accounting")
    restore_rng(saved["rng"])
    return progress


def validation_subset(dataset, cfg):
    generator = torch.Generator().manual_seed(cfg["seed"] + 104729)
    indices = torch.randperm(len(dataset), generator=generator)[: cfg["validation_samples"]]
    indices = indices.sort().values.numpy().astype("<i8")
    lengths = sequence_lengths(dataset)
    selection = {
        "scope": "fixed_validation_subset" if len(indices) < len(dataset) else "full_validation",
        "selection": "seeded uniform chunk sample without replacement, then source order",
        "seed": cfg["seed"] + 104729,
        "chunks": len(indices),
        "total_chunks": len(dataset),
        "expected_targets": int(lengths[indices].sum()),
        "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
    }
    return Subset(dataset, indices.tolist()), selection


@torch.inference_mode()
def validate(model, subset, selection, cfg, device) -> dict:
    model.eval()
    loader = DataLoader(
        subset,
        batch_size=cfg["validation_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=collate_glyph_sequences,
        generator=torch.Generator().manual_seed(cfg["seed"] + 15485863),
    )
    nll_sum = 0.0
    targets_count = exact = tp = fp = fn = 0
    prior_mass = posterior_mass = mode_mass = None
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        with autocast_context(device, cfg["precision"]):
            hidden, targets = valid_hidden(model, batch, frozen=True)
        for start in range(0, len(hidden), cfg["head_chunk_size"]):
            stop = start + cfg["head_chunk_size"]
            real = targets[start:stop]
            with autocast_context(device, cfg["precision"]):
                distribution = model.distribution(hidden[start:stop])
                nll = distribution.nll(real)
                prior = distribution.component_logits.float().softmax(-1)
                posterior = distribution.responsibilities(real)
                if prior_mass is None:
                    prior_mass = torch.zeros(prior.shape[-1], dtype=torch.float64, device=device)
                    posterior_mass = torch.zeros_like(prior_mass)
                    mode_mass = torch.zeros_like(prior_mass)
                prior_mass += prior.double().sum(0)
                posterior_mass += posterior.double().sum(0)
                mode_mass += torch.bincount(prior.argmax(-1), minlength=prior.shape[-1])
                prediction = distribution.decode(
                    threshold=cfg["threshold"], strategy="mode_threshold"
                )
            if not bool(torch.isfinite(nll).all()):
                raise FloatingPointError("Nonfinite validation NLL")
            nll_sum += float(nll.double().sum())
            targets_count += len(real)
            prediction, real = prediction.bool().flatten(1), real.bool().flatten(1)
            exact += int((prediction == real).all(1).sum())
            tp += int((prediction & real).sum())
            fp += int((prediction & ~real).sum())
            fn += int((~prediction & real).sum())
    if targets_count != selection["expected_targets"] or targets_count == 0:
        raise ValueError("Validation did not cover exactly its declared target sample")
    return {
        **selection,
        "targets": targets_count,
        "nll_per_pixel": nll_sum / targets_count,
        "nll_nats_per_grid": 1024 * nll_sum / targets_count,
        "exact_bitmap_match": exact / targets_count,
        "foreground_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "threshold": cfg["threshold"],
        "decode_strategy": "mode_threshold",
        "target_scope": "all valid next-grid targets including EOS; no padding",
        "prior_component_fraction": (prior_mass / targets_count).tolist(),
        "posterior_component_fraction": (posterior_mass / targets_count).tolist(),
        "mode_component_fraction": (mode_mass / targets_count).tolist(),
    }


def optimizer_for(model, cfg):
    # Frozen parameters remain registered so a scheduled unfreeze preserves optimizer identity.
    parameters = list(model.parameters())
    return torch.optim.AdamW(
        [
            {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": cfg["weight_decay"]},
            {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=cfg["learning_rate"],
        betas=(0.9, 0.95),
    )


def train(args: argparse.Namespace) -> None:
    config = effective_config(
        json.loads(Path(args.config).read_text()), mode=args.mode, smoke_tokens=args.smoke_tokens
    )
    cfg = config["training"]
    if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        raise ValueError("run-name must be one directory name")
    if (args.mode == "smoke") != ("smoke" in args.run_name.lower()):
        raise ValueError("Smoke run names must contain smoke; full run names must not")
    device = torch.device(args.device)
    if device.type == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("This experiment is authorized on GPU0: set CUDA_VISIBLE_DEVICES=0")
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    torch.set_float32_matmul_precision("highest")
    log_dir = Path("artifacts/logs") / args.run_name
    checkpoint_dir = Path("artifacts/checkpoints") / args.run_name
    if not args.resume and (log_dir.exists() or checkpoint_dir.exists()):
        raise FileExistsError("Use a new run name or explicitly resume a checkpoint")
    metadata = runtime_metadata(config, Path(args.data), device, mode=args.mode)
    dataset = GlyphSequenceDataset(args.data, "train", cfg["sequence_length"])
    validation = GlyphSequenceDataset(args.data, "validation", cfg["sequence_length"])
    subset, selection = validation_subset(validation, cfg)
    lengths = sequence_lengths(dataset)
    init_path = Path(args.init_checkpoint)
    init_hash = sha256(init_path)
    metadata.update(
        {
            "mode": args.mode,
            "run_name": args.run_name,
            "model_family": "structured_glyph_gpt",
            "head": config["head"],
            "init_checkpoint_sha256": init_hash,
            "model_revision": "transferred weights identified by init_checkpoint_sha256",
            "source_sha256": {
                path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
            },
            "sampler": {
                "algorithm": cfg["sampler"],
                "version": 1,
                "pool_batches": cfg["sortish_pool_batches"],
            },
            "validation_selection": selection,
            "training_targets_per_pass": dataset.target_count,
            "config_file_sha256": sha256(Path(args.config)),
            "adversarial_semantics": "hard grids; detached condition; component gates use NLL",
        }
    )
    model_config = ModelConfig.from_dict(config["model"])
    model = StructuredGlyphGPT(model_config, **config["head"]).to(device)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    if not args.resume:
        initialized = torch.load(init_path, map_location="cpu", weights_only=False)
        original_metadata = validate_initialization(
            initialized, model_config, metadata["data_sha256"]
        )
        if initialized.get("format") == CHECKPOINT_FORMAT:
            model.load_state_dict(initialized["model"], strict=True)
            metadata["init_checkpoint_format"] = CHECKPOINT_FORMAT
        else:
            model.load_v1_state_dict(initialized["model"])
            metadata["init_checkpoint_format"] = "glyph_lm_v1"
        metadata["init_git_commit"] = original_metadata.get("git_commit")
        metadata["init_model_config"] = original_metadata.get("config", {}).get("model")
        del initialized
    model.train()
    configure_backbone(model, frozen=frozen_at(cfg, 0))
    optimizer = optimizer_for(model, cfg)
    amp = device.type == "cuda" and cfg["precision"] == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    discriminator = d_optimizer = d_scaler = None
    if cfg["adversarial_weight"] > 0:
        discriminator = ConditionalGlyphDiscriminator(
            hidden_size=config["model"]["hidden_size"], channels=cfg["discriminator_channels"]
        ).to(device)
        d_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=cfg["discriminator_lr"],
            betas=(0.0, 0.9),
            weight_decay=cfg["discriminator_weight_decay"],
        )
        d_scaler = torch.amp.GradScaler("cuda", enabled=amp)
    metadata["parameters"] = sum(p.numel() for p in model.parameters())
    metadata["initial_trainable_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    metadata["discriminator_parameters"] = (
        sum(p.numel() for p in discriminator.parameters()) if discriminator is not None else 0
    )
    progress = initial_progress()
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        progress = restore_checkpoint(
            saved, model, optimizer, scaler, discriminator, d_optimizer, d_scaler, metadata
        )
        for key in ("init_checkpoint_format", "init_git_commit", "init_model_config"):
            metadata[key] = saved["metadata"][key]
        metadata["resume_from"] = {"checkpoint_sha256": sha256(Path(args.resume))}
        del saved
    configure_backbone(model, frozen=frozen_at(cfg, progress["valid_tokens"]))
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_json(log_dir / "metadata.json", metadata)
    started = time.monotonic()
    previous_seconds = progress["training_seconds"]

    def log(record):
        record = {"time": datetime.now(UTC).isoformat(), **record}
        with (log_dir / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def checkpoint(name):
        progress["training_seconds"] = previous_seconds + time.monotonic() - started
        save_checkpoint(
            checkpoint_dir / name,
            model,
            optimizer,
            scaler,
            discriminator,
            d_optimizer,
            d_scaler,
            progress,
            metadata,
        )

    def validate_and_save():
        metrics = validate(model, subset, selection, cfg, device)
        progress["last_validation_tokens"] = progress["valid_tokens"]
        if (
            progress["best_validation_nll"] is None
            or metrics["nll_per_pixel"] < progress["best_validation_nll"]
        ):
            progress.update(
                best_validation_nll=metrics["nll_per_pixel"],
                best_step=progress["optimizer_steps"],
                best_tokens=progress["valid_tokens"],
            )
            checkpoint("best.pt")
        snapshot = "validation_step{:08d}_tokens{:012d}".format(
            progress["optimizer_steps"], progress["valid_tokens"]
        )
        checkpoint(snapshot + ".pt")
        write_json(
            log_dir / (snapshot + ".json"),
            {
                "metrics": metrics,
                "progress": dict(progress),
                "checkpoint_sha256": sha256(checkpoint_dir / (snapshot + ".pt")),
            },
        )
        model.train()
        configure_backbone(model, frozen=frozen_at(cfg, progress["valid_tokens"]))
        checkpoint("latest.pt")
        log({"kind": "validation", "metrics": metrics, "progress": dict(progress)})

    try:
        (log_dir / "training_complete.json").unlink(missing_ok=True)
        write_json(log_dir / "status.json", {"status": "running", "progress": progress})
        if progress["best_validation_nll"] is None:
            validate_and_save()
        stop_reason = (
            "token_budget" if progress["valid_tokens"] >= cfg["target_tokens"] else "epoch_budget"
        )
        while progress["epoch"] < cfg["max_epochs"] and stop_reason == "epoch_budget":
            sampler = (
                SortishEpochSampler(
                    lengths,
                    cfg["seed"],
                    progress["epoch"],
                    cfg["batch_size"],
                    progress["cursor"],
                    cfg["sortish_pool_batches"],
                )
                if cfg["sampler"] == "sortish"
                else EpochSampler(len(dataset), cfg["seed"], progress["epoch"], progress["cursor"])
            )
            loader = DataLoader(
                dataset,
                batch_size=cfg["batch_size"],
                sampler=sampler,
                num_workers=cfg["num_workers"],
                collate_fn=collate_glyph_sequences,
                pin_memory=device.type == "cuda",
                generator=torch.Generator().manual_seed(cfg["seed"] + progress["epoch"] + 1),
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
                if tokens <= 0:
                    raise ValueError("No supervised targets in training update")
                frozen = frozen_at(cfg, progress["valid_tokens"])
                configure_backbone(model, frozen=frozen)
                optimizer.zero_grad(set_to_none=True)
                lr = learning_rate(progress["valid_tokens"] + tokens, cfg)
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = lr
                update_started = time.monotonic()
                hidden_cache = None
                d_result, d_loss = None, 0.0
                if discriminator is not None:
                    d_optimizer.zero_grad(set_to_none=True)
                    d_loss, hidden_cache = discriminator_backward(
                        model, discriminator, group, cfg, device, d_scaler, tokens, frozen=frozen
                    )
                    d_result = complete_optimizer_step(
                        d_optimizer,
                        d_scaler,
                        list(discriminator.parameters()),
                        cfg["max_grad_norm"],
                    )
                    d_optimizer.zero_grad(set_to_none=True)
                losses = generator_backward(
                    model,
                    discriminator,
                    group,
                    cfg,
                    device,
                    scaler,
                    tokens,
                    frozen=frozen,
                    hidden_cache=hidden_cache,
                )
                g_result = complete_optimizer_step(
                    optimizer,
                    scaler,
                    [p for p in model.parameters() if p.requires_grad],
                    cfg["max_grad_norm"],
                )
                advance_progress(
                    progress,
                    tokens=tokens,
                    samples=sum(batch["glyphs"].shape[0] for batch in group),
                    generator_ok=g_result["succeeded"],
                    discriminator_ok=d_result["succeeded"] if d_result is not None else None,
                )
                del hidden_cache
                overflow = not g_result["succeeded"] or (
                    d_result is not None and not d_result["succeeded"]
                )
                if (
                    overflow
                    or progress["optimizer_steps"] == 1
                    or progress["optimizer_steps"] % cfg["log_every_steps"] == 0
                ):
                    log(
                        {
                            "kind": "overflow" if overflow else "train",
                            "nll_per_pixel": losses["nll_sum"] / tokens,
                            "generator_adversarial_loss": losses["adversarial_sum"] / tokens,
                            "discriminator_loss": d_loss / tokens,
                            "generator_step": g_result,
                            "discriminator_step": d_result,
                            "backbone_frozen": frozen,
                            "learning_rate": lr,
                            "attempted_targets_per_second": tokens
                            / (time.monotonic() - update_started),
                            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
                            if device.type == "cuda"
                            else None,
                            "progress": dict(progress),
                        }
                    )
                if overflow or progress["optimizer_steps"] % cfg["checkpoint_every_steps"] == 0:
                    checkpoint("latest.pt")
                    write_json(log_dir / "status.json", {"status": "running", "progress": progress})
                if (
                    max(progress["overflow_streak"], progress["discriminator_overflow_streak"])
                    >= cfg["max_consecutive_overflows"]
                ):
                    raise FloatingPointError("Repeated AMP skips exceeded the configured limit")
                if progress["valid_tokens"] >= cfg["target_tokens"]:
                    stop_reason = "token_budget"
                    break
                if (
                    progress["valid_tokens"] - progress["last_validation_tokens"]
                    >= cfg["validate_every_tokens"]
                ):
                    validate_and_save()
            if stop_reason == "token_budget":
                break
            progress["epoch"] += 1
            progress["cursor"] = 0
            checkpoint("latest.pt")
        if progress["last_validation_tokens"] != progress["valid_tokens"]:
            validate_and_save()
        checkpoint("final.pt")
        complete = progress["valid_tokens"] >= cfg["target_tokens"]
        completion = {
            "status": "complete" if complete else "budget_not_met",
            "stop_reason": stop_reason,
            "mode": args.mode,
            "progress": progress,
            "target_budget": cfg["target_tokens"],
            "validation_scope": selection,
            "evaluation_complete": False,
            "metadata_sha256": sha256(log_dir / "metadata.json"),
            "best_checkpoint_sha256": sha256(checkpoint_dir / "best.pt"),
            "final_checkpoint_sha256": sha256(checkpoint_dir / "final.pt"),
        }
        write_json(log_dir / "status.json", completion)
        if complete:
            write_json(log_dir / "training_complete.json", completion)
        log({"kind": "training_complete" if complete else "budget_not_met", **completion})
        if not complete:
            raise RuntimeError("Epoch limit reached before the successful-update target budget")
    except BaseException as error:
        # Never checkpoint a partly accumulated G/D cycle; latest.pt is an atomic safe boundary.
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
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--smoke-tokens", type=int, default=32768)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
