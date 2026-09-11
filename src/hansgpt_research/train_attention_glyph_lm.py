"""Train matched A/B/C models from scratch with bounded byte-head activation memory."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hansgpt_research.attention_glyph_lm import GPU_BY_VARIANT, AttentionGlyphGPT
from hansgpt_research.glyph_lm import GlyphSequenceDataset, ModelConfig, collate_glyph_sequences
from hansgpt_research.train_glyph_lm import (
    SortishEpochSampler,
    autocast_context,
    learning_rate,
    move_batch,
    restore_rng,
    runtime_metadata,
    save_checkpoint,
    sequence_lengths,
    sha256,
    write_json,
)
from hansgpt_research.train_structured_glyph_lm import (
    advance_progress,
    complete_optimizer_step,
    effective_config,
    generator_backward,
    initial_progress,
    optimizer_for,
    valid_hidden,
    validation_subset,
)


def check_gpu(variant, device):
    if device.type != "cuda":
        raise RuntimeError("ABC training must run on the training server GPUs")
    if device.index not in (None, 0):
        raise RuntimeError("Use logical cuda:0 after selecting one physical GPU")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU_BY_VARIANT[variant]):
        raise RuntimeError(f"Variant {variant} requires its assigned physical GPU")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("Set CUDA_DEVICE_ORDER=PCI_BUS_ID for physical GPU selection")


@torch.inference_mode()
def validate_nll(model, subset, selection, cfg, device):
    """Exact joint likelihood; never report target-byte-conditioned argmax as generated glyphs."""
    model.eval()
    loader = DataLoader(
        subset,
        batch_size=cfg["validation_batch_size"],
        collate_fn=collate_glyph_sequences,
        num_workers=cfg["num_workers"],
        generator=torch.Generator().manual_seed(cfg["seed"] + 15485863),
    )
    total, count = 0.0, 0
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        with autocast_context(device, cfg["precision"]):
            hidden, targets = valid_hidden(model, batch, frozen=True)
            for start in range(0, len(hidden), cfg["head_chunk_size"]):
                stop = start + cfg["head_chunk_size"]
                nll = model.distribution(hidden[start:stop]).nll(targets[start:stop])
                if not bool(torch.isfinite(nll).all()):
                    raise FloatingPointError("Nonfinite validation NLL")
                total += float(nll.double().sum())
                count += len(nll)
    if count != selection["expected_targets"] or not count:
        raise ValueError("Validation target coverage mismatch")
    return {
        **selection,
        "targets": count,
        "nll_per_pixel": total / count,
        "nll_nats_per_grid": 1024 * total / count,
    }


@torch.inference_mode()
def generation_diagnostic(model, subset, cfg, device, output, step, mode):
    """Small fixed validation diagnostic; raw binary arrays, no projection in feedback."""
    model.eval()
    sample = subset[0]
    valid = int(sample["attention_mask"].sum())
    length = min(16, valid)
    prompt = sample["glyphs"][:length].unsqueeze(0).to(device)
    with autocast_context(device, cfg["precision"]):
        generated = model.generate(prompt, 2 if mode == "smoke" else 32, threshold=0.5)
    values = generated.cpu().numpy()
    if values.dtype != np.uint8 or not np.isin(values, [0, 1]).all():
        raise ValueError("Generated feedback must be binary uint8")
    np.savez_compressed(
        output / f"generation_step{step:08d}.npz", prompt=prompt.cpu().numpy(), generated=values
    )
    bank = subset.dataset.glyph_bank
    controls = set(subset.dataset.control_ids.values())
    gallery = bank[[i for i in range(len(bank)) if i not in controls]].flatten(1).to(device)
    flat = generated.flatten(0, 1).flatten(1)
    distances = []
    for grid in flat:
        distances.append(int((gallery != grid).sum(1).min()))
    repeated = (flat[1:] == flat[:-1]).all(1).float().mean() if len(flat) > 1 else 0.0
    return {
        "scope": "one fixed validation chunk; not a language quality assessment",
        "generated_grids": len(flat),
        "exact_content_glyph_rate": sum(d == 0 for d in distances) / len(distances),
        "nearest_content_hamming_bits": sum(distances) / len(distances),
        "adjacent_exact_repeat_rate": float(repeated),
        "decode": "greedy bytes" if model.byte_decoder else "pixel threshold 0.5",
        "eos_stopping": False,
    }


def train(args):
    config = effective_config(
        json.loads(Path(args.config).read_text("utf-8")),
        mode=args.mode,
        smoke_tokens=args.smoke_tokens,
    )
    cfg = config["training"]
    variant = config["variant"]
    if variant not in GPU_BY_VARIANT or cfg["freeze_backbone"] or cfg["adversarial_weight"]:
        raise ValueError("ABC requires a valid variant, joint training and no GAN")
    if cfg["sampler"] != "sortish":
        raise ValueError("ABC uses the matched sortish sampler")
    if not args.run_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("run-name must contain only letters, digits, underscores or hyphens")
    if (args.mode == "smoke") != ("smoke" in args.run_name.lower()):
        raise ValueError("Only smoke run names must contain smoke")
    device = torch.device(args.device)
    check_gpu(variant, device)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.set_float32_matmul_precision("highest")
    output = Path("artifacts/logs") / args.run_name
    checkpoints = Path("artifacts/checkpoints") / args.run_name
    if not args.resume and (output.exists() or checkpoints.exists()):
        raise FileExistsError("Use a fresh run name or explicitly resume")
    metadata = runtime_metadata(config, Path(args.data), device, mode=args.mode)
    if metadata["data_manifest_sha256"] != config["data_requirements"]["manifest_sha256"]:
        raise ValueError("ABC corpus differs from the pinned v1 dataset")
    dataset = GlyphSequenceDataset(args.data, "train", cfg["sequence_length"])
    validation = GlyphSequenceDataset(args.data, "validation", cfg["sequence_length"])
    subset, selection = validation_subset(validation, cfg)
    lengths = sequence_lengths(dataset)
    model = AttentionGlyphGPT(
        ModelConfig.from_dict(config["model"]), variant, config["encoder"], config["decoder"]
    ).to(device)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    optimizer = optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["precision"] == "fp16")
    metadata.update(
        run_name=args.run_name,
        mode=args.mode,
        model_family="attention_abc_v1",
        validation_selection=selection,
        training_targets_per_pass=dataset.target_count,
        parameters=sum(p.numel() for p in model.parameters()),
        source_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    )
    progress = initial_progress()
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in (
            "config_sha256",
            "data_sha256",
            "data_manifest_sha256",
            "data_verification_sha256",
            "run_name",
            "mode",
            "model_family",
            "validation_selection",
            "source_sha256",
        ):
            if metadata[key] != saved["metadata"].get(key):
                raise ValueError(f"Resume identity mismatch: {key}")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        progress = saved["progress"]
        restore_rng(saved["rng"])
        metadata["resume_from"] = sha256(Path(args.resume))
        del saved
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    write_json(output / "metadata.json", metadata)
    started, previous_seconds = time.monotonic(), progress["training_seconds"]

    def status(state="running", **extra):
        progress["training_seconds"] = previous_seconds + time.monotonic() - started
        write_json(
            output / "status.json",
            {
                "status": state,
                "time": datetime.now(UTC).isoformat(),
                "pid": os.getpid(),
                "variant": variant,
                "progress": progress,
                "target_budget": cfg["target_tokens"],
                **extra,
            },
        )

    def log(kind, **extra):
        record = {
            "time": datetime.now(UTC).isoformat(),
            "kind": kind,
            "progress": dict(progress),
            **extra,
        }
        with (output / "training.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def save(name):
        status(phase="checkpoint")
        save_checkpoint(checkpoints / name, model, optimizer, scaler, dict(progress), metadata)

    def validate():
        status(phase="validation")
        metrics = validate_nll(model, subset, selection, cfg, device)
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
            save("best.pt")
        log("validation", metrics=metrics)
        status(phase="generation_diagnostic")
        diagnostic = generation_diagnostic(
            model, subset, cfg, device, output, progress["optimizer_steps"], args.mode
        )
        log("generation_diagnostic", metrics=diagnostic)
        save("latest.pt")
        model.train()

    try:
        status(phase="initializing")
        if progress["best_validation_nll"] is None:
            validate()
        while progress["valid_tokens"] < cfg["target_tokens"]:
            if progress["epoch"] >= cfg["max_epochs"]:
                raise RuntimeError("Epoch limit before target budget")
            sampler = SortishEpochSampler(
                lengths,
                cfg["seed"],
                progress["epoch"],
                cfg["batch_size"],
                progress["cursor"],
                cfg["sortish_pool_batches"],
            )
            loader = DataLoader(
                dataset,
                batch_size=cfg["batch_size"],
                sampler=sampler,
                num_workers=cfg["num_workers"],
                pin_memory=True,
                collate_fn=collate_glyph_sequences,
                generator=torch.Generator().manual_seed(cfg["seed"] + progress["epoch"]),
            )
            iterator = iter(loader)
            while progress["valid_tokens"] < cfg["target_tokens"]:
                group = []
                for _ in range(cfg["gradient_accumulation_steps"]):
                    try:
                        group.append(next(iterator))
                    except StopIteration:
                        break
                if not group:
                    break
                tokens = sum(int(batch["loss_mask"].sum()) for batch in group)
                status(phase="training")
                optimizer.zero_grad(set_to_none=True)
                lr = learning_rate(progress["valid_tokens"] + tokens, cfg)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                update_started = time.monotonic()
                losses = generator_backward(
                    model, None, group, cfg, device, scaler, tokens, frozen=False
                )
                result = complete_optimizer_step(
                    optimizer, scaler, model.parameters(), cfg["max_grad_norm"]
                )
                advance_progress(
                    progress,
                    tokens=tokens,
                    samples=sum(len(b["glyphs"]) for b in group),
                    generator_ok=result["succeeded"],
                    discriminator_ok=None,
                )
                status(phase="training")
                if (
                    not result["succeeded"]
                    or progress["optimizer_steps"] == 1
                    or progress["optimizer_steps"] % cfg["log_every_steps"] == 0
                ):
                    log(
                        "train" if result["succeeded"] else "overflow",
                        nll_per_pixel=losses["nll_sum"] / tokens,
                        learning_rate=lr,
                        optimizer_result=result,
                        attempted_targets_per_second=tokens / (time.monotonic() - update_started),
                        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device),
                    )
                if progress["overflow_streak"] >= cfg["max_consecutive_overflows"]:
                    raise FloatingPointError("Repeated AMP overflows")
                if (
                    progress["valid_tokens"] - progress["last_validation_tokens"]
                    >= cfg["validate_every_tokens"]
                ):
                    validate()
                elif progress["optimizer_steps"] % cfg["checkpoint_every_steps"] == 0:
                    save("latest.pt")
            if progress["valid_tokens"] < cfg["target_tokens"]:
                progress["epoch"] += 1
                progress["cursor"] = 0
                save("latest.pt")
        if progress["last_validation_tokens"] != progress["valid_tokens"]:
            validate()
        save("final.pt")
        receipt = {
            "status": "complete",
            "progress": dict(progress),
            "mode": args.mode,
            "target_budget": cfg["target_tokens"],
            "evaluation_complete": False,
            "best_checkpoint_sha256": sha256(checkpoints / "best.pt"),
            "final_checkpoint_sha256": sha256(checkpoints / "final.pt"),
            "metadata_sha256": sha256(output / "metadata.json"),
        }
        write_json(output / "training_complete.json", receipt)
        status("complete", phase="finished")
        log("training_complete", receipt=receipt)
    except BaseException as error:
        status("failed", error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--smoke-tokens", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
