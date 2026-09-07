"""Stress GPU0 with full-length binary tile batches; this is not language training."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from hansgpt_research.glyph_lm import GlyphGPT, ModelConfig, pixel_bce_loss
from hansgpt_research.train_glyph_lm import runtime_metadata, sha256, write_json


def memory_statistics(device: torch.device) -> dict:
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "current_allocated_bytes": torch.cuda.memory_allocated(device),
        "current_reserved_bytes": torch.cuda.memory_reserved(device),
        "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
    }


def stress(args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Capacity stress is authorized on GPU0: set CUDA_VISIBLE_DEVICES=0")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU0 capacity stress requires CUDA")
    if min(args.batch_size, args.sequence_length, args.steps) <= 0:
        raise ValueError("Batch size, sequence length and steps must be positive")
    output = Path(args.output).resolve()
    if output.suffix != ".json" or not output.is_relative_to(Path("artifacts/logs").resolve()):
        raise ValueError("Output must be a JSON file under artifacts/logs/")
    if output.exists():
        raise FileExistsError("Choose a new capacity stress output file")

    config_path, data_dir = Path(args.config), Path(args.data)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = ModelConfig.from_dict(config["model"])
    if args.sequence_length > model_config.max_position_embeddings:
        raise ValueError("Requested sequence exceeds the configured context capacity")
    # Record the actual synthetic run independently of the unchanged source JSON.
    config["model"] = model_config.to_dict()
    config["training"].update(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        gradient_accumulation_steps=1,
        precision="fp16",
        gradient_checkpointing=True,
    )
    device = torch.device("cuda:0")
    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")
    metadata = runtime_metadata(config, data_dir, device, mode="synthetic_capacity_stress")
    metadata["source_config_sha256"] = sha256(config_path)
    metadata["source_config_path"] = str(config_path)
    report = {
        "status": "running",
        "mode": "synthetic_capacity_stress",
        "counts_as_training_result": False,
        "sampling": "Uniform verified glyph-bank asset addresses, resolved to pixels before CNN",
        "supervision": "Independent random T+1 tiles shifted once; all B*T targets valid",
        "scope": "GPU memory and finite-gradient capacity only; no language-quality conclusion",
        "metadata": metadata,
        "requested_steps": args.steps,
        "successful_optimizer_steps": 0,
        "overflow_steps": 0,
        "step_metrics": [],
    }
    write_json(output, report)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with np.load(data_dir / "glyph_bank.npz", allow_pickle=False) as archive:
            bitmaps = archive["bitmaps"]
        if bitmaps.ndim == 3:
            bitmaps = bitmaps[:, None]
        if (
            bitmaps.ndim != 4
            or bitmaps.shape[1:] != (1, 32, 32)
            or not len(bitmaps)
            or not np.isin(bitmaps, [0, 1]).all()
        ):
            raise ValueError("Verified glyph bank must contain 32x32 binary assets")
        bank = torch.from_numpy(np.array(bitmaps, dtype=np.uint8, copy=True)).to(device)
        report["glyph_assets"] = len(bank)
        model = GlyphGPT(model_config).to(device).train()
        model.gradient_checkpointing_enable()
        report["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        if report["parameters"] != 78_118_368:
            raise ValueError("This capacity check requires the agreed 78,118,368-parameter Base")
        if not all(layer.gradient_checkpointing for layer in model.backbone.layers):
            raise RuntimeError("Decoder gradient checkpointing did not activate")
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for p in model.parameters() if p.ndim >= 2]},
                {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
            ],
            lr=config["training"]["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=config["training"]["weight_decay"],
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        generator = torch.Generator(device=device).manual_seed(seed)
        mask = torch.ones((args.batch_size, args.sequence_length), device=device, dtype=torch.bool)
        cnn_weight = model.glyph_encoder.convolutions[0].weight
        cnn_before = cnn_weight.detach().clone()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            step_started = time.monotonic()
            asset_addresses = torch.randint(
                len(bank),
                (args.batch_size, args.sequence_length + 1),
                device=device,
                generator=generator,
            )
            tiles = bank[asset_addresses]
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(tiles[:, :-1], attention_mask=mask)
                loss = pixel_bce_loss(logits, tiles[:, 1:], mask)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Synthetic capacity stress produced nonfinite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            cnn_gradients = [parameter.grad for parameter in model.glyph_encoder.parameters()]
            finite_cnn_gradients = all(
                gradient is not None and bool(torch.isfinite(gradient).all())
                for gradient in cnn_gradients
            )
            nonzero_cnn_gradient = cnn_weight.grad is not None and bool(cnn_weight.grad.abs().sum())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["training"]["max_grad_norm"]
            )
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - step_started
            overflow = scaler.get_scale() < old_scale
            report["overflow_steps"] += int(overflow)
            report["successful_optimizer_steps"] += int(not overflow)
            report["step_metrics"].append(
                {
                    "step": step + 1,
                    "seconds": elapsed,
                    "synthetic_valid_tiles": mask.numel(),
                    "valid_position_fraction": 1.0,
                    "bce_per_pixel": float(loss.detach()),
                    "grad_norm": float(grad_norm),
                    "finite_cnn_gradients": finite_cnn_gradients,
                    "nonzero_cnn_gradient": nonzero_cnn_gradient,
                    "grad_scaler_scale": scaler.get_scale(),
                    "optimizer_step_skipped": overflow,
                    **memory_statistics(device),
                }
            )
            write_json(output, report)
            if (
                not finite_cnn_gradients
                or not nonzero_cnn_gradient
                or not torch.isfinite(grad_norm)
            ):
                raise FloatingPointError("Synthetic capacity stress failed its CNN gradient check")
        if report["successful_optimizer_steps"] != args.steps:
            raise FloatingPointError("Capacity test did not finish every requested optimizer step")
        report["cnn_weight_max_absolute_change"] = float(
            (cnn_weight.detach() - cnn_before).abs().max()
        )
        if report["cnn_weight_max_absolute_change"] <= 0:
            raise RuntimeError("CNN weights did not change after the optimizer steps")
        report["optimizer_state_tensors"] = sum(
            torch.is_tensor(value) for state in optimizer.state.values() for value in state.values()
        )
        if not report["optimizer_state_tensors"]:
            raise RuntimeError("AdamW optimizer states were not allocated")
        report["status"] = "complete"
        report["passed"] = True
    except BaseException as error:
        report.update(
            status="failed",
            passed=False,
            error_type=type(error).__name__,
            error=str(error),
            out_of_memory=isinstance(error, torch.cuda.OutOfMemoryError),
        )
    finally:
        report["wall_seconds"] = time.monotonic() - started
        report["memory"] = memory_statistics(device)
        write_json(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "mode": report["mode"],
                "output": str(output),
                "successful_optimizer_steps": report["successful_optimizer_steps"],
                "memory": report["memory"],
            }
        ),
        flush=True,
    )
    if report["status"] != "complete":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/experiments/hansgpt_binary_v1.json")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", required=True)
    stress(parser.parse_args())


if __name__ == "__main__":
    main()
