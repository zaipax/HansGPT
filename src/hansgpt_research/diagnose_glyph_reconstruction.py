"""D1: reconstruct known glyph images from a frozen v1 CNN, without language generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor, nn

from hansgpt_research.evaluate_glyph_lm import target_frequencies
from hansgpt_research.glyph_lm import GlyphEncoder, GlyphSequenceDataset, ModelConfig
from hansgpt_research.train_glyph_lm import autocast_context, runtime_metadata, sha256, write_json
from hansgpt_research.train_structured_glyph_lm import complete_optimizer_step

COHORTS = ("decoder_fit", "decoder_validation", "decoder_holdout")


def tensor_hash(value: Tensor) -> str:
    array = value.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{array.dtype}:{tuple(array.shape)}".encode())
    digest.update(array.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def module_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor_hash(value).encode())
    return digest.hexdigest()


def select_decoder_glyphs(
    bitmaps: Tensor, inventory: dict, train_counts: np.ndarray, seed: int
) -> dict:
    """Only corpus-train content glyphs enter any diagnostic cohort; split pixel groups.

    Held-out means unseen by these new decoders' supervision, not unseen by the
    pretrained v1 encoder. Characters with equal pixels never cross cohorts.
    """
    if bitmaps.ndim != 4 or tuple(bitmaps.shape[1:]) != (1, 32, 32):
        raise ValueError("Expected a binary glyph bank with shape [assets,1,32,32]")
    if not bool(((bitmaps == 0) | (bitmaps == 1)).all()):
        raise ValueError("Glyph bank must contain only binary pixels")
    if len(train_counts) != len(bitmaps) or np.any(train_counts < 0):
        raise ValueError("Training frequencies must align with the glyph bank")
    controls = {int(index) for index in inventory["controls"]}
    content = sorted({int(index) for index in inventory["characters"].values()} - controls)
    eligible = [index for index in content if train_counts[index] > 0]
    excluded = [index for index in content if train_counts[index] == 0]
    groups = defaultdict(list)
    hashes = {}
    for index in eligible:
        digest = tensor_hash(bitmaps[index].to(torch.uint8))
        groups[digest].append(index)
        hashes[index] = digest
    ranked = sorted(
        groups, key=lambda digest: hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()
    )
    if len(ranked) < 3:
        raise ValueError("Need at least three distinct training pixel groups for decoder splits")
    validation_count = holdout_count = max(1, len(ranked) // 10)
    fit_count = len(ranked) - validation_count - holdout_count
    selected = {
        "decoder_fit": ranked[:fit_count],
        "decoder_validation": ranked[fit_count : fit_count + validation_count],
        "decoder_holdout": ranked[fit_count + validation_count :],
    }
    cohorts = {
        name: sorted(index for digest in pixel_groups for index in groups[digest])
        for name, pixel_groups in selected.items()
    }
    return {
        "seed": seed,
        "cohorts": cohorts,
        "eligible_asset_ids": eligible,
        "excluded_train_zero_asset_ids": excluded,
        "excluded_control_asset_ids": sorted(controls),
        "pixel_sha256": {str(index): hashes[index] for index in eligible},
        "pixel_hash_definition": "SHA256 of dtype/shape header followed by contiguous pixel bytes",
        "collision_groups": [members for members in groups.values() if len(members) > 1],
        "unique_pixel_groups": len(groups),
        "split_rule": "Seeded pixel-hash ranking, approximately 80/10/10 by whole collision groups",
        "supervision": "Only decoder_fit glyphs; each successful sweep presents each asset once",
        "heldout_scope": "Decoder-held-out corpus-train characters; not v1-unseen characters",
    }


class BalancedGlyphOrder:
    """Uniform without-replacement sweeps, preserving the final partial sweep."""

    def __init__(self, indices, seed: int):
        self.indices = np.asarray(indices, dtype=np.int64)
        if not len(self.indices) or len(np.unique(self.indices)) != len(self.indices):
            raise ValueError("Balanced sampling needs nonempty distinct fit positions")
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.indices)
        self.cursor = 0

    def take(self, count: int) -> np.ndarray:
        if count < 1:
            raise ValueError("Batch size must be positive")
        pieces = []
        while count:
            available = min(count, len(self.order) - self.cursor)
            pieces.append(self.order[self.cursor : self.cursor + available])
            self.cursor += available
            count -= available
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(self.indices)
                self.cursor = 0
        return np.concatenate(pieces)


def fixed_bitflips(clean: Tensor, count: int, seed: int) -> Tensor:
    """Flip exactly count distinct bits per image; this is not an identity guarantee."""
    if clean.ndim != 4 or tuple(clean.shape[1:]) != (1, 32, 32):
        raise ValueError("Bitflip input must be [glyphs,1,32,32]")
    if not bool(((clean == 0) | (clean == 1)).all()) or not 0 <= count <= 1024:
        raise ValueError("Need binary images and a bitflip count between 0 and 1024")
    corrupted = clean.detach().cpu().to(torch.uint8).clone().flatten(1)
    if count:
        generator = torch.Generator().manual_seed(seed)
        ranks = torch.rand(corrupted.shape, generator=generator)
        selected = ranks.topk(count, dim=1).indices
        corrupted.scatter_(1, selected, 1 - corrupted.gather(1, selected))
    return corrupted.reshape_as(clean).to(clean.device)


@torch.no_grad()
def encode_images(encoder, images: Tensor, device, batch_size: int, precision: str) -> Tensor:
    """Cache normal detached CPU tensors, never inference tensors needed by a decoder backward."""
    encoder.eval()
    features = []
    for start in range(0, len(images), batch_size):
        with autocast_context(device, precision):
            encoded = encoder(images[start : start + batch_size].to(device))
        if not bool(torch.isfinite(encoded).all()):
            raise FloatingPointError("Frozen encoder produced nonfinite visual features")
        features.append(encoded.float().cpu())
    return torch.cat(features)


class ReconstructionDecoder(nn.Module):
    """Fresh decoder of a known image embedding; no contextual next-token operation."""

    def __init__(self, hidden_size: int, kind: str):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.network = nn.Sequential(
                nn.Linear(hidden_size, 1024), nn.Unflatten(-1, (1, 32, 32))
            )
        elif kind == "spatial":
            self.network = nn.Sequential(
                nn.Linear(hidden_size, 128 * 4 * 4),
                nn.SiLU(),
                nn.Unflatten(-1, (128, 4, 4)),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(128, 64, 3, padding=1),
                nn.SiLU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(64, 32, 3, padding=1),
                nn.SiLU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(16, 1, 3, padding=1),
            )
        else:
            raise ValueError("Decoder must be linear or spatial")

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


def glyph_metrics(prediction: Tensor, target: Tensor, nll: Tensor | None = None) -> dict:
    """Per-glyph and macro/micro foreground metrics, with an explicit zero-division rule."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Prediction and reference must align as [glyphs,1,32,32]")
    if tuple(target.shape[1:]) != (1, 32, 32) or not len(target):
        raise ValueError("Metrics require nonempty 32x32 glyphs")
    for images in (prediction, target):
        if not bool(((images == 0) | (images == 1)).all()):
            raise ValueError("Reconstruction metrics require binary images")
    predicted, expected = prediction.bool().flatten(1), target.bool().flatten(1)
    tp = (predicted & expected).sum(1).double()
    fp = (predicted & ~expected).sum(1).double()
    fn = (~predicted & expected).sum(1).double()
    f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    iou = tp / (tp + fp + fn).clamp_min(1)
    exact = (predicted == expected).all(1)
    result = {
        "glyphs": len(target),
        "macro_foreground_f1": float(f1.mean()),
        "macro_dice": float(f1.mean()),
        "macro_iou": float(iou.mean()),
        "micro_foreground_f1": float(2 * tp.sum() / (2 * tp + fp + fn).sum().clamp_min(1)),
        "micro_iou": float(tp.sum() / (tp + fp + fn).sum().clamp_min(1)),
        "exact_bitmap_match": float(exact.double().mean()),
        "mean_hamming_bits": float((fp + fn).mean()),
        "zero_division": "Foreground F1/IoU are 0 when both images have no foreground",
        "per_glyph": {
            "foreground_f1": f1.cpu().tolist(),
            "dice": f1.cpu().tolist(),
            "iou": iou.cpu().tolist(),
            "exact_bitmap_match": exact.cpu().tolist(),
            "hamming_bits": (fp + fn).long().cpu().tolist(),
        },
    }
    if nll is not None:
        if nll.shape != (len(target),) or not bool(torch.isfinite(nll).all()):
            raise ValueError("Need one finite reconstruction NLL per image")
        result["reconstruction_bce_nats_per_pixel"] = float(nll.double().mean())
        result["per_glyph"]["reconstruction_bce_nats_per_pixel"] = nll.cpu().tolist()
    return result


@torch.no_grad()
def predict_images(decoder, features, targets, device, batch_size, precision):
    decoder.eval()
    predictions, losses = [], []
    for start in range(0, len(features), batch_size):
        with autocast_context(device, precision):
            logits = decoder(features[start : start + batch_size].to(device)).float()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("Decoder produced nonfinite logits")
        reference = targets[start : start + batch_size].to(device).float()
        loss = (
            F.binary_cross_entropy_with_logits(logits, reference, reduction="none")
            .flatten(1)
            .mean(1)
        )
        predictions.append((logits.sigmoid() >= 0.5).to(torch.uint8).cpu())
        losses.append(loss.cpu())
    return torch.cat(predictions), torch.cat(losses)


def train_decoder(decoder, features, targets, fit_positions, args, device, precision) -> dict:
    """Train fixed-budget clean reconstruction only; replay overflowed batches unchanged."""
    order = BalancedGlyphOrder(fit_positions, args.seed + 101)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.001, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    features, targets = features.to(device), targets.to(device)
    exposures = np.zeros(len(features), dtype=np.int64)
    training = {"successful_steps": 0, "attempts": 0, "overflow_skips": 0, "curve": []}
    started = time.monotonic()
    decoder.train()
    for step in range(args.steps):
        selected = order.take(args.batch_size)
        indices = torch.from_numpy(selected).to(device)
        while True:
            if training["attempts"] >= args.steps + 20:
                raise FloatingPointError(
                    "Reconstruction decoder exhausted its AMP calibration budget"
                )
            training["attempts"] += 1
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                logits = decoder(features[indices])
                loss = F.binary_cross_entropy_with_logits(logits.float(), targets[indices].float())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite reconstruction training loss")
            scaler.scale(loss).backward()
            outcome = complete_optimizer_step(optimizer, scaler, list(decoder.parameters()), 1.0)
            if outcome["succeeded"]:
                break
            training["overflow_skips"] += 1
        training["successful_steps"] += 1
        exposures += np.bincount(selected, minlength=len(exposures))
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            point = {
                "step": step + 1,
                "bce_nats_per_pixel": float(loss.detach()),
                "grad_norm": outcome["grad_norm"],
                "scale": outcome["scale"],
            }
            training["curve"].append(point)
            print(json.dumps({"decoder": decoder.kind, **point}), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training.update(
        seconds=time.monotonic() - started,
        optimizer="AdamW lr 0.001 weight_decay 0, clip 1; fixed successful-step budget",
        successful_glyph_presentations=int(exposures.sum()),
        fit_exposure_min=int(exposures[fit_positions].min()),
        fit_exposure_max=int(exposures[fit_positions].max()),
        exposures_sha256=hashlib.sha256(exposures.astype("<i8").tobytes()).hexdigest(),
        decoder_validation_or_holdout_presentations=int(
            exposures.sum() - exposures[fit_positions].sum()
        ),
    )
    return training


def reconstruction_sheet(path, clean, corrupted, prediction, noisy_prediction, characters):
    scale, row_height, left = 3, 108, 96
    canvas = Image.new("RGB", (left + 4 * 108, 30 + len(clean) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(
        ("reference", "clean reconstruction", "bitflip input", "reconstruction")
    ):
        draw.text((left + column * 108, 8), label, fill="black")
    for row, character in enumerate(characters):
        y = 30 + row * row_height
        draw.text((2, y + 38), f"U+{ord(character):04X}", fill="black")
        for column, values in enumerate((clean, prediction, corrupted, noisy_prediction)):
            array = values[row, 0].numpy()
            if not np.isin(array, [0, 1]).all():
                raise ValueError("Contact sheet requires raw binary predictions")
            tile = Image.fromarray((255 * (1 - array)).astype(np.uint8))
            tile = tile.resize((32 * scale, 32 * scale), resample=Image.Resampling.NEAREST)
            canvas.paste(tile, (left + column * 108, y))
    canvas.save(path)


def write_report(output: Path, result: dict) -> None:
    lines = [
        "# D1：冻结 v1 字形编码器的重构诊断",
        "",
        "本实验输入已知目标字的图像，重构同一张图。没有运行语言上下文或下一字生成；"
        "复制成功不代表能够续写中文。编码器完全冻结，两个解码器均从随机权重训练。",
        "",
        "三个组都只来自语言训练语料中出现过的内容字符。decoder_fit 用于新解码器监督；"
        "decoder_validation 和 decoder_holdout 不参与拟合。本轮采用固定步数最终权重和固定0.5阈值，"
        "未用后二组挑权重或阈值。保留组是新解码器的留出字符，不能称为 v1 未见字泛化。",
        "",
        "像素碰撞按组拆分，语言 train 中零次出现的资产和控制格全部排除。"
        "训练按字符地址等频遍历，语言语料原始频次不作为重复权重。",
        "",
        "| 解码器 | 输入 | 组 | 字形数 | 重构BCE nats/pixel | "
        "macro前景F1 | micro前景F1 | 整格匹配 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for decoder, values in result["decoders"].items():
        for mode, cohorts in values["evaluation"].items():
            for cohort, metrics in cohorts.items():
                lines.append(
                    f"| {decoder} | {mode} | {cohort} | {metrics['glyphs']} | "
                    f"{metrics['reconstruction_bce_nats_per_pixel']:.6f} | "
                    f"{metrics['macro_foreground_f1']:.4f} | "
                    f"{metrics['micro_foreground_f1']:.4f} | {metrics['exact_bitmap_match']:.4f} |"
                )
    lines.extend(
        [
            "",
            "bitflip 是单独的输入扰动诊断，解码器只训练干净图像；"
            f"每图固定翻转{result['bitflip_count']}个不同像素，仍以原干净图为参考。"
            "翻位可能改变字符身份，所以其成绩不等于语义不变去噪能力。",
            "",
            "成功重构说明冻结向量中存在本解码器可利用的信息。失败也可能来自解码器结构、"
            "优化步数或精度，不能单独证明编码器丢失了笔画信息。两种解码器参数量和时间单独报告；"
            "相同更新次数不等于相同计算预算。精确向量碰撞检查也不能证明鲁棒性或线性可逆性。",
            "",
            "metrics.json 保存固定选择、编码器/检查点/源码/数据哈希、均衡采样计数及训练曲线；"
            "逐字符数值见各decoder的CSV，NPZ保存未经字库投影的严格0/1原图。",
        ]
    )
    for decoder in result["decoders"]:
        for cohort in COHORTS:
            lines.extend(["", f"![{decoder} {cohort}]({decoder}_{cohort}.png)"])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnose(args: argparse.Namespace) -> None:
    if args.steps < 1 or args.batch_size < 1 or not 0 <= args.bitflip_count <= 32:
        raise ValueError("Need positive steps/batch size and at most 32 flipped pixels")
    device = torch.device(args.device)
    if device.type == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("D1 GPU execution is authorized only with CUDA_VISIBLE_DEVICES=0")
    output = Path(args.output).resolve()
    if output.exists() or not output.is_relative_to(Path("artifacts").resolve()):
        raise ValueError("Choose a new output directory under artifacts/")
    checkpoint_path = Path(args.checkpoint)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    original = saved["metadata"]
    if saved.get("format") == "structured_glyph_lm_v2" or original.get("model_family"):
        raise ValueError("This diagnostic requires the original frozen v1 encoder checkpoint")
    model_config = ModelConfig.from_dict(original["config"]["model"])
    precision = "fp16" if device.type == "cuda" else "fp32"
    config = {
        "model": model_config.to_dict(),
        "training": {"seed": args.seed, "precision": precision},
        "reconstruction": vars(args),
    }
    metadata = runtime_metadata(
        config, Path(args.data), device, mode="glyph_reconstruction_diagnostic"
    )
    if original["data_sha256"] != metadata["data_sha256"]:
        raise ValueError("D1 corpus identity differs from the v1 encoder checkpoint")
    encoder = GlyphEncoder(model_config)
    encoder.load_state_dict(
        {
            name.removeprefix("glyph_encoder."): value
            for name, value in saved["model"].items()
            if name.startswith("glyph_encoder.")
        },
        strict=True,
    )
    encoder.requires_grad_(False).eval().to(device)
    encoder_hash = module_hash(encoder)
    metadata.update(
        model_revision="frozen v1 checkpoint CNN; randomly initialized reconstruction decoders",
        checkpoint_sha256=sha256(checkpoint_path),
        checkpoint_git_commit=original["git_commit"],
        checkpoint_completed_corpus_passes=saved.get("progress", {}).get("epoch"),
        encoder_state_sha256=encoder_hash,
        source_sha256={
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "diagnose_glyph_reconstruction.py",
                "glyph_lm.py",
                "train_glyph_lm.py",
                "train_structured_glyph_lm.py",
                "evaluate_glyph_lm.py",
            )
        },
    )
    del saved
    dataset = GlyphSequenceDataset(args.data, "train", sequence_length=32)
    counts = target_frequencies(dataset)
    selection = select_decoder_glyphs(dataset.glyph_bank, dataset.inventory, counts, args.seed)
    asset_ids = selection["eligible_asset_ids"]
    positions = {asset: index for index, asset in enumerate(asset_ids)}
    cohort_positions = {
        name: np.array([positions[asset] for asset in selection["cohorts"][name]], dtype=np.int64)
        for name in COHORTS
    }
    clean = dataset.glyph_bank[asset_ids].clone()
    corrupted = fixed_bitflips(clean, args.bitflip_count, args.seed + 211)
    output.mkdir(parents=True)
    write_json(output / "selection.json", selection)
    write_json(output / "metadata.json", metadata)
    result = {
        "status": "running",
        "mode": "frozen_encoder_same_glyph_reconstruction",
        "language_generation_evaluated": False,
        "metadata": metadata,
        "selection": selection,
        "selection_sha256": sha256(output / "selection.json"),
        "bitflip_count": args.bitflip_count,
        "bitflip_seed": args.seed + 211,
        "threshold": 0.5,
        "checkpoint_selection": "fixed successful-step final weights; no held-out selection",
        "decoders": {},
    }
    write_json(output / "status.json", {"status": "running", "stage": "encoding"})
    started = time.monotonic()
    try:
        clean_features = encode_images(encoder, clean, device, args.batch_size, precision)
        noisy_features = encode_images(encoder, corrupted, device, args.batch_size, precision)
        feature_groups = defaultdict(set)
        for index, asset in enumerate(asset_ids):
            feature_groups[tensor_hash(clean_features[index])].add(
                selection["pixel_sha256"][str(asset)]
            )
        result["feature_diagnostic"] = {
            "saved_dtype": str(clean_features.dtype),
            "shape": list(clean_features.shape),
            "exact_distinct_pixel_embedding_collision_groups": sum(
                len(values) > 1 for values in feature_groups.values()
            ),
            "scope": "Exact collisions at the recorded encoder computation precision only",
        }
        torch.save(
            {
                "clean": clean_features,
                "bitflip": noisy_features,
                "asset_ids": asset_ids,
                "encoder_state_sha256": encoder_hash,
            },
            output / "visual_features.pt",
        )
        result["feature_cache_sha256"] = sha256(output / "visual_features.pt")
        labels = {
            int(asset): character for character, asset in dataset.inventory["characters"].items()
        }
        for kind in ("linear", "spatial"):
            torch.manual_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed)
                torch.cuda.reset_peak_memory_stats(device)
            decoder = ReconstructionDecoder(model_config.hidden_size, kind).to(device)
            value = {"parameters": sum(parameter.numel() for parameter in decoder.parameters())}
            initial_hash = module_hash(decoder)
            fit_indices = cohort_positions["decoder_fit"]
            initial_predictions, initial_nll = predict_images(
                decoder,
                clean_features[fit_indices],
                clean[fit_indices],
                device,
                args.batch_size,
                precision,
            )
            initial_metrics = glyph_metrics(initial_predictions, clean[fit_indices], initial_nll)
            initial_metrics.pop("per_glyph")
            value["random_decoder_initial_clean_fit"] = initial_metrics
            value["training"] = train_decoder(
                decoder,
                clean_features,
                clean,
                cohort_positions["decoder_fit"],
                args,
                device,
                precision,
            )
            value["decoder_weights_changed"] = module_hash(decoder) != initial_hash
            if not value["decoder_weights_changed"]:
                raise RuntimeError("Fresh reconstruction decoder weights did not update")
            predictions, nll = predict_images(
                decoder, clean_features, clean, device, args.batch_size, precision
            )
            noisy_predictions, noisy_nll = predict_images(
                decoder, noisy_features, clean, device, args.batch_size, precision
            )
            value["evaluation"] = {"clean": {}, "bitflip": {}}
            rows = {
                asset: {
                    "asset_id": asset,
                    "character": labels[asset],
                    "train_frequency": int(counts[asset]),
                    "pixel_sha256": selection["pixel_sha256"][str(asset)],
                }
                for asset in asset_ids
            }
            for cohort, indices in cohort_positions.items():
                for mode, predicted, loss in (
                    ("clean", predictions, nll),
                    ("bitflip", noisy_predictions, noisy_nll),
                ):
                    metrics = glyph_metrics(predicted[indices], clean[indices], loss[indices])
                    per_glyph = metrics.pop("per_glyph")
                    value["evaluation"][mode][cohort] = metrics
                    for offset, position in enumerate(indices):
                        row = rows[asset_ids[position]]
                        row["cohort"] = cohort
                        row.update(
                            {
                                f"{mode}_{metric}": entries[offset]
                                for metric, entries in per_glyph.items()
                            }
                        )
                shown = sorted(
                    indices,
                    key=lambda position: hashlib.sha256(
                        f"{args.seed}:show:{asset_ids[position]}".encode()
                    ).hexdigest(),
                )[:16]
                reconstruction_sheet(
                    output / f"{kind}_{cohort}.png",
                    clean[shown],
                    corrupted[shown],
                    predictions[shown],
                    noisy_predictions[shown],
                    [labels[asset_ids[position]] for position in shown],
                )
            with (output / f"{kind}_per_glyph.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(next(iter(rows.values()))))
                writer.writeheader()
                writer.writerows(rows.values())
            np.savez_compressed(
                output / f"{kind}_binary_arrays.npz",
                asset_ids=np.array(asset_ids, dtype=np.int64),
                reference=clean.numpy(),
                bitflip_input=corrupted.numpy(),
                prediction=predictions.numpy(),
                bitflip_prediction=noisy_predictions.numpy(),
            )
            torch.save(
                {
                    "kind": kind,
                    "hidden_size": model_config.hidden_size,
                    "decoder": decoder.state_dict(),
                },
                output / f"{kind}_decoder.pt",
            )
            value["peak_cuda_allocated_bytes"] = (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            )
            value["decoder_sha256"] = sha256(output / f"{kind}_decoder.pt")
            result["decoders"][kind] = value
            write_json(output / "metrics.json", result)
            del decoder
        if module_hash(encoder) != encoder_hash or any(
            parameter.grad is not None for parameter in encoder.parameters()
        ):
            raise RuntimeError("The supposedly frozen encoder changed or received gradients")
        result.update(
            status="complete", encoder_unchanged=True, wall_seconds=time.monotonic() - started
        )
        result["artifacts_sha256"] = {
            path.name: sha256(path)
            for path in sorted(output.iterdir())
            if path.suffix in {".pt", ".csv", ".npz", ".png"}
        }
        write_json(output / "metrics.json", result)
        write_report(output, result)
        write_json(
            output / "status.json",
            {"status": "complete", "metrics_sha256": sha256(output / "metrics.json")},
        )
    except BaseException as error:
        write_json(
            output / "status.json",
            {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--bitflip-count", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    diagnose(parser.parse_args())


if __name__ == "__main__":
    main()
