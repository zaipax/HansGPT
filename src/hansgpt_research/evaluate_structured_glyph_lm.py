"""Complete-split teacher-forced scoring of v1 and structured binary glyph models."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hansgpt_research.diagnose_glyph_generation import IndependentPrediction, load_model
from hansgpt_research.evaluate_glyph_lm import (
    BinaryMetrics,
    glyph_categories,
    nearest_glyphs,
    new_retrieval_counts,
    safe_ratio,
)
from hansgpt_research.glyph_lm import GlyphSequenceDataset, collate_glyph_sequences
from hansgpt_research.structured_glyph_lm import DECODE_STRATEGIES
from hansgpt_research.train_glyph_lm import (
    autocast_context,
    canonical_hash,
    move_batch,
    runtime_metadata,
    sha256,
    write_json,
)


def forward_hidden(model, glyphs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_hidden"):
        return model.forward_hidden(glyphs, attention_mask=attention_mask, use_cache=False)
    # This is exactly v1's forward path before its pixel head, allowing bounded head scoring.
    embeddings = model.encode_glyphs(glyphs)
    return model.backbone(
        inputs_embeds=embeddings, attention_mask=attention_mask, use_cache=False, return_dict=True
    ).last_hidden_state


def distribution_for_hidden(model, hidden: torch.Tensor):
    if hasattr(model, "distribution"):
        return model.distribution(hidden)
    return IndependentPrediction(model.pixel_head(hidden).reshape(-1, 1, 32, 32))


def decode_and_score(
    distribution,
    targets: torch.Tensor,
    *,
    threshold: float,
    strategy: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    nll = distribution.nll(targets).float()
    if nll.shape != targets.shape[:1] or not bool(torch.isfinite(nll).all()):
        raise FloatingPointError("Invalid or nonfinite exact next-grid NLL")
    if bool((nll < -1e-6).any()):
        raise FloatingPointError("Negative NLL from a normalized binary distribution")
    binary = distribution.decode(threshold=threshold, strategy=strategy, generator=generator)
    if (
        binary.shape != targets.shape
        or binary.dtype != torch.uint8
        or not bool(((binary == 0) | (binary == 1)).all())
    ):
        raise ValueError("Scoring requires strict uint8 binary 32x32 decoder outputs")
    return binary, nll


def content_candidates(
    ids: torch.Tensor, distances: torch.Tensor, control_ids: torch.Tensor, content_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover content Top-k from full Top-(k + number_of_controls), preserving order."""
    eligible = ~torch.isin(ids, control_ids)
    if content_k < 1 or bool((eligible.sum(-1) < content_k).any()):
        raise ValueError("Full-gallery candidates do not contain enough content glyphs")
    positions = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
    selected = (
        torch.where(eligible, positions, ids.shape[1])
        .topk(content_k, largest=False, sorted=True)
        .indices
    )
    return ids.gather(1, selected), distances.gather(1, selected)


class SplitAccumulator:
    def __init__(
        self, dataset, device: torch.device, *, query_chunk: int = 256, gallery_chunk: int = 2048
    ):
        self.bank = dataset.glyph_bank.to(device)
        self.full_ids = torch.arange(len(self.bank), device=device)
        self.controls = torch.tensor(list(dataset.control_ids.values()), device=device)
        self.content_ids = torch.tensor(dataset.gallery_ids, device=device)
        if not len(self.content_ids):
            raise ValueError("Evaluation needs a nonempty content gallery")
        categories = glyph_categories(dataset.inventory)
        self.category_ids = {
            name: torch.tensor(ids, dtype=torch.long, device=device)
            for name, ids in categories.items()
        }
        self.metrics = {
            name: BinaryMetrics()
            for name in (
                "all_targets",
                "content_only",
                "han_only",
                "punctuation_only",
                "controls_only",
            )
        }
        self.retrieval = {name: new_retrieval_counts() for name in self.metrics}
        self.query_chunk, self.gallery_chunk = query_chunk, gallery_chunk

    @torch.inference_mode()
    def add(
        self,
        prediction: torch.Tensor,
        targets: torch.Tensor,
        target_ids: torch.Tensor,
        nll_per_pixel: torch.Tensor,
    ) -> None:
        if len(prediction) != len(target_ids) or nll_per_pixel.shape != target_ids.shape:
            raise ValueError("Predictions, target IDs and exact likelihood counts must align")
        control = torch.isin(target_ids, self.controls)
        masks = {
            "all_targets": torch.ones_like(control),
            "content_only": ~control,
            "controls_only": control,
            **{name: torch.isin(target_ids, ids) for name, ids in self.category_ids.items()},
        }
        full_k = min(len(self.full_ids), 5 + len(self.controls))
        all_ids, all_distances = nearest_glyphs(
            prediction,
            self.bank,
            self.full_ids,
            query_chunk=self.query_chunk,
            gallery_chunk=self.gallery_chunk,
            k=full_k,
        )
        content_ids, content_distances = content_candidates(
            all_ids, all_distances, self.controls, min(5, len(self.content_ids))
        )
        for name, mask in masks.items():
            # BinaryMetrics aggregates nats/grid, regardless of the output distribution family.
            self.metrics[name].add(prediction[mask], targets[mask], nll_per_pixel[mask] * 1024)
            candidates, distances = (
                (all_ids[:, :5], all_distances[:, :5])
                if name in {"all_targets", "controls_only"}
                else (content_ids, content_distances)
            )
            actual = target_ids[mask]
            selected = candidates[mask]
            counts = self.retrieval[name]
            counts["targets"] += len(actual)
            counts["top1"] += int((selected[:, 0] == actual).sum())
            counts["top5"] += int((selected == actual[:, None]).any(-1).sum())
            counts["nearest_hamming_sum"] += float(distances[mask, 0].double().sum())
            counts["legal_content_bitmaps"] += int((distances[mask, 0] == 0).sum())

    def result(self) -> dict:
        output = {}
        for name, metrics in self.metrics.items():
            values = metrics.result()
            values["nll_per_pixel"] = values.pop("bce_per_pixel")
            # Absence of a target category is not a measured zero score.
            if not metrics.tiles:
                values = {
                    key: value if key in {"tiles", "tp", "fp", "fn", "tn"} else None
                    for key, value in values.items()
                }
            counts = self.retrieval[name]
            all_gallery = name in {"all_targets", "controls_only"}
            denominator = counts["targets"]
            retrieval = {
                "targets": denominator,
                "top1_count": counts["top1"],
                "top5_count": counts["top5"],
                "top1_accuracy": safe_ratio(counts["top1"], denominator) if denominator else None,
                "top5_accuracy": safe_ratio(counts["top5"], denominator) if denominator else None,
                "mean_nearest_hamming_bits": safe_ratio(counts["nearest_hamming_sum"], denominator)
                if denominator
                else None,
                "exact_gallery_bitmap_rate": safe_ratio(
                    counts["legal_content_bitmaps"], denominator
                )
                if denominator
                else None,
                "gallery_size": len(self.full_ids) if all_gallery else len(self.content_ids),
                "gallery_scope": "all content and all four control glyphs"
                if all_gallery
                else "all content glyphs; all controls excluded, matching v1 content retrieval",
                "tie_break": "lowest inventory ID; identical bitmaps cannot distinguish labels",
            }
            output[name] = {**values, "retrieval": retrieval}
        if (
            output["content_only"]["tiles"] + output["controls_only"]["tiles"]
            != output["all_targets"]["tiles"]
            or output["han_only"]["tiles"] + output["punctuation_only"]["tiles"]
            != output["content_only"]["tiles"]
        ):
            raise ValueError("Target category partitions are incomplete or overlapping")
        return output


@torch.inference_mode()
def score_full_split(
    model,
    dataset,
    device: torch.device,
    *,
    threshold: float = 0.45,
    strategy: str = "mode_threshold",
    batch_size: int = 32,
    head_chunk_size: int = 256,
    num_workers: int = 0,
    precision: str = "fp16",
    seed: int = 20260908,
    query_chunk: int = 256,
    gallery_chunk: int = 2048,
) -> dict:
    if not 0 < threshold < 1 or strategy not in DECODE_STRATEGIES:
        raise ValueError("Need a fixed threshold in (0,1) and supported decoding strategy")
    if min(batch_size, head_chunk_size, query_chunk, gallery_chunk) < 1 or num_workers < 0:
        raise ValueError("Invalid evaluation batching configuration")
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_glyph_sequences,
        generator=torch.Generator().manual_seed(seed),
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    accumulator = SplitAccumulator(
        dataset, device, query_chunk=query_chunk, gallery_chunk=gallery_chunk
    )
    blocks_seen = batches_seen = targets_seen = 0
    started = time.monotonic()
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        valid = batch["loss_mask"].bool()
        with autocast_context(device, precision):
            hidden = forward_hidden(model, batch["glyphs"], batch["attention_mask"])[valid]
        targets, target_ids = batch["targets"][valid], batch["target_ids"][valid]
        if not bool(torch.isfinite(hidden).all()):
            raise FloatingPointError("Nonfinite hidden states in complete-split evaluation")
        for start in range(0, len(hidden), head_chunk_size):
            end = start + head_chunk_size
            with autocast_context(device, precision):
                distribution = distribution_for_hidden(model, hidden[start:end])
                prediction, nll = decode_and_score(
                    distribution,
                    targets[start:end],
                    threshold=threshold,
                    strategy=strategy,
                    generator=generator,
                )
            accumulator.add(prediction, targets[start:end], target_ids[start:end], nll)
            targets_seen += len(prediction)
        blocks_seen += len(batch["glyphs"])
        batches_seen += 1
        if batches_seen == 1 or batches_seen % 100 == 0:
            print(
                json.dumps(
                    {
                        "stage": "full_split_evaluation",
                        "batches": batches_seen,
                        "blocks": blocks_seen,
                        "targets": targets_seen,
                        "expected_targets": dataset.target_count,
                    }
                ),
                flush=True,
            )
    groups = accumulator.result()
    if (
        targets_seen != dataset.target_count
        or blocks_seen != len(dataset)
        or groups["all_targets"]["tiles"] != dataset.target_count
    ):
        raise ValueError("Evaluation did not cover the full split exactly once")
    return {
        "status": "complete",
        "split": dataset.split,
        "scope": "entire split; sequential document-isolated teacher-forced next-grid evaluation",
        "coverage": {
            "blocks_seen": blocks_seen,
            "expected_blocks": len(dataset),
            "targets_seen": targets_seen,
            "expected_targets": dataset.target_count,
            "batches_seen": batches_seen,
            "full_split_verified": True,
        },
        "threshold": threshold,
        "strategy": strategy,
        "threshold_selection": (
            "provided to CLI; no threshold or model selection is performed on this split"
        ),
        "likelihood": (
            "exact normalized whole-grid model NLL divided by 1024; mixture models sum "
            "pixel log-probabilities within components, then marginalize components; "
            "not BCE of component-averaged pixel probabilities"
        ),
        "decoding_note": (
            "NLL scores the full model distribution independently of decoder choice; "
            "binary metrics score one decoded grid per real reference prefix"
        ),
        "elapsed_seconds": time.monotonic() - started,
        "baselines": {
            "status": "not_run",
            "note": "This evaluation scores the supplied candidate only.",
        },
        **groups,
    }


def validate_checkpoint_identity(saved: dict, current_metadata: dict) -> dict:
    original = saved["metadata"]
    if original.get("config_sha256") != canonical_hash(original["config"]):
        raise ValueError("Checkpoint config does not match its recorded configuration hash")
    for key in ("data_sha256", "data_manifest_sha256", "data_verification_sha256"):
        if original.get(key) != current_metadata.get(key):
            raise ValueError(f"Checkpoint and current verified dataset differ: {key}")
    return {
        "training_git_commit": original["git_commit"],
        "training_run_name": original.get("run_name"),
        "training_config_sha256": original["config_sha256"],
        "checkpoint_progress": saved.get("progress", {}),
        "training_mode": original.get("mode"),
        "training_completion": (
            "not inferred from a supplied checkpoint; verify separate training receipt"
        ),
        "config_and_data_identity_verified": True,
    }


def markdown_report(result: dict, path: Path) -> None:
    lines = [
        "# 完整拆分二值字形评估",
        "",
        f"完整遍历 `{result['split']}`，共 {result['coverage']['targets_seen']:,} 个有效目标。",
        f"阈值 `{result['threshold']}`；策略 `{result['strategy']}`。评估器没有在该拆分选择阈值。",
        "NLL 为模型的精确整格负对数似然除以 1024；混合模型不是对平均像素概率计算 BCE。",
        "",
        "| 范围 | 目标数 | NLL / pixel | 前景 F1 | IoU | 整格匹配 | Top1 | Top5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("all_targets", "content_only", "han_only", "punctuation_only", "controls_only"):
        metrics = result[name]
        values = [
            metrics[key] for key in ("nll_per_pixel", "foreground_f1", "iou", "exact_bitmap_match")
        ]
        values.extend(metrics["retrieval"][key] for key in ("top1_accuracy", "top5_accuracy"))
        rendered = " | ".join("N/A" if value is None else f"{value:.6f}" for value in values)
        lines.append(f"| {name} | {metrics['tiles']:,} | {rendered} |")
    lines.extend(
        [
            "",
            "内容／汉字／标点检索使用同一完整内容字形库，排除控制格；"
            "全部目标和控制格检索使用包括控制格的完整库。各项候选范围保存在 JSON。",
            "这是给定真实前文的完整拆分评分，不代表原始图反馈的自由生成或语义验收已经通过。",
            "本次未重新评测基线；NLL 与解码策略分开解释。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    output, data = Path(args.output), Path(args.data)
    if output.exists():
        raise FileExistsError("Choose a new versioned full-split evaluation output directory")
    device = torch.device(args.device)
    if device.type == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0 for the authorized GPU")
    torch.set_float32_matmul_precision("highest")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = saved["metadata"]["config"]
    metadata = runtime_metadata(config, data, device, mode="evaluation")
    identity = validate_checkpoint_identity(saved, metadata)
    seed = config["training"]["seed"] if args.seed is None else args.seed
    metadata.update(
        arguments=vars(args),
        seed=seed,
        checkpoint_identity=identity,
        checkpoint_sha256=sha256(Path(args.checkpoint)),
        model_family=saved["metadata"].get("model_family", "glyph_gpt_v1"),
        inference_only=True,
        training_config_seed=config["training"]["seed"],
    )
    metadata["model_revision"] = metadata["checkpoint_sha256"]
    model = load_model(saved, device)
    dataset = GlyphSequenceDataset(data, args.split, config["training"]["sequence_length"])
    metadata["evaluation_sequence_length"] = dataset.sequence_length
    metadata["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    output.mkdir(parents=True)
    write_json(output / "metadata.json", metadata)
    try:
        result = score_full_split(
            model,
            dataset,
            device,
            threshold=args.threshold,
            strategy=args.strategy,
            batch_size=args.batch_size,
            head_chunk_size=args.head_chunk_size,
            num_workers=args.num_workers,
            precision=config["training"]["precision"],
            seed=seed,
            query_chunk=args.retrieval_query_chunk,
            gallery_chunk=args.retrieval_gallery_chunk,
        )
        write_json(output / "metrics.json", result)
        markdown_report(result, output / "REPORT.md")
        write_json(
            output / "evaluation_complete.json",
            {
                "status": "complete",
                "scope": "full split",
                "split": args.split,
                "coverage": result["coverage"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "metadata_sha256": sha256(output / "metadata.json"),
                "metrics_sha256": sha256(output / "metrics.json"),
                "report_sha256": sha256(output / "REPORT.md"),
                "checkpoint_data_binding_verified": True,
                "semantic_or_free_generation_acceptance": "not evaluated by this command",
            },
        )
    except BaseException as error:
        write_json(
            output / "evaluation_failed.json",
            {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--strategy", choices=sorted(DECODE_STRATEGIES), default="mode_threshold")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-chunk-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--retrieval-query-chunk", type=int, default=256)
    parser.add_argument("--retrieval-gallery-chunk", type=int, default=2048)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
