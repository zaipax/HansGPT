"""Full held-out evaluation and binary-feedback generation for glyph-native GPT."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import regex
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from hansgpt_research.glyph_lm import (
    GlyphGPT,
    GlyphSequenceDataset,
    ModelConfig,
    collate_glyph_sequences,
)
from hansgpt_research.train_glyph_lm import (
    autocast_context,
    move_batch,
    runtime_metadata,
    sha256,
    write_json,
)


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def glyph_categories(inventory: dict) -> dict[str, list[int]]:
    """Classify evaluation labels only; character IDs never enter the model."""
    categories = {"han_only": [], "punctuation_only": []}
    for character, identifier in inventory["characters"].items():
        if regex.fullmatch(r"[\p{Unified_Ideograph}〇]", character):
            category = "han_only"
        elif regex.fullmatch(r"\p{P}", character):
            category = "punctuation_only"
        else:
            raise ValueError("Content inventory label is neither one Han character nor punctuation")
        categories[category].append(int(identifier))
    return {name: sorted(identifiers) for name, identifiers in categories.items()}


def category_bitmap_sets(glyph_bank: torch.Tensor, categories: dict) -> dict[str, set[bytes]]:
    """Exact packed bitmap membership preserves collisions, including across categories."""
    bitmaps = np.packbits(glyph_bank.detach().cpu().numpy().reshape(len(glyph_bank), 1024), axis=1)
    return {
        name: {bitmaps[index].tobytes() for index in identifiers}
        for name, identifiers in categories.items()
    }


def exact_category_memberships(
    predictions: torch.Tensor, bitmap_sets: dict
) -> dict[str, np.ndarray]:
    packed = np.packbits(predictions.detach().cpu().numpy().reshape(len(predictions), 1024), axis=1)
    keys = [bitmap.tobytes() for bitmap in packed]
    return {
        name: np.array([key in members for key in keys], dtype=bool)
        for name, members in bitmap_sets.items()
    }


def new_retrieval_counts() -> dict:
    return {
        "targets": 0,
        "top1": 0,
        "top5": 0,
        "nearest_hamming_sum": 0.0,
        "legal_content_bitmaps": 0,
    }


def summarize_retrieval(counts: dict, gallery_size: int) -> dict:
    return {
        **counts,
        "top1_accuracy": safe_ratio(counts["top1"], counts["targets"]),
        "top5_accuracy": safe_ratio(counts["top5"], counts["targets"]),
        "mean_nearest_hamming_bits": safe_ratio(counts["nearest_hamming_sum"], counts["targets"]),
        "legal_content_bitmap_rate": safe_ratio(counts["legal_content_bitmaps"], counts["targets"]),
        "gallery_size": gallery_size,
        "gallery_scope": "all rendered corpus content glyphs; all controls excluded",
        "tie_break": (
            "lowest glyph inventory ID; pixel collisions cannot identify a unique character"
        ),
    }


class BinaryMetrics:
    """Micro pixel counts; only true target positions contribute to denominators."""

    def __init__(self):
        self.tiles = self.tp = self.fp = self.fn = self.tn = self.exact = 0
        self.bce = 0.0

    def add(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        per_tile_bce: torch.Tensor | None = None,
    ) -> None:
        prediction, target = prediction.bool().flatten(1), target.bool().flatten(1)
        self.tiles += len(prediction)
        self.tp += int((prediction & target).sum().item())
        self.fp += int((prediction & ~target).sum().item())
        self.fn += int((~prediction & target).sum().item())
        self.tn += int((~prediction & ~target).sum().item())
        self.exact += int((prediction == target).all(-1).sum().item())
        if per_tile_bce is not None:
            self.bce += float(per_tile_bce.double().sum().item())

    def result(self) -> dict:
        pixels = self.tiles * 1024
        f1 = safe_ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn)
        return {
            "tiles": self.tiles,
            "bce_per_pixel": safe_ratio(self.bce, pixels),
            "nll_nats_per_grid": safe_ratio(self.bce, self.tiles),
            "foreground_precision": safe_ratio(self.tp, self.tp + self.fp),
            "foreground_recall": safe_ratio(self.tp, self.tp + self.fn),
            "foreground_f1": f1,
            "dice": f1,
            "iou": safe_ratio(self.tp, self.tp + self.fp + self.fn),
            "pixel_accuracy": safe_ratio(self.tp + self.tn, pixels),
            "exact_bitmap_match": safe_ratio(self.exact, self.tiles),
            "hamming_bits_per_grid": safe_ratio(self.fp + self.fn, self.tiles),
            "foreground_rate": safe_ratio(self.tp + self.fp, pixels),
            "target_foreground_rate": safe_ratio(self.tp + self.fn, pixels),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


def make_loader(dataset, config: dict, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config["training"]["seed"]),
        collate_fn=collate_glyph_sequences,
    )


@torch.inference_mode()
def validate(
    model, dataset, config: dict, device: torch.device, max_batches: int | None = None
) -> dict:
    model.eval()
    thresholds = config["evaluation"]["threshold_candidates"]
    counts = {threshold: BinaryMetrics() for threshold in thresholds}
    controls = torch.tensor(list(dataset.control_ids.values()), device=device)
    content_counts = {threshold: BinaryMetrics() for threshold in thresholds}
    total = BinaryMetrics()
    for batch_index, cpu_batch in enumerate(make_loader(dataset, config, device)):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(cpu_batch, device)
        with autocast_context(device, config["training"]["precision"]):
            logits = model(batch["glyphs"], attention_mask=batch["attention_mask"])
        mask = batch["loss_mask"].bool()
        targets = batch["targets"][mask].float().flatten(1)
        logits = logits[mask].float().flatten(1)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite validation logits")
        per_tile_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").sum(-1)
        probability = logits.sigmoid()
        content = ~torch.isin(batch["target_ids"][mask], controls)
        total.add(probability >= 0.5, targets, per_tile_bce)
        for threshold in thresholds:
            binary = probability >= threshold
            counts[threshold].add(binary, targets, per_tile_bce)
            content_counts[threshold].add(binary[content], targets[content], per_tile_bce[content])
    if total.tiles == 0:
        raise ValueError("Empty validation targets")
    if max_batches is None and total.tiles != dataset.target_count:
        raise ValueError("Validation did not cover every valid next-grid target exactly once")
    # Deterministic tie break prefers the threshold nearest 0.5, then the lower threshold.
    selected = max(
        thresholds, key=lambda t: (content_counts[t].result()["foreground_f1"], -abs(t - 0.5), -t)
    )
    return {
        **total.result(),
        "selected_threshold": selected,
        "threshold_selection": "maximum validation content-pixel micro F1; controls excluded",
        "threshold_metrics": {
            str(t): {"all_targets": counts[t].result(), "content_only": content_counts[t].result()}
            for t in thresholds
        },
        "scope": "full validation" if max_batches is None else f"first {max_batches} batches",
    }


def target_frequencies(dataset) -> np.ndarray:
    """Count valid next-grid labels from one specified split, including EOS."""
    bank_size = len(dataset.glyph_bank)
    counts = np.zeros(bank_size, dtype=np.int64)
    tokens = np.memmap(dataset.token_path, mode="r", dtype="<u2")
    for start in range(0, len(tokens), 1_000_000):
        counts += np.bincount(
            tokens[start : start + 1_000_000].astype(np.int64), minlength=bank_size
        )
    # BOS/PAD are never valid next-grid targets in the paragraph format.
    for name in ("BOS", "PAD"):
        if name in dataset.control_ids:
            counts[dataset.control_ids[name]] = 0
    if int(counts.sum()) != dataset.target_count:
        raise ValueError("Target frequency count does not match dataset supervised positions")
    return counts


def select_constant_threshold(
    probability: torch.Tensor,
    validation_counts: torch.Tensor,
    glyph_bank: torch.Tensor,
    control_ids: list[int],
    thresholds: list[float],
) -> dict:
    """Tune a fixed-pixel baseline from validation sufficient statistics only."""
    counts = validation_counts.clone().double()
    counts[control_ids] = 0
    tiles = int(counts.sum().item())
    if tiles <= 0:
        raise ValueError("Baseline threshold selection needs validation content targets")
    foreground = counts @ glyph_bank.double().flatten(1)
    candidates = {}
    for threshold in thresholds:
        prediction = probability.flatten() >= threshold
        true_positive = float(foreground[prediction].sum())
        target_positive = float(foreground.sum())
        predicted_positive = int(prediction.sum()) * tiles
        candidates[str(threshold)] = {
            "foreground_f1": safe_ratio(2 * true_positive, predicted_positive + target_positive),
            "iou": safe_ratio(true_positive, predicted_positive + target_positive - true_positive),
        }
    selected = max(
        thresholds,
        key=lambda t: (candidates[str(t)]["foreground_f1"], -abs(t - 0.5), -t),
    )
    return {
        "selected_threshold": selected,
        "selection_scope": "entire validation content targets; all controls excluded",
        "selection_metric": "maximum validation content-pixel micro F1",
        "validation_content_tiles": tiles,
        "threshold_metrics": candidates,
        "probability_fit_scope": "training next-grid targets only, including EOS",
    }


@torch.inference_mode()
def nearest_glyphs(
    predictions: torch.Tensor,
    gallery: torch.Tensor,
    gallery_ids: torch.Tensor,
    *,
    query_chunk: int = 256,
    gallery_chunk: int = 2048,
    k: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact binary Hamming search; gallery chunks prevent query×vocabulary×pixels allocation."""
    if not len(gallery):
        raise ValueError("Empty content glyph gallery")
    k = min(k, len(gallery))
    query = predictions.float().flatten(1)
    gallery = gallery.float().flatten(1)
    id_outputs, distance_outputs = [], []
    for offset in range(0, len(query), query_chunk):
        part = query[offset : offset + query_chunk]
        distances = torch.full((len(part), k), float("inf"), device=part.device)
        indices = torch.zeros((len(part), k), dtype=torch.long, device=part.device)
        for start in range(0, len(gallery), gallery_chunk):
            candidate = gallery[start : start + gallery_chunk]
            hamming = part.sum(-1, keepdim=True) + candidate.sum(-1)[None] - 2 * part @ candidate.T
            # Integer-valued distances in float32; deterministic gallery index breaks exact ties.
            candidate_ids = gallery_ids[start : start + gallery_chunk].expand(len(part), -1)
            all_distances = torch.cat((distances, hamming), dim=1)
            all_ids = torch.cat((indices, candidate_ids), dim=1)
            tie = all_distances.double() + all_ids.double() * 1e-9
            selected = tie.topk(k, largest=False, sorted=True).indices
            distances = all_distances.gather(1, selected)
            indices = all_ids.gather(1, selected)
        id_outputs.append(indices)
        distance_outputs.append(distances)
    if not id_outputs:
        return (
            torch.empty((0, k), dtype=torch.long, device=query.device),
            torch.empty((0, k), device=query.device),
        )
    return torch.cat(id_outputs), torch.cat(distance_outputs)


@torch.inference_mode()
def evaluate_split(
    model,
    dataset,
    train_dataset,
    validation_dataset,
    config: dict,
    device: torch.device,
    threshold: float,
) -> dict:
    model.eval()
    counts = target_frequencies(train_dataset)
    bank = dataset.glyph_bank.to(device).float().flatten(1)
    weights = torch.tensor(counts, device=device, dtype=torch.float64)
    pixel_frequency = (weights @ bank.double() / weights.sum()).float().clamp(1e-6, 1 - 1e-6)
    baseline_selection = select_constant_threshold(
        pixel_frequency,
        torch.tensor(target_frequencies(validation_dataset), device=device),
        bank,
        list(dataset.control_ids.values()),
        config["evaluation"]["threshold_candidates"],
    )
    baseline_threshold = baseline_selection["selected_threshold"]
    frequency_logits = torch.logit(pixel_frequency)
    gallery_ids = torch.tensor(dataset.gallery_ids, device=device, dtype=torch.long)
    gallery = bank[gallery_ids]
    controls = torch.tensor(list(dataset.control_ids.values()), device=device)
    all_metrics, content_metrics = BinaryMetrics(), BinaryMetrics()
    categories = glyph_categories(dataset.inventory)
    category_id_tensors = {
        name: torch.tensor(identifiers, dtype=torch.long, device=device)
        for name, identifiers in categories.items()
    }
    category_metrics = {name: BinaryMetrics() for name in categories}
    baselines = {"all_background": BinaryMetrics(), "train_pixel_frequency": BinaryMetrics()}
    baseline_content = {name: BinaryMetrics() for name in baselines}
    baseline_categories = {
        baseline: {name: BinaryMetrics() for name in categories} for baseline in baselines
    }
    bins = {
        name: BinaryMetrics()
        for name in ("unseen", "rare_1_99", "medium_100_9999", "head_10000_plus")
    }
    retrieval = new_retrieval_counts()
    category_retrieval = {name: new_retrieval_counts() for name in categories}
    retrieval_bins = {name: {"targets": 0, "top1": 0, "top5": 0} for name in bins}
    train_counts = torch.tensor(counts, device=device)
    for index, cpu_batch in enumerate(make_loader(dataset, config, device)):
        batch = move_batch(cpu_batch, device)
        with autocast_context(device, config["training"]["precision"]):
            logits = model(batch["glyphs"], attention_mask=batch["attention_mask"])
        mask = batch["loss_mask"].bool()
        targets = batch["targets"][mask].float().flatten(1)
        target_ids = batch["target_ids"][mask]
        logits = logits[mask].float().flatten(1)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite test logits")
        tile_bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").sum(-1)
        binary = logits.sigmoid() >= threshold
        content = ~torch.isin(target_ids, controls)
        category_masks = {
            name: torch.isin(target_ids, identifiers)
            for name, identifiers in category_id_tensors.items()
        }
        all_metrics.add(binary, targets, tile_bce)
        content_metrics.add(binary[content], targets[content], tile_bce[content])
        for name, selected in category_masks.items():
            category_metrics[name].add(binary[selected], targets[selected], tile_bce[selected])
        for name, probability in (
            ("all_background", torch.full_like(pixel_frequency, 1e-6)),
            ("train_pixel_frequency", pixel_frequency),
        ):
            baseline_logits = (
                torch.logit(probability) if name == "all_background" else frequency_logits
            )
            baseline_bce = F.binary_cross_entropy_with_logits(
                baseline_logits.expand_as(targets), targets, reduction="none"
            ).sum(-1)
            binary_baseline = (
                torch.zeros_like(targets, dtype=torch.bool)
                if name == "all_background"
                else (probability >= baseline_threshold).expand_as(targets)
            )
            baselines[name].add(binary_baseline, targets, baseline_bce)
            baseline_content[name].add(
                binary_baseline[content], targets[content], baseline_bce[content]
            )
            for category, selected in category_masks.items():
                baseline_categories[name][category].add(
                    binary_baseline[selected], targets[selected], baseline_bce[selected]
                )
        predicted_ids, distances = nearest_glyphs(
            binary[content],
            gallery,
            gallery_ids,
            query_chunk=config["evaluation"]["retrieval_query_chunk"],
            gallery_chunk=config["evaluation"]["retrieval_gallery_chunk"],
        )
        actual = target_ids[content]
        top1 = predicted_ids[:, 0] == actual
        top5 = (predicted_ids == actual[:, None]).any(-1)
        retrieval["targets"] += len(actual)
        retrieval["top1"] += int(top1.sum())
        retrieval["top5"] += int(top5.sum())
        retrieval["nearest_hamming_sum"] += float(distances[:, 0].sum())
        retrieval["legal_content_bitmaps"] += int((distances[:, 0] == 0).sum())
        for name, mask in category_masks.items():
            selected = mask[content]
            category_retrieval[name]["targets"] += int(selected.sum())
            category_retrieval[name]["top1"] += int(top1[selected].sum())
            category_retrieval[name]["top5"] += int(top5[selected].sum())
            category_retrieval[name]["nearest_hamming_sum"] += float(distances[selected, 0].sum())
            category_retrieval[name]["legal_content_bitmaps"] += int(
                (distances[selected, 0] == 0).sum()
            )
        frequencies = train_counts[target_ids]
        bin_masks = {
            "unseen": frequencies == 0,
            "rare_1_99": (frequencies > 0) & (frequencies < 100),
            "medium_100_9999": (frequencies >= 100) & (frequencies < 10000),
            "head_10000_plus": frequencies >= 10000,
        }
        for name, bin_mask in bin_masks.items():
            selected = bin_mask & content
            bins[name].add(binary[selected], targets[selected], tile_bce[selected])
            retrieved = bin_mask[content]
            retrieval_bins[name]["targets"] += int(retrieved.sum())
            retrieval_bins[name]["top1"] += int(top1[retrieved].sum())
            retrieval_bins[name]["top5"] += int(top5[retrieved].sum())
        if index % 100 == 0:
            print(
                json.dumps(
                    {"stage": "test", "batches": index + 1, "valid_tiles": all_metrics.tiles}
                ),
                flush=True,
            )
    if not all_metrics.tiles:
        raise ValueError("Empty test split")
    if all_metrics.tiles != dataset.target_count:
        raise ValueError("Test did not cover every valid next-grid target exactly once")
    if sum(metrics.tiles for metrics in category_metrics.values()) != content_metrics.tiles:
        raise ValueError("Han and punctuation target slices do not partition all content targets")
    return {
        "scope": "entire test split; teacher-forced next-grid prediction",
        "threshold": threshold,
        "all_targets": all_metrics.result(),
        "content_only": content_metrics.result(),
        "retrieval": summarize_retrieval(retrieval, len(gallery_ids)),
        **{
            name: {
                **metrics.result(),
                "retrieval": summarize_retrieval(category_retrieval[name], len(gallery_ids)),
            }
            for name, metrics in category_metrics.items()
        },
        "category_definition": {
            "han_only": "inventory label matches exactly one Unified_Ideograph or 〇",
            "punctuation_only": "inventory label matches exactly one Unicode Punctuation",
            "classification_scope": "reference target labels only; model and gallery unchanged",
            "inventory_counts": {
                name: len(identifiers) for name, identifiers in categories.items()
            },
            "collision_note": (
                "Categories partition reference labels, not necessarily bitmap identities."
            ),
        },
        "baselines": {
            name: {
                **value.result(),
                "content_only": baseline_content[name].result(),
                **{
                    category: metrics.result()
                    for category, metrics in baseline_categories[name].items()
                },
            }
            for name, value in baselines.items()
        },
        "baseline_threshold_selection": {
            "train_pixel_frequency": baseline_selection,
            "all_background": {
                "selected_threshold": None,
                "selection_scope": "none; fixed all-zero binary image",
            },
        },
        "baseline_note": (
            "Per-position independent train pixel frequencies; "
            "baseline independently tunes its own validation threshold; "
            "background BCE uses probability 1e-6, not exact zero."
        ),
        "frequency_bins": {
            name: {**value.result(), "retrieval": retrieval_bins[name]}
            for name, value in bins.items()
        },
        "frequency_definition": "training next-target occurrences; controls excluded from bins",
    }


def contact_sheet(
    path: Path, prompt: torch.Tensor, generated: torch.Tensor, *, columns: int = 32
) -> None:
    tiles = torch.cat((prompt, generated)).detach().cpu().numpy().reshape(-1, 32, 32)
    cell, margin = 36, 24
    rows = math.ceil(len(tiles) / columns)
    canvas = Image.new("RGB", (columns * cell, rows * cell + margin), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (2, 4),
        f"Prompt: first {len(prompt)} tiles (blue border); raw binary continuation",
        fill="black",
    )
    for index, tile in enumerate(tiles):
        x, y = (index % columns) * cell + 2, (index // columns) * cell + margin + 2
        canvas.paste(Image.fromarray((255 * (1 - tile)).astype(np.uint8)), (x, y))
        if index < len(prompt):
            draw.rectangle((x - 1, y - 1, x + 32, y + 32), outline="blue")
    canvas.save(path)


@torch.inference_mode()
def generate_examples(
    model, dataset, config: dict, device: torch.device, threshold: float, output: Path
) -> list[dict]:
    settings = config["evaluation"]
    length = max(settings["generation_lengths"])
    gallery_ids = torch.tensor(dataset.gallery_ids, device=device)
    bank = dataset.glyph_bank.to(device).float()
    gallery = bank[gallery_ids]
    bitmap_sets = category_bitmap_sets(dataset.glyph_bank, glyph_categories(dataset.inventory))
    labels = {
        int(identifier): character
        for character, identifier in dataset.inventory["characters"].items()
    }
    examples = []
    for index in range(min(settings["generation_examples"], len(dataset))):
        record = dataset[index]
        valid = int(record["attention_mask"].sum())
        prompt_length = min(settings["generation_prompt_tokens"], max(valid - 1, 1))
        prompt = record["glyphs"][:prompt_length].unsqueeze(0).to(device)
        # Fixed horizons intentionally disable EOS stop so long-range collapse is measurable.
        with autocast_context(device, config["training"]["precision"]):
            generated = model.generate(
                prompt, max_new_tokens=length, threshold=threshold, eos_glyph=None, use_cache=True
            )
        if generated.shape[1] != length or not ((generated == 0) | (generated == 1)).all():
            raise ValueError("Generation did not produce the required strict binary sequence")
        predicted_ids, distances = nearest_glyphs(generated[0], gallery, gallery_ids)
        memberships = exact_category_memberships(generated[0], bitmap_sets)
        example = {
            "test_sequence_index": index,
            "prompt_tiles": prompt_length,
            "feedback": "raw thresholded 0/1 predictions; no nearest-glyph projection",
            "decoding": "deterministic threshold; no character vocabulary sampling",
            "category_legality_note": (
                "Independent exact bitmap membership; Han/punctuation rates may overlap if "
                "their font bitmaps collide. Membership does not identify a unique character."
            ),
            "horizons": {},
        }
        for horizon in settings["generation_lengths"]:
            sequence = generated[0, :horizon]
            flat = sequence.flatten(1)
            image_name = f"generation_{index}_{horizon}.png"
            contact_sheet(output / image_name, prompt[0], sequence)
            diagnostic_text = "".join(
                labels[int(identifier)] for identifier in predicted_ids[:horizon, 0].cpu()
            )
            example["horizons"][str(horizon)] = {
                "tiles": horizon,
                "legal_content_bitmap_rate": float((distances[:horizon, 0] == 0).float().mean()),
                "exact_han_bitmap_rate": float(memberships["han_only"][:horizon].mean()),
                "exact_punctuation_bitmap_rate": float(
                    memberships["punctuation_only"][:horizon].mean()
                ),
                "mean_nearest_hamming_bits": float(distances[:horizon, 0].mean()),
                "foreground_rate": float(sequence.float().mean()),
                "adjacent_exact_repeat_rate": float((flat[1:] == flat[:-1]).all(-1).float().mean())
                if horizon > 1
                else 0,
                "unique_bitmaps": int(torch.unique(flat, dim=0).shape[0]),
                "nearest_glyph_transcription_diagnostic_only": diagnostic_text,
                "contact_sheet": image_name,
            }
            reference_length = min(horizon, valid - prompt_length + 1)
            reference = record["targets"][
                prompt_length - 1 : prompt_length - 1 + reference_length
            ].to(device)
            reference_metrics = BinaryMetrics()
            reference_metrics.add(sequence[:reference_length], reference)
            diagnostic = reference_metrics.result()
            # Binary-only generation exposes no normalized sequence probability.
            del diagnostic["bce_per_pixel"], diagnostic["nll_nats_per_grid"]
            example["horizons"][str(horizon)]["same_reference_diagnostic"] = diagnostic
        np.savez_compressed(
            output / f"generation_{index}.npz",
            prompt=prompt.cpu().numpy().astype(np.uint8),
            generated=generated.cpu().numpy().astype(np.uint8),
        )
        examples.append(example)
        print(json.dumps({"stage": "generation", "example_complete": index + 1}), flush=True)
    return examples


def loss_curve(log_path: Path, output: Path) -> bool:
    if not log_path.exists():
        return False
    points = {"train": [], "validation": []}
    for line in log_path.read_text().splitlines():
        record = json.loads(line)
        kind = record.get("kind")
        if kind in points:
            value = (
                record.get("bce_per_pixel")
                if kind == "train"
                else record["metrics"]["bce_per_pixel"]
            )
            points[kind].append((record["valid_tokens"], value))
    all_points = [point for series in points.values() for point in series]
    if not all_points:
        return False
    maximum_x = max(max(x for x, _ in all_points), 1)
    minimum_y, maximum_y = min(y for _, y in all_points), max(y for _, y in all_points)
    maximum_y = max(maximum_y, minimum_y + 0.01)
    width, height, padding = 960, 480, 65
    x_scale = (width - 2 * padding) / maximum_x
    y_scale = (height - 2 * padding) / (maximum_y - minimum_y)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="65" y="25" font-family="sans-serif" font-size="18">'
        "HansGPT unweighted binary cross entropy per pixel</text>",
    ]
    for i in range(6):
        fraction = i / 5
        x, y = (
            padding + fraction * (width - 2 * padding),
            height - padding - fraction * (height - 2 * padding),
        )
        svg.extend(
            [
                f'<text x="{x}" y="{height - 35}" font-size="12">'
                f"{maximum_x * fraction / 1e6:.1f}M</text>",
                f'<text x="8" y="{y}" font-size="12">'
                f"{minimum_y + fraction * (maximum_y - minimum_y):.4f}</text>",
                f'<line x1="{padding}" y1="{y}" x2="{width - padding}" y2="{y}" stroke="#ddd"/>',
            ]
        )
    for name, color in (("train", "#2368bd"), ("validation", "#d1522c")):
        coordinates = " ".join(
            f"{padding + x * x_scale:.2f},{height - padding - (y - minimum_y) * y_scale:.2f}"
            for x, y in points[name]
        )
        svg.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        svg.append(
            f'<text x="{width - 190}" y="{30 if name == "train" else 50}" '
            f'fill="{color}" font-size="14">{escape(name)}</text>'
        )
    svg.append(
        '<text x="370" y="465" font-size="14">Successful-update valid next-grid tokens</text></svg>'
    )
    output.write_text("\n".join(svg), encoding="utf-8")
    return True


def markdown_report(result: dict, path: Path) -> None:
    training, test = result["training_completion"], result["test"]
    progress = training["progress"]
    target_tokens = result["checkpoint_metadata"]["config"]["training"]["target_tokens"]
    lines = [
        "# HansGPT 二值字形语言模型实验结果",
        "",
        "每个输入和输出位置是一张 32×32 二值图。CNN 和因果 Transformer 从零联合训练；"
        "输出为 1024 个 Bernoulli logits，无字符分类头。",
        "",
        f"- 实验：`{result['run_name']}`；训练模式：`{training['mode']}`。",
        f"- 训练提交：`{result['checkpoint_metadata']['git_commit']}`。",
        f"- 评估提交：`{result['evaluation_metadata']['git_commit']}`。",
        f"- 参数：{result['checkpoint_metadata']['parameters']:,}；"
        f"GPU：{result['checkpoint_metadata']['gpu']}；"
        f"精度：{result['checkpoint_metadata']['precision']}。",
        f"- 已更新有效 next-grid Token：{progress['valid_tokens']:,}；"
        f"更新步数：{progress['optimizer_steps']:,}；溢出跳过：{progress['overflow_steps']}。",
        f"- 预声明预算：{target_tokens:,} Token；停止原因：`{training['stop_reason']}`。",
        f"- 训练含验证与存盘用时：{progress['training_seconds'] / 3600:.3f} 小时。",
        f"- 选择检查点：验证 BCE 最优，step={progress['best_step']}，"
        f"Token={progress['best_tokens']:,}。",
        f"- 二值化阈值：{result['validation']['selected_threshold']}，"
        "只使用完整验证集内容像素 F1 选择。",
        "",
        "## 完整测试集：给定真实前文的下一格预测",
        "",
        "| 指标 | 所有有效目标（含 EOS 等控制格） | 内容字形（不含控制格） |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("目标格数", "tiles"),
        ("BCE / pixel", "bce_per_pixel"),
        ("NLL nats / grid", "nll_nats_per_grid"),
        ("前景 F1", "foreground_f1"),
        ("IoU", "iou"),
        ("Dice", "dice"),
        ("整格完全匹配率", "exact_bitmap_match"),
        ("Hamming bits / grid", "hamming_bits_per_grid"),
        ("像素准确率（次要）", "pixel_accuracy"),
    ):
        lines.append(
            f"| {label} | {test['all_targets'][key]:.6f} | {test['content_only'][key]:.6f} |"
        )
    retrieval = test["retrieval"]
    baseline_threshold = test["baseline_threshold_selection"]["train_pixel_frequency"][
        "selected_threshold"
    ]
    lines.extend(
        [
            "",
            f"内容字形库大小：{retrieval['gallery_size']}。按预测二值图的 Hamming 距离检索，"
            f"Top1={retrieval['top1_accuracy']:.6f}，Top5={retrieval['top5_accuracy']:.6f}；"
            f"预测图精确落入内容字形库比例={retrieval['legal_content_bitmap_rate']:.6f}。",
            "字符 ID 仅用于评估标签，模型不接收该 ID。同像素字形碰撞采用最小 ID 打破平局，"
            "不能凭像素区分这些字符。",
            "",
            "## 汉字与标点分别计分",
            "",
            "汉字定义为 Unicode Unified_Ideograph 加〇；标点定义为 Unicode Punctuation。"
            "只按真实目标标签划分，检索仍使用同一个完整内容字形库；"
            "把汉字预测成标点会保留为错误，不会通过缩小候选库被隐藏。",
            "",
            "| 指标 | 汉字目标 | 标点目标 |",
            "|---|---:|---:|",
        ]
    )
    for label, key in (
        ("目标格数", "tiles"),
        ("BCE / pixel", "bce_per_pixel"),
        ("NLL nats / grid", "nll_nats_per_grid"),
        ("前景 F1", "foreground_f1"),
        ("IoU", "iou"),
        ("Dice", "dice"),
        ("整格完全匹配率", "exact_bitmap_match"),
        ("Hamming bits / grid", "hamming_bits_per_grid"),
        ("像素准确率（次要）", "pixel_accuracy"),
    ):
        lines.append(
            f"| {label} | {test['han_only'][key]:.6f} | {test['punctuation_only'][key]:.6f} |"
        )
    for label, key in (
        ("检索目标数", "targets"),
        ("检索 Top1", "top1_accuracy"),
        ("检索 Top5", "top5_accuracy"),
    ):
        lines.append(
            f"| {label} | {test['han_only']['retrieval'][key]:.6f} | "
            f"{test['punctuation_only']['retrieval'][key]:.6f} |"
        )
    lines.extend(
        [
            "",
            "同一位图对应多个标签时，检索仍按最小 ID 打破平局；标签分类不消除字体碰撞。",
            "",
            "## 基线（相同测试参考目标，全部有效目标含控制格）",
            "",
            "| 基线 | BCE / pixel | 前景 F1 | IoU | 整格匹配 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in test["baselines"].items():
        lines.append(
            f"| {name} | {metrics['bce_per_pixel']:.6f} | {metrics['foreground_f1']:.6f} | "
            f"{metrics['iou']:.6f} | {metrics['exact_bitmap_match']:.6f} |"
        )
    lines.extend(
        [
            "",
            "全背景基线的 BCE 以每像素前景概率 10⁻⁶ 计算；"
            "训练像素频率基线仅统计训练 next-target，不使用测试集拟合。",
            f"像素频率基线独立选择阈值 {baseline_threshold}，选择范围为完整验证集的内容目标，"
            "优化内容像素 micro F1；模型阈值独立选择。两者均未使用测试集调参。"
            "基线内容、纯汉字和标点指标分别另存于 metrics.json 的 baselines.* 下。",
            "",
            "## 自由生成：原始二值图反馈",
            "",
            "每条样本生成至 512 格，报告 32、128、512 格前缀。阈值二值化后的图直接反馈 CNN，"
            "不投影到合法字形库；最近字形转写只用于诊断，"
            "不能当作模型实际生成的字符或语义正确率。",
            "",
            "| 样本 | 格数 | 合法内容字形率 | 精确汉字位图率 | 精确标点位图率 | "
            "最近 Hamming | 相邻重复率 | 不同位图数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, example in enumerate(result["generation"]):
        for horizon, metrics in example["horizons"].items():
            lines.append(
                f"| {index} | {horizon} | {metrics['legal_content_bitmap_rate']:.6f} | "
                f"{metrics['exact_han_bitmap_rate']:.6f} | "
                f"{metrics['exact_punctuation_bitmap_rate']:.6f} | "
                f"{metrics['mean_nearest_hamming_bits']:.3f} | "
                f"{metrics['adjacent_exact_repeat_rate']:.6f} | {metrics['unique_bitmaps']} |"
            )
        lines.extend(["", f"![样本 {index} 的原始二值格](generation_{index}_512.png)", ""])
    lines.extend(
        [
            "## 解释范围与复现",
            "",
            "- 测试使用文档/来源隔离，字符允许重叠；这不是未见字符的组合泛化证明。"
            "实际去重、数据来源、清洗数量及限制以 metrics.json 内的数据 manifest 为准。",
            "- BCE 是独立像素条件概率的负对数似然，不是词表语言模型的字符困惑度；"
            "Dice 与此处定义的前景 F1 相同。",
            "- 下一字可能有多个合理答案；单参考字形的 F1、匹配率与检索率不是开放生成语义质量。"
            "独立像素阈值解码可能出现混合轮廓和重复塌缩。",
            "- 字形合法性只表示落入固定字体的位图集合，不能证明语法、事实或对话能力。"
            "尚未进行指令微调、跨字体或未见字评测。",
            "- 自由生成的汉字／标点率分别做精确位图集合匹配；若两类字形发生像素碰撞，"
            "同一生成格可计入两类，因此两率不保证相加等于合法内容率，且不能证明字符身份。",
            "- metrics.json 保存完整配置、源数据与字体 manifest、数据数组校验和、软件版本、"
            "提交、检查点 SHA-256、频次分层和生成诊断；generation_*.npz 保存严格 0/1 原图。",
            "",
            "![训练与验证损失](loss_curve.svg)",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata, config = saved["metadata"], saved["metadata"]["config"]
    data_dir, output = Path(args.data), Path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new evaluation output directory")
    output.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("This experiment is authorized on GPU0: set CUDA_VISIBLE_DEVICES=0")
    evaluation_metadata = runtime_metadata(config, data_dir, device, mode=metadata["mode"])
    if evaluation_metadata["data_sha256"] != metadata["data_sha256"]:
        raise ValueError("Evaluation data does not match the training checkpoint")
    if evaluation_metadata["data_manifest_sha256"] != metadata["data_manifest_sha256"]:
        raise ValueError("Evaluation manifest does not match the training checkpoint")
    completion_path = Path("artifacts/logs") / metadata["run_name"] / "training_complete.json"
    completion = json.loads(completion_path.read_text())
    if completion["status"] != "complete":
        raise ValueError("Training has no successful completion marker")
    if sha256(checkpoint_path) != completion["best_checkpoint_sha256"]:
        raise ValueError("Final evaluation must use the checkpoint selected by validation")
    model = GlyphGPT(ModelConfig.from_dict(config["model"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    datasets = {
        split: GlyphSequenceDataset(data_dir, split, config["training"]["sequence_length"])
        for split in ("train", "validation", "test")
    }
    validation = validate(model, datasets["validation"], config, device)
    write_json(output / "validation.json", validation)
    threshold = validation["selected_threshold"]
    test = evaluate_split(
        model,
        datasets["test"],
        datasets["train"],
        datasets["validation"],
        config,
        device,
        threshold,
    )
    write_json(output / "test.json", test)
    generation = generate_examples(model, datasets["test"], config, device, threshold, output)
    result = {
        "status": "complete",
        "run_name": metadata["run_name"],
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_metadata": metadata,
        "evaluation_metadata": evaluation_metadata,
        "training_completion": completion,
        "validation": validation,
        "test": test,
        "generation": generation,
    }
    write_json(output / "metrics.json", result)
    loss_curve(completion_path.parent / "metrics.jsonl", output / "loss_curve.svg")
    markdown_report(result, output / "REPORT.md")
    write_json(
        output / "evaluation_complete.json",
        {
            "status": "complete",
            "scope": "full held-out split",
            "test_tiles": test["all_targets"]["tiles"],
            "metrics_sha256": sha256(output / "metrics.json"),
            "report_sha256": sha256(output / "REPORT.md"),
        },
    )
    print(
        json.dumps({"status": "complete", "output": str(output), "test": test["content_only"]}),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
