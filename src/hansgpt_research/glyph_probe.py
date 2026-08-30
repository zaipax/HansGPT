from __future__ import annotations

import argparse
import bz2
import hashlib
import importlib.metadata
import json
import math
import random
import subprocess
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import regex
import torch
import torch.nn.functional as F
import transformers
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

SPLIT_NAMES = {0: "train", 1: "validation", 2: "test"}
HAN_CHARACTER = regex.compile(r"\A\p{Script=Han}\Z")
HAN_IN_TEXT = regex.compile(r"\p{Script=Han}")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    model_id: str
    model_revision: str | None
    unihan_zip: str
    wikipedia_dump: str
    font_path: str
    output_dir: str
    max_characters: int = 8000
    wikipedia_max_pages: int | None = None
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    seed: int = 20260830
    glyph_size: int = 32
    font_size: int = 30
    render_threshold: int = 128
    feature_batch_size: int = 128
    train_batch_size: int = 256
    max_epochs: int = 120
    early_stopping_patience: int = 15
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    dice_loss_weight: float = 0.5
    bootstrap_samples: int = 1000

    @classmethod
    def from_json(cls, path: Path) -> ExperimentConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def validate(self) -> None:
        if self.glyph_size != 32:
            raise ValueError("This experiment requires exactly 32x32 glyph targets")
        if self.max_characters < 100:
            raise ValueError("max_characters must be at least 100")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train and validation fractions must leave a test split")


@dataclass
class DatasetBundle:
    characters: list[str]
    codepoints: np.ndarray
    frequencies: np.ndarray
    token_ids: np.ndarray
    bitmaps: np.ndarray
    splits: np.ndarray


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(bundle: DatasetBundle) -> str:
    digest = hashlib.sha256()
    for array in (
        bundle.codepoints,
        bundle.frequencies,
        bundle.token_ids,
        bundle.bitmaps,
        bundle.splits,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _unihan_codepoints(unihan_zip: Path) -> set[int]:
    with zipfile.ZipFile(unihan_zip) as archive:
        member = next(
            (name for name in archive.namelist() if name.endswith("Unihan_IRGSources.txt")),
            None,
        )
        if member is None:
            raise ValueError("Unihan_IRGSources.txt is missing from the Unihan archive")
        codepoints: set[int] = set()
        with archive.open(member) as handle:
            for raw_line in handle:
                if not raw_line.startswith(b"U+"):
                    continue
                field = raw_line.split(b"\t", 1)[0]
                codepoints.add(int(field[2:], 16))
    return codepoints


def _font_codepoints(font_path: Path) -> set[int]:
    font = TTFont(font_path, lazy=True)
    try:
        cmap = font.getBestCmap()
        if cmap is None:
            raise ValueError(f"No Unicode cmap found in {font_path}")
        return set(cmap)
    finally:
        font.close()


def count_wikipedia_characters(
    dump_path: Path,
    cache_path: Path,
    max_pages: int | None,
) -> tuple[Counter[str], dict[str, int]]:
    if cache_path.exists():
        print(f"[dataset] Reusing Wikipedia frequency cache: {cache_path}", flush=True)
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return Counter(payload["frequencies"]), payload["stats"]

    print(f"[dataset] Scanning Wikipedia dump: {dump_path}", flush=True)
    frequencies: Counter[str] = Counter()
    page_count = 0
    text_character_count = 0
    with bz2.open(dump_path, "rb") as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag.endswith("text") and element.text:
                matches = HAN_IN_TEXT.findall(element.text)
                frequencies.update(matches)
                text_character_count += len(matches)
            if element.tag.endswith("page"):
                page_count += 1
                element.clear()
                if page_count % 10_000 == 0:
                    print(
                        f"[dataset] Scanned {page_count:,} pages and "
                        f"{text_character_count:,} Han occurrences",
                        flush=True,
                    )
                if max_pages is not None and page_count >= max_pages:
                    break

    stats = {
        "pages_scanned": page_count,
        "han_character_occurrences": text_character_count,
        "unique_han_characters": len(frequencies),
    }
    _json_dump(cache_path, {"frequencies": dict(frequencies), "stats": stats})
    return frequencies, stats


def render_glyph(
    character: str,
    font: ImageFont.FreeTypeFont,
    size: int,
    threshold: int,
) -> np.ndarray:
    image = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = draw.textbbox((0, 0), character, font=font)
    width = right - left
    height = bottom - top
    x = (size - width) / 2 - left
    y = (size - height) / 2 - top
    draw.text((x, y), character, font=font, fill=255)
    bitmap = (np.asarray(image, dtype=np.uint8) >= threshold).astype(np.uint8)
    if not bitmap.any():
        raise ValueError(f"Character {character!r} rendered as an empty bitmap")
    return bitmap.reshape(-1)


def _split_indices(count: int, config: ExperimentConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(count)
    train_end = round(count * config.train_fraction)
    validation_end = train_end + round(count * config.validation_fraction)
    splits = np.full(count, 2, dtype=np.uint8)
    splits[order[:train_end]] = 0
    splits[order[train_end:validation_end]] = 1
    return splits


def prepare_dataset(
    config: ExperimentConfig, output_dir: Path
) -> tuple[DatasetBundle, dict[str, Any]]:
    dataset_path = output_dir / "dataset" / "hanziglyph.npz"
    stats_path = output_dir / "dataset" / "dataset_stats.json"
    if dataset_path.exists() and stats_path.exists():
        print(f"[dataset] Reusing prepared dataset: {dataset_path}", flush=True)
        return load_dataset(dataset_path), json.loads(stats_path.read_text(encoding="utf-8"))

    unihan_zip = Path(config.unihan_zip)
    font_path = Path(config.font_path)
    wikipedia_dump = Path(config.wikipedia_dump)
    for source_path in (unihan_zip, font_path, wikipedia_dump):
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

    print("[dataset] Loading tokenizer and selecting single-token Han characters", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        use_fast=True,
    )
    frequencies, wikipedia_stats = count_wikipedia_characters(
        wikipedia_dump,
        output_dir / "dataset" / "wikipedia_frequencies.json",
        config.wikipedia_max_pages,
    )
    candidates = _unihan_codepoints(unihan_zip) & _font_codepoints(font_path)
    valid_codepoints = [
        codepoint
        for codepoint in candidates
        if unicodedata.category(chr(codepoint)) == "Lo"
        and HAN_CHARACTER.fullmatch(chr(codepoint)) is not None
    ]
    valid_codepoints.sort(key=lambda value: (-frequencies[chr(value)], value))

    selected: list[tuple[int, int]] = []
    multi_token_count = 0
    for codepoint in valid_codepoints:
        token_ids = tokenizer.encode(chr(codepoint), add_special_tokens=False)
        if len(token_ids) != 1:
            multi_token_count += 1
            continue
        selected.append((codepoint, token_ids[0]))
        if len(selected) == config.max_characters:
            break
    if len(selected) < config.max_characters:
        raise RuntimeError(
            f"Only {len(selected)} font-supported single-token characters were available; "
            f"requested {config.max_characters}"
        )

    print(f"[dataset] Rendering {len(selected):,} deterministic 32x32 glyphs", flush=True)
    font = ImageFont.truetype(str(font_path), size=config.font_size)
    bitmaps = np.stack(
        [
            render_glyph(chr(codepoint), font, config.glyph_size, config.render_threshold)
            for codepoint, _ in selected
        ]
    )
    codepoints = np.asarray([item[0] for item in selected], dtype=np.int32)
    token_ids = np.asarray([item[1] for item in selected], dtype=np.int32)
    selected_frequencies = np.asarray(
        [frequencies[chr(codepoint)] for codepoint in codepoints],
        dtype=np.int64,
    )
    splits = _split_indices(len(selected), config)
    characters = [chr(codepoint) for codepoint in codepoints]
    bundle = DatasetBundle(
        characters=characters,
        codepoints=codepoints,
        frequencies=selected_frequencies,
        token_ids=token_ids,
        bitmaps=bitmaps,
        splits=splits,
    )

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        codepoints=codepoints,
        frequencies=selected_frequencies,
        token_ids=token_ids,
        bitmaps=bitmaps,
        splits=splits,
    )
    with (dataset_path.parent / "characters.jsonl").open("w", encoding="utf-8") as handle:
        for index, character in enumerate(characters):
            handle.write(
                json.dumps(
                    {
                        "char_id": index,
                        "char": character,
                        "codepoint": f"U+{codepoints[index]:04X}",
                        "frequency": int(selected_frequencies[index]),
                        "token_id": int(token_ids[index]),
                        "split": SPLIT_NAMES[int(splits[index])],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    foreground = bitmaps.mean(axis=1)
    stats: dict[str, Any] = {
        "character_count": len(characters),
        "split_counts": {
            name: int((splits == split_id).sum()) for split_id, name in SPLIT_NAMES.items()
        },
        "single_token_only": True,
        "multi_token_candidates_skipped_before_limit": multi_token_count,
        "foreground_fraction_mean": float(foreground.mean()),
        "foreground_fraction_min": float(foreground.min()),
        "foreground_fraction_max": float(foreground.max()),
        "frequency_min": int(selected_frequencies.min()),
        "frequency_median": float(np.median(selected_frequencies)),
        "frequency_max": int(selected_frequencies.max()),
        "wikipedia": wikipedia_stats,
        "dataset_content_sha256": content_sha256(bundle),
        "sources": {
            "unihan": {"path": str(unihan_zip), "sha256": sha256_file(unihan_zip)},
            "font": {"path": str(font_path), "sha256": sha256_file(font_path)},
            "wikipedia": {
                "path": str(wikipedia_dump),
                "sha256": sha256_file(wikipedia_dump),
            },
        },
        "rendering": {
            "glyph_size": config.glyph_size,
            "font_size": config.font_size,
            "threshold": config.render_threshold,
            "pillow_version": importlib.metadata.version("pillow"),
            "fonttools_version": importlib.metadata.version("fonttools"),
        },
    }
    _json_dump(stats_path, stats)
    print(
        f"[dataset] Ready: {stats['split_counts']}, content SHA-256 "
        f"{stats['dataset_content_sha256']}",
        flush=True,
    )
    return bundle, stats


def load_dataset(path: Path) -> DatasetBundle:
    with np.load(path) as data:
        codepoints = data["codepoints"]
        return DatasetBundle(
            characters=[chr(value) for value in codepoints],
            codepoints=codepoints,
            frequencies=data["frequencies"],
            token_ids=data["token_ids"],
            bitmaps=data["bitmaps"],
            splits=data["splits"],
        )


def extract_features(
    config: ExperimentConfig,
    bundle: DatasetBundle,
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    feature_dir = output_dir / "features"
    hidden_path = feature_dir / "last_hidden.npy"
    embedding_path = feature_dir / "input_embedding.npy"
    metadata_path = feature_dir / "extraction_metadata.json"
    if hidden_path.exists() and embedding_path.exists() and metadata_path.exists():
        print(f"[features] Reusing cached features: {feature_dir}", flush=True)
        return (
            {
                "last_hidden": np.load(hidden_path, mmap_mode="r"),
                "input_embedding": np.load(embedding_path, mmap_mode="r"),
            },
            json.loads(metadata_path.read_text(encoding="utf-8")),
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for feature extraction")
    print(f"[features] Loading frozen model {config.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()
    model.requires_grad_(False)

    hidden_batches: list[np.ndarray] = []
    embedding_batches: list[np.ndarray] = []
    for start in range(0, len(bundle.characters), config.feature_batch_size):
        batch_characters = bundle.characters[start : start + config.feature_batch_size]
        encoded = tokenizer(
            batch_characters,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded = {name: tensor.to("cuda") for name, tensor in encoded.items()}
        lengths = encoded["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(batch_characters), device="cuda")
        with torch.inference_mode():
            embeddings = model.get_input_embeddings()(encoded["input_ids"])
            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            selected_embeddings = embeddings[rows, lengths]
            selected_hidden = outputs.hidden_states[-1][rows, lengths]
        hidden_batches.append(selected_hidden.float().cpu().numpy())
        embedding_batches.append(selected_embeddings.float().cpu().numpy())
        completed = min(start + config.feature_batch_size, len(bundle.characters))
        if completed == len(bundle.characters) or completed % 1024 == 0:
            print(f"[features] Extracted {completed:,}/{len(bundle.characters):,}", flush=True)

    features = {
        "last_hidden": np.concatenate(hidden_batches),
        "input_embedding": np.concatenate(embedding_batches),
    }
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(hidden_path, features["last_hidden"].astype(np.float16))
    np.save(embedding_path, features["input_embedding"].astype(np.float16))
    metadata = {
        "model_id": config.model_id,
        "requested_revision": config.model_revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None) or config.model_revision,
        "model_class": type(model).__name__,
        "hidden_size": int(features["last_hidden"].shape[1]),
        "dtype": "bfloat16",
        "attention": "sdpa",
        "feature_batch_size": config.feature_batch_size,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    _json_dump(metadata_path, metadata)
    print(
        f"[features] Cached hidden_size={metadata['hidden_size']}, peak GPU "
        f"{metadata['gpu_peak_allocated_gib']:.3f} GiB",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return features, metadata


class LinearGlyphHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, 1024)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def _loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: torch.Tensor,
    dice_weight: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)
    return bce + dice_weight * dice_loss(logits, targets)


def _predict_probabilities(
    head: LinearGlyphHead,
    features: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    batches: list[np.ndarray] = []
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batches.append(head(features[start : start + batch_size]).sigmoid().cpu().numpy())
    return np.concatenate(batches)


def choose_threshold(probabilities: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -math.inf
    for threshold in np.linspace(0.05, 0.95, 19):
        prediction = probabilities >= threshold
        truth = targets.astype(bool)
        true_positive = np.logical_and(prediction, truth).sum()
        false_positive = np.logical_and(prediction, ~truth).sum()
        false_negative = np.logical_and(~prediction, truth).sum()
        f1 = (2 * true_positive) / max(2 * true_positive + false_positive + false_negative, 1)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _retrieval_hits(prediction: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prediction_tensor = torch.from_numpy(prediction.astype(np.float32)).to("cuda")
    target_tensor = torch.from_numpy(targets.astype(np.float32)).to("cuda")
    target_ones = target_tensor.sum(dim=1)
    top1_parts: list[np.ndarray] = []
    top5_parts: list[np.ndarray] = []
    for start in range(0, len(prediction_tensor), 256):
        chunk = prediction_tensor[start : start + 256]
        distances = chunk.sum(dim=1, keepdim=True) + target_ones - 2 * chunk @ target_tensor.T
        nearest = distances.topk(k=min(5, len(target_tensor)), largest=False).indices
        expected = torch.arange(start, start + len(chunk), device="cuda")
        top1_parts.append((nearest[:, 0] == expected).cpu().numpy())
        top5_parts.append((nearest == expected[:, None]).any(dim=1).cpu().numpy())
    return np.concatenate(top1_parts), np.concatenate(top5_parts)


def _bootstrap_intervals(
    true_positive: np.ndarray,
    false_positive: np.ndarray,
    false_negative: np.ndarray,
    exact: np.ndarray,
    hamming: np.ndarray,
    top1: np.ndarray,
    top5: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(true_positive), size=(samples, len(true_positive)))
    tp = true_positive[draw].sum(axis=1)
    fp = false_positive[draw].sum(axis=1)
    fn = false_negative[draw].sum(axis=1)
    values = {
        "foreground_f1": 2 * tp / np.maximum(2 * tp + fp + fn, 1),
        "iou": tp / np.maximum(tp + fp + fn, 1),
        "exact_match": exact[draw].mean(axis=1),
        "hamming_fraction": hamming[draw].mean(axis=1),
        "retrieval_top1": top1[draw].mean(axis=1),
        "retrieval_top5": top5[draw].mean(axis=1),
    }
    return {
        name: [float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))]
        for name, value in values.items()
    }


def _per_character_counts(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = probabilities >= threshold
    truth = targets.astype(bool)
    return (
        np.logical_and(prediction, truth).sum(axis=1),
        np.logical_and(prediction, ~truth).sum(axis=1),
        np.logical_and(~prediction, truth).sum(axis=1),
    )


def paired_f1_difference(
    first_probabilities: np.ndarray,
    first_threshold: float,
    second_probabilities: np.ndarray,
    second_threshold: float,
    targets: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    first_tp, first_fp, first_fn = _per_character_counts(
        first_probabilities, targets, first_threshold
    )
    second_tp, second_fp, second_fn = _per_character_counts(
        second_probabilities, targets, second_threshold
    )

    def f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
        return 2 * tp / np.maximum(2 * tp + fp + fn, 1)

    first_f1 = float(f1(first_tp.sum(), first_fp.sum(), first_fn.sum()))
    second_f1 = float(f1(second_tp.sum(), second_fp.sum(), second_fn.sum()))
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(targets), size=(samples, len(targets)))
    first_bootstrap = f1(
        first_tp[draw].sum(axis=1),
        first_fp[draw].sum(axis=1),
        first_fn[draw].sum(axis=1),
    )
    second_bootstrap = f1(
        second_tp[draw].sum(axis=1),
        second_fp[draw].sum(axis=1),
        second_fn[draw].sum(axis=1),
    )
    difference = first_bootstrap - second_bootstrap
    return {
        "first_foreground_f1": first_f1,
        "second_foreground_f1": second_f1,
        "foreground_f1_difference": first_f1 - second_f1,
        "confidence_interval_95": [
            float(np.quantile(difference, 0.025)),
            float(np.quantile(difference, 0.975)),
        ],
        "bootstrap_probability_first_greater": float((difference > 0).mean()),
        "bootstrap_samples": samples,
    }


def pixel_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    bootstrap_samples: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    prediction = probabilities >= threshold
    truth = targets.astype(bool)
    true_positive = np.logical_and(prediction, truth).sum(axis=1)
    false_positive = np.logical_and(prediction, ~truth).sum(axis=1)
    false_negative = np.logical_and(~prediction, truth).sum(axis=1)
    true_negative = np.logical_and(~prediction, ~truth).sum(axis=1)
    tp = int(true_positive.sum())
    fp = int(false_positive.sum())
    fn = int(false_negative.sum())
    tn = int(true_negative.sum())
    exact = (prediction == truth).all(axis=1)
    hamming = (prediction != truth).mean(axis=1)
    top1, top5 = _retrieval_hits(prediction, truth)
    metrics: dict[str, Any] = {
        "threshold": threshold,
        "foreground_precision": tp / max(tp + fp, 1),
        "foreground_recall": tp / max(tp + fn, 1),
        "foreground_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "pixel_accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "exact_match": float(exact.mean()),
        "hamming_fraction": float(hamming.mean()),
        "hamming_pixels": float(hamming.mean() * targets.shape[1]),
        "retrieval_top1": float(top1.mean()),
        "retrieval_top5": float(top5.mean()),
        "character_count": len(targets),
    }
    if bootstrap_samples:
        metrics["confidence_intervals_95"] = _bootstrap_intervals(
            true_positive,
            false_positive,
            false_negative,
            exact,
            hamming,
            top1,
            top5,
            bootstrap_samples,
            seed,
        )
    return metrics


def _frequency_bucket_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    frequencies: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    low_cut, high_cut = np.quantile(frequencies, [1 / 3, 2 / 3])
    masks = {
        "low": frequencies <= low_cut,
        "medium": np.logical_and(frequencies > low_cut, frequencies <= high_cut),
        "high": frequencies > high_cut,
    }
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        metrics = pixel_metrics(probabilities[mask], targets[mask], threshold)
        result[name] = {
            key: metrics[key]
            for key in (
                "character_count",
                "foreground_f1",
                "iou",
                "pixel_accuracy",
                "hamming_fraction",
            )
        }
    result["cutoffs"] = {"low_max": float(low_cut), "medium_max": float(high_cut)}
    return result


def train_probe(
    name: str,
    features: np.ndarray,
    bundle: DatasetBundle,
    config: ExperimentConfig,
    output_dir: Path,
    shuffle_training_labels: bool = False,
) -> tuple[dict[str, Any], np.ndarray]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    device = torch.device("cuda")
    print(f"[training:{name}] Starting linear probe", flush=True)
    train_indices = np.flatnonzero(bundle.splits == 0)
    validation_indices = np.flatnonzero(bundle.splits == 1)
    test_indices = np.flatnonzero(bundle.splits == 2)
    feature_array = np.asarray(features).astype(np.float32)
    feature_mean = feature_array[train_indices].mean(axis=0, keepdims=True)
    feature_std = feature_array[train_indices].std(axis=0, keepdims=True)
    feature_std = np.maximum(feature_std, 1e-6)
    feature_array = (feature_array - feature_mean) / feature_std
    feature_tensor = torch.from_numpy(feature_array).to(device)
    target_array = bundle.bitmaps.astype(np.float32)
    training_targets = target_array.copy()
    if shuffle_training_labels:
        rng = np.random.default_rng(config.seed + 1)
        training_targets[train_indices] = training_targets[rng.permutation(train_indices)]
    target_tensor = torch.from_numpy(training_targets).to(device)
    train_index_tensor = torch.from_numpy(train_indices).to(device)
    validation_index_tensor = torch.from_numpy(validation_indices).to(device)
    test_index_tensor = torch.from_numpy(test_indices).to(device)
    positive_count = target_tensor[train_index_tensor].sum()
    total_count = target_tensor[train_index_tensor].numel()
    positive_weight = ((total_count - positive_count) / positive_count).clamp(1, 20)

    head = LinearGlyphHead(feature_tensor.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    remaining_patience = config.early_stopping_patience
    history: list[dict[str, float | int]] = []

    validation_features = feature_tensor[validation_index_tensor]
    validation_targets = torch.from_numpy(target_array[validation_indices]).to(device)
    for epoch in range(1, config.max_epochs + 1):
        head.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator).numpy()]
        train_loss_sum = 0.0
        for start in range(0, len(order), config.train_batch_size):
            indices = order[start : start + config.train_batch_size]
            index_tensor = torch.from_numpy(indices).to(device)
            logits = head(feature_tensor[index_tensor])
            loss = _loss(
                logits,
                target_tensor[index_tensor],
                positive_weight,
                config.dice_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(indices)

        head.eval()
        with torch.inference_mode():
            validation_loss = float(
                _loss(
                    head(validation_features),
                    validation_targets,
                    positive_weight,
                    config.dice_loss_weight,
                )
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / len(train_indices),
                "validation_loss": validation_loss,
            }
        )
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[training:{name}] epoch={epoch} "
                f"train={history[-1]['train_loss']:.5f} val={validation_loss:.5f}",
                flush=True,
            )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
            remaining_patience = config.early_stopping_patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    validation_probabilities = _predict_probabilities(
        head,
        feature_tensor[validation_index_tensor],
        config.train_batch_size,
    )
    threshold, validation_f1 = choose_threshold(
        validation_probabilities,
        target_array[validation_indices],
    )
    test_probabilities = _predict_probabilities(
        head,
        feature_tensor[test_index_tensor],
        config.train_batch_size,
    )
    metrics = pixel_metrics(
        test_probabilities,
        target_array[test_indices],
        threshold,
        config.bootstrap_samples,
        config.seed + 2,
    )
    metrics.update(
        {
            "probe": name,
            "feature_size": int(feature_tensor.shape[1]),
            "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "validation_foreground_f1_at_selected_threshold": validation_f1,
            "positive_weight": float(positive_weight),
            "training_labels_shuffled": shuffle_training_labels,
            "feature_standardization": "training-split per-dimension mean and standard deviation",
            "feature_standard_deviation_min": float(feature_std.min()),
            "frequency_buckets": _frequency_bucket_metrics(
                test_probabilities,
                target_array[test_indices],
                bundle.frequencies[test_indices],
                threshold,
            ),
            "history": history,
        }
    )
    probe_dir = output_dir / "probes" / name
    probe_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "feature_mean": torch.from_numpy(feature_mean.squeeze(0)),
            "feature_std": torch.from_numpy(feature_std.squeeze(0)),
        },
        probe_dir / "head.pt",
    )
    _json_dump(probe_dir / "metrics.json", metrics)
    np.save(probe_dir / "test_probabilities.npy", test_probabilities.astype(np.float16))
    print(
        f"[training:{name}] Complete at epoch {best_epoch}: "
        f"test F1={metrics['foreground_f1']:.4f}, IoU={metrics['iou']:.4f}",
        flush=True,
    )
    return metrics, test_probabilities


def mean_glyph_baseline(
    bundle: DatasetBundle,
    config: ExperimentConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    train_targets = bundle.bitmaps[bundle.splits == 0]
    validation_targets = bundle.bitmaps[bundle.splits == 1]
    test_mask = bundle.splits == 2
    mean_glyph = train_targets.mean(axis=0)
    validation_probabilities = np.repeat(mean_glyph[None, :], len(validation_targets), axis=0)
    threshold, validation_f1 = choose_threshold(validation_probabilities, validation_targets)
    test_probabilities = np.repeat(mean_glyph[None, :], int(test_mask.sum()), axis=0)
    metrics = pixel_metrics(
        test_probabilities,
        bundle.bitmaps[test_mask],
        threshold,
        config.bootstrap_samples,
        config.seed + 2,
    )
    metrics.update(
        {
            "probe": "mean_glyph",
            "feature_size": 0,
            "trainable_parameters": 0,
            "best_epoch": 0,
            "validation_foreground_f1_at_selected_threshold": validation_f1,
            "training_labels_shuffled": False,
            "frequency_buckets": _frequency_bucket_metrics(
                test_probabilities,
                bundle.bitmaps[test_mask],
                bundle.frequencies[test_mask],
                threshold,
            ),
        }
    )
    return metrics, test_probabilities


def save_prediction_sheet(
    bundle: DatasetBundle,
    probabilities: np.ndarray,
    threshold: float,
    font_path: Path,
    output_path: Path,
) -> None:
    test_indices = np.flatnonzero(bundle.splits == 2)[:24]
    cell = 32
    label_width = 42
    row_height = 40
    sheet = Image.new("RGB", (label_width + cell * 2 + 20, row_height * len(test_indices)), "white")
    draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.truetype(str(font_path), 20)
    for row, dataset_index in enumerate(test_indices):
        target = (bundle.bitmaps[dataset_index].reshape(32, 32) * 255).astype(np.uint8)
        prediction = (probabilities[row].reshape(32, 32) >= threshold).astype(np.uint8) * 255
        y = row * row_height + 4
        draw.text((4, y + 4), bundle.characters[dataset_index], font=label_font, fill="black")
        target_image = Image.fromarray(255 - target, mode="L").convert("RGB")
        prediction_image = Image.fromarray(255 - prediction, mode="L").convert("RGB")
        sheet.paste(target_image, (label_width, y))
        sheet.paste(prediction_image, (label_width + cell + 10, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _format_interval(metrics: dict[str, Any], name: str) -> str:
    value = metrics[name]
    interval = metrics.get("confidence_intervals_95", {}).get(name)
    if interval is None:
        return f"{value:.4f}"
    return f"{value:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"


def build_report(
    config: ExperimentConfig,
    dataset_stats: dict[str, Any],
    extraction_metadata: dict[str, Any],
    results: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    output_dir: Path,
) -> Path:
    primary = results["last_hidden"]
    shuffled = results["shuffled_labels"]
    mean_baseline = results["mean_glyph"]
    improvement = primary["foreground_f1"] - shuffled["foreground_f1"]
    hidden_vs_shuffled = comparisons["last_hidden_minus_shuffled_labels"]
    hidden_vs_mean = comparisons["last_hidden_minus_mean_glyph"]
    if (
        hidden_vs_shuffled["confidence_interval_95"][0] > 0
        and hidden_vs_mean["confidence_interval_95"][0] > 0
    ):
        conclusion = (
            "最后层隐藏状态在线性探针上明显优于打乱标签对照，支持冻结模型隐藏状态中存在"
            "可线性解码、能够跨未见字符泛化的字形信号。"
        )
    elif hidden_vs_mean["confidence_interval_95"][1] < 0:
        conclusion = (
            "最后层隐藏状态的线性探针显著低于训练集平均字形基线；即使它可能略高于"
            "打乱标签对照，本轮结果仍不支持最后层存在有实际效用的跨未见字符字形信号。"
        )
    else:
        conclusion = (
            "最后层隐藏状态未明显优于打乱标签对照，本轮结果不足以证明存在可跨未见字符"
            "泛化的线性字形信号。"
        )
    lines = [
        f"# {config.experiment_name} 完整实验报告",
        "",
        f"- 生成时间（UTC）：{datetime.now(UTC).isoformat()}",
        f"- Git 提交：`{_git_commit()}`",
        f"- 模型：`{config.model_id}`",
        f"- 模型解析 revision：`{extraction_metadata['resolved_revision']}`",
        f"- GPU：{extraction_metadata['gpu_name']}",
        f"- 数据集内容 SHA-256：`{dataset_stats['dataset_content_sha256']}`",
        "",
        "## 摘要",
        "",
        (
            f"本实验对 {dataset_stats['character_count']:,} 个互不重复的汉字建立固定 32×32 二值"
            "字形标签，按字符划分训练/验证/测试集。预训练模型全程冻结，只训练一个"
            f" `{primary['feature_size']}→1024` 线性输出头。测试字符从未参与输出头训练。"
        ),
        "",
        conclusion,
        "",
        "## 实验设计",
        "",
        f"- 字符拆分：{dataset_stats['split_counts']}",
        "- 输入：单个汉字，且所有样本均为 tokenizer 单 Token。",
        "- 目标：Noto Sans CJK SC Regular 确定性渲染的 32×32 二值点阵。",
        "- 主实验：冻结模型最后层隐藏状态 + 单线性层。",
        "- 对照：输入 Embedding 线性探针、训练标签随机打乱、训练集平均字形。",
        "- 特征标准化：只使用训练字符计算逐维均值与标准差，验证/测试不参与统计。",
        "- 损失：加权 BCEWithLogitsLoss + Dice Loss。",
        "- 阈值：只用验证集在 0.05～0.95 网格中选择。",
        f"- 置信区间：按测试字符 bootstrap {config.bootstrap_samples:,} 次，95% 区间。",
        "",
        "## 主要结果",
        "",
        "括号为 95% bootstrap 置信区间。",
        "",
        "| 方法 | 前景 F1 | IoU | 像素准确率 | Hamming | 检索 Top-1 | 检索 Top-5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("last_hidden", "input_embedding", "shuffled_labels", "mean_glyph"):
        metrics = results[key]
        lines.append(
            "| "
            + key
            + " | "
            + " | ".join(
                [
                    _format_interval(metrics, "foreground_f1"),
                    _format_interval(metrics, "iou"),
                    f"{metrics['pixel_accuracy']:.4f}",
                    _format_interval(metrics, "hamming_fraction"),
                    _format_interval(metrics, "retrieval_top1"),
                    _format_interval(metrics, "retrieval_top5"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 配对差值检验",
            "",
            "正值表示前一个方法更好；区间按同一批测试字符配对 bootstrap。",
            "",
            "| 对比 | 前景 F1 差值 | 95% CI | P(差值>0) |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, comparison in comparisons.items():
        interval = comparison["confidence_interval_95"]
        lines.append(
            f"| {name} | {comparison['foreground_f1_difference']:+.4f} | "
            f"[{interval[0]:+.4f}, {interval[1]:+.4f}] | "
            f"{comparison['bootstrap_probability_first_greater']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 字频分桶",
            "",
            "| 方法 | 字频桶 | 字符数 | 前景 F1 | IoU | Hamming |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for key in ("last_hidden", "input_embedding", "shuffled_labels", "mean_glyph"):
        for bucket in ("low", "medium", "high"):
            value = results[key]["frequency_buckets"].get(bucket)
            if value is None:
                continue
            lines.append(
                f"| {key} | {bucket} | {value['character_count']} | "
                f"{value['foreground_f1']:.4f} | {value['iou']:.4f} | "
                f"{value['hamming_fraction']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## 数据与环境记录",
            "",
            f"- Wikipedia 扫描统计：{dataset_stats['wikipedia']}",
            f"- 平均前景像素比例：{dataset_stats['foreground_fraction_mean']:.4f}",
            f"- 字体 SHA-256：`{dataset_stats['sources']['font']['sha256']}`",
            f"- Unihan SHA-256：`{dataset_stats['sources']['unihan']['sha256']}`",
            f"- Wikipedia SHA-256：`{dataset_stats['sources']['wikipedia']['sha256']}`",
            f"- PyTorch：{extraction_metadata['torch_version']}",
            f"- Transformers：{extraction_metadata['transformers_version']}",
            f"- 特征提取峰值显存：{extraction_metadata['gpu_peak_allocated_gib']:.3f} GiB",
            "",
            "## 结论",
            "",
            conclusion,
            "",
            (
                f"主实验相对打乱标签对照的前景 F1 差值为 {improvement:+.4f}；相对平均字形"
                f"基线差值为 {primary['foreground_f1'] - mean_baseline['foreground_f1']:+.4f}。"
            ),
            "",
            "## 限制与下一步",
            "",
            "- 本轮只验证 Qwen3.5-2B-Base，不能直接外推到 4B 主模型。",
            "- 随机字符不重叠拆分不能替代基于 IDS 构件的组合拆分。",
            "- 当前任务是“输入字符本身→字形”，尚未测试纯中文上下文预测下一字字形。",
            "- 下一轮应在单 Token 交集上加入 Qwen3.5-4B-Base 与 Qwen3-4B-Base。",
            "",
            "## 复现命令",
            "",
            "```bash",
            "git pull --ff-only origin main",
            "uv sync --frozen",
            "uv run hansgpt-glyph-probe --config configs/experiments/qwen35_2b_hanziglyph_5k.json",
            "```",
            "",
            "完整配置、训练曲线、探针权重、逐实验指标和预测图保存在同一 artifacts 目录。",
            "",
        ]
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_experiment(config: ExperimentConfig) -> Path:
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(output_dir / "resolved_config.json", asdict(config))
    print(f"[experiment] Starting {config.experiment_name}", flush=True)
    dataset, dataset_stats = prepare_dataset(config, output_dir)
    features, extraction_metadata = extract_features(config, dataset, output_dir)

    results: dict[str, dict[str, Any]] = {}
    probabilities: dict[str, np.ndarray] = {}
    results["last_hidden"], probabilities["last_hidden"] = train_probe(
        "last_hidden", features["last_hidden"], dataset, config, output_dir
    )
    results["input_embedding"], probabilities["input_embedding"] = train_probe(
        "input_embedding", features["input_embedding"], dataset, config, output_dir
    )
    results["shuffled_labels"], probabilities["shuffled_labels"] = train_probe(
        "shuffled_labels",
        features["last_hidden"],
        dataset,
        config,
        output_dir,
        shuffle_training_labels=True,
    )
    results["mean_glyph"], probabilities["mean_glyph"] = mean_glyph_baseline(dataset, config)
    test_targets = dataset.bitmaps[dataset.splits == 2]
    comparison_pairs = (
        ("last_hidden", "shuffled_labels"),
        ("last_hidden", "mean_glyph"),
        ("last_hidden", "input_embedding"),
        ("input_embedding", "mean_glyph"),
        ("input_embedding", "shuffled_labels"),
    )
    comparisons = {
        f"{first}_minus_{second}": paired_f1_difference(
            probabilities[first],
            results[first]["threshold"],
            probabilities[second],
            results[second]["threshold"],
            test_targets,
            config.bootstrap_samples,
            config.seed + 100 + pair_index,
        )
        for pair_index, (first, second) in enumerate(comparison_pairs)
    }
    _json_dump(output_dir / "metrics.json", {"methods": results, "comparisons": comparisons})
    save_prediction_sheet(
        dataset,
        probabilities["last_hidden"],
        results["last_hidden"]["threshold"],
        Path(config.font_path),
        output_dir / "predictions.png",
    )
    report_path = build_report(
        config,
        dataset_stats,
        extraction_metadata,
        results,
        comparisons,
        output_dir,
    )
    print(f"[experiment] Report written to {report_path}", flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen-backbone 32x32 Hanzi glyph linear-probe experiment."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config)
    report_path = run_experiment(config)
    print(f"Experiment complete. Report: {report_path}")


if __name__ == "__main__":
    main()
