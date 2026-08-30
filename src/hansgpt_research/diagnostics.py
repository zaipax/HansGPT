from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from hansgpt_research.glyph_probe import (
    DatasetBundle,
    _json_dump,
    _loss,
    choose_threshold,
    load_dataset,
    pixel_metrics,
)

IDS_OPERATORS = ("⿰", "⿱", "⿲", "⿳", "⿴", "⿵", "⿶", "⿷", "⿸", "⿹", "⿺", "⿻")
STRUCTURE_NAMES = ("atomic", *IDS_OPERATORS, "other")


@dataclass(frozen=True)
class DiagnosticConfig:
    suite_name: str
    model_id: str
    model_revision: str
    base_experiment_dir: str
    output_dir: str
    font_path: str
    unihan_zip: str
    chise_ids_dir: str
    layer_indices: list[int]
    seed: int = 20260830
    feature_batch_size: int = 128
    train_batch_size: int = 256
    max_epochs: int = 180
    seen_max_epochs: int = 400
    autoencoder_max_epochs: int = 250
    early_stopping_patience: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    dice_loss_weight: float = 0.5
    bootstrap_samples: int = 1000
    latent_size: int = 64

    @classmethod
    def from_json(cls, path: Path) -> DiagnosticConfig:
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def validate(self) -> None:
        if self.layer_indices != sorted(set(self.layer_indices)):
            raise ValueError("layer_indices must be sorted and unique")
        if not self.layer_indices or self.layer_indices[0] != 0:
            raise ValueError("layer_indices must include embedding output layer 0")
        if self.latent_size <= 0:
            raise ValueError("latent_size must be positive")


class LinearBitmapHead(nn.Module):
    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(feature_size, 1024)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class MLPBitmapHead(nn.Module):
    def __init__(self, feature_size: int, hidden_size: int = 1024) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1024),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class MultiTaskGlyphHead(nn.Module):
    def __init__(self, feature_size: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.glyph = nn.Linear(1024, 1024)
        self.structure = nn.Linear(1024, len(STRUCTURE_NAMES))
        self.radical = nn.Linear(1024, 214)
        self.strokes = nn.Linear(1024, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.trunk(features)
        return {
            "glyph": self.glyph(shared),
            "structure": self.structure(shared),
            "radical": self.radical(shared),
            "strokes": self.strokes(shared).squeeze(-1),
        }


class GlyphAutoencoder(nn.Module):
    def __init__(self, latent_size: int) -> None:
        super().__init__()
        self.encoder_convolution = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.encoder_projection = nn.Linear(64 * 4 * 4, latent_size)
        self.decoder_projection = nn.Linear(latent_size, 64 * 4 * 4)
        self.decoder_convolution = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def encode(self, bitmaps: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder_convolution(bitmaps)
        return self.encoder_projection(encoded.flatten(1))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder_projection(latents).reshape(-1, 64, 4, 4)
        return self.decoder_convolution(decoded).flatten(1)

    def forward(self, bitmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(bitmaps)
        return self.decode(latents), latents


class LatentMapper(nn.Module):
    def __init__(self, feature_size: int, latent_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, latent_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split_indices(bundle: DatasetBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.flatnonzero(bundle.splits == value) for value in (0, 1, 2))  # type: ignore[return-value]


def _normalise_features(
    features: np.ndarray,
    training_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(features).astype(np.float32)
    mean = array[training_indices].mean(axis=0, keepdims=True)
    standard_deviation = np.maximum(
        array[training_indices].std(axis=0, keepdims=True),
        1e-6,
    )
    return (array - mean) / standard_deviation, mean, standard_deviation


def _positive_weight(targets: torch.Tensor) -> torch.Tensor:
    positive = targets.sum()
    return ((targets.numel() - positive) / positive.clamp_min(1)).clamp(1, 20)


def _predict_head(
    model: nn.Module,
    features: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            logits = model(features[start : start + batch_size])
            if isinstance(logits, dict):
                logits = logits["glyph"]
            output.append(logits.sigmoid().cpu().numpy())
    return np.concatenate(output)


def _checkpoint_payload(
    state: dict[str, torch.Tensor],
    mean: np.ndarray,
    standard_deviation: np.ndarray,
) -> dict[str, Any]:
    return {
        "state_dict": state,
        "feature_mean": torch.from_numpy(mean.squeeze(0)),
        "feature_std": torch.from_numpy(standard_deviation.squeeze(0)),
    }


def extract_layer_features(
    config: DiagnosticConfig,
    bundle: DatasetBundle,
    output_dir: Path,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    feature_dir = output_dir / "features"
    metadata_path = feature_dir / "metadata.json"
    paths = {layer: feature_dir / f"layer_{layer:02d}.npy" for layer in config.layer_indices}
    if metadata_path.exists() and all(path.exists() for path in paths.values()):
        print("[features] Reusing all cached layer features", flush=True)
        return (
            {layer: np.load(path, mmap_mode="r") for layer, path in paths.items()},
            json.loads(metadata_path.read_text(encoding="utf-8")),
        )

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
    model.eval().requires_grad_(False)
    layer_batches: dict[int, list[np.ndarray]] = {layer: [] for layer in config.layer_indices}

    for start in range(0, len(bundle.characters), config.feature_batch_size):
        characters = bundle.characters[start : start + config.feature_batch_size]
        encoded = tokenizer(
            characters,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded = {name: value.to("cuda") for name, value in encoded.items()}
        lengths = encoded["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(characters), device="cuda")
        with torch.inference_mode():
            hidden_states = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            ).hidden_states
        for layer in config.layer_indices:
            if layer >= len(hidden_states):
                raise ValueError(
                    f"Requested layer {layer}, model returned {len(hidden_states)} states"
                )
            selected = hidden_states[layer][rows, lengths]
            layer_batches[layer].append(selected.float().cpu().numpy())
        completed = min(start + config.feature_batch_size, len(bundle.characters))
        if completed == len(bundle.characters) or completed % 1024 == 0:
            print(f"[features] {completed:,}/{len(bundle.characters):,}", flush=True)

    feature_dir.mkdir(parents=True, exist_ok=True)
    features: dict[int, np.ndarray] = {}
    for layer, batches in layer_batches.items():
        features[layer] = np.concatenate(batches)
        np.save(paths[layer], features[layer].astype(np.float16))
    metadata = {
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "model_class": type(model).__name__,
        "hidden_state_count": len(hidden_states),
        "layer_indices": config.layer_indices,
        "feature_size": int(features[config.layer_indices[0]].shape[1]),
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    _json_dump(metadata_path, metadata)
    del model
    torch.cuda.empty_cache()
    return features, metadata


def render_grayscale_targets(
    bundle: DatasetBundle,
    font_path: Path,
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    target_path = output_dir / "targets" / "grayscale.npy"
    metadata_path = output_dir / "targets" / "grayscale_metadata.json"
    if target_path.exists() and metadata_path.exists():
        return np.load(target_path, mmap_mode="r"), json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    font = ImageFont.truetype(str(font_path), 30)
    targets: list[np.ndarray] = []
    for character in bundle.characters:
        image = Image.new("L", (32, 32), color=0)
        draw = ImageDraw.Draw(image)
        left, top, right, bottom = draw.textbbox((0, 0), character, font=font)
        x = (32 - (right - left)) / 2 - left
        y = (32 - (bottom - top)) / 2 - top
        draw.text((x, y), character, font=font, fill=255)
        targets.append(np.asarray(image, dtype=np.float32).reshape(-1) / 255)
    array = np.stack(targets)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(target_path, array.astype(np.float16))
    fractional = np.logical_and(array > 0, array < 1)
    metadata = {
        "target_count": len(array),
        "fractional_pixel_fraction": float(fractional.mean()),
        "mean_intensity": float(array.mean()),
    }
    _json_dump(metadata_path, metadata)
    return array, metadata


def structure_class(decomposition: str, character: str) -> int:
    if not decomposition or decomposition == character:
        return 0
    if decomposition[0] in IDS_OPERATORS:
        return IDS_OPERATORS.index(decomposition[0]) + 1
    return len(STRUCTURE_NAMES) - 1


def _load_ids(chise_dir: Path) -> dict[int, str]:
    decompositions: dict[int, str] = {}
    for path in sorted(chise_dir.rglob("IDS*.txt")):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith("U+"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                try:
                    codepoint = int(parts[0][2:], 16)
                except ValueError:
                    continue
                decompositions.setdefault(codepoint, parts[2].split(";")[0].strip())
    return decompositions


def _load_unihan_structure(unihan_zip: Path) -> tuple[dict[int, int], dict[int, int]]:
    radicals: dict[int, int] = {}
    strokes: dict[int, int] = {}
    with zipfile.ZipFile(unihan_zip) as archive:
        for member in archive.namelist():
            if not member.endswith(".txt"):
                continue
            with archive.open(member) as handle:
                for raw_line in handle:
                    if not raw_line.startswith(b"U+"):
                        continue
                    parts = raw_line.decode("utf-8", errors="replace").rstrip().split("\t")
                    if len(parts) < 3:
                        continue
                    codepoint = int(parts[0][2:], 16)
                    if parts[1] == "kRSUnicode":
                        match = re.search(r"(\d{1,3})['’]?\.", parts[2])
                        if match:
                            radical = int(match.group(1))
                            if 1 <= radical <= 214:
                                radicals.setdefault(codepoint, radical - 1)
                    elif parts[1] == "kTotalStrokes":
                        match = re.search(r"\d+", parts[2])
                        if match:
                            strokes.setdefault(codepoint, int(match.group()))
    return radicals, strokes


def load_structure_targets(
    config: DiagnosticConfig,
    bundle: DatasetBundle,
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    target_path = output_dir / "targets" / "structure.npz"
    metadata_path = output_dir / "targets" / "structure_metadata.json"
    if target_path.exists() and metadata_path.exists():
        with np.load(target_path) as data:
            return (
                {name: data[name] for name in ("structure", "radical", "strokes")},
                json.loads(metadata_path.read_text(encoding="utf-8")),
            )
    chise_dir = Path(config.chise_ids_dir)
    if not chise_dir.is_dir():
        raise FileNotFoundError(f"CHISE IDS directory is missing: {chise_dir}")
    ids = _load_ids(chise_dir)
    radicals, strokes = _load_unihan_structure(Path(config.unihan_zip))
    structure_array = np.full(len(bundle.characters), -1, dtype=np.int64)
    radical_array = np.full(len(bundle.characters), -1, dtype=np.int64)
    stroke_array = np.full(len(bundle.characters), np.nan, dtype=np.float32)
    for index, codepoint in enumerate(bundle.codepoints.tolist()):
        if codepoint in ids:
            structure_array[index] = structure_class(ids[codepoint], bundle.characters[index])
        if codepoint in radicals:
            radical_array[index] = radicals[codepoint]
        if codepoint in strokes:
            stroke_array[index] = strokes[codepoint]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target_path,
        structure=structure_array,
        radical=radical_array,
        strokes=stroke_array,
    )
    metadata = {
        "structure_coverage": float((structure_array >= 0).mean()),
        "radical_coverage": float((radical_array >= 0).mean()),
        "stroke_coverage": float(np.isfinite(stroke_array).mean()),
        "structure_names": STRUCTURE_NAMES,
        "chise_commit": subprocess.run(
            ["git", "-C", str(chise_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }
    _json_dump(metadata_path, metadata)
    return {
        "structure": structure_array,
        "radical": radical_array,
        "strokes": stroke_array,
    }, metadata


def train_bitmap_head(
    name: str,
    features: np.ndarray,
    training_targets: np.ndarray,
    binary_targets: np.ndarray,
    config: DiagnosticConfig,
    output_dir: Path,
    head_kind: str,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    max_epochs: int | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    experiment_dir = output_dir / "experiments" / name
    metrics_path = experiment_dir / "metrics.json"
    probability_path = experiment_dir / "probabilities.npy"
    if metrics_path.exists() and probability_path.exists():
        print(f"[experiment:{name}] Reusing completed result", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8")), np.load(probability_path)

    _set_seed(config.seed)
    normalised, mean, standard_deviation = _normalise_features(features, training_indices)
    feature_tensor = torch.from_numpy(normalised).to("cuda")
    target_tensor = torch.from_numpy(np.asarray(training_targets).astype(np.float32)).to("cuda")
    training_index_tensor = torch.from_numpy(training_indices).to("cuda")
    validation_index_tensor = torch.from_numpy(validation_indices).to("cuda")
    evaluation_index_tensor = torch.from_numpy(evaluation_indices).to("cuda")
    positive_weight = _positive_weight(target_tensor[training_index_tensor])
    if head_kind == "linear":
        head: nn.Module = LinearBitmapHead(feature_tensor.shape[1]).to("cuda")
    elif head_kind == "mlp":
        head = MLPBitmapHead(feature_tensor.shape[1]).to("cuda")
    else:
        raise ValueError(f"Unknown head kind: {head_kind}")
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = config.early_stopping_patience
    history: list[dict[str, float | int]] = []
    epoch_limit = max_epochs or config.max_epochs
    print(f"[experiment:{name}] Training {head_kind} head", flush=True)

    for epoch in range(1, epoch_limit + 1):
        head.train()
        order = training_indices[torch.randperm(len(training_indices), generator=generator).numpy()]
        training_loss = 0.0
        for start in range(0, len(order), config.train_batch_size):
            batch = torch.from_numpy(order[start : start + config.train_batch_size]).to("cuda")
            logits = head(feature_tensor[batch])
            loss = _loss(
                logits,
                target_tensor[batch],
                positive_weight,
                config.dice_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach()) * len(batch)
        head.eval()
        with torch.inference_mode():
            validation_loss = float(
                _loss(
                    head(feature_tensor[validation_index_tensor]),
                    target_tensor[validation_index_tensor],
                    positive_weight,
                    config.dice_loss_weight,
                )
            )
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss / len(training_indices),
                "validation_loss": validation_loss,
            }
        )
        if epoch == 1 or epoch % 20 == 0:
            print(
                f"[experiment:{name}] epoch={epoch} train={history[-1]['training_loss']:.5f} "
                f"val={validation_loss:.5f}",
                flush=True,
            )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in head.state_dict().items()
            }
            patience = config.early_stopping_patience
        else:
            patience -= 1
            if patience == 0:
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint produced for {name}")
    head.load_state_dict(best_state)
    validation_probabilities = _predict_head(
        head,
        feature_tensor[validation_index_tensor],
        config.train_batch_size,
    )
    threshold, validation_f1 = choose_threshold(
        validation_probabilities,
        binary_targets[validation_indices],
    )
    probabilities = _predict_head(
        head,
        feature_tensor[evaluation_index_tensor],
        config.train_batch_size,
    )
    metrics = pixel_metrics(
        probabilities,
        binary_targets[evaluation_indices],
        threshold,
        config.bootstrap_samples,
        config.seed + 500,
    )
    metrics.update(
        {
            "name": name,
            "head_kind": head_kind,
            "training_character_count": len(training_indices),
            "validation_character_count": len(validation_indices),
            "evaluation_character_count": len(evaluation_indices),
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "validation_f1": validation_f1,
            "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
            "soft_training_targets": bool(np.any((training_targets > 0) & (training_targets < 1))),
            "history": history,
        }
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(best_state, mean, standard_deviation),
        experiment_dir / "head.pt",
    )
    _json_dump(metrics_path, metrics)
    np.save(probability_path, probabilities.astype(np.float16))
    print(
        f"[experiment:{name}] F1={metrics['foreground_f1']:.4f} "
        f"Top1={metrics['retrieval_top1']:.4f}",
        flush=True,
    )
    del head, feature_tensor, target_tensor
    torch.cuda.empty_cache()
    return metrics, probabilities


def train_multitask_head(
    features: np.ndarray,
    bundle: DatasetBundle,
    auxiliary_targets: dict[str, np.ndarray],
    config: DiagnosticConfig,
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    name = "multitask_ids_radical_strokes"
    experiment_dir = output_dir / "experiments" / name
    metrics_path = experiment_dir / "metrics.json"
    probability_path = experiment_dir / "probabilities.npy"
    if metrics_path.exists() and probability_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8")), np.load(probability_path)
    _set_seed(config.seed)
    train_indices, validation_indices, test_indices = _split_indices(bundle)
    normalised, mean, standard_deviation = _normalise_features(features, train_indices)
    feature_tensor = torch.from_numpy(normalised).to("cuda")
    bitmap_tensor = torch.from_numpy(bundle.bitmaps.astype(np.float32)).to("cuda")
    structure = torch.from_numpy(auxiliary_targets["structure"]).to("cuda")
    radical = torch.from_numpy(auxiliary_targets["radical"]).to("cuda")
    strokes_numpy = auxiliary_targets["strokes"]
    stroke_mean = float(np.nanmean(strokes_numpy[train_indices]))
    stroke_std = float(np.nanstd(strokes_numpy[train_indices]))
    normalised_strokes = (strokes_numpy - stroke_mean) / max(stroke_std, 1e-6)
    strokes = torch.from_numpy(normalised_strokes.astype(np.float32)).to("cuda")
    positive_weight = _positive_weight(bitmap_tensor[torch.from_numpy(train_indices).to("cuda")])
    head = MultiTaskGlyphHead(feature_tensor.shape[1]).to("cuda")
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    def multitask_loss(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = head(feature_tensor[indices])
        glyph_loss = _loss(
            output["glyph"],
            bitmap_tensor[indices],
            positive_weight,
            config.dice_loss_weight,
        )
        auxiliary = torch.zeros((), device="cuda")
        structure_mask = structure[indices] >= 0
        if structure_mask.any():
            auxiliary = auxiliary + 0.1 * F.cross_entropy(
                output["structure"][structure_mask], structure[indices][structure_mask]
            )
        radical_mask = radical[indices] >= 0
        if radical_mask.any():
            auxiliary = auxiliary + 0.05 * F.cross_entropy(
                output["radical"][radical_mask], radical[indices][radical_mask]
            )
        stroke_mask = torch.isfinite(strokes[indices])
        if stroke_mask.any():
            auxiliary = auxiliary + 0.05 * F.smooth_l1_loss(
                output["strokes"][stroke_mask], strokes[indices][stroke_mask]
            )
        return glyph_loss + auxiliary, glyph_loss

    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = config.early_stopping_patience
    history: list[dict[str, float | int]] = []
    validation_tensor = torch.from_numpy(validation_indices).to("cuda")
    print("[experiment:multitask] Training IDS/radical/stroke auxiliary head", flush=True)
    for epoch in range(1, config.max_epochs + 1):
        head.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator).numpy()]
        training_loss = 0.0
        for start in range(0, len(order), config.train_batch_size):
            batch = torch.from_numpy(order[start : start + config.train_batch_size]).to("cuda")
            loss, _ = multitask_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach()) * len(batch)
        head.eval()
        with torch.inference_mode():
            validation_loss, validation_glyph_loss = multitask_loss(validation_tensor)
        value = float(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss / len(train_indices),
                "validation_loss": value,
                "validation_glyph_loss": float(validation_glyph_loss),
            }
        )
        if value < best_loss - 1e-5:
            best_loss = value
            best_epoch = epoch
            best_state = {
                key: tensor.detach().cpu().clone() for key, tensor in head.state_dict().items()
            }
            patience = config.early_stopping_patience
        else:
            patience -= 1
            if patience == 0:
                break
    if best_state is None:
        raise RuntimeError("Multitask training produced no checkpoint")
    head.load_state_dict(best_state).eval()
    validation_probabilities = _predict_head(
        head,
        feature_tensor[validation_tensor],
        config.train_batch_size,
    )
    threshold, validation_f1 = choose_threshold(
        validation_probabilities, bundle.bitmaps[validation_indices]
    )
    test_tensor = torch.from_numpy(test_indices).to("cuda")
    probabilities = _predict_head(head, feature_tensor[test_tensor], config.train_batch_size)
    metrics = pixel_metrics(
        probabilities,
        bundle.bitmaps[test_indices],
        threshold,
        config.bootstrap_samples,
        config.seed + 600,
    )
    with torch.inference_mode():
        output = head(feature_tensor[test_tensor])
    structure_mask = auxiliary_targets["structure"][test_indices] >= 0
    radical_mask = auxiliary_targets["radical"][test_indices] >= 0
    stroke_mask = np.isfinite(auxiliary_targets["strokes"][test_indices])
    auxiliary_metrics = {
        "structure_accuracy": float(
            (
                output["structure"].argmax(dim=1).cpu().numpy()[structure_mask]
                == auxiliary_targets["structure"][test_indices][structure_mask]
            ).mean()
        ),
        "radical_accuracy": float(
            (
                output["radical"].argmax(dim=1).cpu().numpy()[radical_mask]
                == auxiliary_targets["radical"][test_indices][radical_mask]
            ).mean()
        ),
        "stroke_mae": float(
            np.abs(
                (output["strokes"].cpu().numpy() * stroke_std + stroke_mean)[stroke_mask]
                - auxiliary_targets["strokes"][test_indices][stroke_mask]
            ).mean()
        ),
    }
    metrics.update(
        {
            "name": name,
            "head_kind": "shared_mlp_multitask",
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "validation_f1": validation_f1,
            "trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
            "auxiliary_metrics": auxiliary_metrics,
            "history": history,
        }
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(best_state, mean, standard_deviation),
        experiment_dir / "head.pt",
    )
    _json_dump(metrics_path, metrics)
    np.save(probability_path, probabilities.astype(np.float16))
    return metrics, probabilities


def train_autoencoder(
    bundle: DatasetBundle,
    config: DiagnosticConfig,
    output_dir: Path,
) -> tuple[GlyphAutoencoder, np.ndarray, dict[str, Any], np.ndarray]:
    experiment_dir = output_dir / "experiments" / "glyph_autoencoder"
    checkpoint_path = experiment_dir / "autoencoder.pt"
    metrics_path = experiment_dir / "metrics.json"
    probability_path = experiment_dir / "probabilities.npy"
    train_indices, validation_indices, test_indices = _split_indices(bundle)
    _set_seed(config.seed)
    model = GlyphAutoencoder(config.latent_size).to("cuda")
    bitmap_tensor = torch.from_numpy(bundle.bitmaps.astype(np.float32)).to("cuda")
    images = bitmap_tensor.reshape(-1, 1, 32, 32)
    positive_weight = _positive_weight(bitmap_tensor[torch.from_numpy(train_indices).to("cuda")])
    if checkpoint_path.exists() and metrics_path.exists() and probability_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda"))
        model.eval()
        with torch.inference_mode():
            latents = model.encode(images).cpu().numpy()
        return (
            model,
            latents,
            json.loads(metrics_path.read_text(encoding="utf-8")),
            np.load(probability_path),
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = config.early_stopping_patience
    history: list[dict[str, float | int]] = []
    validation_tensor = torch.from_numpy(validation_indices).to("cuda")
    print("[experiment:autoencoder] Training glyph-only latent model", flush=True)
    for epoch in range(1, config.autoencoder_max_epochs + 1):
        model.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator).numpy()]
        training_loss = 0.0
        for start in range(0, len(order), config.train_batch_size):
            batch = torch.from_numpy(order[start : start + config.train_batch_size]).to("cuda")
            logits, _ = model(images[batch])
            loss = _loss(
                logits,
                bitmap_tensor[batch],
                positive_weight,
                config.dice_loss_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach()) * len(batch)
        model.eval()
        with torch.inference_mode():
            validation_logits, _ = model(images[validation_tensor])
            validation_loss = float(
                _loss(
                    validation_logits,
                    bitmap_tensor[validation_tensor],
                    positive_weight,
                    config.dice_loss_weight,
                )
            )
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss / len(train_indices),
                "validation_loss": validation_loss,
            }
        )
        if epoch == 1 or epoch % 20 == 0:
            print(
                f"[experiment:autoencoder] epoch={epoch} val={validation_loss:.5f}",
                flush=True,
            )
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            patience = config.early_stopping_patience
        else:
            patience -= 1
            if patience == 0:
                break
    if best_state is None:
        raise RuntimeError("Autoencoder training produced no checkpoint")
    model.load_state_dict(best_state).eval()
    with torch.inference_mode():
        validation_probabilities = model(images[validation_tensor])[0].sigmoid().cpu().numpy()
        test_tensor = torch.from_numpy(test_indices).to("cuda")
        probabilities = model(images[test_tensor])[0].sigmoid().cpu().numpy()
        latents = model.encode(images).cpu().numpy()
    threshold, validation_f1 = choose_threshold(
        validation_probabilities, bundle.bitmaps[validation_indices]
    )
    metrics = pixel_metrics(
        probabilities,
        bundle.bitmaps[test_indices],
        threshold,
        config.bootstrap_samples,
        config.seed + 700,
    )
    metrics.update(
        {
            "name": "glyph_autoencoder_reconstruction",
            "latent_size": config.latent_size,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "validation_f1": validation_f1,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "history": history,
        }
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    _json_dump(metrics_path, metrics)
    np.save(probability_path, probabilities.astype(np.float16))
    return model, latents, metrics, probabilities


def train_latent_mapper(
    features: np.ndarray,
    latents: np.ndarray,
    decoder: GlyphAutoencoder,
    bundle: DatasetBundle,
    config: DiagnosticConfig,
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    name = "language_to_autoencoder_latent"
    experiment_dir = output_dir / "experiments" / name
    metrics_path = experiment_dir / "metrics.json"
    probability_path = experiment_dir / "probabilities.npy"
    if metrics_path.exists() and probability_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8")), np.load(probability_path)
    _set_seed(config.seed)
    train_indices, validation_indices, test_indices = _split_indices(bundle)
    normalised, feature_mean, feature_std = _normalise_features(features, train_indices)
    latent_mean = latents[train_indices].mean(axis=0, keepdims=True)
    latent_std = np.maximum(latents[train_indices].std(axis=0, keepdims=True), 1e-6)
    normalised_latents = (latents - latent_mean) / latent_std
    feature_tensor = torch.from_numpy(normalised).to("cuda")
    latent_tensor = torch.from_numpy(normalised_latents.astype(np.float32)).to("cuda")
    mapper = LatentMapper(feature_tensor.shape[1], config.latent_size).to("cuda")
    optimizer = torch.optim.AdamW(
        mapper.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = config.early_stopping_patience
    history: list[dict[str, float | int]] = []
    validation_tensor = torch.from_numpy(validation_indices).to("cuda")
    print("[experiment:latent] Training language-to-glyph latent mapper", flush=True)
    for epoch in range(1, config.max_epochs + 1):
        mapper.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator).numpy()]
        training_loss = 0.0
        for start in range(0, len(order), config.train_batch_size):
            batch = torch.from_numpy(order[start : start + config.train_batch_size]).to("cuda")
            loss = F.mse_loss(mapper(feature_tensor[batch]), latent_tensor[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach()) * len(batch)
        mapper.eval()
        with torch.inference_mode():
            validation_loss = float(
                F.mse_loss(
                    mapper(feature_tensor[validation_tensor]),
                    latent_tensor[validation_tensor],
                )
            )
        history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss / len(train_indices),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()
            }
            patience = config.early_stopping_patience
        else:
            patience -= 1
            if patience == 0:
                break
    if best_state is None:
        raise RuntimeError("Latent mapper training produced no checkpoint")
    mapper.load_state_dict(best_state).eval()
    latent_mean_tensor = torch.from_numpy(latent_mean).to("cuda")
    latent_std_tensor = torch.from_numpy(latent_std).to("cuda")

    def decoded_probabilities(indices: np.ndarray) -> np.ndarray:
        output: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(indices), config.train_batch_size):
                batch = torch.from_numpy(indices[start : start + config.train_batch_size]).to(
                    "cuda"
                )
                predicted = mapper(feature_tensor[batch])
                predicted = predicted * latent_std_tensor + latent_mean_tensor
                output.append(decoder.decode(predicted).sigmoid().cpu().numpy())
        return np.concatenate(output)

    validation_probabilities = decoded_probabilities(validation_indices)
    threshold, validation_f1 = choose_threshold(
        validation_probabilities, bundle.bitmaps[validation_indices]
    )
    probabilities = decoded_probabilities(test_indices)
    metrics = pixel_metrics(
        probabilities,
        bundle.bitmaps[test_indices],
        threshold,
        config.bootstrap_samples,
        config.seed + 800,
    )
    metrics.update(
        {
            "name": name,
            "head_kind": "mlp_to_frozen_autoencoder_decoder",
            "latent_size": config.latent_size,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "validation_f1": validation_f1,
            "trainable_parameters": sum(parameter.numel() for parameter in mapper.parameters()),
            "history": history,
        }
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "feature_mean": torch.from_numpy(feature_mean.squeeze(0)),
            "feature_std": torch.from_numpy(feature_std.squeeze(0)),
            "latent_mean": torch.from_numpy(latent_mean.squeeze(0)),
            "latent_std": torch.from_numpy(latent_std.squeeze(0)),
        },
        experiment_dir / "mapper.pt",
    )
    _json_dump(metrics_path, metrics)
    np.save(probability_path, probabilities.astype(np.float16))
    return metrics, probabilities


def _metric_row(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {metrics['foreground_f1']:.4f} | {metrics['iou']:.4f} | "
        f"{metrics['hamming_fraction']:.4f} | {metrics['retrieval_top1']:.4f} | "
        f"{metrics['retrieval_top5']:.4f} | {metrics['exact_match']:.4f} |"
    )


def build_diagnostic_report(
    config: DiagnosticConfig,
    dataset_stats: dict[str, Any],
    feature_metadata: dict[str, Any],
    grayscale_metadata: dict[str, Any],
    structure_metadata: dict[str, Any],
    results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> Path:
    unseen_linear = results["layer_24_linear"]
    seen_linear = results["seen_linear"]
    seen_mlp = results["seen_mlp"]
    unseen_mlp = results["last_layer_mlp"]
    best_layer_name = max(
        (f"layer_{layer:02d}_linear" for layer in config.layer_indices),
        key=lambda name: results[name]["retrieval_top1"],
    )
    best_layer = results[best_layer_name]
    multitask = results["multitask_ids_radical_strokes"]
    grayscale = results["last_layer_grayscale"]
    autoencoder = results["glyph_autoencoder"]
    latent = results["language_to_autoencoder_latent"]
    diagnosis = [
        (
            f"见过字符时，线性/MLP 的 F1 为 {seen_linear['foreground_f1']:.4f}/"
            f"{seen_mlp['foreground_f1']:.4f}；未见字符最后层线性/MLP 为 "
            f"{unseen_linear['foreground_f1']:.4f}/{unseen_mlp['foreground_f1']:.4f}。"
        ),
        (
            f"分层检索最强的是 {best_layer_name}，Top-1={best_layer['retrieval_top1']:.4f}；"
            f"最后层为 {unseen_linear['retrieval_top1']:.4f}。"
        ),
        (
            f"结构多任务相对同结构 MLP 的 F1 变化为 "
            f"{multitask['foreground_f1'] - unseen_mlp['foreground_f1']:+.4f}。"
        ),
        (
            f"灰度监督相对二值线性头的 F1 变化为 "
            f"{grayscale['foreground_f1'] - unseen_linear['foreground_f1']:+.4f}。"
        ),
        (
            f"字形自编码器直接重建测试字 F1={autoencoder['foreground_f1']:.4f}，"
            f"语言特征预测 latent 后 F1={latent['foreground_f1']:.4f}。"
        ),
    ]
    lines = [
        f"# {config.suite_name}完整报告",
        "",
        f"- 生成时间（UTC）：{datetime.now(UTC).isoformat()}",
        f"- Git 提交：`{_git_commit()}`",
        f"- 模型 revision：`{config.model_revision}`",
        f"- 数据集 SHA-256：`{dataset_stats['dataset_content_sha256']}`",
        f"- GPU 峰值：{feature_metadata['peak_gpu_gib']:.3f} GiB",
        "",
        "## 诊断摘要",
        "",
        *[f"- {item}" for item in diagnosis],
        "",
        "## 全部实验结果",
        "",
        "| 实验 | 前景 F1 | IoU | Hamming | 检索 Top-1 | 检索 Top-5 | Exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    ordered_names = [
        "seen_linear",
        "seen_mlp",
        *[f"layer_{layer:02d}_linear" for layer in config.layer_indices],
        "last_layer_mlp",
        "multitask_ids_radical_strokes",
        "last_layer_grayscale",
        "glyph_autoencoder",
        "language_to_autoencoder_latent",
    ]
    lines.extend(_metric_row(name, results[name]) for name in ordered_names)
    lines.extend(
        [
            "",
            "## 结构辅助任务",
            "",
            f"- CHISE revision：`{structure_metadata['chise_commit']}`",
            f"- IDS 结构覆盖率：{structure_metadata['structure_coverage']:.2%}",
            f"- 部首覆盖率：{structure_metadata['radical_coverage']:.2%}",
            f"- 笔画数覆盖率：{structure_metadata['stroke_coverage']:.2%}",
            f"- 辅助任务测试指标：{multitask['auxiliary_metrics']}",
            "",
            "## 灰度与字形 latent",
            "",
            f"- 灰度目标非二值像素比例：{grayscale_metadata['fractional_pixel_fraction']:.4f}",
            f"- 自编码器 latent 维度：{config.latent_size}",
            "",
            "## 归因结论",
            "",
            (
                "本报告用容量上限、层深、非线性、结构监督、目标平滑和字形先验六个方向"
                "区分瓶颈。最终判断应结合上表：见过字符结果用于判断输出头容量，分层结果"
                "用于判断语言层是否损失字形信号，自编码器直接重建与语言到 latent 的差距"
                "用于判断瓶颈在字形生成器还是语言表示。"
            ),
            "",
            "## 复现",
            "",
            "```bash",
            "git pull --ff-only origin main",
            "uv sync --frozen",
            (
                "uv run hansgpt-diagnostics --config "
                "configs/experiments/qwen35_2b_diagnostic_suite.json"
            ),
            "```",
            "",
        ]
    )
    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_diagnostics(config: DiagnosticConfig) -> Path:
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _json_dump(output_dir / "resolved_config.json", asdict(config))
    base_dir = Path(config.base_experiment_dir)
    bundle = load_dataset(base_dir / "dataset" / "hanziglyph.npz")
    dataset_stats = json.loads(
        (base_dir / "dataset" / "dataset_stats.json").read_text(encoding="utf-8")
    )
    features, feature_metadata = extract_layer_features(config, bundle, output_dir)
    grayscale_targets, grayscale_metadata = render_grayscale_targets(
        bundle, Path(config.font_path), output_dir
    )
    structure_targets, structure_metadata = load_structure_targets(config, bundle, output_dir)
    train_indices, validation_indices, test_indices = _split_indices(bundle)
    all_indices = np.arange(len(bundle.characters), dtype=np.int64)
    results: dict[str, dict[str, Any]] = {}
    probabilities: dict[str, np.ndarray] = {}

    results["seen_linear"], probabilities["seen_linear"] = train_bitmap_head(
        "seen_linear",
        features[config.layer_indices[-1]],
        bundle.bitmaps,
        bundle.bitmaps,
        config,
        output_dir,
        "linear",
        all_indices,
        all_indices,
        all_indices,
        config.seen_max_epochs,
    )
    results["seen_mlp"], probabilities["seen_mlp"] = train_bitmap_head(
        "seen_mlp",
        features[config.layer_indices[-1]],
        bundle.bitmaps,
        bundle.bitmaps,
        config,
        output_dir,
        "mlp",
        all_indices,
        all_indices,
        all_indices,
        config.seen_max_epochs,
    )
    for layer in config.layer_indices:
        name = f"layer_{layer:02d}_linear"
        results[name], probabilities[name] = train_bitmap_head(
            name,
            features[layer],
            bundle.bitmaps,
            bundle.bitmaps,
            config,
            output_dir,
            "linear",
            train_indices,
            validation_indices,
            test_indices,
        )
    results["last_layer_mlp"], probabilities["last_layer_mlp"] = train_bitmap_head(
        "last_layer_mlp",
        features[config.layer_indices[-1]],
        bundle.bitmaps,
        bundle.bitmaps,
        config,
        output_dir,
        "mlp",
        train_indices,
        validation_indices,
        test_indices,
    )
    results["multitask_ids_radical_strokes"], probabilities["multitask_ids_radical_strokes"] = (
        train_multitask_head(
            features[config.layer_indices[-1]],
            bundle,
            structure_targets,
            config,
            output_dir,
        )
    )
    results["last_layer_grayscale"], probabilities["last_layer_grayscale"] = train_bitmap_head(
        "last_layer_grayscale",
        features[config.layer_indices[-1]],
        grayscale_targets,
        bundle.bitmaps,
        config,
        output_dir,
        "linear",
        train_indices,
        validation_indices,
        test_indices,
    )
    autoencoder, latents, results["glyph_autoencoder"], probabilities["glyph_autoencoder"] = (
        train_autoencoder(bundle, config, output_dir)
    )
    results["language_to_autoencoder_latent"], probabilities["language_to_autoencoder_latent"] = (
        train_latent_mapper(
            features[config.layer_indices[-1]],
            latents,
            autoencoder,
            bundle,
            config,
            output_dir,
        )
    )
    _json_dump(output_dir / "metrics.json", results)
    report_path = build_diagnostic_report(
        config,
        dataset_stats,
        feature_metadata,
        grayscale_metadata,
        structure_metadata,
        results,
        output_dir,
    )
    print(f"Diagnostic suite complete: {report_path}", flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete glyph bottleneck suite")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_diagnostics(DiagnosticConfig.from_json(args.config))


if __name__ == "__main__":
    main()
