"""Soft font similarity and raw-image review artifacts; never generation feedback."""

from collections import Counter
from pathlib import Path

import numpy as np
import regex
import torch
from PIL import Image, ImageDraw

from hansgpt_research.train_glyph_lm import write_json


@torch.inference_mode()
def score_glyphs(predictions, gallery, device="cpu", chunk_size=128):
    """Compare binary glyphs without calling soft similarity a human-readable rate."""
    reference = torch.as_tensor(gallery, device=device, dtype=torch.float32).flatten(1)
    pixels = torch.as_tensor(predictions, device=device, dtype=torch.float32).flatten(1)
    if len(reference) < 2:
        raise ValueError("Need at least two distinct reference glyphs")
    counts = reference.sum(1)
    rows = []
    with torch.autocast(torch.device(device).type, enabled=False):
        for batch in pixels.split(chunk_size):
            foreground = batch.sum(1)
            intersection = batch @ reference.T
            denominator = foreground[:, None] + counts[None]
            f1 = 2 * intersection / denominator.clamp_min(1)
            distance = denominator - 2 * intersection
            scores, indices = f1.topk(2, dim=1)
            index = indices[:, 0]
            chosen = torch.arange(len(batch), device=reference.device)
            overlap = intersection[chosen, index]
            iou = overlap / (denominator[chosen, index] - overlap).clamp_min(1)
            for i in range(len(batch)):
                rows.append(
                    {
                        "foreground_pixels": int(foreground[i]),
                        "min_hamming": int(distance[i].min()),
                        "best_f1": float(scores[i, 0]),
                        "best_iou": float(iou[i]),
                        "best_index": int(indices[i, 0]),
                        "second_index": int(indices[i, 1]),
                        "f1_margin": float(scores[i, 0] - scores[i, 1]),
                        "hamming_at_best_f1": int(distance[i, index[i]]),
                    }
                )
    return rows


def _key(grid):
    return np.packbits(np.asarray(grid, dtype=np.uint8).reshape(-1)).tobytes()


def analyze_readability(dataset, prompts, generated, generation_samples, output, device="cpu"):
    """Save similarity distributions, uncertain nearest-font guesses and blind review sheets."""
    output = Path(output)
    groups = {}
    for char, index in dataset.inventory["characters"].items():
        bitmap = dataset.glyph_bank[int(index)].numpy()
        entry = groups.setdefault(_key(bitmap), {"bitmap": bitmap, "labels": []})
        entry["labels"].append(char)
    entries = [groups[k] for k in sorted(groups)]
    gallery = np.stack([e["bitmap"] for e in entries])
    controls = {
        _key(dataset.glyph_bank[i].numpy()): name for name, i in dataset.control_ids.items()
    }
    images, addresses = [], []
    for sample, record in enumerate(generation_samples):
        for position in range(record["body_grids"]):
            images.append(generated[sample, position])
            addresses.append((sample, position))
    rows = score_glyphs(np.stack(images), gallery, device) if images else []
    per_sample = [[] for _ in generation_samples]
    for row, image, (sample, position) in zip(rows, images, addresses, strict=True):
        best, second = entries[row.pop("best_index")], entries[row.pop("second_index")]
        row.update(
            sample=sample,
            position=position,
            nearest_labels=best["labels"],
            alternative_labels=second["labels"],
            control=controls.get(_key(image)),
        )
        row["nearest_is_han"] = any(
            regex.fullmatch(r"[\p{Unified_Ideograph}〇]", c) for c in best["labels"]
        )
        per_sample[sample].append(row)
    content = [r for r in rows if r["control"] is None and r["foreground_pixels"] > 0]

    def rate(predicate):
        return sum(predicate(r) for r in content) / len(rows) if rows else None

    report = {
        "scope": "Actual generated body grids; EOS and post-EOS fill excluded",
        "warning": "Font similarity and guesses are diagnostics, not OCR accuracy or human labels",
        "body_grids": len(rows),
        "control_counts": dict(Counter(r["control"] for r in rows if r["control"])),
        "blank_grids": sum(r["foreground_pixels"] == 0 for r in rows),
        "exact_content_rate": rate(lambda r: r["min_hamming"] == 0),
        "hamming_proximity_rates": {
            str(k): rate(lambda r, k=k: r["min_hamming"] <= k) for k in [1, 2, 4, 8, 16, 32]
        },
        "best_f1_similarity_rates": {
            str(k): rate(lambda r, k=k: r["best_f1"] >= k) for k in [0.7, 0.8, 0.9, 0.95]
        },
        "best_f1_at_least_0_8_and_margin_0_05_rate": rate(
            lambda r: r["best_f1"] >= 0.8 and r["f1_margin"] >= 0.05
        ),
        "noncontrol_best_f1_quantiles": np.quantile(
            [r["best_f1"] for r in content], [0.1, 0.5, 0.9, 0.99]
        ).tolist()
        if content
        else [],
        "rate_denominator": "all body grids; blank/control outputs never count as similar content",
        "glyphs": per_sample,
    }
    write_json(output / "glyph_similarity.json", report)
    lines = ["NEAREST-FONT DIAGNOSTIC ONLY. These are guesses, not raw model text.", ""]
    for i, (record, scores) in enumerate(zip(generation_samples, per_sample, strict=True)):
        diagnostic = "".join(
            "<" + r["control"] + ">"
            if r["control"]
            else r["nearest_labels"][0]
            if r["best_f1"] >= 0.8 and r["f1_margin"] >= 0.05
            else "□"
            for r in scores
        )
        lines.extend(
            [f"Sample {i}: {record['prompt']}", "Conservative font guess: " + diagnostic, ""]
        )
    (output / "nearest_font_diagnostic.txt").write_text("\n".join(lines), encoding="utf-8")
    # Output-only sheets reduce prompt-induced guessing during visual review.
    for page_start in range(0, len(generated), 8):
        count = min(8, len(generated) - page_start)
        sheet = Image.new("RGB", (16 * 50 + 16, count * 124 + 24), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (8, 4),
            "Raw outputs only: first 32 body grids; no nearest-font replacement",
            fill="black",
        )
        for row, sample in enumerate(range(page_start, page_start + count)):
            top = 24 + row * 124
            draw.text((8, top), f"Sample {sample:02d}", fill="black")
            n = min(32, generation_samples[sample]["body_grids"])
            for pos in range(n):
                tile = Image.fromarray(
                    (255 * (1 - generated[sample, pos].reshape(32, 32))).astype(np.uint8)
                )
                tile = tile.resize((48, 48), Image.Resampling.NEAREST)
                sheet.paste(tile, (8 + (pos % 16) * 50, top + 16 + (pos // 16) * 50))
        sheet.save(output / f"blind_review_{page_start:02d}.png")
    write_json(
        output / "human_review_template.json",
        {
            "status": "pending visual review; similarity metrics are not human labels",
            "rubric": {
                "glyph_legibility": "identifiable / partly identifiable / unclear / empty",
                "semantic_coherence": "coherent continuation / fragmentary / incoherent / empty",
                "note": "Inspect pixels first, then prompts and full outputs; mark uncertainty",
            },
            "samples": [
                {"sample": i, "glyph_legibility": None, "semantic_coherence": None, "notes": ""}
                for i in range(len(generated))
            ],
        },
    )
    return {k: v for k, v in report.items() if k != "glyphs"}
