"""Broad raw-image inference for a Qwen3-style C checkpoint.

The outer model predicts complete binary glyph grids. This diagnostic keeps
those grids as the feedback signal and never projects them through a glyph
gallery, OCR system, or font renderer. It evaluates several prefix lengths,
greedy decoding, and categorical byte sampling at two temperatures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw

from hansgpt_research.cvae_fixed_step import install_xformers
from hansgpt_research.packed_glyph_data import PackedGlyphSequenceDataset
from hansgpt_research.train_attention_glyph_lm import model_from_config
from hansgpt_research.train_glyph_lm import autocast_context, sha256, write_json


PROMPT_LENGTHS = (8, 32, 128, 256, 512, 1024)
HORIZONS = (8, 32, 128, 256)
CONDITIONS = (
    ("greedy", 1.0),
    ("sample", 0.7),
    ("sample", 1.0),
)


def bitmap_key(grid: np.ndarray) -> bytes:
    return np.packbits(np.asarray(grid, dtype=np.uint8).reshape(-1)).tobytes()


def label_lookup(dataset) -> tuple[dict[bytes, list[str]], dict[bytes, str]]:
    labels: dict[bytes, list[str]] = {}
    for character, index in dataset.inventory["characters"].items():
        labels.setdefault(bitmap_key(dataset.glyph_bank[index].numpy()), []).append(character)
    controls = {
        bitmap_key(dataset.glyph_bank[index].numpy()): name
        for name, index in dataset.control_ids.items()
    }
    if set(labels) & set(controls):
        raise ValueError("Content/control bitmap collision")
    return labels, controls


def transcribe(grids: np.ndarray, labels: dict[bytes, list[str]], controls: dict[bytes, str]) -> str:
    values = []
    for grid in grids:
        key = bitmap_key(grid)
        if key in controls:
            values.append(f"<{controls[key]}>")
        elif key not in labels:
            values.append("?")
        elif len(labels[key]) == 1:
            values.append(labels[key][0])
        else:
            values.append("[" + "/".join(labels[key]) + "]")
    return "".join(values)


def choose_documents(dataset, *, count: int, minimum_length: int, seed: int) -> list[int]:
    """Select one deterministic long validation document per source page."""
    parquet = dataset.data_dir / f"{dataset.split}.parquet"
    rows = pq.read_table(parquet, columns=["source_page_id"]).column(0).to_pylist()
    if len(rows) != len(dataset.offsets) - 1:
        raise ValueError("Validation Parquet and glyph offsets disagree")
    candidates: dict[str, tuple[str, int]] = {}
    for index, page in enumerate(rows):
        length = int(dataset.offsets[index + 1] - dataset.offsets[index])
        if length < minimum_length:
            continue
        page_id = str(page)
        rank = hashlib.sha256(f"{seed}:document:{index}:{page_id}".encode()).hexdigest()
        previous = candidates.get(page_id)
        if previous is None or rank < previous[0]:
            candidates[page_id] = (rank, index)
    selected = [index for _, index in sorted(candidates.values())[:count]]
    if len(selected) != count:
        raise ValueError(
            f"Requested {count} validation documents of length >= {minimum_length}; "
            f"only {len(selected)} eligible documents exist"
        )
    return selected


def document_glyphs(dataset, document: int, length: int) -> torch.Tensor:
    start = int(dataset.offsets[document])
    stop = start + length
    ids = np.asarray(dataset.tokens[start:stop], dtype=np.int64)
    if len(ids) != length:
        raise ValueError("Selected document is shorter than the requested prompt")
    return dataset.glyph_bank[torch.from_numpy(ids)]


def _longest_true_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def repetition_metrics(grids: np.ndarray, near_hamming: int = 4, max_period: int = 16) -> dict:
    flat = grids.reshape(len(grids), -1)
    if not len(flat):
        return {
            "adjacent_exact_repeat_rate": 0.0,
            "adjacent_near_repeat_rate": 0.0,
            "max_exact_run": 0,
            "unique_bitmaps": 0,
            "has_short_cycle": False,
            "longest_short_cycle_tiles": 0,
        }
    distances = np.count_nonzero(flat[1:] != flat[:-1], axis=1)
    exact_run = _longest_true_run(distances == 0) + (1 if len(flat) else 0)
    longest_cycle = 0
    for period in range(1, min(max_period, len(flat) - 1) + 1):
        cycle_distances = np.count_nonzero(flat[period:] != flat[:-period], axis=1)
        run = _longest_true_run(cycle_distances <= near_hamming)
        longest_cycle = max(longest_cycle, run + period if run else 0)
    return {
        "adjacent_exact_repeat_rate": float((distances == 0).mean()) if len(distances) else 0.0,
        "adjacent_near_repeat_rate": float((distances <= near_hamming).mean())
        if len(distances)
        else 0.0,
        "max_exact_run": int(exact_run),
        "unique_bitmaps": int(len(np.unique(flat, axis=0))),
        "has_short_cycle": bool(longest_cycle >= 8),
        "longest_short_cycle_tiles": int(longest_cycle),
    }


def eos_positions(generated: np.ndarray, eos_grid: np.ndarray) -> list[int | None]:
    eos_key = bitmap_key(eos_grid)
    values = []
    for output in generated:
        values.append(next((i + 1 for i, grid in enumerate(output) if bitmap_key(grid) == eos_key), None))
    return values


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = prediction.astype(bool)
    actual = target.astype(bool)
    tp = int(np.logical_and(predicted, actual).sum())
    fp = int(np.logical_and(predicted, ~actual).sum())
    fn = int(np.logical_and(~predicted, actual).sum())
    union = tp + fp + fn
    return {
        "foreground_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "iou": tp / max(1, union),
        "dice": 2 * tp / max(1, 2 * tp + fp + fn),
        "exact_bitmap_match": float(np.all(prediction == target, axis=tuple(range(2, prediction.ndim))).mean()),
        "pixel_accuracy": float(np.equal(prediction, target).mean()),
    }


def generation_summary(
    prompts: np.ndarray,
    generated: np.ndarray,
    references: np.ndarray,
    labels: dict[bytes, list[str]],
    controls: dict[bytes, str],
    eos_grid: np.ndarray,
) -> dict:
    eos = eos_positions(generated, eos_grid)
    samples = []
    for index, output in enumerate(generated):
        stop = eos[index]
        body = output if stop is None else output[: stop - 1]
        body_keys = [bitmap_key(grid) for grid in body]
        legal = sum(key in labels for key in body_keys)
        samples.append(
            {
                "sample": index,
                "prompt_transcription": transcribe(prompts[index], labels, controls),
                "generated_transcription_before_eos": transcribe(body, labels, controls),
                "body_grids_before_eos": len(body),
                "eos_position": stop,
                "exact_content_rate_before_eos": legal / len(body) if len(body) else None,
                "foreground_rate": float(body.mean()) if len(body) else None,
                "blank_grid_count": int(np.all(body == 0, axis=(1, 2, 3)).sum()) if len(body) else 0,
                "repetition": repetition_metrics(body),
            }
        )
    horizon_metrics = {}
    for horizon in HORIZONS:
        count = min(horizon, generated.shape[1], references.shape[1])
        horizon_metrics[str(horizon)] = (
            {"available": 0}
            if count == 0
            else {"available": count, **binary_metrics(generated[:, :count], references[:, :count])}
        )
    body_grids = sum(item["body_grids_before_eos"] for item in samples)
    legal_grids = sum(
        int(item["exact_content_rate_before_eos"] * item["body_grids_before_eos"])
        for item in samples
        if item["exact_content_rate_before_eos"] is not None
    )
    return {
        "samples": samples,
        "aggregate": {
            "prompt_count": len(samples),
            "generated_grids": int(generated.shape[0] * generated.shape[1]),
            "body_grids_before_eos": body_grids,
            "exact_content_rate_before_eos": legal_grids / max(1, body_grids),
            "eos_rate": sum(value is not None for value in eos) / max(1, len(eos)),
            "mean_eos_position": float(np.mean([value for value in eos if value is not None]))
            if any(value is not None for value in eos)
            else None,
            "mean_foreground_rate": float(np.mean([item["foreground_rate"] for item in samples if item["foreground_rate"] is not None]))
            if any(item["foreground_rate"] is not None for item in samples)
            else None,
            "total_blank_grids": sum(item["blank_grid_count"] for item in samples),
        },
        "horizons": horizon_metrics,
        "protocol": "raw binary feedback; no glyph lookup, OCR, or font projection",
    }


def draw_contact_sheet(path: Path, prompts: np.ndarray, generated: np.ndarray, title: str) -> None:
    cell = 34
    columns = 32
    title_height = 28
    sample_rows = [max(1, math.ceil((len(prompt) + len(output)) / columns)) for prompt, output in zip(prompts, generated, strict=True)]
    canvas = Image.new("RGB", (columns * cell + 16, title_height + sum(rows * cell + 8 for rows in sample_rows)), "white")
    pen = ImageDraw.Draw(canvas)
    pen.text((8, 6), title, fill="black")
    y_base = title_height
    for prompt, output, rows in zip(prompts, generated, sample_rows, strict=True):
        combined = np.concatenate((prompt, output), axis=0)
        for index, grid in enumerate(combined):
            x = 8 + (index % columns) * cell
            y = y_base + (index // columns) * cell
            tile = Image.fromarray((255 * (1 - grid.reshape(32, 32))).astype(np.uint8), mode="L").convert("RGB")
            canvas.paste(tile, (x, y))
            if index < len(prompt):
                pen.rectangle((x - 1, y - 1, x + 32, y + 32), outline="blue")
        y_base += rows * cell + 8
    canvas.save(path)


@torch.inference_mode()
def rollout(
    model,
    prompt: torch.Tensor,
    *,
    max_new: int,
    strategy: str,
    temperature: float,
    generator: torch.Generator,
    precision: str,
    device: torch.device,
    use_cuda_graph: bool = False,
) -> torch.Tensor:
    """Generate fixed-length raw grids while retaining EOS for analysis."""
    context = prompt.to(device=device, dtype=torch.uint8)
    model_input = context[:, -model.config.max_position_embeddings :]
    cache = None
    generated: list[torch.Tensor] = []
    for _ in range(max_new):
        with autocast_context(device, precision):
            hidden, next_cache = model.forward_hidden(
                model_input, past_key_values=cache, use_cache=True, return_cache=True
            )
            distribution = model.distribution(hidden[:, -1:])
            tile = distribution.decode(
                strategy=strategy,
                temperature=temperature,
                generator=generator,
                use_cache=True,
                use_cuda_graph=use_cuda_graph,
            )
        if tile.dtype != torch.uint8 or not bool(((tile == 0) | (tile == 1)).all()):
            raise ValueError("Raw rollout must return strict binary uint8 grids")
        generated.append(tile)
        context = torch.cat((context, tile), dim=1)
        if context.shape[1] <= model.config.max_position_embeddings:
            cache, model_input = next_cache, tile
        else:
            cache = None
            model_input = context[:, -model.config.max_position_embeddings :]
    return torch.cat(generated, dim=1)


def parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not values or min(values) < 1:
        raise ValueError("Expected positive comma-separated integers")
    return values


def run(args: argparse.Namespace) -> None:
    prompt_lengths = parse_ints(args.prompt_lengths)
    if args.max_new < max(HORIZONS):
        raise ValueError(f"max-new must be at least {max(HORIZONS)}")
    if args.prompt_count < 1:
        raise ValueError("prompt-count must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("This diagnostic requires a CUDA device")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match --physical-gpu")

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    saved = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    metadata = saved.get("metadata", {})
    config = metadata.get("config")
    if not isinstance(config, dict) or config.get("variant") != "C":
        raise ValueError("Checkpoint is not a Qwen3-style C model")
    training = config["training"]
    precision = training.get("precision", "fp16")
    dataset = PackedGlyphSequenceDataset(config["data"], args.split, training["sequence_length"])
    minimum_length = max(prompt_lengths) + args.max_new + 1
    documents = choose_documents(
        dataset, count=args.prompt_count, minimum_length=minimum_length, seed=args.seed
    )
    labels, controls = label_lookup(dataset)
    eos_grid = dataset.glyph_bank[dataset.control_ids["EOS"]].numpy()

    model = model_from_config(config).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    checkpoint_progress = saved.get("progress")
    checkpoint_git_commit = metadata.get("git_commit")
    del saved
    acceleration = "native_sdpa"
    try:
        install_xformers()
        acceleration = "xformers_cutlass"
    except Exception as error:  # pragma: no cover - depends on the server wheel
        acceleration = f"native_sdpa:{type(error).__name__}"

    output.mkdir(parents=True)
    prompts_by_length: dict[int, torch.Tensor] = {}
    references_by_length: dict[int, torch.Tensor] = {}
    for length in prompt_lengths:
        prompts = torch.stack([document_glyphs(dataset, index, length) for index in documents])
        references = torch.stack(
            [document_glyphs(dataset, index, length + args.max_new) for index in documents]
        )[:, length:]
        prompts_by_length[length] = prompts
        references_by_length[length] = references

    report: dict = {
        "status": "running",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_git_commit": checkpoint_git_commit,
        "checkpoint_progress": checkpoint_progress,
        "model_family": config.get("model_family", "attention_abc_v1"),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_config": config["model"],
        "decoder_config": config["decoder"],
        "data": str(dataset.data_dir),
        "split": args.split,
        "seed": args.seed,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "precision": precision,
        "attention_backend": acceleration,
        "prompt_lengths": list(prompt_lengths),
        "max_new": args.max_new,
        "documents": documents,
        "conditions": {},
    }

    started = time.monotonic()
    for length in prompt_lengths:
        prompt = prompts_by_length[length].to(device)
        reference = references_by_length[length].numpy()
        for strategy, temperature in CONDITIONS:
            tag = "greedy" if strategy == "greedy" else f"sample_t{temperature:g}".replace(".", "p")
            name = f"prompt{length:04d}_{tag}"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            generator = torch.Generator(device=device).manual_seed(
                args.seed + length * 1009 + int(temperature * 100) + (0 if strategy == "greedy" else 17)
            )
            torch.cuda.synchronize(device)
            condition_started = time.monotonic()
            raw = rollout(
                model,
                prompt,
                max_new=args.max_new,
                strategy=strategy,
                temperature=temperature,
                generator=generator,
                precision=precision,
                device=device,
                use_cuda_graph=getattr(args, "use_cuda_graph", False),
            )
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - condition_started
            raw_cpu = raw.cpu().numpy()
            prompt_cpu = prompts_by_length[length].numpy()
            summary = generation_summary(
                prompt_cpu, raw_cpu, reference, labels, controls, eos_grid
            )
            summary.update(
                {
                    "prompt_length": length,
                    "strategy": strategy,
                    "temperature": temperature,
                    "seconds": elapsed,
                    "generated_grids_per_second": raw_cpu.shape[0] * raw_cpu.shape[1] / elapsed,
                    "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            )
            np.savez_compressed(
                output / f"{name}.npz",
                prompts=prompt_cpu,
                generated=raw_cpu,
                reference=reference[:, : args.max_new],
            )
            draw_contact_sheet(
                output / f"{name}.png",
                prompt_cpu,
                raw_cpu,
                f"{config['experiment']} | prompt={length} | {tag} | raw glyphs",
            )
            write_json(output / f"{name}.json", summary)
            report["conditions"][name] = summary
            print(
                json.dumps(
                    {
                        "condition": name,
                        "seconds": elapsed,
                        "eos_rate": summary["aggregate"]["eos_rate"],
                        "exact_content_rate": summary["aggregate"]["exact_content_rate_before_eos"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    report["status"] = "complete"
    report["seconds"] = time.monotonic() - started
    write_json(output / "result.json", report)
    write_json(
        output / "manifest.json",
        {
            "status": "complete",
            "result": "result.json",
            "images": sorted(path.name for path in output.glob("*.png")),
            "arrays": sorted(path.name for path in output.glob("*.npz")),
        },
    )
    print(json.dumps({"status": "complete", "seconds": report["seconds"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--prompt-count", type=int, default=8)
    parser.add_argument("--prompt-lengths", default=",".join(map(str, PROMPT_LENGTHS)))
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="Use CUDA Graph for accelerated inner 128-byte greedy decoding",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
