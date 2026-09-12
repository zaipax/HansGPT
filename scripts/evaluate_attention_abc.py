"""Full-test likelihood and fixed independent-page raw-image generation for ABC."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw

from hansgpt_research.attention_glyph_lm import AttentionGlyphGPT
from hansgpt_research.diagnose_glyph_generation import repetition_metrics
from hansgpt_research.evaluate_structured_glyph_lm import SplitAccumulator
from hansgpt_research.glyph_lm import GlyphSequenceDataset, ModelConfig
from hansgpt_research.train_attention_glyph_lm import check_gpu, validate_nll
from hansgpt_research.train_glyph_lm import autocast_context, runtime_metadata, sha256, write_json


def select_documents(dataset, count, seed):
    pages = pq.read_table(dataset.data_dir / "test.parquet", columns=["source_page_id"])
    page_ids = pages["source_page_id"].to_pylist()
    if len(page_ids) != len(dataset.offsets) - 1:
        raise ValueError("Parquet documents and glyph offsets disagree")
    chosen, seen = [], set()
    for document in np.random.default_rng(seed).permutation(len(page_ids)):
        if (
            page_ids[document] in seen
            or dataset.offsets[document + 1] - dataset.offsets[document] < 25
        ):
            continue
        chosen.append(int(document))
        seen.add(page_ids[document])
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError("Insufficient independent eligible test pages")
    return chosen, [page_ids[i] for i in chosen]


def bitmap_key(grid):
    return np.packbits(np.asarray(grid, dtype=np.uint8).reshape(-1)).tobytes()


def label_lookup(dataset):
    lookup = {}
    for char, index in dataset.inventory["characters"].items():
        lookup.setdefault(bitmap_key(dataset.glyph_bank[index].numpy()), []).append(char)
    controls = {
        bitmap_key(dataset.glyph_bank[index].numpy()): name
        for name, index in dataset.control_ids.items()
    }
    if set(lookup) & set(controls):
        raise ValueError("Content/control bitmap collision")
    return lookup, controls


def transcribe(grids, labels, controls):
    result = []
    for grid in grids:
        key = bitmap_key(grid)
        if key in controls:
            result.append(f"<{controls[key]}>")
        elif key not in labels:
            result.append("□")
        elif len(labels[key]) == 1:
            result.append(labels[key][0])
        else:
            result.append("[" + "/".join(labels[key]) + "]")
    return "".join(result)


def generation_summary(prompt, generated, labels, controls):
    summaries = []
    totals = {
        "body_grids": 0,
        "exact_content_grids": 0,
        "other_control_grids": 0,
        "terminated": 0,
        "short_under_eight": 0,
        "fully_legal_nonempty": 0,
        "repeated_adjacent_pairs": 0,
        "adjacent_pairs": 0,
        "exact_cycle_samples": 0,
        "near_cycle_samples": 0,
        "phrase_cycle_samples": 0,
    }
    for index, (prefix, output) in enumerate(zip(prompt, generated, strict=True)):
        keys = [bitmap_key(grid) for grid in output]
        eos_at = next((i for i, key in enumerate(keys) if controls.get(key) == "EOS"), None)
        body = output[:eos_at] if eos_at is not None else output
        body_keys = keys[: len(body)]
        legal = sum(key in labels for key in body_keys)
        pairs = max(0, len(body) - 1)
        repeats = sum(a == b for a, b in zip(body_keys[:-1], body_keys[1:], strict=True))
        exact_repetition = repetition_metrics(body, near_hamming=0)
        near_repetition = repetition_metrics(body, near_hamming=4)
        phrase_repetition = repetition_metrics(body, near_hamming=0, max_period=16)
        summaries.append(
            {
                "sample": index,
                "prompt": transcribe(prefix, labels, controls),
                "exact_bitmap_transcription": transcribe(body, labels, controls),
                "body_grids": len(body),
                "eos_position": eos_at,
                "exact_content_rate": legal / len(body) if len(body) else None,
                "adjacent_repeat_rate": repeats / pairs if pairs else 0,
                "exact_repetition": exact_repetition,
                "near_repetition": near_repetition,
                "phrase_repetition": phrase_repetition,
            }
        )
        totals["body_grids"] += len(body)
        totals["exact_content_grids"] += legal
        totals["other_control_grids"] += sum(key in controls for key in body_keys)
        totals["terminated"] += int(eos_at is not None)
        totals["short_under_eight"] += int(len(body) < 8)
        totals["fully_legal_nonempty"] += int(bool(len(body)) and legal == len(body))
        totals["repeated_adjacent_pairs"] += repeats
        totals["adjacent_pairs"] += pairs
        totals["exact_cycle_samples"] += int(exact_repetition["has_short_cycle"])
        totals["near_cycle_samples"] += int(near_repetition["has_short_cycle"])
        totals["phrase_cycle_samples"] += int(phrase_repetition["has_short_cycle"])
    totals["exact_content_rate"] = totals["exact_content_grids"] / max(1, totals["body_grids"])
    totals["adjacent_repeat_rate"] = totals["repeated_adjacent_pairs"] / max(
        1, totals["adjacent_pairs"]
    )
    return {
        "summary": totals,
        "samples": summaries,
        "transcription": "Exact bitmap lookup; □ = no exact match; aliases shown explicitly",
        "protocol": "32 test pages; 16-grid prompts; up to 128 grids; exact EOS stopping",
        "semantic_review": "not yet reviewed; legality/repetition are not semantic scores",
    }


def draw_samples(path, prompts, generated, title):
    # Four sample pages per image; all raw pixels, with prompts outlined in blue.
    cell, columns = 34, 24
    rows_per_sample = 7
    canvas = Image.new(
        "RGB", (columns * cell + 16, len(prompts) * rows_per_sample * cell + 32), "white"
    )
    pen = ImageDraw.Draw(canvas)
    pen.text((8, 6), title, fill="black")
    for sample, (prefix, output) in enumerate(zip(prompts, generated, strict=True)):
        grids = np.concatenate((prefix, output))
        base = 28 + sample * rows_per_sample * cell
        for j, grid in enumerate(grids):
            tile = Image.fromarray((255 * (1 - grid.reshape(32, 32))).astype(np.uint8))
            x, y = 8 + (j % columns) * cell, base + (j // columns) * cell
            canvas.paste(tile, (x, y))
            if j < len(prefix):
                pen.rectangle((x - 1, y - 1, x + 32, y + 32), outline="blue")
    canvas.save(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=list("ABC"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    check_gpu(args.variant, device)
    if args.output.exists():
        raise FileExistsError("Evaluation output must be fresh")
    name = "hansgpt_abc_r2_" + args.variant.lower()
    logs = Path("artifacts/logs") / name
    receipt = json.loads((logs / "training_complete.json").read_text())
    checkpoint = Path("artifacts/checkpoints") / name / "best.pt"
    if (
        receipt["status"] != "complete"
        or receipt["mode"] != "full"
        or receipt["metadata_sha256"] != sha256(logs / "metadata.json")
        or receipt["best_checkpoint_sha256"] != sha256(checkpoint)
    ):
        raise ValueError("Training completion or checkpoint identity mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = saved["metadata"]
    config, cfg = metadata["config"], metadata["config"]["training"]
    data = Path("data/processed/modelscope_zhwiki_full_v1")
    evaluation_metadata = runtime_metadata(config, data, device, mode="full")
    for key in ("data_manifest_sha256", "data_sha256", "data_verification_sha256"):
        if evaluation_metadata[key] != metadata[key]:
            raise ValueError(f"Evaluation data mismatch: {key}")
    model = (
        AttentionGlyphGPT(
            ModelConfig.from_dict(config["model"]),
            args.variant,
            config["encoder"],
            config["decoder"],
        )
        .to(device)
        .eval()
    )
    model.load_state_dict(saved["model"], strict=True)
    del saved
    dataset = GlyphSequenceDataset(data, "test", cfg["sequence_length"])
    args.output.mkdir(parents=True)
    write_json(
        args.output / "metadata.json",
        {
            "training": metadata,
            "evaluation": evaluation_metadata,
            "checkpoint_sha256": receipt["best_checkpoint_sha256"],
        },
    )
    started = time.monotonic()
    selection = {
        "scope": "full_test",
        "expected_targets": dataset.target_count,
        "chunks": len(dataset),
    }
    full_test = validate_nll(model, dataset, selection, cfg, device)
    write_json(args.output / "full_test_nll.json", full_test)
    print("Full-test likelihood complete", full_test, flush=True)
    documents, pages = select_documents(dataset, 32, 20260913)
    records = [dataset[int(dataset.chunk_offsets[document])] for document in documents]
    inputs = torch.stack([r["glyphs"][:23] for r in records]).to(device)
    targets = torch.stack([r["targets"][15:23] for r in records]).reshape(-1, 1, 32, 32).to(device)
    target_ids = torch.stack([r["target_ids"][15:23] for r in records]).reshape(-1).to(device)
    with autocast_context(device, cfg["precision"]):
        hidden = model.forward_hidden(inputs)[:, 15:23].reshape(-1, config["model"]["hidden_size"])
        distribution = model.distribution(hidden)
        nll = distribution.nll(targets)
        shuffled_nll = model.distribution(hidden.roll(8, 0)).nll(targets)
        prediction = model.decode_grid(hidden, threshold=0.5)
    accumulator = SplitAccumulator(dataset, device)
    accumulator.add(prediction, targets, target_ids, nll)
    paired = {
        "scope": "256 positions: 8 per selected independent test page; true outer prefixes",
        "metrics": accumulator.result(),
        "true_context_nll": float(nll.mean()),
        "wrong_page_context_nll": float(shuffled_nll.mean()),
        "decode": "greedy bytes" if model.byte_decoder else "pixel threshold 0.5",
    }
    write_json(args.output / "paired_generation.json", paired)
    print("Paired generation complete", flush=True)
    prompts = inputs[:, :16]
    with autocast_context(device, cfg["precision"]):
        generated = model.generate(
            prompts, 128, threshold=0.5, eos_glyph=dataset.glyph_bank[dataset.control_ids["EOS"]]
        )
    raw, prefix = generated.cpu().numpy(), prompts.cpu().numpy()
    if raw.dtype != np.uint8 or not np.isin(raw, [0, 1]).all():
        raise ValueError("Generation must remain raw binary uint8")
    np.savez_compressed(args.output / "raw_generation.npz", prompts=prefix, generated=raw)
    labels, controls = label_lookup(dataset)
    generation = generation_summary(prefix, raw, labels, controls)
    generation["documents"], generation["source_pages"] = documents, pages
    write_json(args.output / "generation.json", generation)
    for start in range(0, len(prefix), 4):
        draw_samples(
            args.output / f"samples_{start:02d}.png",
            prefix[start : start + 4],
            raw[start : start + 4],
            f"{args.variant}: samples {start}-{start + 3}; blue=prompt",
        )
    result = {
        "status": "complete",
        "variant": args.variant,
        "full_test": full_test,
        "paired_generation": paired,
        "generation_summary": generation["summary"],
        "seconds": time.monotonic() - started,
        "metadata_sha256": sha256(args.output / "metadata.json"),
        "outputs": {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()},
    }
    write_json(args.output / "evaluation_complete.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
