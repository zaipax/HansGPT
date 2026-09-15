"""Compare validated and optimized cached byte decoding in one GPU process."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from hansgpt_research.byte_glyph_decoder import (
    BYTE_BOS,
    BYTE_VALUES,
    GRID_BYTES,
    unpack_glyph_bytes,
)
from hansgpt_research.cvae_fixed_step import install_xformers
from hansgpt_research.packed_glyph_data import PackedGlyphSequenceDataset
from hansgpt_research.train_attention_glyph_lm import model_from_config
from hansgpt_research.train_glyph_lm import autocast_context, sha256, write_json
from infer_qwen3_checkpoint import choose_documents, document_glyphs


@torch.inference_mode()
def validated_cached_decode(decoder, hidden: torch.Tensor) -> torch.Tensor:
    """Reproduce the pre-optimization path with all public API checks per byte."""
    leading = tuple(hidden.shape[:-1])
    token = torch.full((*leading, 1), BYTE_BOS, dtype=torch.long, device=hidden.device)
    cache = None
    generated = []
    for _ in range(GRID_BYTES):
        logits, cache = decoder(hidden, token, cache=cache, use_cache=True)
        scores = logits[..., -1, :].float()
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("Cannot decode nonfinite byte logits")
        value = scores.argmax(-1, keepdim=True)
        generated.append(value.to(torch.uint8))
        token = value
    return unpack_glyph_bytes(torch.cat(generated, dim=-1))


def run(args: argparse.Namespace) -> None:
    if args.rounds < 2 or args.warmup < 1 or args.prompt_length < 1:
        raise ValueError("rounds must be at least two; warmup and prompt length must be positive")
    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(","))
    if not batch_sizes or min(batch_sizes) < 1 or len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("batch sizes must be unique positive integers")
    if not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
        raise RuntimeError("Run on physical GPU0 with CUDA_VISIBLE_DEVICES=0")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    checkpoint = Path(args.checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    metadata = saved.get("metadata", {})
    config = metadata.get("config")
    if not isinstance(config, dict) or config.get("variant") != "C":
        raise ValueError("Checkpoint is not a Qwen3-style C model")
    precision = config["training"].get("precision", "fp16")
    dataset = PackedGlyphSequenceDataset(
        config["data"], args.split, config["training"]["sequence_length"]
    )
    documents = choose_documents(
        dataset,
        count=max(batch_sizes),
        minimum_length=args.prompt_length,
        seed=args.seed,
    )
    model = model_from_config(config).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    del saved
    try:
        install_xformers()
        attention_backend = "xformers_cutlass"
    except Exception as error:  # pragma: no cover - depends on the server wheel
        attention_backend = f"native_sdpa:{type(error).__name__}"

    all_prompts = torch.stack(
        [document_glyphs(dataset, document, args.prompt_length) for document in documents]
    ).to(device)
    report = {
        "status": "running",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_progress": metadata.get("progress"),
        "checkpoint_git_commit": metadata.get("git_commit"),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "gpu": torch.cuda.get_device_name(device),
        "precision": precision,
        "attention_backend": attention_backend,
        "prompt_length": args.prompt_length,
        "batch_sizes": list(batch_sizes),
        "warmup": args.warmup,
        "rounds": args.rounds,
        "cases": {},
    }
    output.mkdir(parents=True)
    for batch_size in batch_sizes:
        prompt = all_prompts[:batch_size]
        with autocast_context(device, precision):
            hidden = model.forward_hidden(prompt)[:, -1:]

        def current_fast():
            with autocast_context(device, precision):
                return model.byte_decoder.generate(hidden, strategy="greedy", use_cache=True)

        def validated_reference():
            with autocast_context(device, precision):
                return validated_cached_decode(model.byte_decoder, hidden)

        paths = {"current_fast": current_fast, "validated_reference": validated_reference}
        for _ in range(args.warmup):
            for function in paths.values():
                function()
        expected = validated_reference()
        actual = current_fast()
        if not torch.equal(actual, expected):
            raise AssertionError("Optimized and validated cached decoding differ")

        timings = {name: [] for name in paths}
        for round_index in range(args.rounds):
            order = tuple(paths) if round_index % 2 == 0 else tuple(reversed(paths))
            for name in order:
                torch.cuda.synchronize(device)
                started = time.monotonic()
                paths[name]()
                torch.cuda.synchronize(device)
                timings[name].append(time.monotonic() - started)
        case = {}
        for name, seconds in timings.items():
            median_seconds = statistics.median(seconds)
            case[name] = {
                "seconds": seconds,
                "median_seconds_per_decoder_call": median_seconds,
                "median_aggregate_grids_per_second": batch_size / median_seconds,
                "median_per_sequence_grids_per_second": 1 / median_seconds,
            }
        case["speedup"] = (
            case["validated_reference"]["median_seconds_per_decoder_call"]
            / case["current_fast"]["median_seconds_per_decoder_call"]
        )
        report["cases"][f"batch{batch_size}"] = case
        write_json(output / "result.json", report)
        print(json.dumps({"batch_size": batch_size, **case}), flush=True)
    report["status"] = "complete"
    write_json(output / "result.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-sizes", default="1,8")
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260915)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
