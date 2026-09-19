"""Benchmark raw Qwen3-style C glyph generation on one CUDA device."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from hansgpt_research.cvae_fixed_step import install_xformers
from hansgpt_research.packed_glyph_data import PackedGlyphSequenceDataset
from hansgpt_research.train_attention_glyph_lm import model_from_config
from hansgpt_research.train_glyph_lm import autocast_context, sha256, write_json
from infer_qwen3_checkpoint import choose_documents, document_glyphs, rollout


def parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or min(values) < 1 or len(set(values)) != len(values):
        raise ValueError("Expected unique positive comma-separated integers")
    return values


@torch.inference_mode()
def profile_rollout(
    model,
    prompt: torch.Tensor,
    *,
    max_new: int,
    precision: str,
    device: torch.device,
    use_cuda_graph: bool = False,
) -> dict:
    """Split rollout wall time into outer-model and byte-decoder CUDA intervals."""
    context = prompt.to(device=device, dtype=torch.uint8)
    model_input = context[:, -model.config.max_position_embeddings :]
    cache = None
    generated = []
    events = []
    generator = torch.Generator(device=device).manual_seed(20260915)
    torch.cuda.synchronize(device)
    wall_started = time.monotonic()
    for _ in range(max_new):
        outer_started = torch.cuda.Event(enable_timing=True)
        outer_finished = torch.cuda.Event(enable_timing=True)
        decoder_finished = torch.cuda.Event(enable_timing=True)
        outer_started.record()
        with autocast_context(device, precision):
            hidden, next_cache = model.forward_hidden(
                model_input, past_key_values=cache, use_cache=True, return_cache=True
            )
        outer_finished.record()
        with autocast_context(device, precision):
            distribution = model.distribution(hidden[:, -1:])
            tile = distribution.decode(
                strategy="greedy",
                temperature=1.0,
                generator=generator,
                use_cache=True,
                use_cuda_graph=use_cuda_graph,
            )
        decoder_finished.record()
        if tile.dtype != torch.uint8 or not bool(((tile == 0) | (tile == 1)).all()):
            raise ValueError("Raw rollout must return strict binary uint8 grids")
        generated.append(tile)
        context = torch.cat((context, tile), dim=1)
        if context.shape[1] <= model.config.max_position_embeddings:
            cache, model_input = next_cache, tile
        else:
            cache = None
            model_input = context[:, -model.config.max_position_embeddings :]
        events.append((outer_started, outer_finished, decoder_finished))
    torch.cuda.synchronize(device)
    wall_seconds = time.monotonic() - wall_started
    outer_seconds = sum(start.elapsed_time(finish) for start, finish, _ in events) / 1000
    decoder_seconds = sum(finish.elapsed_time(end) for _, finish, end in events) / 1000
    output = torch.cat(generated, dim=1)
    return {
        "steps": max_new,
        "generated_grids": int(output.shape[0] * output.shape[1]),
        "wall_seconds": wall_seconds,
        "outer_cuda_seconds": outer_seconds,
        "byte_decoder_cuda_seconds": decoder_seconds,
        "unattributed_seconds": max(0.0, wall_seconds - outer_seconds - decoder_seconds),
        "outer_fraction": outer_seconds / wall_seconds,
        "byte_decoder_fraction": decoder_seconds / wall_seconds,
        "unattributed_fraction": max(0.0, wall_seconds - outer_seconds - decoder_seconds)
        / wall_seconds,
    }


def run(args: argparse.Namespace) -> None:
    batch_sizes = parse_ints(args.batch_sizes)
    prompt_lengths = parse_ints(args.prompt_lengths)
    if args.max_new < 1 or args.warmup_new < 1 or args.profile_new < 1 or args.repeats < 2:
        raise ValueError("Generation lengths must be positive and repeats must be at least two")
    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("This benchmark requires CUDA")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError(f"Set CUDA_VISIBLE_DEVICES={args.physical_gpu}")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
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
        minimum_length=max(prompt_lengths),
        seed=args.seed,
    )
    model = model_from_config(config).to(device).eval()
    model.load_state_dict(saved["model"], strict=True)
    del saved
    attention_backend = "native_sdpa"
    try:
        install_xformers()
        attention_backend = "xformers_cutlass"
    except Exception as error:  # pragma: no cover - depends on the server wheel
        attention_backend = f"native_sdpa:{type(error).__name__}"

    prompts = {
        length: torch.stack(
            [document_glyphs(dataset, document, length) for document in documents]
        )
        for length in prompt_lengths
    }
    report = {
        "status": "running",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_progress": metadata.get("progress"),
        "checkpoint_git_commit": metadata.get("git_commit"),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(device),
        "precision": precision,
        "attention_backend": attention_backend,
        "batch_sizes": list(batch_sizes),
        "prompt_lengths": list(prompt_lengths),
        "max_new": args.max_new,
        "warmup_new": args.warmup_new,
        "repeats": args.repeats,
        "profile_new": args.profile_new,
        "cases": {},
    }
    output.mkdir(parents=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for length in prompt_lengths:
        for batch_size in batch_sizes:
            name = f"prompt{length:04d}_batch{batch_size}"
            prompt = prompts[length][:batch_size].to(device)
            rollout(
                model,
                prompt,
                max_new=args.warmup_new,
                strategy="greedy",
                temperature=1.0,
                generator=generator,
                precision=precision,
                device=device,
                use_cuda_graph=args.use_cuda_graph,
            )
            measurements = []
            first_output = None
            torch.cuda.reset_peak_memory_stats(device)
            for repeat in range(args.repeats):
                torch.cuda.synchronize(device)
                started = time.monotonic()
                raw = rollout(
                    model,
                    prompt,
                    max_new=args.max_new,
                    strategy="greedy",
                    temperature=1.0,
                    generator=generator,
                    precision=precision,
                    device=device,
                    use_cuda_graph=args.use_cuda_graph,
                )
                torch.cuda.synchronize(device)
                elapsed = time.monotonic() - started
                if first_output is None:
                    first_output = raw
                elif not torch.equal(first_output, raw):
                    raise AssertionError("Greedy output changed between benchmark repeats")
                measurements.append(
                    {
                        "repeat": repeat,
                        "seconds": elapsed,
                        "aggregate_grids_per_second": batch_size * args.max_new / elapsed,
                        "per_sequence_grids_per_second": args.max_new / elapsed,
                    }
                )
            throughputs = [item["aggregate_grids_per_second"] for item in measurements]
            case = {
                "prompt_length": length,
                "batch_size": batch_size,
                "measurements": measurements,
                "median_aggregate_grids_per_second": statistics.median(throughputs),
                "mean_aggregate_grids_per_second": statistics.mean(throughputs),
                "min_aggregate_grids_per_second": min(throughputs),
                "max_aggregate_grids_per_second": max(throughputs),
                "median_per_sequence_grids_per_second": statistics.median(throughputs)
                / batch_size,
                "profile": profile_rollout(
                    model,
                    prompt,
                    max_new=args.profile_new,
                    precision=precision,
                    device=device,
                    use_cuda_graph=args.use_cuda_graph,
                ),
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            report["cases"][name] = case
            write_json(output / "result.json", report)
            print(
                json.dumps(
                    {
                        "case": name,
                        "aggregate_grids_per_second": case[
                            "median_aggregate_grids_per_second"
                        ],
                        "per_sequence_grids_per_second": case[
                            "median_per_sequence_grids_per_second"
                        ],
                        "profile": case["profile"],
                    }
                ),
                flush=True,
            )
    report["status"] = "complete"
    write_json(output / "result.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-sizes", default="1,8")
    parser.add_argument("--prompt-lengths", default="8,512,1024")
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--warmup-new", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile-new", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="Use CUDA Graph for inner 128-byte greedy decoding",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
