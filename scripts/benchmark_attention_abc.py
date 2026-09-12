"""Same-real-batch GPU throughput and capacity measurements, isolated from formal runs."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import subprocess
import time
from pathlib import Path

import torch

from hansgpt_research.glyph_lm import GlyphSequenceDataset, collate_glyph_sequences
from hansgpt_research.train_attention_glyph_lm import check_gpu, model_from_config
from hansgpt_research.train_glyph_lm import sequence_lengths, write_json
from hansgpt_research.train_structured_glyph_lm import (
    complete_optimizer_step,
    generator_backward,
    optimizer_for,
)


def run_case(config, samples, case, device, steps):
    torch.manual_seed(config["training"]["seed"])
    cfg = dict(config["training"])
    cfg["head_chunk_size"] = case["head_chunk_size"]
    cfg["fused_adamw"] = case.get("fused_adamw", False)
    architecture = dict(config["model"])
    architecture.update(case.get("model", {}))
    if "encode_chunk_size" in case:
        architecture["glyph_encode_chunk_size"] = case["encode_chunk_size"]
    construction = copy.deepcopy(config)
    construction["model"] = architecture
    model = model_from_config(construction).to(device).train()
    if case["checkpointing"]:
        model.gradient_checkpointing_enable()
    optimizer = optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    group = [
        collate_glyph_sequences(samples[i : i + case["batch_size"]])
        for i in range(0, len(samples), case["batch_size"])
    ]
    tokens = sum(int(b["loss_mask"].sum()) for b in group)
    measured, successful, norms = [], 0, []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(steps + 2):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        loss = generator_backward(model, None, group, cfg, device, scaler, tokens, frozen=False)
        outcome = complete_optimizer_step(
            optimizer, scaler, model.parameters(), cfg["max_grad_norm"]
        )
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        if step >= 2:
            measured.append(seconds)
            successful += int(outcome["succeeded"])
            norms.append(outcome["grad_norm"])
    return {
        "case": case,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds_per_update": sum(measured) / len(measured),
        "attempted_targets_per_second": tokens * len(measured) / sum(measured),
        "successful_targets_per_second": tokens * successful / sum(measured),
        "successful_steps": successful,
        "measured_steps": len(measured),
        "tokens_per_update": tokens,
        "final_nll_per_pixel": loss["nll_sum"] / tokens,
        "gradient_norms": norms,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=list("ABC"), required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    device = torch.device("cuda:0")
    check_gpu(args.variant, device)
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Benchmark requires clean committed source")
    config = json.loads(
        Path(f"configs/experiments/hansgpt_attention_{args.variant.lower()}.json").read_text()
    )
    dataset = GlyphSequenceDataset("data/processed/modelscope_zhwiki_full_v1", "train", 256)
    # Deterministic full-length real chunks stress actual capacity, not short padded batches.
    indices = (sequence_lengths(dataset) == 256).nonzero()[0][:32]
    samples = [dataset[int(index)] for index in indices]
    if len(samples) != 32:
        raise ValueError("Need 32 full-length real training chunks")
    cases = [
        {"name": "baseline", "batch_size": 4, "head_chunk_size": 32, "checkpointing": True},
        {"name": "batch32", "batch_size": 32, "head_chunk_size": 32, "checkpointing": True},
        {"name": "head256", "batch_size": 32, "head_chunk_size": 256, "checkpointing": True},
        {"name": "no_recompute", "batch_size": 32, "head_chunk_size": 256, "checkpointing": False},
        {
            "name": "encode256",
            "batch_size": 32,
            "head_chunk_size": 256,
            "checkpointing": False,
            "encode_chunk_size": 256,
        },
        {
            "name": "large",
            "batch_size": 32,
            "head_chunk_size": 256,
            "checkpointing": False,
            "encode_chunk_size": 256,
            "model": {
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "intermediate_size": 2816,
            },
        },
    ]
    report = {
        "variant": args.variant,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu": torch.cuda.get_device_name(device),
        "sample_indices": indices.tolist(),
        "torch_threads": torch.get_num_threads(),
        "scope": "32 real chunks; includes transfer/backward/update, excludes loading/saving",
        "cases": [],
    }
    for case in cases:
        if args.only and case["name"] not in args.only:
            continue
        try:
            result = run_case(config, samples, case, device, args.steps)
        except torch.OutOfMemoryError:
            result = {"case": case, "error": "cuda_out_of_memory"}
        report["cases"].append(result)
        print(json.dumps(result), flush=True)
        write_json(args.output, report)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
