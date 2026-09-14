"""Measure synchronized Qwen3-style C training throughput on selected GPUs.

The benchmark uses real full-length windows from the pinned packed corpus. Glyph
bytes are prepared and transferred before timing; each measured update includes
the outer forward/backward, chunked byte loss, FP32 gradient all-reduce, and the
optimizer update. It never writes checkpoints or mutates training artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from hansgpt_research.byte_training import ByteBackward, ByteCollator
from hansgpt_research.cvae_fixed_step import install_xformers
from hansgpt_research.distributed_sync import (
    DEFAULT_GRADIENT_REDUCE_BUCKET_MIB,
    allocate_gradient_reduce_buffer,
    sync_gradients,
)
from hansgpt_research.packed_glyph_data import PackedGlyphSequenceDataset
from hansgpt_research.train_attention_glyph_lm import model_from_config
from hansgpt_research.train_glyph_lm import runtime_metadata, sequence_lengths, sha256, write_json
from hansgpt_research.train_structured_glyph_lm import complete_optimizer_step, optimizer_for


def _all_gather_count(value: int, device: torch.device, world: int) -> list[int]:
    local = torch.tensor(value, dtype=torch.long, device=device)
    gathered = [torch.zeros_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)
    return [int(item) for item in gathered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/hansgpt_qwen3_c_1p5b_gpu0_3.json"),
    )
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-bucket-mib", type=int, default=None)
    parser.add_argument("--physical-gpus", default="0,1,2,3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = args.physical_gpus.split(",")
    if len(gpu_ids) < 2 or len(set(gpu_ids)) != len(gpu_ids) or any(
        not item.isdigit() for item in gpu_ids
    ):
        raise ValueError(
            "--physical-gpus must contain at least two distinct comma-separated GPU IDs"
        )
    if visible != args.physical_gpus:
        raise RuntimeError(f"Set CUDA_VISIBLE_DEVICES={args.physical_gpus}")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("Set CUDA_DEVICE_ORDER=PCI_BUS_ID for physical GPU selection")
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != len(gpu_ids) or local not in range(world):
        raise RuntimeError("WORLD_SIZE must match the number of selected physical GPUs")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = config["training"]
    context = int(cfg["sequence_length"])
    batch_size = int(args.batch_size if args.batch_size is not None else cfg["batch_size"])
    steps = int(args.steps if args.steps is not None else cfg["benchmark_steps"])
    warmup = int(args.warmup if args.warmup is not None else cfg["benchmark_warmup_steps"])
    bucket_mib = int(
        args.gradient_bucket_mib
        if args.gradient_bucket_mib is not None
        else cfg.get("gradient_reduce_bucket_mib", DEFAULT_GRADIENT_REDUCE_BUCKET_MIB)
    )
    if min(batch_size, steps, warmup, bucket_mib) < 1:
        raise ValueError("batch size, measured steps and warmup steps must be positive")
    if config.get("variant") != "C" or config.get("architecture") != "qwen3_dense_attention_only":
        raise ValueError("The benchmark requires the Qwen3-style pure C configuration")
    if cfg.get("world_size") != world or cfg.get("gradient_accumulation_steps") != 1:
        raise ValueError("Configuration world size and gradient accumulation do not match")
    data_dir = args.data or Path(config["data"])
    output = args.output
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}")

    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        install_xformers()

        dataset = PackedGlyphSequenceDataset(data_dir, "train", context)
        lengths = sequence_lengths(dataset)
        full_indices = np.flatnonzero(lengths == context)
        required = world * batch_size
        if len(full_indices) < required:
            raise ValueError(f"Need {required} full-length windows, found {len(full_indices)}")
        local_indices = full_indices[rank * batch_size : (rank + 1) * batch_size]
        cpu = ByteCollator()([dataset[int(index)] for index in local_indices])
        local_targets = int(cpu["mask"].sum())

        model = model_from_config(config).to(device).train()
        if cfg["gradient_checkpointing"]:
            model.gradient_checkpointing_enable()
        parameters = list(model.parameters())
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=30))
        counts = _all_gather_count(local_targets, device, world)
        global_targets = sum(counts)
        if not local_targets or not global_targets:
            raise ValueError("Benchmark requires nonempty target counts on every rank")
        for parameter in parameters:
            dist.broadcast(parameter.data, src=0)
        optimizer = optimizer_for(model, cfg)
        scaler = torch.amp.GradScaler("cuda")
        scaler.scale(torch.ones((), device=device))
        backward = ByteBackward(model, batch_size, context, cfg["head_chunk_size"], compiled=True)
        data = {
            key: cpu[key].to(device, non_blocking=True)
            for key in ("tiles", "indices", "byte_targets", "mask")
        }
        reduce_buffer = allocate_gradient_reduce_buffer(device, bucket_mib)

        def update() -> dict:
            optimizer.zero_grad(set_to_none=True)
            sums = backward(
                **data,
                scale=scaler._get_scale_async() * (local_targets / global_targets),
            )
            dist.all_reduce(sums)
            if not bool(torch.isfinite(sums).all()):
                raise FloatingPointError("Nonfinite distributed byte loss")
            sync_started = torch.cuda.Event(enable_timing=True)
            sync_finished = torch.cuda.Event(enable_timing=True)
            sync_started.record()
            sync_gradients(parameters, reduce_buffer)
            sync_finished.record()
            result = complete_optimizer_step(optimizer, scaler, parameters, cfg["max_grad_norm"])
            success = torch.tensor(int(result["succeeded"]), device=device)
            dist.all_reduce(success, op=dist.ReduceOp.MIN)
            if int(success) != 1:
                raise FloatingPointError("AMP overflow during throughput benchmark")
            return {
                "loss_sum": float(sums),
                "optimizer": result,
                "sync_events": (sync_started, sync_finished),
            }

        for _ in range(warmup):
            update()
        torch.cuda.synchronize(device)
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        rows = []
        for step in range(steps):
            row_started = time.perf_counter()
            result = update()
            torch.cuda.synchronize(device)
            rows.append(
                {
                    "step": step + 1,
                    "seconds": time.perf_counter() - row_started,
                    "global_targets": global_targets,
                    "nll_per_pixel": result["loss_sum"] / (global_targets * 1024),
                    "gradient_sync_seconds": result["sync_events"][0].elapsed_time(
                        result["sync_events"][1]
                    )
                    / 1000,
                }
            )
        torch.cuda.synchronize(device)
        local_elapsed = time.perf_counter() - started
        elapsed = torch.tensor(local_elapsed, dtype=torch.float64, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        local_mean_sync = torch.tensor(
            sum(row["gradient_sync_seconds"] for row in rows) / len(rows),
            dtype=torch.float64,
            device=device,
        )
        mean_syncs = [torch.zeros_like(local_mean_sync) for _ in range(world)]
        dist.all_gather(mean_syncs, local_mean_sync)
        memory = torch.tensor(
            [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
            dtype=torch.float64,
            device=device,
        )
        memories = [torch.zeros_like(memory) for _ in range(world)]
        dist.all_gather(memories, memory)

        if rank == 0:
            metadata = runtime_metadata(config, data_dir, device, mode="benchmark")
            metadata.update(
                benchmark_config_sha256=sha256(args.config),
                physical_gpus=visible,
                rank_sample_indices=full_indices[:required].tolist(),
            )
            report = {
                "status": "complete",
                "metadata": metadata,
                "architecture": config["architecture"],
                "parameters": sum(parameter.numel() for parameter in parameters),
                "world_size": world,
                "batch_size_per_rank": batch_size,
                "global_batch_size": world * batch_size,
                "context": context,
                "head_chunk_size": cfg["head_chunk_size"],
                "gradient_reduce_bucket_mib": bucket_mib,
                "gradient_reduce_bucket_elements": reduce_buffer.numel(),
                "warmup_steps": warmup,
                "measured_steps": steps,
                "global_targets_per_update": global_targets,
                "measured_seconds": float(elapsed),
                "global_targets_per_second": global_targets * steps / float(elapsed),
                "per_rank_mean_gradient_sync_seconds": [float(item) for item in mean_syncs],
                "max_mean_gradient_sync_seconds": max(float(item) for item in mean_syncs),
                "gradient_sync_fraction_of_step": max(float(item) for item in mean_syncs)
                / (float(elapsed) / steps),
                "per_rank_peak_allocated_gib": [float(item[0]) / 2**30 for item in memories],
                "per_rank_peak_reserved_gib": [float(item[1]) / 2**30 for item in memories],
                "rows": rows,
                "timing_scope": (
                    "Real full-length packed corpus windows, data already on GPU; includes "
                    "outer forward/backward, byte-head loss, NCCL gradient all-reduce and "
                    "fused AdamW update; excludes model construction and data preparation."
                ),
                "gpu_isolation": f"CUDA_VISIBLE_DEVICES={visible}",
            }
            write_json(output, report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
