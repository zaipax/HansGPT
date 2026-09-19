"""Isolated single-GPU end-to-end training ablations; never saves weights."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hansgpt_research.byte_training import ByteBackward, ByteCollator
from hansgpt_research.cvae_fixed_step import install_xformers
from hansgpt_research.packed_glyph_data import PackedGlyphSequenceDataset
from hansgpt_research.train_attention_glyph_lm import model_from_config
from hansgpt_research.train_glyph_lm import runtime_metadata, sequence_lengths, write_json
from hansgpt_research.train_structured_glyph_lm import complete_optimizer_step, optimizer_for

KEYS = ("tiles", "indices", "byte_targets", "mask")


class PackedByteDataset(PackedGlyphSequenceDataset):
    """Benchmark candidate: preserve packed-window semantics without dense glyphs."""

    def __getitem__(self, index):
        start = int(index) * self.sequence_length
        ids = np.array(self.tokens[start : start + self.sequence_length + 1], dtype=np.int64)
        valid = len(ids) - 1
        inputs = np.full(self.sequence_length, self.control_ids["PAD"], dtype=np.int64)
        targets = inputs.copy()
        inputs[:valid], targets[:valid] = ids[:-1], ids[1:]
        mask = (np.arange(self.sequence_length) < valid) & (targets != self.control_ids["BOS"])
        return inputs, targets, mask


class PackedByteCollator:
    def __init__(self, bank):
        pixels = bank.numpy().reshape(len(bank), 1024)
        if not np.isin(pixels, [0, 1]).all():
            raise ValueError("Binary glyph bank required")
        self.bank = np.packbits(pixels, axis=1)

    def __call__(self, samples):
        inputs, targets, mask = (
            np.stack(items).reshape(-1) for items in zip(*samples, strict=True)
        )
        unique, inverse = np.unique(self.bank[inputs], axis=0, return_inverse=True)
        return dict(
            tiles=torch.from_numpy(np.unpackbits(unique, axis=1).reshape(-1, 1, 32, 32)),
            indices=torch.from_numpy(inverse),
            byte_targets=torch.from_numpy(self.bank[targets]),
            mask=torch.from_numpy(mask),
            target_ids=torch.from_numpy(targets),
        )


def run(args):
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "2"
        or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
    ):
        raise RuntimeError("Set CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2")
    config = json.loads(args.config.read_text())
    cfg = config["training"]
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    install_xformers(opaque_backward=args.opaque_backward)
    device = torch.device("cuda:0")
    data_dir = Path(config["data"])
    metadata = runtime_metadata(config, data_dir, device, mode="benchmark")
    context, batch_size = cfg["sequence_length"], cfg["batch_size"]
    ds = PackedGlyphSequenceDataset(data_dir, "train", context)
    indices = np.flatnonzero(sequence_lengths(ds) == context)[:batch_size].tolist()
    if len(indices) != batch_size:
        raise ValueError("Insufficient full windows")
    cpu = ByteCollator()([ds[i] for i in indices])
    targets = int(cpu["mask"].sum())
    batch_indices = [indices] * (args.warmup + args.steps + 4)
    if args.vary_data:
        full = np.flatnonzero(sequence_lengths(ds) == context)
        chosen = np.random.default_rng(cfg["seed"]).choice(
            full, size=len(batch_indices) * batch_size, replace=False
        )
        batch_indices = chosen.reshape(-1, batch_size).tolist()
    collator = ByteCollator()
    if args.packed_data:
        packed = PackedByteDataset(data_dir, "train", context)
        collator = PackedByteCollator(packed.glyph_bank)
        # Include EOS transitions and the final partial window in equivalence checks.
        check_ids = list(dict.fromkeys(indices + list(range(32)) + [len(ds) - 1]))
        for offset in range(0, len(check_ids), batch_size):
            chosen = check_ids[offset : offset + batch_size]
            reference = ByteCollator()([ds[i] for i in chosen])
            candidate = collator([packed[i] for i in chosen])
            for key in reference:
                if not torch.equal(reference[key], candidate[key]):
                    raise AssertionError(f"Packed input mismatch: {key}, {chosen}")
        ds = packed
    loader = DataLoader(
        ds,
        batch_sampler=batch_indices,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
        multiprocessing_context="spawn",
        persistent_workers=True,
    )
    batches = iter(loader)
    model = model_from_config(config).to(device).train()
    if not args.no_checkpoint:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if args.compile_backbone:
        model.backbone = torch.compile(model.backbone, fullgraph=True, dynamic=False)
    elif args.compile_layers:
        # Keep HF's configuration/output wrapper eager; compile the actual blocks.
        for layer in model.backbone.layers:
            layer.forward = torch.compile(layer.forward, fullgraph=True, dynamic=False)
    parameters = list(model.parameters())
    optimizer = optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    scaler.scale(torch.ones((), device=device))
    backward = ByteBackward(model, batch_size, context, args.chunk, compiled=True)

    def update():
        started = time.perf_counter()
        cpu_batch = next(batches)
        loaded = time.perf_counter()
        data = {key: cpu_batch[key].to(device, non_blocking=True) for key in KEYS}
        optimizer.zero_grad(set_to_none=True)
        loss = backward(**data, scale=scaler._get_scale_async(), checked=True)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite loss")
        result = complete_optimizer_step(optimizer, scaler, parameters, cfg["max_grad_norm"])
        if not result["succeeded"]:
            raise FloatingPointError("AMP overflow; no valid throughput result")
        torch.cuda.synchronize()
        return dict(
            seconds=time.perf_counter() - started,
            loader_wait_seconds=loaded - started,
            nll_per_pixel=float(loss) / (targets * 1024),
            **result,
        )

    started = time.perf_counter()
    warmup = []
    for step in range(args.warmup):
        row = update()
        warmup.append(row)
        print(json.dumps(dict(phase="warmup", step=step, **row)), flush=True)
    warmup_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    rows = []
    for step in range(args.steps):
        row = update()
        rows.append(row)
        print(json.dumps(dict(phase="measure", step=step, **row)), flush=True)
    elapsed = time.perf_counter() - started
    report = dict(
        status="complete",
        metadata=metadata,
        options=vars(args) | {"config": str(args.config), "output": str(args.output)},
        sample_indices=batch_indices,
        targets_per_step=targets,
        rows=rows,
        warmup=warmup,
        warmup_seconds_including_compile=warmup_seconds,
        measured_seconds=elapsed,
        targets_per_second=targets * args.steps / elapsed,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        scope=(
            "Real full windows; includes loader wait, H2D, "
            "checked forward/backward, clip and fused AdamW. No NCCL, checkpoint I/O, "
            "validation or recovery retry. Random initialization; not convergence evidence."
        ),
    )
    write_json(args.output, report)
    if args.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as prof:
            update()
        args.output.with_suffix(".profile.txt").write_text(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
        )
    print(
        json.dumps({k: report[k] for k in ("status", "targets_per_second", "peak_allocated_gib")}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/hansgpt_qwen3_parallel_byte_gpu3_100m_recovery.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--compile-backbone", action="store_true")
    parser.add_argument("--compile-layers", action="store_true")
    parser.add_argument("--packed-data", action="store_true")
    parser.add_argument("--vary-data", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--opaque-backward", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.steps, args.warmup) < 1:
        raise ValueError("Positive steps required")
    try:
        run(args)
    except Exception as exc:
        write_json(
            args.output, dict(status="failed", error=str(exc), traceback=traceback.format_exc())
        )
        raise


if __name__ == "__main__":
    main()

