"""Bounded, fresh-process memory probes using the production CVAE training graph."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hansgpt_research.conditional_glyph_vae import (
    ConditionalGlyphVAE,
    gaussian_kl,
    sample_gaussian,
)
from hansgpt_research.glyph_lm import collate_glyph_sequences
from hansgpt_research.packed_glyph_data import PackedGlyphSequenceDataset
from hansgpt_research.train_glyph_lm import move_batch, runtime_metadata, write_json
from hansgpt_research.train_structured_glyph_lm import complete_optimizer_step, optimizer_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--head-chunk-size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.head_chunk_size) <= 0 or args.steps < 3:
        raise ValueError("Positive batch/head sizes and at least three successful steps required")
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu)
        or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
    ):
        raise ValueError("Use the explicitly selected physical GPU and PCI_BUS_ID order")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    config = json.loads(Path("configs/experiments/conditional_vae_24l_1m.json").read_text())
    config["data"] = "data/processed/chinese_document_v3"
    config["data_requirements"] = dict(
        corpus_type="chinese_document_packed_v3",
        manifest_sha256="9962a55afc778caf74ef12a24fe017e594c2a683729738e68bce364ab11298a9",
    )
    cfg = config["training"]
    cfg.update(
        sequence_length=1024,
        packing="eos_causal",
        batch_size=args.batch_size,
        head_chunk_size=args.head_chunk_size,
    )
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda:0")
    metadata = runtime_metadata(config, Path(config["data"]), device)
    write_json(args.output / "metadata.json", metadata)
    result = dict(
        status="running",
        batch_size=args.batch_size,
        head_chunk_size=args.head_chunk_size,
        context=1024,
        steps=[],
        gradient_checkpointing=False,
        precision="fp16",
        beta=1.0,
        note="Full model, Adam state and backwards; first successful step is warmup",
    )
    try:
        dataset = PackedGlyphSequenceDataset(config["data"], "train", 1024)
        model = ConditionalGlyphVAE(config).to(device).train()
        result["parameters"] = sum(p.numel() for p in model.parameters())
        result["trainable_parameters"] = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        optimizer = optimizer_for(model, cfg)
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        successes = attempts = 0
        while successes < args.steps and attempts < args.steps + 12:
            # Vary real source windows, rather than reusing a single easy batch.
            base = (attempts * 104729) % (len(dataset) - args.batch_size)
            cpu = collate_glyph_sequences([dataset[base + i] for i in range(args.batch_size)])
            torch.cuda.synchronize()
            started = time.perf_counter()
            batch = move_batch(cpu, device)
            mask = batch["loss_mask"].bool()
            total = int(mask.sum())
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                hidden = model.forward_hidden(batch["glyphs"], batch["attention_mask"])[mask]
            targets = batch["targets"][mask]
            leaf = hidden.detach().requires_grad_(True)
            bce_sum = kl_sum = 0.0
            for start in range(0, total, args.head_chunk_size):
                h = leaf[start : start + args.head_chunk_size]
                y = targets[start : start + len(h)]
                with torch.autocast("cuda", dtype=torch.float16):
                    pm, pl = model.prior(h)
                    qm, ql = model.posterior(h, y)
                    logits = model.decode(h, sample_gaussian(qm, ql))
                    rec = (
                        F.binary_cross_entropy_with_logits(
                            logits.float(), y.float(), reduction="none"
                        )
                        .flatten(1)
                        .sum(1)
                    )
                    kl = gaussian_kl(qm, ql, pm, pl)
                    loss = (rec + kl).sum() / (total * 1024)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite CVAE loss")
                scaler.scale(loss).backward()
                bce_sum += float(rec.detach().sum())
                kl_sum += float(kl.detach().sum())
            if leaf.grad is None:
                raise RuntimeError("Missing context gradient")
            hidden.backward(leaf.grad)
            update = complete_optimizer_step(
                optimizer, scaler, model.parameters(), cfg["max_grad_norm"]
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            attempts += 1
            successes += int(update["succeeded"])
            entry = dict(
                attempt=attempts,
                succeeded=update["succeeded"],
                seconds=elapsed,
                targets=total,
                targets_per_second=total / elapsed,
                bce_per_pixel=bce_sum / (total * 1024),
                kl_per_glyph=kl_sum / total,
                optimizer=update,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            )
            result["steps"].append(entry)
            print(json.dumps(entry), flush=True)
            write_json(args.output / "progress.json", result)
        if successes != args.steps:
            raise FloatingPointError("Too many skipped AMP updates")
        result["status"] = "passed"
        timed = [s for s in result["steps"] if s["succeeded"]][1:]
        result["steady_targets_per_second"] = sum(s["targets"] for s in timed) / sum(
            s["seconds"] for s in timed
        )
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        result["optimizer_state_tensors"] = sum(len(v) for v in optimizer.state.values())
    except torch.cuda.OutOfMemoryError:
        result["status"] = "out_of_memory"
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.output / "result.json", result)
        print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
