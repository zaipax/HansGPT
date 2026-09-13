"""Actual full-model fixed-shape training probes, isolated per GPU/process."""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from hansgpt_research.conditional_glyph_vae import ConditionalGlyphVAE
from hansgpt_research.cvae_fixed_step import FixedBackward, install_xformers, prepare_pixels
from hansgpt_research.glyph_lm import collate_glyph_sequences
from hansgpt_research.packed_glyph_data import PackedGlyphSequenceDataset
from hansgpt_research.train_glyph_lm import runtime_metadata, write_json
from hansgpt_research.train_structured_glyph_lm import complete_optimizer_step, optimizer_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--variant",
        choices=["fixed", "xformers", "graph", "apex", "combined", "xformers_graph"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("Wrong physical GPU selection")
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path("configs/experiments/conditional_vae_24l_10m_optimized.json").read_text())
    tc = cfg["training"]
    tc["apex_fused_adam"] = args.variant in ["apex", "combined"]
    torch.set_num_threads(4)
    torch.manual_seed(tc["seed"])
    np.random.seed(tc["seed"])
    meta = runtime_metadata(cfg, Path(cfg["data"]), torch.device("cuda"))
    meta.update(variant=args.variant, physical_gpu=args.gpu, compile_head=not args.no_compile)
    write_json(args.output / "metadata.json", meta)
    if args.variant in ["xformers", "combined", "xformers_graph"]:
        install_xformers()
    report = dict(
        status="running",
        variant=args.variant,
        gpu=args.gpu,
        steps=[],
        scope=(
            "Fixed CPU pixel dedup + shared differentiable encoder; full-model training. "
            "Graph captures backward, optimizer stays eager."
        ),
    )
    try:
        ds = PackedGlyphSequenceDataset(cfg["data"], "train", 1024)
        prepared = []
        cpu_times = []
        for i in range(args.steps + 3):
            tick = time.perf_counter()
            start = (i * 104729) % (len(ds) - 8)
            batch = collate_glyph_sequences([ds[start + j] for j in range(8)])
            prepared.append(prepare_pixels(batch))
            cpu_times.append(time.perf_counter() - tick)
        bucket = ((max(p["unique"] for p in prepared) + 127) // 128) * 128
        for p in prepared:
            p["tiles"] = torch.cat(
                (p["tiles"], p["tiles"].new_zeros(bucket - len(p["tiles"]), 1, 32, 32))
            )
        report.update(unique_bucket=bucket, cpu_preparation_seconds_mean=float(np.mean(cpu_times)))
        model = ConditionalGlyphVAE(cfg).cuda().train()
        optimizer = optimizer_for(model, tc)
        scaler = torch.amp.GradScaler("cuda")
        scaler.scale(torch.ones((), device="cuda"))
        step = FixedBackward(model, 8, 1024, 256, compiled=not args.no_compile)
        static = {
            k: prepared[0][k].cuda() for k in ["tiles", "x_index", "y_index", "targets", "mask"]
        }
        noise = torch.randn(8192, 64, device="cuda")
        scale = torch.ones((), device="cuda") * 65536
        beta = torch.ones((), device="cuda")

        def backward():
            return step(**static, noise=noise, scale=scale, beta=beta)

        def fill(p):
            for key in static:
                static[key].copy_(p[key])
            noise.normal_()
            scale.copy_(scaler._get_scale_async())

        warm = time.perf_counter()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            sums = backward()
            update = complete_optimizer_step(optimizer, scaler, model.parameters(), 1.0)
            if not update["succeeded"]:
                raise ValueError("Warmup overflow")
            scale.copy_(scaler._get_scale_async())
        torch.cuda.synchronize()
        report["warmup_seconds"] = time.perf_counter() - warm
        graph = None
        if args.variant in ["graph", "combined", "xformers_graph"]:
            optimizer.zero_grad(set_to_none=True)
            del sums
            gc.collect()
            torch.cuda.empty_cache()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    sums = backward()
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
        report["setup_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["setup_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        free, total_memory = torch.cuda.mem_get_info()
        peak_device_used = (total_memory - free) / 2**30
        torch.cuda.reset_peak_memory_stats()
        for i, p in enumerate(prepared[2 : 2 + args.steps]):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            fill(p)
            if graph is None:
                optimizer.zero_grad(set_to_none=True)
                sums = backward()
            else:
                graph.replay()
            metrics = sums.tolist()
            if not all(np.isfinite(metrics)):
                raise FloatingPointError("Nonfinite loss; no update")
            update = complete_optimizer_step(optimizer, scaler, model.parameters(), 1.0)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - tick
            free, total_memory = torch.cuda.mem_get_info()
            peak_device_used = max(peak_device_used, (total_memory - free) / 2**30)
            targets = int(p["mask"].sum())
            row = dict(
                step=i,
                seconds=seconds,
                targets=targets,
                targets_per_second=targets / seconds,
                bce=metrics[0] / (targets * 1024),
                kl=metrics[1] / targets,
                optimizer=update,
            )
            report["steps"].append(row)
            print(json.dumps(row), flush=True)
            if not update["succeeded"]:
                raise FloatingPointError("AMP overflow in measured run")
        report.update(
            status="passed",
            parameters=sum(p.numel() for p in model.parameters()),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            peak_device_used_gib=peak_device_used,
            targets_per_second=sum(r["targets"] for r in report["steps"])
            / sum(r["seconds"] for r in report["steps"]),
        )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.output / "result.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "steps"}), flush=True)


if __name__ == "__main__":
    main()
