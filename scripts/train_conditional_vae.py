"""From-scratch CVAE experiments; exact successful-Han budget, prior-only generation."""

import argparse
import json
import math
import os
import runpy
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import regex
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hansgpt_research.conditional_glyph_vae import (
    ConditionalGlyphVAE,
    gaussian_kl,
    gaussian_log_prob,
    sample_gaussian,
)
from hansgpt_research.glyph_lm import GlyphSequenceDataset, collate_glyph_sequences
from hansgpt_research.train_glyph_lm import (
    SortishEpochSampler,
    learning_rate,
    move_batch,
    runtime_metadata,
    save_checkpoint,
    sequence_lengths,
    sha256,
    write_json,
)
from hansgpt_research.train_structured_glyph_lm import (
    complete_optimizer_step,
    optimizer_for,
    validation_subset,
)


def trim_han_budget(mask, is_han, remaining):
    if remaining <= 0:
        return torch.zeros_like(mask)
    han = (mask.bool() & is_han.bool()).flatten()
    if int(han.sum()) <= remaining:
        return mask.clone()
    last = int(han.nonzero()[remaining - 1, 0])
    return mask * (torch.arange(mask.numel(), device=mask.device).reshape_as(mask) <= last)


@torch.inference_mode()
def evaluate_elbo(model, subset, selection, cfg, device):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 77)
    loader = DataLoader(
        subset,
        batch_size=cfg["validation_batch_size"],
        collate_fn=collate_glyph_sequences,
        num_workers=0,
    )
    recon = kl = count = tp = fp = fn = exact = 0
    for cpu in loader:
        batch = move_batch(cpu, device)
        mask = batch["loss_mask"].bool()
        with torch.autocast("cuda", dtype=torch.float16):
            hidden = model.forward_hidden(batch["glyphs"], batch["attention_mask"])[mask]
        targets = batch["targets"][mask]
        for start in range(0, len(hidden), cfg["head_chunk_size"]):
            h = hidden[start : start + cfg["head_chunk_size"]]
            y = targets[start : start + len(h)]
            with torch.autocast("cuda", dtype=torch.float16):
                pm, pl = model.prior(h)
                qm, ql = model.posterior(h, y)
                logits = model.decode(h, sample_gaussian(qm, ql, generator))
            recon += float(
                F.binary_cross_entropy_with_logits(logits.float(), y.float(), reduction="sum")
            )
            kl += float(gaussian_kl(qm, ql, pm, pl).sum())
            count += len(h)
            pred = logits >= 0
            truth = y.bool()
            tp += int((pred & truth).sum())
            fp += int((pred & ~truth).sum())
            fn += int((~pred & truth).sum())
            exact += int((pred == truth).flatten(1).all(1).sum())
    assert count == selection["expected_targets"]
    return dict(
        scope=selection["scope"],
        targets=count,
        posterior_reconstruction_bce_per_pixel=recon / (count * 1024),
        kl_nats_per_glyph=kl / count,
        negative_elbo_per_pixel=(recon + kl) / (count * 1024),
        posterior_sample_f1=2 * tp / max(1, 2 * tp + fp + fn),
        posterior_sample_exact=exact / count,
        warning="Posterior sees targets. Reconstruction is not generation; ELBO is not exact NLL.",
    )


def toy(model, ds, output, steps):
    device = torch.device("cuda:0")
    lookup = ds.inventory["characters"]
    bank = ds.glyph_bank.to(device)
    prefix_ids = [ds.control_ids["BOS"]] + [lookup[c] for c in "我想喝一杯"]
    prompt = bank[prefix_ids].unsqueeze(0)
    targets = bank[[lookup["茶"], lookup["水"]]]
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.95), weight_decay=0)
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator(device=device).manual_seed(519)
    rows = []
    for step in range(steps):
        model.train()
        opt.zero_grad(set_to_none=True)
        y = targets[torch.arange(64, device=device) % 2]
        with torch.autocast("cuda", dtype=torch.float16):
            h = model.forward_hidden(prompt)[:, -1].expand(64, -1)
            pm, pl = model.prior(h)
            qm, ql = model.posterior(h, y)
            z = sample_gaussian(qm, ql, generator)
            logits = model.decode(h, z)
            rec = F.binary_cross_entropy_with_logits(logits.float(), y.float())
            kl = gaussian_kl(qm, ql, pm, pl).mean()
            loss = rec + min(1, (step + 1) / 500) * kl / 1024
        scaler.scale(loss).backward()
        result = complete_optimizer_step(opt, scaler, model.parameters(), 1)
        if not result["succeeded"] and step > 100:
            raise FloatingPointError("Toy AMP overflow")
        if (step + 1) % 250 == 0:
            row = dict(
                step=step + 1, reconstruction_bce=float(rec.detach()), kl_nats=float(kl.detach())
            )
            rows.append(row)
            print("TOY", json.dumps(row), flush=True)
            write_json(output / "toy_progress.json", rows)
    model.eval()
    raw = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        h = model.forward_hidden(prompt)[:, -1].expand(64, -1)
        pm, pl = model.prior(h)
        for _ in range(8):
            raw.append((model.decode(h, sample_gaussian(pm, pl, generator)) >= 0).to(torch.uint8))
        y = targets.repeat(32, 1, 1, 1)
        qm, ql = model.posterior(h, y)
        oracle = model.decode(h, sample_gaussian(qm, ql, generator)) >= 0
        zero = model.decode(h, pm) >= 0
    raw = torch.cat(raw)
    counts = [int((raw == t).flatten(1).all(1).sum()) for t in targets]
    averaged = (targets.float().mean(0) >= 0.5).to(torch.uint8)
    report = dict(
        prior_samples=len(raw),
        exact_tea=counts[0],
        exact_water=counts[1],
        other=len(raw) - sum(counts),
        prior_valid_rate=sum(counts) / len(raw),
        posterior_sample_exact=float((oracle == y).flatten(1).all(1).float().mean()),
        prior_mean_unique_bitmaps=int(torch.unique(zero.flatten(1), dim=0).shape[0]),
        analytic_pixel_mean_matches_either=bool(any(torch.equal(averaged, t) for t in targets)),
        mechanism_passed=min(counts) >= len(raw) * 0.05 and sum(counts) >= len(raw) * 0.8,
        scope="Independent random-init toy; no weights transfer to the corpus experiment",
    )
    np.savez_compressed(
        output / "toy_samples.npz",
        targets=targets.cpu().numpy(),
        prior_samples=raw.cpu().numpy(),
        pixel_mean=averaged.cpu().numpy(),
    )
    helpers = runpy.run_path("scripts/evaluate_attention_abc.py")
    helpers["draw_samples"](
        output / "toy_samples.png",
        [prompt.cpu().numpy()[0]],
        [raw[:128].cpu().numpy()],
        "CVAE toy: blue=shared prompt; independent prior draws",
    )
    write_json(output / "complete.json", report)
    print("TOY_RESULT", json.dumps(report), flush=True)


@torch.inference_mode()
def final_evaluation(model, ds, cfg, output, smoke=False):
    model.eval()
    device = torch.device("cuda:0")
    helpers = runpy.run_path("scripts/evaluate_attention_abc.py")
    count = 2 if smoke else 32
    documents, pages = helpers["select_documents"](ds, count, cfg["seed"])
    records = [ds[int(ds.chunk_offsets[i])] for i in documents]
    x = torch.stack([r["glyphs"][:23] for r in records]).to(device)
    y = torch.stack([r["targets"][15:23] for r in records]).reshape(-1, 1, 32, 32).to(device)
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 991)
    with torch.autocast("cuda", dtype=torch.float16):
        h = model.forward_hidden(x)[:, 15:23].reshape(-1, model.config.hidden_size)
        pm, pl = model.prior(h)
        qm, ql = model.posterior(h, y)
        posterior = model.decode(h, qm)
        shuffled = model.decode(h, qm.roll(1, 0))
        prior_pixels = model.decode(h, sample_gaussian(pm, pl, generator)) >= 0
        repeated_context = h[:1].expand(64, -1)
        repeated_mean, repeated_logvar = model.prior(repeated_context)
        diverse = (
            model.decode(
                repeated_context, sample_gaussian(repeated_mean, repeated_logvar, generator)
            )
            >= 0
        )
    weights = []
    for _ in range(4 if smoke else 64):
        z = sample_gaussian(qm, ql, generator)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model.decode(h, z)
        logpx = (
            -F.binary_cross_entropy_with_logits(logits.float(), y.float(), reduction="none")
            .flatten(1)
            .sum(1)
        )
        weights.append(logpx + gaussian_log_prob(z, pm, pl) - gaussian_log_prob(z, qm, ql))
    weights = torch.stack(weights)
    estimate = -(torch.logsumexp(weights, dim=0) - math.log(len(weights))).mean() / 1024
    with torch.autocast("cuda", dtype=torch.float16):
        generated = model.generate(
            x[:, :16],
            4 if smoke else 128,
            generator=generator,
            eos_glyph=ds.glyph_bank[ds.control_ids["EOS"]],
        )
    p, g = x[:, :16].cpu().numpy(), generated.cpu().numpy()
    labels, controls = helpers["label_lookup"](ds)
    correct = y.bool()
    tp = int((prior_pixels & correct).sum())
    fp = int((prior_pixels & ~correct).sum())
    fn = int((~prior_pixels & correct).sum())
    flat = prior_pixels.flatten(1).float()
    gallery = ds.glyph_bank.flatten(1).to(device=device, dtype=torch.float32)
    distances = flat.sum(1, keepdim=True) + gallery.sum(1)[None] - 2 * flat @ gallery.T
    target_ids = torch.stack([r["target_ids"][15:23] for r in records]).flatten().to(device)
    retrieved = distances.argmin(1)
    diversity_keys = [helpers["bitmap_key"](grid) for grid in diverse.cpu().numpy()]
    result = helpers["generation_summary"](p, g, labels, controls)
    result.update(
        protocol=f"{count} test pages; one prior z per glyph; threshold .5; raw feedback only",
        source_pages=pages,
    )
    write_json(output / "generation.json", result)
    np.savez_compressed(output / "generation.npz", prompts=p, generated=g)
    for start in range(0, count, 4):
        helpers["draw_samples"](
            output / f"samples_{start:02d}.png",
            p[start : start + 4],
            g[start : start + 4],
            "CVAE: prior-only generation",
        )
    evidence = dict(
        importance_samples=len(weights),
        evaluated_positions=len(y),
        iwae_estimate_nats_per_pixel=float(estimate),
        importance_scope="Finite-sample likelihood estimate on fixed positions; not full-test NLL",
        posterior_mean_bce=float(F.binary_cross_entropy_with_logits(posterior.float(), y.float())),
        shuffled_posterior_latent_bce=float(
            F.binary_cross_entropy_with_logits(shuffled.float(), y.float())
        ),
        kl_nats_per_glyph=float(gaussian_kl(qm, ql, pm, pl).mean()),
        prior_single_draw_next_grid={
            "exact": float((prior_pixels == correct).flatten(1).all(1).float().mean()),
            "foreground_f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "foreground_dice": 2 * tp / max(1, 2 * tp + fp + fn),
            "foreground_iou": tp / max(1, tp + fp + fn),
            "hamming": float((prior_pixels != correct).flatten(1).sum(1).float().mean()),
            "nearest_glyph_top1": float((retrieved == target_ids).float().mean()),
            "retrieval_scope": "all glyphs and controls; lowest-index ties; scoring only",
        },
        same_context_prior_draws={
            "draws": 64,
            "unique_bitmaps": len(set(diversity_keys)),
            "exact_content_glyphs": sum(key in labels for key in diversity_keys),
        },
        generation=result["summary"],
    )
    write_json(output / "prior_evaluation.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["toy", "smoke", "full"], default="full")
    parser.add_argument("--toy-steps", type=int, default=2000)
    args = parser.parse_args()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "5"
        or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
    ):
        raise RuntimeError("Use physical GPU5 with explicit PCI ordering")
    config = json.loads(Path("configs/experiments/conditional_vae_1m.json").read_text())
    cfg = config["training"]
    if args.mode == "smoke":
        cfg.update(target_han=8192, target_tokens=8192, validation_samples=8)
    if args.mode == "toy":
        config["model"].update(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
        )
        config["encoder"].update(width=32, layers=1)
        config["decoders"].update(
            width=64, semantic_layers=1, glyph_layers=2, heads=4, intermediate_size=192
        )
        config["vae"]["latent_dim"] = 8
    name = config["experiment"] + "_" + args.mode
    output = Path("artifacts/reports") / name
    logs = Path("artifacts/logs") / name
    checkpoints = Path("artifacts/checkpoints") / name
    for p in [output, logs, checkpoints]:
        p.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    metadata = runtime_metadata(config, Path(config["data"]), device, mode="full")
    if metadata["data_manifest_sha256"] != config["data_requirements"]["manifest_sha256"]:
        raise ValueError("Pinned corpus mismatch")
    model = ConditionalGlyphVAE(config).to(device)
    metadata.update(
        initialization="all model weights random; no checkpoint loaded",
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        mode=args.mode,
        objective="(pixel BCE sum + beta * KL(q||p)) / 1024; beta warms up by successful Han count",
        budget="successful Han next-glyph targets; punctuation/EOS trained and counted separately",
    )
    write_json(output / "metadata.json", metadata)
    dataset = GlyphSequenceDataset(config["data"], "train", cfg["sequence_length"])
    if args.mode == "toy":
        toy(model, dataset, output, args.toy_steps)
        return
    validation = GlyphSequenceDataset(config["data"], "validation", cfg["sequence_length"])
    subset, selection = validation_subset(validation, cfg)
    lengths = sequence_lengths(dataset)
    han_lookup = torch.zeros(len(dataset.glyph_bank), dtype=torch.bool, device=device)
    for char, index in dataset.inventory["characters"].items():
        if regex.fullmatch(r"[\p{Unified_Ideograph}〇]", char):
            han_lookup[int(index)] = True
    optimizer = optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler("cuda")
    progress = dict(
        epoch=0,
        cursor=0,
        steps=0,
        han=0,
        all_targets=0,
        attempted_han=0,
        overflows=0,
        last_validation_han=0,
    )
    started = time.monotonic()
    best = float("inf")
    streak = 0

    def status(phase, state="running", **extra):
        write_json(
            logs / "status.json",
            dict(
                status=state,
                phase=phase,
                pid=os.getpid(),
                time=datetime.now(UTC).isoformat(),
                progress=progress,
                target_han=cfg["target_han"],
                seconds=time.monotonic() - started,
                **extra,
            ),
        )

    def log(kind, **extra):
        with (logs / "training.jsonl").open("a") as f:
            f.write(json.dumps(dict(kind=kind, progress=dict(progress), **extra)) + "\n")

    def validate():
        nonlocal best
        status("validation")
        metrics = evaluate_elbo(model, subset, selection, cfg, device)
        log("validation", metrics=metrics)
        progress["last_validation_han"] = progress["han"]
        if metrics["negative_elbo_per_pixel"] < best:
            best = metrics["negative_elbo_per_pixel"]
            save_checkpoint(
                checkpoints / "best.pt", model, optimizer, scaler, dict(progress), metadata
            )
        model.train()

    try:
        validate()
        while progress["han"] < cfg["target_han"]:
            sampler = SortishEpochSampler(
                lengths, cfg["seed"], progress["epoch"], cfg["batch_size"], progress["cursor"], 64
            )
            loader = DataLoader(
                dataset,
                batch_size=cfg["batch_size"],
                sampler=sampler,
                num_workers=cfg["num_workers"],
                pin_memory=True,
                collate_fn=collate_glyph_sequences,
                generator=torch.Generator().manual_seed(cfg["seed"] + progress["epoch"]),
            )
            for cpu in loader:
                if progress["han"] >= cfg["target_han"]:
                    break
                batch = move_batch(cpu, device)
                is_han = han_lookup[batch["target_ids"]]
                mask = trim_han_budget(
                    batch["loss_mask"], is_han, cfg["target_han"] - progress["han"]
                ).bool()
                han = int(is_han[mask].sum())
                total = int(mask.sum())
                if not total:
                    continue
                optimizer.zero_grad(set_to_none=True)
                model.train()
                beta = config["vae"]["beta_max"] * min(
                    1, (progress["han"] + han) / config["vae"]["kl_warmup_han"]
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(progress["han"] + han, cfg)
                with torch.autocast("cuda", dtype=torch.float16):
                    hidden = model.forward_hidden(batch["glyphs"], batch["attention_mask"])[mask]
                targets = batch["targets"][mask]
                leaf = hidden.detach().requires_grad_(True)
                recon_sum = kl_sum = 0.0
                for start in range(0, total, cfg["head_chunk_size"]):
                    h = leaf[start : start + cfg["head_chunk_size"]]
                    y = targets[start : start + len(h)]
                    with torch.autocast("cuda", dtype=torch.float16):
                        pm, pl = model.prior(h)
                        qm, ql = model.posterior(h, y)
                        z = sample_gaussian(qm, ql)
                        logits = model.decode(h, z)
                        rec = (
                            F.binary_cross_entropy_with_logits(
                                logits.float(), y.float(), reduction="none"
                            )
                            .flatten(1)
                            .sum(1)
                        )
                        kl = gaussian_kl(qm, ql, pm, pl)
                        loss = (rec + beta * kl).sum() / (total * 1024)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite CVAE loss")
                    scaler.scale(loss).backward()
                    recon_sum += float(rec.detach().sum())
                    kl_sum += float(kl.detach().sum())
                if leaf.grad is None:
                    raise RuntimeError("Missing context gradient")
                hidden.backward(leaf.grad)
                update = complete_optimizer_step(
                    optimizer, scaler, model.parameters(), cfg["max_grad_norm"]
                )
                progress["cursor"] += len(cpu["glyphs"])
                progress["attempted_han"] += han
                if update["succeeded"]:
                    progress["steps"] += 1
                    progress["han"] += han
                    progress["all_targets"] += total
                    streak = 0
                else:
                    progress["overflows"] += 1
                    streak += 1
                if streak >= 20:
                    raise FloatingPointError("Repeated AMP overflow")
                if progress["steps"] % 10 == 0:
                    log(
                        "train",
                        posterior_bce=recon_sum / (total * 1024),
                        kl_nats_per_glyph=kl_sum / total,
                        beta=beta,
                    )
                status("training")
                if progress["han"] - progress["last_validation_han"] >= cfg["validate_every_han"]:
                    validate()
            if progress["han"] < cfg["target_han"]:
                progress["epoch"] += 1
                progress["cursor"] = 0
        validate()
        assert progress["han"] == cfg["target_han"]
        save_checkpoint(
            checkpoints / "final.pt", model, optimizer, scaler, dict(progress), metadata
        )
        status("prior_evaluation")
        test = GlyphSequenceDataset(config["data"], "test", cfg["sequence_length"])
        evidence = final_evaluation(model, test, cfg, output, args.mode == "smoke")
        write_json(
            output / "complete.json",
            dict(
                status="complete",
                progress=progress,
                evaluation=evidence,
                final_sha256=sha256(checkpoints / "final.pt"),
                initialization=metadata["initialization"],
            ),
        )
        status("finished", "complete")
    except BaseException as error:
        status("failed", "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
