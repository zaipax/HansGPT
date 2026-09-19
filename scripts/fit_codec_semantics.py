"""Short semantic alignment pilot with immutable GPT input and glyph codec."""

import argparse
import json
import os
import runpy
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hansgpt_research.codec_language_lm import CodecAlignedGlyphGPT, latent_contrastive_loss
from hansgpt_research.glyph_codec import load_glyph_codec
from hansgpt_research.glyph_lm import GlyphSequenceDataset, collate_glyph_sequences
from hansgpt_research.train_attention_glyph_lm import model_from_config, validate_nll
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
    advance_progress,
    complete_optimizer_step,
    initial_progress,
    optimizer_for,
    valid_hidden,
    validation_subset,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "5":
        raise RuntimeError("Use GPU5")
    name = "codec_semantic_alignment_v1" + ("_smoke" if args.smoke else "")
    output = Path("artifacts/reports") / name
    logs = Path("artifacts/logs") / name
    checkpoints = Path("artifacts/checkpoints") / name
    for p in [output, logs, checkpoints]:
        p.mkdir(parents=True, exist_ok=False)
    receipt = json.loads(
        Path("artifacts/reports/glyph_codec_controls_v1/complete.json").read_text()
    )
    if not receipt["gate_passed"]:
        raise RuntimeError("Codec/control readiness gate failed")
    device = torch.device("cuda:0")
    torch.manual_seed(20260914)
    torch.set_num_threads(4)
    codec_path = Path("artifacts/checkpoints/glyph_codec_controls_v1/codec.pt")
    codec = load_glyph_codec(codec_path, device, expected_sha256=receipt["checkpoint_sha256"])
    source = Path("artifacts/checkpoints/hansgpt_dual_r1/best.pt")
    if sha256(source) != "67f91aee5aecf54237cdbfb9159f8fa2dc30b22bfd0b3c916f58781791edadb2":
        raise ValueError("Original checkpoint changed")
    saved = torch.load(source, map_location="cpu", weights_only=False)
    config = saved["metadata"]["config"]
    cfg = config["training"]
    cfg.update(
        target_tokens=8192 if args.smoke else 1000000,
        learning_rate=1e-4,
        minimum_learning_rate_ratio=0.3,
        warmup_tokens=50000,
        head_chunk_size=256,
        validate_every_tokens=250000,
        validation_samples=8 if args.smoke else 128,
    )
    data = Path("data/processed/modelscope_zhwiki_full_v1")
    metadata = runtime_metadata(config, data, device, mode="smoke" if args.smoke else "full")
    for key in ["data_sha256", "data_manifest_sha256", "data_verification_sha256"]:
        if metadata[key] != saved["metadata"][key]:
            raise ValueError("Corpus identity mismatch")
    base = model_from_config(config)
    base.load_state_dict(saved["model"], strict=True)
    del saved
    model = CodecAlignedGlyphGPT(base, codec).to(device)
    dataset = GlyphSequenceDataset(data, "train", 256)
    validation = GlyphSequenceDataset(data, "validation", 256)
    subset, selection = validation_subset(validation, cfg)
    lengths = sequence_lengths(dataset)
    bank = dataset.glyph_bank.cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        latent_bank = torch.cat([codec.encode(x) for x in bank.split(256)]).detach()
    _, aliases = np.unique(dataset.glyph_bank.flatten(1).numpy(), axis=0, return_inverse=True)
    aliases = torch.as_tensor(aliases, device=device)
    metadata.update(
        model_family="codec_aligned_glyph_gpt_v1",
        codec_sha256=receipt["checkpoint_sha256"],
        codec_interface=codec.interface_version,
        initial_sha256=sha256(source),
        frozen=["original glyph encoder", "original GPT backbone", "entire glyph codec"],
        trainable=["semantic decoder", "permanent semantic mapping", "latent normalization"],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        latent_contrastive_weight=0.05,
        latent_contrastive_temperature=0.1,
        target_scope="future glyph latents are detached labels only, never generation inputs",
    )
    write_json(output / "metadata.json", metadata)
    optimizer = optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler("cuda")
    progress = initial_progress()
    started = time.monotonic()

    def status(phase, state="running", **extra):
        write_json(
            logs / "status.json",
            dict(
                status=state,
                phase=phase,
                time=datetime.now(UTC).isoformat(),
                pid=os.getpid(),
                progress=progress,
                wall_seconds=time.monotonic() - started,
                **extra,
            ),
        )

    def log(kind, **extra):
        with (logs / "training.jsonl").open("a") as f:
            f.write(json.dumps(dict(kind=kind, progress=dict(progress), **extra)) + "\n")

    def validate():
        status("validation")
        metrics = validate_nll(model, subset, selection, cfg, device)
        progress["last_validation_tokens"] = progress["valid_tokens"]
        log("validation", metrics=metrics)
        if (
            progress["best_validation_nll"] is None
            or metrics["nll_per_pixel"] < progress["best_validation_nll"]
        ):
            progress.update(
                best_validation_nll=metrics["nll_per_pixel"],
                best_step=progress["optimizer_steps"],
                best_tokens=progress["valid_tokens"],
            )
            save_checkpoint(
                checkpoints / "best.pt", model, optimizer, scaler, dict(progress), metadata
            )
        model.train()

    try:
        validate()
        while progress["valid_tokens"] < cfg["target_tokens"]:
            sampler = SortishEpochSampler(
                lengths, cfg["seed"], progress["epoch"], 32, progress["cursor"], 64
            )
            loader = DataLoader(
                dataset,
                batch_size=32,
                sampler=sampler,
                num_workers=2,
                pin_memory=True,
                collate_fn=collate_glyph_sequences,
                generator=torch.Generator().manual_seed(cfg["seed"] + progress["epoch"]),
            )
            for cpu in loader:
                if progress["valid_tokens"] >= cfg["target_tokens"]:
                    break
                model.train()
                optimizer.zero_grad(set_to_none=True)
                batch = move_batch(cpu, device)
                mask = batch["loss_mask"].bool()
                ids = batch["target_ids"][mask]
                tokens = len(ids)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(progress["valid_tokens"] + tokens, cfg)
                with torch.autocast("cuda", dtype=torch.float16):
                    hidden, targets = valid_hidden(model, batch, frozen=False)
                leaf = hidden.detach().requires_grad_(True)
                pixel_sum = alignment_sum = 0.0
                for start in range(0, tokens, 256):
                    stop = start + 256
                    with torch.autocast("cuda", dtype=torch.float16):
                        latents = model.latents(leaf[start:stop])
                        pixels = codec.decode(latents)
                        pixel = (
                            F.binary_cross_entropy_with_logits(
                                pixels.float(), targets[start:stop].float(), reduction="none"
                            )
                            .flatten(1)
                            .mean(1)
                        )
                        alignment = latent_contrastive_loss(
                            latents, latent_bank[ids[start:stop]], aliases[ids[start:stop]]
                        )
                        loss = (pixel.sum() + 0.05 * alignment.sum()) / tokens
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite alignment loss")
                    scaler.scale(loss).backward()
                    pixel_sum += float(pixel.detach().sum())
                    alignment_sum += float(alignment.detach().sum())
                if leaf.grad is None:
                    raise RuntimeError("Semantic gradient missing")
                hidden.backward(leaf.grad)
                result = complete_optimizer_step(optimizer, scaler, model.parameters(), 1)
                advance_progress(
                    progress,
                    tokens=tokens,
                    samples=len(cpu["glyphs"]),
                    generator_ok=result["succeeded"],
                    discriminator_ok=None,
                )
                if progress["overflow_streak"] >= 20:
                    raise FloatingPointError("Repeated alignment overflow")
                status("training")
                if progress["optimizer_steps"] % 10 == 0:
                    log("train", pixel_nll=pixel_sum / tokens, latent_loss=alignment_sum / tokens)
                if (
                    progress["valid_tokens"] - progress["last_validation_tokens"]
                    >= cfg["validate_every_tokens"]
                ):
                    validate()
            if progress["valid_tokens"] < cfg["target_tokens"]:
                progress["epoch"] += 1
                progress["cursor"] = 0
        validate()
        save_checkpoint(
            checkpoints / "final.pt", model, optimizer, scaler, dict(progress), metadata
        )
        status("evaluation")
        model.eval()
        test = GlyphSequenceDataset(data, "test", 256)
        evaluated = test
        test_selection = {"scope": "full_test", "expected_targets": test.target_count}
        if args.smoke:
            evaluated, test_selection = validation_subset(test, cfg)
        metrics = validate_nll(model, evaluated, test_selection, cfg, device)
        helpers = runpy.run_path("scripts/evaluate_attention_abc.py")
        paired_helpers = runpy.run_path("scripts/run_dual_ablation.py")
        count = 2 if args.smoke else 32
        paired = paired_helpers["paired_metrics"](model, test, count, cfg["seed"], cfg, device)
        documents, pages = helpers["select_documents"](test, count, cfg["seed"])
        ids = np.stack(
            [test.tokens[int(test.offsets[i]) : int(test.offsets[i]) + 16] for i in documents]
        ).astype(np.int64)
        prompts = test.glyph_bank[ids].cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            generated = model.generate(
                prompts,
                4 if args.smoke else 128,
                threshold=0.5,
                eos_glyph=test.glyph_bank[test.control_ids["EOS"]],
            )
        p, g = prompts.cpu().numpy(), generated.cpu().numpy()
        labels, controls = helpers["label_lookup"](test)
        result = helpers["generation_summary"](p, g, labels, controls)
        result["source_pages"] = pages
        result["protocol"] = f"{count} independent test pages; raw bitmap feedback; 0.5 threshold"
        write_json(output / "generation.json", result)
        np.savez_compressed(output / "generation.npz", prompts=p, generated=g)
        for start in range(0, count, 4):
            helpers["draw_samples"](
                output / f"samples_{start:02d}.png",
                p[start : start + 4],
                g[start : start + 4],
                name,
            )
        write_json(
            output / "complete.json",
            dict(
                status="complete",
                full_test=metrics,
                paired_generation=paired,
                generation=result["summary"],
                progress=progress,
                codec_sha256=receipt["checkpoint_sha256"],
                final_sha256=sha256(checkpoints / "final.pt"),
            ),
        )
        status("finished", "complete")
    except BaseException as error:
        status("failed", "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
