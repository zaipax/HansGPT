"""Train and gate glyph-only repairs; never silently attach them to an old GPT."""

import argparse
import json
import math
import os
import runpy
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hansgpt_research.attention_glyph_lm import AttentionGlyphEncoder
from hansgpt_research.dual_decoder_glyph_lm import DualDecoderConfig, SpatialGlyphDecoder
from hansgpt_research.glyph_codec import GlyphCodec
from hansgpt_research.glyph_lm import GlyphSequenceDataset
from hansgpt_research.train_glyph_lm import sha256, verify_data_readiness, write_json
from hansgpt_research.train_structured_glyph_lm import complete_optimizer_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["fixed", "spatial"], required=True)
    parser.add_argument("--config", default="configs/experiments/glyph_codec_repair.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.smoke:
        cfg.update(small_steps=2, full_steps=2, small_exact_gate=0)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(cfg["gpus"][args.arm]):
        raise RuntimeError("Wrong GPU")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("GPU order must be explicit")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Clean source required")
    name = cfg["name"] + ("_smoke" if args.smoke else "") + "_" + args.arm
    output = Path("artifacts/reports") / name
    logs = Path("artifacts/logs") / name
    checkpoints = Path("artifacts/checkpoints") / name
    for p in [output, logs, checkpoints]:
        p.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    initial = Path(cfg["initial_checkpoint"])
    assert sha256(initial) == cfg["initial_sha256"]
    saved = torch.load(initial, map_location="cpu", weights_only=False)
    base = saved["metadata"]["config"]
    readiness = verify_data_readiness(Path(cfg["data"]))
    if readiness["manifest_sha256"] != saved["metadata"]["data_manifest_sha256"]:
        raise ValueError("Initial checkpoint and corpus manifest differ")
    if readiness["model_consumed_sha256"] != saved["metadata"]["data_sha256"]:
        raise ValueError("Initial checkpoint and glyph assets differ")
    e = AttentionGlyphEncoder(1024, **base["encoder"])
    d = SpatialGlyphDecoder(DualDecoderConfig(**base["decoders"]))
    e.load_state_dict(
        {
            k[len("glyph_encoder.") :]: v
            for k, v in saved["model"].items()
            if k.startswith("glyph_encoder.")
        },
        strict=True,
    )
    d.load_state_dict(
        {
            k[len("glyph_decoder.") :]: v
            for k, v in saved["model"].items()
            if k.startswith("glyph_decoder.")
        },
        strict=True,
    )
    model = GlyphCodec(args.arm, e, d).cuda()
    del saved, e, d
    ds = GlyphSequenceDataset(cfg["data"], "train", 256)
    freq = np.bincount(ds.tokens.astype(np.int64), minlength=len(ds.glyph_bank))
    eligible = [i for i in np.flatnonzero(freq) if i not in ds.control_ids.values()]
    helpers = runpy.run_path("scripts/run_dual_ablation.py")
    train_ids, heldout_ids = helpers["glyph_partition"](
        ds.glyph_bank, eligible, cfg["seed"], cfg["validation_fraction"]
    )
    # Reserve part of the old reconstruction-held-out set for a final audit.
    # Neither subset receives gradients; the audit subset is not used for early stopping.
    val_ids, audit_ids = heldout_ids[::2], heldout_ids[1::2]
    small_ids = train_ids[: cfg["small_glyphs"]]
    metadata = dict(
        config=cfg,
        base_config=base,
        arm=args.arm,
        interface=model.interface_version,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        initial_checkpoint_sha256=cfg["initial_sha256"],
        data_manifest_sha256=sha256(Path(cfg["data"]) / "manifest.json"),
        data_sha256=readiness["model_consumed_sha256"],
        data_verification_sha256=readiness["verification_sha256"],
        gpu=torch.cuda.get_device_name(),
        precision="fp16",
        torch_version=torch.__version__,
        train_ids=train_ids,
        validation_ids=val_ids,
        audit_ids=audit_ids,
        small_ids=small_ids,
        scope="glyph codec only; reconstruction-held-out characters may occur in the original LM",
    )
    write_json(output / "metadata.json", metadata)
    bank = ds.glyph_bank.cuda()
    rng = np.random.default_rng(cfg["seed"])
    start = time.monotonic()
    compute = 0.0
    features = None
    if args.arm == "fixed":
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            features = torch.cat([model.encoder(x) for x in bank.split(256)]).detach()
    frozen_before = (
        {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
        if args.arm == "fixed"
        else None
    )

    def predict(ids):
        if features is not None:
            return model.decode(model.adapter(features[ids]).reshape(-1, 4, 256))
        return model(bank[ids])

    def status(phase, state="running", **extra):
        write_json(
            logs / "status.json",
            dict(
                status=state,
                phase=phase,
                pid=os.getpid(),
                arm=args.arm,
                time=datetime.now(UTC).isoformat(),
                compute_seconds=compute,
                wall_seconds=time.monotonic() - start,
                **extra,
            ),
        )

    @torch.inference_mode()
    def evaluate(ids):
        model.eval()
        total = exact = errors = tp = fp = fn = nll = 0
        for offset in range(0, len(ids), 256):
            indices = ids[offset : offset + 256]
            true = bank[indices]
            with torch.autocast("cuda", dtype=torch.float16):
                logits = predict(indices)
            pred = logits > 0
            y = true.bool()
            difference = (pred != y).flatten(1).sum(1)
            total += len(indices)
            exact += int((difference == 0).sum())
            errors += int(difference.sum())
            tp += int((pred & y).sum())
            fp += int((pred & ~y).sum())
            fn += int((~pred & y).sum())
            nll += float(
                F.binary_cross_entropy_with_logits(logits.float(), true.float(), reduction="sum")
            )
        return dict(
            glyphs=total,
            nll=nll / (total * 1024),
            exact=exact / total,
            exact_count=exact,
            hamming=errors / total,
            f1=2 * tp / max(1, 2 * tp + fp + fn),
        )

    params = [p for p in model.parameters() if p.requires_grad]
    phase_results = {}
    overflow_streak = 0
    try:
        for phase, ids, max_steps in [
            ("small", small_ids, cfg["small_steps"]),
            ("full", train_ids, cfg["full_steps"]),
        ]:
            optimizer = torch.optim.AdamW(
                params, lr=cfg["learning_rate"], betas=(0.9, 0.95), weight_decay=0
            )
            scaler = torch.amp.GradScaler("cuda")
            best = float("inf")
            stale = 0
            small_pass = False
            for step in range(max_steps + 1):
                if step % 500 == 0 or step == max_steps:
                    train = evaluate(ids)
                    val = evaluate(val_ids)
                    row = dict(
                        phase=phase, step=step, train=train, validation=val, compute_seconds=compute
                    )
                    with (logs / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
                    score = train["nll"] if phase == "small" else val["nll"]
                    if score < best:
                        best = score
                        stale = 0
                        path = checkpoints / (phase + "_best.pt")
                        pending = path.with_suffix(".tmp")
                        torch.save(
                            dict(
                                model=model.state_dict(),
                                metadata=metadata,
                                phase=phase,
                                step=step,
                                optimizer=optimizer.state_dict(),
                                scaler=scaler.state_dict(),
                                numpy_rng=rng.bit_generator.state,
                                torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state_all(),
                            ),
                            pending,
                        )
                        pending.replace(path)
                    else:
                        stale += 1
                    phase_results[phase] = row
                    status(phase, step=step, maximum_steps=max_steps, metrics=row)
                    if phase == "small" and train["exact"] >= cfg["small_exact_gate"]:
                        small_pass = True
                        break
                    if phase == "full" and stale >= cfg["full_patience_evaluations"]:
                        break
                if step == max_steps:
                    break
                model.train()
                optimizer.zero_grad(set_to_none=True)
                ratio = cfg["minimum_learning_rate_ratio"]
                lr = cfg["learning_rate"] * (
                    ratio + (1 - ratio) * (1 + math.cos(math.pi * step / max_steps)) / 2
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                selected = np.asarray(ids)[rng.integers(len(ids), size=cfg["batch_size"])].tolist()
                torch.cuda.synchronize()
                tick = time.monotonic()
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = predict(selected)
                    loss = F.binary_cross_entropy_with_logits(
                        logits.float(), bank[selected].float()
                    )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite loss")
                scaler.scale(loss).backward()
                result = complete_optimizer_step(optimizer, scaler, params, 1)
                overflow_streak = 0 if result["succeeded"] else overflow_streak + 1
                if overflow_streak >= 20:
                    raise FloatingPointError("Repeated AMP overflow")
                torch.cuda.synchronize()
                compute += time.monotonic() - tick
                if step % 100 == 0:
                    status(phase, step=step + 1, maximum_steps=max_steps)
            if phase == "small" and not small_pass:
                receipt = dict(
                    status="complete",
                    gate_passed=False,
                    reason="small_memorization_gate_failed",
                    phases=phase_results,
                )
                write_json(output / "complete.json", receipt)
                status("small_gate_failed", "complete")
                return
        checkpoint = checkpoints / "full_best.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        final_train = evaluate(train_ids)
        final_val = evaluate(val_ids)
        audit = evaluate(audit_ids)
        if frozen_before is not None:
            assert all(
                torch.equal(v.detach().cpu(), frozen_before[k])
                for k, v in model.encoder.state_dict().items()
            )
        passed = (
            not args.smoke
            and final_train["exact"] >= cfg["full_train_exact_gate"]
            and final_val["exact"] >= cfg["heldout_exact_gate"]
            and final_val["hamming"] <= cfg["heldout_hamming_gate"]
            and audit["exact"] >= cfg["heldout_exact_gate"]
            and audit["hamming"] <= cfg["heldout_hamming_gate"]
        )
        # Inference must traverse the saved permanent interface, not the training cache.
        model.eval()
        indices = audit_ids[:16]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            raw = model(bank[indices])
            cached = predict(indices)
        assert torch.equal(raw > 0, cached > 0), "Training/inference interface disagreement"
        image = Image.new("L", (32 * 16, 64), 255)
        for i, (real, pred) in enumerate(
            zip(bank[indices].cpu().numpy(), (raw > 0).cpu().numpy(), strict=True)
        ):
            image.paste(
                Image.fromarray((255 * (1 - real.reshape(32, 32))).astype("uint8")), (i * 32, 0)
            )
            image.paste(
                Image.fromarray((255 * (1 - pred.reshape(32, 32).astype("uint8"))).astype("uint8")),
                (i * 32, 32),
            )
        image.resize((1024, 128), Image.Resampling.NEAREST).save(output / "audit_examples.png")
        receipt = dict(
            status="complete",
            gate_passed=passed,
            language_training_ready=passed,
            train=final_train,
            validation=final_val,
            audit=audit,
            compute_seconds=compute,
            selected_step=saved["step"],
            checkpoint_sha256=sha256(checkpoint),
            phases=phase_results,
            frozen_encoder_unchanged=frozen_before is not None,
            interface_roundtrip_passed=True,
        )
        write_json(output / "complete.json", receipt)
        status("finished", "complete", gate_passed=passed)
    except BaseException as error:
        status("failed", "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
