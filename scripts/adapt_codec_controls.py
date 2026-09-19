"""Teach reserved control bitmaps while freezing the successful glyph encoder."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hansgpt_research.glyph_codec import load_glyph_codec
from hansgpt_research.glyph_lm import GlyphSequenceDataset
from hansgpt_research.train_glyph_lm import sha256, write_json
from hansgpt_research.train_structured_glyph_lm import complete_optimizer_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "5":
        raise RuntimeError("Use GPU5")
    name = "glyph_codec_controls_v1" + ("_smoke" if args.smoke else "")
    output = Path("artifacts/reports") / name
    checkpoints = Path("artifacts/checkpoints") / name
    output.mkdir(parents=True, exist_ok=False)
    checkpoints.mkdir(parents=True, exist_ok=False)
    source = Path("artifacts/checkpoints/glyph_codec_spatial16_v1_spatial/full_best.pt")
    receipt = json.loads(
        Path("artifacts/reports/glyph_codec_spatial16_v1_spatial/complete.json").read_text()
    )
    assert receipt["gate_passed"]
    model = load_glyph_codec(source, "cuda", expected_sha256=receipt["checkpoint_sha256"])
    original = torch.load(source, map_location="cpu", weights_only=False)
    metadata = original["metadata"]
    del original
    metadata = {
        **metadata,
        "control_adaptation": {
            "parent_sha256": receipt["checkpoint_sha256"],
            "learning_rate": 3e-5,
            "batch_size": 256,
            "control_examples_per_batch": 32,
            "encoder_frozen": True,
        },
    }
    ds = GlyphSequenceDataset(metadata["config"]["data"], "train", 256)
    bank = ds.glyph_bank.cuda()
    train_ids = metadata["train_ids"]
    val_ids = metadata["validation_ids"]
    audit_ids = metadata["audit_ids"]
    controls = list(ds.control_ids.values())
    torch.manual_seed(519)
    torch.set_num_threads(4)
    rng = np.random.default_rng(519)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        features = torch.cat([model.encode(x) for x in bank.split(256)]).detach()
    model.decoder.requires_grad_(True)
    params = list(model.decoder.parameters())
    opt = torch.optim.AdamW(params, lr=3e-5, betas=(0.9, 0.95), weight_decay=0)
    scaler = torch.amp.GradScaler("cuda")
    started = time.monotonic()

    @torch.inference_mode()
    def evaluate(ids):
        model.eval()
        exact = errors = count = 0
        for i in range(0, len(ids), 256):
            batch = ids[i : i + 256]
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model.decode(features[batch]) > 0
            difference = (pred != bank[batch].bool()).flatten(1).sum(1)
            count += len(batch)
            exact += int((difference == 0).sum())
            errors += int(difference.sum())
        return dict(exact=exact / count, hamming=errors / count, glyphs=count)

    baseline = evaluate(val_ids)
    rows = []
    passed = False
    overflow = 0
    for step in range((2 if args.smoke else args.steps) + 1):
        if step % 100 == 0 or (args.smoke and step == 2):
            row = dict(step=step, validation=evaluate(val_ids), controls=evaluate(controls))
            rows.append(row)
            write_json(
                output / "progress.json", dict(rows=rows, seconds=time.monotonic() - started)
            )
            print(json.dumps(row), flush=True)
            if (
                not args.smoke
                and row["controls"]["exact"] == 1
                and row["validation"]["exact"] >= max(0.95, baseline["exact"] - 0.01)
            ):
                passed = True
                break
        if step == (2 if args.smoke else args.steps):
            break
        model.train()
        opt.zero_grad(set_to_none=True)
        ids = np.asarray(train_ids)[rng.integers(len(train_ids), size=224)].tolist() + controls * 8
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model.decode(features[ids])
            loss = F.binary_cross_entropy_with_logits(logits.float(), bank[ids].float())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite control loss")
        scaler.scale(loss).backward()
        r = complete_optimizer_step(opt, scaler, params, 1)
        overflow = 0 if r["succeeded"] else overflow + 1
        if overflow >= 20:
            raise FloatingPointError("Repeated control AMP overflow")
    audit = evaluate(audit_ids)
    train = evaluate(train_ids)
    passed = (
        passed and audit["exact"] >= 0.95 and audit["hamming"] <= 0.1 and train["exact"] >= 0.95
    )
    # The permanent encoder and inference path must match cached training features.
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        direct = model(bank[controls]) > 0
        cached = model.decode(features[controls]) > 0
    assert torch.equal(direct, cached)
    path = checkpoints / "codec.pt"
    torch.save(dict(model=model.state_dict(), metadata=metadata, step=step), path)
    write_json(
        output / "complete.json",
        dict(
            status="complete",
            gate_passed=passed,
            train=train,
            validation=rows[-1]["validation"],
            audit=audit,
            controls=evaluate(controls),
            checkpoint_sha256=sha256(path),
            parent_sha256=receipt["checkpoint_sha256"],
            step=step,
        ),
    )


if __name__ == "__main__":
    main()
