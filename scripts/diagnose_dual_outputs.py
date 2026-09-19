"""Controlled diagnostics of a completed dual decoder; never save modified weights."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hansgpt_research.glyph_lm import GlyphSequenceDataset
from hansgpt_research.train_attention_glyph_lm import model_from_config


def sample(dataset, seed, count=256):
    eligible = np.flatnonzero(np.diff(dataset.offsets) >= 19)
    chosen = np.random.default_rng(seed).choice(eligible, count, replace=False)
    ids = np.stack([dataset.tokens[int(dataset.offsets[i]):int(dataset.offsets[i])+18]
                    for i in chosen]).astype(np.int64)
    return dataset.glyph_bank[ids[:, :17]].cuda(), dataset.glyph_bank[ids[:, 17]].cuda()


def metrics(logits, targets, gallery):
    p = logits.float().sigmoid().flatten(1)
    y = targets.float().flatten(1)
    report = {"nll": float(F.binary_cross_entropy_with_logits(logits.float(), targets.float())),
              "mean_probability": float(p.mean()), "target_foreground": float(y.mean()),
              "uncertain_pixel_fraction": float(((p > .1) & (p < .9)).float().mean())}
    sweep = {}
    for threshold in [.15, .2, .25, .3, .4, .5, .6]:
        b = (p >= threshold).float()
        distance = b.sum(1, keepdim=True) + gallery.sum(1)[None] - 2*b @ gallery.T
        tp = (b*y).sum()
        sweep[str(threshold)] = {
            "foreground": float(b.mean()), "f1": float(2*tp/(b.sum()+y.sum())),
            "exact_target": float((b == y).all(1).float().mean()),
            "exact_legal_glyph": float((distance.min(1).values == 0).float().mean()),
            "nearest_hamming": float(distance.min(1).values.mean()),
        }
    # Diagnostic only: best complete glyph under the model's Bernoulli score.
    # This is not used for autoregressive feedback or represented as raw output.
    scores = logits.float().flatten(1) @ gallery.T
    selected = gallery[scores.argmax(1)]
    report["gallery_constrained_exact_target"] = float((selected == y).all(1).float().mean())
    report["thresholds"] = sweep
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overfit-steps", type=int, default=300)
    args = parser.parse_args()
    torch.manual_seed(519)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = saved["metadata"]["config"]
    model = model_from_config(config).cuda().eval()
    model.load_state_dict(saved["model"], strict=True)
    del saved
    ds = {s: GlyphSequenceDataset("data/processed/modelscope_zhwiki_full_v1", s, 256)
          for s in ["train", "test"]}
    gallery = ds['train'].glyph_bank.float().cuda().flatten(1)
    samples = {s: sample(d, 519) for s, d in ds.items()}
    report = {"checkpoint": str(args.checkpoint), "protocol":
              "256 seeded paragraphs per split, 17-grid true prefix, one held-out next grid; FP32"}

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2)+'\n')

    def predict(x):
        return model.distribution(model.forward_hidden(x)[:, -1]).pixel_logits.reshape(-1,1,32,32)

    with torch.inference_mode():
        for split, (x, y) in samples.items():
            logits = torch.cat([predict(b) for b in x.split(16)])
            report[split] = metrics(logits, y, gallery)
            if split == 'test':
                wrong = torch.cat([predict(b) for b in x.roll(1, 0).split(16)])
                report['wrong_context_nll'] = float(F.binary_cross_entropy_with_logits(wrong,y.float()))
                np.savez_compressed(args.output/'test_probabilities.npz',
                                    probabilities=logits.sigmoid().cpu().numpy(), targets=y.cpu().numpy())
            print(split, json.dumps(report[split]), flush=True)
        x, y = samples['test']; x=x[:4]
        full = predict(x)
        cache = None
        for i in range(x.shape[1]):
            hidden, cache = model.forward_hidden(x[:, i:i+1], past_key_values=cache,
                                                  use_cache=True, return_cache=True)
        cached = model.distribution(hidden[:, -1]).pixel_logits.reshape_as(full)
        with torch.autocast('cuda',dtype=torch.float16):
            half = predict(x)
        report['numerics'] = {
            'cache_max_probability_difference':float((full.sigmoid()-cached.sigmoid()).abs().max()),
            'cache_binary_disagreement':float(((full>0)!=(cached>0)).float().mean()),
            'fp16_max_probability_difference':float((full.sigmoid()-half.float().sigmoid()).abs().max()),
            'fp16_binary_disagreement':float(((full>0)!=(half>0)).float().mean()),
        }
        frequencies=np.bincount(ds['train'].tokens.astype(np.int64),minlength=len(gallery))
        frequencies[ds['train'].control_ids['BOS']]=0
        frequencies[ds['train'].control_ids['PAD']]=0
        weights=torch.tensor(frequencies,device='cuda',dtype=torch.float32)
        prior=(weights @ gallery / weights.sum()).clamp(1e-5,1-1e-5).reshape(1,1,32,32)
        report['unconditional_training_pixel_prior_nll']=float(F.binary_cross_entropy(
            prior.expand_as(samples['test'][1]),samples['test'][1].float()))
    save()
    print('NUMERICS',report['numerics'],flush=True)

    # Full-path memorization control: 32 fixed contexts, one unambiguous target each.
    # Uses a diagnostic copy in memory; never writes weights or training state.
    x,y=(v[:32] for v in samples['train'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=0)
    scaler=torch.amp.GradScaler('cuda')
    curve=[]
    for step in range(args.overfit_steps+1):
        model.eval()
        if step % 25 == 0 or step == args.overfit_steps:
            with torch.inference_mode():
                logits=predict(x)
                row={'step':step, **metrics(logits,y,gallery)}
            curve.append(row)
            print('MEMORIZATION',step,row['nll'],row['thresholds']['0.5'],flush=True)
        if step==args.overfit_steps:break
        model.train();optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            loss=F.binary_cross_entropy_with_logits(predict(x).float(),y.float())
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        scaler.step(optimizer);scaler.update()
    report['memorization_control']=curve
    save()


if __name__=='__main__':
    main()
