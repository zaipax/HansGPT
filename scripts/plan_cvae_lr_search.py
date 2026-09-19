"""Verify one identical ordered training prefix for all eight LR candidates."""

import copy
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import regex
import torch

from hansgpt_research.packed_glyph_data import PackedGlyphSequenceDataset
from hansgpt_research.train_glyph_lm import sequence_lengths, write_json


def main():
    torch.set_num_threads(4)
    configs = [
        json.loads(p.read_text())
        for p in sorted(Path("configs/experiments/lr_search_v1").glob("gpu*.json"))
    ]
    assert len(configs) == 8

    def normalize(config):
        c = copy.deepcopy(config)
        c.pop("gpu")
        c.pop("experiment")
        c["training"].pop("learning_rate")
        return c

    assert all(normalize(c) == normalize(configs[0]) for c in configs)
    c = configs[0]
    tc = c["training"]
    ds = PackedGlyphSequenceDataset(c["data"], "train", tc["sequence_length"])
    order = np.asarray(
        list(
            runpy.run_path("scripts/train_conditional_vae.py")["epoch_sampler"](
                sequence_lengths(ds), tc, dict(epoch=0, cursor=0)
            )
        ),
        dtype="<i8",
    )
    han_table = np.zeros(len(ds.glyph_bank), dtype=bool)
    for char, index in ds.inventory["characters"].items():
        if regex.fullmatch(r"[\p{Unified_Ideograph}〇]", char):
            han_table[int(index)] = True
    remaining = tc["target_han"]
    total = 0
    max_unique = 0
    max_cursor = 0
    counts = []
    digest = hashlib.sha256()
    for cursor in range(0, len(order), tc["batch_size"]):
        blocks = [
            np.asarray(ds.tokens[int(i) * 1024 : int(i) * 1024 + 1025])
            for i in order[cursor : cursor + tc["batch_size"]]
        ]
        unique = len(np.unique(np.concatenate(blocks)))
        counts.append(unique)
        if unique > max_unique:
            max_unique, max_cursor = unique, cursor
        targets = np.concatenate([b[1:] for b in blocks])
        targets = targets[targets != ds.control_ids["BOS"]]
        flags = han_table[targets]
        n = int(flags.sum())
        if n >= remaining:
            stop = int(np.flatnonzero(flags)[remaining - 1]) + 1
            targets = targets[:stop]
            n = remaining
        digest.update(targets.astype("<u2").tobytes())
        total += len(targets)
        remaining -= n
        if not remaining:
            break
    assert remaining == 0
    report = dict(
        status="verified",
        candidates=[
            dict(gpu=c["gpu"], peak_lr=c["training"]["learning_rate"], experiment=c["experiment"])
            for c in configs
        ],
        target_han=tc["target_han"],
        effective_targets=total,
        ordered_target_sha256=digest.hexdigest(),
        epoch_order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),
        batches=len(counts),
        max_unique_asset_upper_bound=max_unique,
        max_unique_batch_cursor=max_cursor,
        unique_asset_p95=float(np.quantile(counts, 0.95)),
        scope="Asset counts bound pixel uniqueness; worst-batch smoke uses the recorded cursor",
    )
    out = Path("artifacts/reports/cvae_lr_search_v1")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "plan.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
