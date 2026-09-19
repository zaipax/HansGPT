"""Validate a byte-training continuation without changing its data or LR trajectory."""

import copy
import random

import numpy as np
import torch


def validate_resume(saved, config, world):
    if saved.get("diagnostic_only"):
        raise ValueError("Diagnostic failure snapshots are not validated training checkpoints")
    previous = copy.deepcopy(saved["metadata"]["config"])
    current = copy.deepcopy(config)
    for value in (previous, current):
        value.pop("experiment", None)
        for key in (
            "target_tokens",
            "checkpoint_every_positions",
            "keep_recent_checkpoints",
            "retain_every_positions",
            "validation_samples",
            "nonfinite_fp32_retry",
        ):
            value["training"].pop(key, None)
    if previous != current:
        raise ValueError("Resume must preserve architecture, data, optimizer and LR schedule")
    if len(saved["rank_rng"]) != world:
        raise ValueError("Resume world size changed")
    progress = saved["progress"]
    for key in ("epoch", "cursor", "batch_skip", "steps", "all_targets", "han", "overflows"):
        if type(progress.get(key)) is not int or progress[key] < 0:
            raise ValueError(f"Invalid saved progress: {key}")
    if not 0 < progress["all_targets"] < config["training"]["target_tokens"]:
        raise ValueError("Resume requires a larger cumulative stopping budget")
    return dict(progress)


def restore_rank_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"], device)
