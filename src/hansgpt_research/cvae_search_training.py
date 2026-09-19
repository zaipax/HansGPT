"""Production data preparation and bounded storage for controlled CVAE searches."""

import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import default_collate

from hansgpt_research.cvae_fixed_step import prepare_pixels
from hansgpt_research.train_glyph_lm import save_checkpoint


@dataclass
class SharedGlyphCollator:
    batch_size: int
    context: int
    pad_id: int = 0

    def __call__(self, samples):
        batch = default_collate(samples)
        actual = len(samples)
        if batch["glyphs"].shape[1] != self.context or actual > self.batch_size:
            raise ValueError("Unexpected source batch shape")
        if actual < self.batch_size:
            for key, value in batch.items():
                fill = self.pad_id if key == "target_ids" else 0
                padded = torch.full((self.batch_size, *value.shape[1:]), fill, dtype=value.dtype)
                padded[:actual] = value
                batch[key] = padded
        # With causal attention, right-padding cannot influence earlier valid
        # outputs. Zero-weight trailing queries are retained for fixed head shapes.
        prepared = prepare_pixels(batch, allow_trailing_padding=True)
        prepared["target_ids"] = batch["target_ids"].flatten()
        prepared["source_batch_size"] = actual
        return prepared


def checkpoint_latest(path, model, optimizer, scaler, progress, metadata):
    """One full resumable file per run, with serialized atomic writes across GPUs."""
    import fcntl

    path = Path(path)
    lock_path = path.parent.parent / ".lr-search-checkpoint.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        estimate = sum(p.numel() * p.element_size() for p in model.parameters()) * 3
        if shutil.disk_usage(path.parent).free < estimate + 2**30:
            raise OSError("Insufficient space for atomic checkpoint replacement plus reserve")
        save_checkpoint(path, model, optimizer, scaler, progress, metadata)
