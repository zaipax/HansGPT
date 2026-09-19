"""EOS-delimited causal packing without synthetic cross-document targets."""

import numpy as np
import torch

from hansgpt_research.glyph_lm import GlyphSequenceDataset


class PackedGlyphSequenceDataset(GlyphSequenceDataset):
    """Stream windows overlap one input; EOS -> BOS loss is always masked.

    Attention is ordinary causal attention across EOS (GPT-style packing), NOT
    document-isolated attention. Position indices run continuously within a pack.
    Source split order is deterministic; shuffle windows in the training sampler.
    Asset IDs are used only to retrieve bitmaps, never as learned token inputs.
    """

    def __len__(self):
        return (len(self.tokens) - 2) // self.sequence_length + 1

    def effective_lengths(self):
        lengths = np.full(len(self), self.sequence_length, dtype=np.int32)
        lengths[-1] = (len(self.tokens) - 2) % self.sequence_length + 1
        boundaries = (self.offsets[1:-1] - 1) // self.sequence_length
        lengths -= np.bincount(boundaries, minlength=len(self)).astype(np.int32)
        if int(lengths.sum()) != self.target_count:
            raise ValueError("Packed target accounting mismatch")
        return lengths

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.sequence_length
        ids = np.array(
            self.tokens[start : start + self.sequence_length + 1], copy=True, dtype=np.int64
        )
        valid = len(ids) - 1
        inputs = torch.full((self.sequence_length,), self.control_ids["PAD"], dtype=torch.long)
        targets = inputs.clone()
        inputs[:valid] = torch.from_numpy(ids[:-1])
        targets[:valid] = torch.from_numpy(ids[1:])
        attention = torch.arange(self.sequence_length) < valid
        loss = attention & (targets != self.control_ids["BOS"])
        return dict(
            glyphs=self.glyph_bank[inputs],
            targets=self.glyph_bank[targets],
            attention_mask=attention,
            loss_mask=loss,
            target_ids=targets,
        )


def packing_statistics(offsets, context=1024):
    lengths = np.diff(offsets) - 1
    total = int(offsets[-1]) - 1
    windows = (total + context - 1) // context
    targets = int(lengths.sum())
    return dict(
        context=context,
        windows=windows,
        effective_targets=targets,
        mean_effective_targets=targets / windows,
        target_utilization=targets / (windows * context),
        padding_positions=windows * context - total,
        masked_document_transitions=len(lengths) - 1,
        document_mean_targets=float(lengths.mean()),
        document_p50=float(np.quantile(lengths, 0.5)),
        document_p95=float(np.quantile(lengths, 0.95)),
        document_p99=float(np.quantile(lengths, 0.99)),
        document_max=int(lengths.max()),
        genuine_full_document_windows=int((lengths // context).sum()),
        long_document_targets=int(lengths[lengths >= context].sum()),
    )
