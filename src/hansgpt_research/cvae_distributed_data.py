"""Rank sharding and exact global valid-position budgets for CVAE training."""

import itertools

import torch
from torch.utils.data import Dataset


class PaddedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if index >= 0:
            return self.dataset[index]
        return {k: torch.zeros_like(v) for k, v in self.dataset[0].items()}


class RankBatches:
    def __init__(self, sampler, batch, rank, world):
        self.sampler, self.batch, self.rank, self.world = sampler, batch, rank, world

    def __iter__(self):
        source = iter(self.sampler)
        size = self.batch * self.world
        while group := list(itertools.islice(source, size)):
            group.extend([-1] * (size - len(group)))
            yield group[self.rank * self.batch:(self.rank + 1) * self.batch]


def select_global_positions(mask, counts, rank, skip, take):
    """Select [skip, skip+take) in rank-major valid-target order, without holes."""
    if skip < 0 or take < 0 or skip + take > sum(counts):
        raise ValueError('Invalid global position interval')
    valid = mask.bool().flatten().nonzero().flatten()
    if len(valid) != counts[rank]:
        raise ValueError('Local/global target counts disagree')
    base = sum(counts[:rank])
    lower = max(0, min(len(valid), skip - base))
    upper = max(0, min(len(valid), skip + take - base))
    selected = torch.zeros_like(mask, dtype=torch.bool).flatten()
    selected[valid[lower:upper]] = True
    return selected.reshape_as(mask)
