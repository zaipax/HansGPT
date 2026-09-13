import torch
import pytest

from hansgpt_research.cvae_distributed_data import RankBatches, select_global_positions


def test_rank_shards_cover_order_once_with_padded_tail():
    order = [7, 1, 8, 3, 2, 9, 4, 6, 0, 5, 10]
    shards = [list(RankBatches(order, 2, rank, 3)) for rank in range(3)]
    rebuilt = [x for step in range(2) for rank in range(3) for x in shards[rank][step]]
    assert rebuilt == order + [-1]


@pytest.mark.parametrize('boundary', [1, 3, 4, 6, 8])
def test_checkpoint_boundary_preserves_all_targets_without_overlap(boundary):
    masks = [torch.tensor([1, 0, 1, 1]), torch.tensor([0, 0, 0, 0]),
             torch.tensor([1, 1, 0, 1]), torch.tensor([0, 1, 1, 0])]
    counts = [int(m.sum()) for m in masks]
    first = [select_global_positions(m, counts, r, 0, boundary) for r,m in enumerate(masks)]
    rest = [select_global_positions(m, counts, r, boundary, 8-boundary) for r,m in enumerate(masks)]
    assert sum(int(m.sum()) for m in first) == boundary
    for m,a,b in zip(masks, first, rest):
        assert not (a & b).any()
        assert torch.equal(a | b, m.bool())


def test_invalid_budget_rejected():
    with pytest.raises(ValueError):
        select_global_positions(torch.ones(2), [2], 0, 1, 2)
