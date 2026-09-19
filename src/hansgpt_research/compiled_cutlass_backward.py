"""Opt-in compiler boundary for Torch 2.8's mismatched CUTLASS backward meta schema.

The CUDA implementation is unchanged. Only fixed-length, bias-free, dropout-free
attention is supported. This is a first-order training operator.
"""

import torch
from torch import Tensor


@torch.library.custom_op("hansgpt::cutlass_backward", mutates_args=())
def cutlass_backward(
    grad: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    out: Tensor,
    logsumexp: Tensor,
    seed: Tensor,
    offset: Tensor,
    mask_type: int,
    scale: float | None,
    splits: int | None,
    window: int | None,
) -> tuple[Tensor, Tensor, Tensor]:
    dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward.default(
        grad,
        query,
        key,
        value,
        None,
        out,
        None,
        None,
        query.shape[1],
        key.shape[1],
        logsumexp,
        0.0,
        seed,
        offset,
        mask_type,
        False,
        scale=scale,
        num_splits_key=splits,
        window_size=window,
    )
    return dq, dk, dv


@cutlass_backward.register_fake
def _fake(grad, query, key, value, out, logsumexp, seed, offset, mask_type, scale, splits, window):
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)
