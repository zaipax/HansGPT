"""Synchronous FP32 gradient reduction for the custom multi-GPU runners."""

from __future__ import annotations

import torch
import torch.distributed as dist

DEFAULT_GRADIENT_REDUCE_BUCKET_MIB = 256


def gradient_bucket_elements(bucket_mib: int) -> int:
    """Return the number of FP32 elements in a validated bucket size."""
    if isinstance(bucket_mib, bool) or not isinstance(bucket_mib, int) or bucket_mib < 1:
        raise ValueError("gradient reduce bucket size must be a positive integer MiB")
    return bucket_mib * 1024 * 1024 // 4


def allocate_gradient_reduce_buffer(device: torch.device, bucket_mib: int) -> torch.Tensor:
    """Allocate the reusable FP32 scratch bucket used by ``sync_gradients``."""
    return torch.empty(gradient_bucket_elements(bucket_mib), dtype=torch.float32, device=device)


def sync_gradients(parameters, buffer: torch.Tensor) -> None:
    """SUM globally weighted gradients with bounded FP32 scratch space."""
    if buffer.dtype != torch.float32 or buffer.ndim != 1 or not buffer.numel():
        raise ValueError("Gradient reduction buffer must be a nonempty one-dimensional FP32 tensor")
    views, used = [], 0

    def flush() -> None:
        if not views:
            return
        dist.all_reduce(buffer[:used])
        for grad, start, size in views:
            grad.copy_(buffer[start : start + size])

    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("Missing gradient: cannot silently change collective order")
        grad = parameter.grad.view(-1)
        for start in range(0, grad.numel(), buffer.numel()):
            piece = grad[start : start + buffer.numel()]
            if used + piece.numel() > buffer.numel():
                flush()
                views, used = [], 0
            buffer[used : used + piece.numel()].copy_(piece)
            views.append((piece, used, piece.numel()))
            used += piece.numel()
    flush()
