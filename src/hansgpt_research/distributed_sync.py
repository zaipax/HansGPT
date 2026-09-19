"""Synchronous and pipelined FP32 gradient reduction for multi-GPU runners."""

from __future__ import annotations

import torch
import torch.distributed as dist

DEFAULT_GRADIENT_REDUCE_BUCKET_MIB = 256


def gradient_bucket_elements(bucket_mib: int) -> int:
    """Return the number of FP32 elements in a validated bucket size."""
    if isinstance(bucket_mib, bool) or not isinstance(bucket_mib, int) or bucket_mib < 1:
        raise ValueError("gradient reduce bucket size must be a positive integer MiB")
    return bucket_mib * 1024 * 1024 // 4


def allocate_gradient_reduce_buffer(
    device: torch.device, bucket_mib: int, *, double_buffered: bool = False
) -> torch.Tensor:
    """Allocate the reusable FP32 scratch bucket(s) used by ``sync_gradients``."""
    elements = gradient_bucket_elements(bucket_mib)
    shape = (2, elements) if double_buffered else (elements,)
    return torch.empty(shape, dtype=torch.float32, device=device)


def sync_gradients(parameters, buffer: torch.Tensor) -> None:
    """SUM globally weighted gradients with bounded FP32 scratch space.

    Supports standard single buffer (1D) or double-buffered pipelined reduction (2D, shape [2, N]).
    """
    if buffer.dtype != torch.float32 or not buffer.numel():
        raise ValueError("Gradient reduction buffer must be a nonempty FP32 tensor")
    if buffer.ndim == 1:
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
    elif buffer.ndim == 2 and buffer.shape[0] == 2:
        capacity = buffer.shape[1]
        buf_idx = 0
        views, used = [], 0
        in_flight = None

        def flush_inflight() -> None:
            nonlocal in_flight
            if in_flight is None:
                return
            old_idx, _, old_views, handle = in_flight
            if handle is not None:
                handle.wait()
            for grad, start, size in old_views:
                grad.copy_(buffer[old_idx, start : start + size])
            in_flight = None

        def launch_current() -> None:
            nonlocal buf_idx, views, used, in_flight
            if not views:
                return
            flush_inflight()
            handle = dist.all_reduce(buffer[buf_idx, :used], async_op=True)
            in_flight = (buf_idx, used, views, handle)
            buf_idx = 1 - buf_idx
            views, used = [], 0

        for parameter in parameters:
            if parameter.grad is None:
                raise RuntimeError("Missing gradient: cannot silently change collective order")
            grad = parameter.grad.view(-1)
            for start in range(0, grad.numel(), capacity):
                piece = grad[start : start + capacity]
                if used + piece.numel() > capacity:
                    launch_current()
                buffer[buf_idx, used : used + piece.numel()].copy_(piece)
                views.append((piece, used, piece.numel()))
                used += piece.numel()

        launch_current()
        flush_inflight()
    else:
        raise ValueError("Gradient reduction buffer must have shape (N,) or (2, N)")
