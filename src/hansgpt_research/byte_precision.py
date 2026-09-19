"""Single-rank byte backward recovery boundary."""

import torch


class NonfiniteByteForward(FloatingPointError):
    """A forward tensor or loss is nonfinite before any optimizer step."""


def checked_byte_backward(backward, optimizer, data, scale, on_retry=None):
    device = data["tiles"].device
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        return backward(**data, scale=scale, checked=True)
    except NonfiniteByteForward as error:
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        if on_retry:
            on_retry(str(error))
    # Full FP32 eager recomputation, with identical data, RNG, loss scale and mask.
    # No position is consumed until the caller successfully updates the optimizer.
    try:
        return backward(**data, scale=scale, fp32=True, checked=True)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
