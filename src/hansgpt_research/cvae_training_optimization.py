"""Equivalent CVAE training kernels; checkpoints retain the original parameter names."""

import math

import torch
import torch.nn.functional as F

from hansgpt_research.conditional_glyph_vae import gaussian_kl


def head_sums(model, hidden, targets, features, noise, valid):
    """Pure fixed-shape region: no uniqueness, scalar reads or random draws."""
    pm, pl = model.prior(hidden)
    context = model.context_tokens(hidden)
    target = model.target_projection(features)
    tokens = torch.cat((model.posterior_query.expand(len(context), -1, -1), context, target), 1)
    qm, ql = model.posterior_head(model.posterior_transformer(tokens)[:, 0]).chunk(2, -1)
    qm, ql = qm.float(), ql.float().clamp(-6, 2)
    z = qm + torch.exp(0.5 * ql) * noise
    logits = model.decode(hidden, z)
    rec = (
        F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
        .flatten(1)
        .sum(1)
    )
    kl = gaussian_kl(qm, ql, pm, pl)
    return (rec * valid).sum(), (kl * valid).sum()


def make_head_kernel(model, *, compiled=False):
    def kernel(hidden, targets, features, noise, valid):
        return head_sums(model, hidden, targets, features, noise, valid)

    if compiled:
        return torch.compile(
            kernel, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
        )
    return kernel


def pad_batch(tensor, length):
    if len(tensor) == length:
        return tensor
    return torch.cat((tensor, tensor.new_zeros((length - len(tensor), *tensor.shape[1:]))))


def optimized_backward(
    model, hidden, targets, scaler, chunk_size, beta, kernel, *, autocast_enabled=True
):
    """One metrics transfer per batch, with all finite checks before optimizer updates.

    Unique glyph encoding stays eager. Tail padding affects only independent
    glyphs and has zero loss weight. Random draws are made for real targets only,
    outside compilation, matching the baseline's per-chunk RNG consumption.
    """
    leaf = hidden.detach().requires_grad_(True)
    statistics = torch.zeros(2, device=hidden.device, dtype=torch.float64)
    total = len(hidden)
    for start in range(0, total, chunk_size):
        h = leaf[start : start + chunk_size]
        y = targets[start : start + len(h)]
        n = len(h)
        with torch.autocast(hidden.device.type, dtype=torch.float16, enabled=autocast_enabled):
            features = model.glyph_features(y)
            noise = torch.randn(
                (n, model.spec["vae"]["latent_dim"]), device=hidden.device, dtype=torch.float32
            )
            valid = torch.arange(chunk_size, device=hidden.device) < n
            rec, kl = kernel(
                pad_batch(h, chunk_size),
                pad_batch(y, chunk_size),
                pad_batch(features, chunk_size),
                pad_batch(noise, chunk_size),
                valid,
            )
            loss = (rec + beta * kl) / (total * 1024)
        scaler.scale(loss).backward()
        statistics.add_(torch.stack((rec.detach(), kl.detach())).double())
    rec_sum, kl_sum = statistics.tolist()
    if not math.isfinite(rec_sum) or not math.isfinite(kl_sum):
        raise FloatingPointError("Nonfinite CVAE loss statistics; optimizer not updated")
    if leaf.grad is None:
        raise RuntimeError("Missing context gradient")
    hidden.backward(leaf.grad)
    return rec_sum, kl_sum
