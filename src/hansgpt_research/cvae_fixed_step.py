"""Fixed-shape CVAE backward, with pixel-derived unique glyphs prepared on CPU."""

import numpy as np
import torch
import torch.nn.functional as F

from hansgpt_research.cvae_training_optimization import make_head_kernel


def prepare_pixels(batch, bucket=None, *, allow_trailing_padding=False):
    x = batch["glyphs"].numpy()
    y = batch["targets"].numpy()
    if not np.isin(x, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Only binary inputs are supported")
    attention = batch["attention_mask"].bool()
    if not bool(attention.all()):
        if not allow_trailing_padding:
            raise ValueError("This probe requires full input windows; loss masks remain separate")
        if bool((~attention[:, :-1] & attention[:, 1:]).any()):
            raise ValueError("Only trailing padding is safe with omitted causal attention masks")
        if bool((batch["loss_mask"].bool() & ~attention).any()):
            raise ValueError("Padding cannot contribute training loss")
    joined = np.concatenate((x.reshape(-1, 1024), y.reshape(-1, 1024)))
    packed = np.packbits(joined, axis=1)
    unique, inverse = np.unique(packed, axis=0, return_inverse=True)
    n = len(unique)
    bucket = n if bucket is None else bucket
    if bucket < n:
        raise ValueError("Unique-glyph bucket overflow; never truncate")
    tiles = np.zeros((bucket, 1, 32, 32), dtype=np.uint8)
    tiles[:n] = np.unpackbits(unique, axis=1).reshape(-1, 1, 32, 32)
    count = x.shape[0] * x.shape[1]
    return dict(
        tiles=torch.from_numpy(tiles),
        x_index=torch.from_numpy(inverse[:count]),
        y_index=torch.from_numpy(inverse[count:]),
        targets=batch["targets"].reshape(-1, 1, 32, 32),
        mask=batch["loss_mask"].reshape(-1),
        unique=n,
    )


class FixedBackward:
    """Shared encoder activations receive both context and posterior gradients.

    Captures forward/backward only. AMP unscale, finite-gradient checks, clipping
    and optimizer updates remain outside capture, with a device scale buffer.
    """

    def __init__(self, model, batch_size, context, chunk, *, compiled=True):
        if batch_size * context % chunk:
            raise ValueError("Full positional shape must divide into chunks")
        self.model, self.batch, self.context, self.chunk = model, batch_size, context, chunk
        self.kernel = make_head_kernel(model, compiled=compiled)

    def __call__(self, tiles, x_index, y_index, targets, mask, noise, scale, beta):
        model = self.model
        with torch.autocast(tiles.device.type, dtype=torch.float16, enabled=tiles.is_cuda):
            encoded = model.encoder(tiles)
            features = encoded.detach().requires_grad_(True)
            x = features[x_index].flatten(1)
            inputs = model.input_norm(model.input_projection(x)).reshape(
                self.batch, self.context, -1
            )
            outer = model.backbone(
                inputs_embeds=inputs, attention_mask=None, use_cache=False, return_dict=True
            ).last_hidden_state
            hidden = model.semantic(outer, None, None, False)[0].reshape(
                -1, model.config.hidden_size
            )
        leaf = hidden.detach().requires_grad_(True)
        sums = torch.zeros(2, device=tiles.device, dtype=torch.float64)
        denominator = mask.sum().clamp_min(1) * 1024
        for start in range(0, len(targets), self.chunk):
            stop = start + self.chunk
            with torch.autocast(tiles.device.type, dtype=torch.float16, enabled=tiles.is_cuda):
                rec, kl = self.kernel(
                    leaf[start:stop],
                    targets[start:stop],
                    features[y_index[start:stop]],
                    noise[start:stop],
                    mask[start:stop],
                )
                loss = (rec + beta * kl) / denominator
            (loss * scale).backward()
            sums.add_(torch.stack((rec.detach(), kl.detach())).double())
        hidden.backward(leaf.grad)
        encoded.backward(features.grad)
        return sums


def install_xformers():
    """Route SDPA calls in both HF Llama and torch Transformer modules through CUTLASS.

    Process-local replacement; no fallback. Current benchmark has no dropout or
    padding attention mask. Boolean/additive masks are supported explicitly.
    """
    import xformers.ops as xo
    from xformers.ops.fmha import cutlass

    original = F.scaled_dot_product_attention

    def backward_operator(*args, **kwargs):
        # xFormers 0.0.32 supplies empty CPU RNG placeholders when dropout is
        # disabled. Torch 2.8 AOTAutograd cannot infer a common device from those
        # and CUDA Q/K/V. These arguments are unused at p=0; keep them on CUDA.
        if kwargs.get("dropout_p", 0.0) == 0.0:
            placeholder = args[1].new_empty((0,), dtype=torch.int64)
            kwargs["philox_seed"] = placeholder
            kwargs["philox_offset"] = placeholder
        # ``out`` is an input here, not an out= destination. Pass it positionally
        # so Dynamo does not misinterpret a multi-output operator's keyword.
        return torch.ops.aten._efficient_attention_backward.default(
            *args,
            kwargs["bias"],
            kwargs["out"],
            kwargs["cu_seqlens_q"],
            kwargs["cu_seqlens_k"],
            kwargs["max_seqlen_q"],
            kwargs["max_seqlen_k"],
            kwargs["logsumexp"],
            kwargs["dropout_p"],
            kwargs["philox_seed"],
            kwargs["philox_offset"],
            kwargs["custom_mask_type"],
            kwargs["bias_requires_grad"],
            scale=kwargs.get("scale"),
            num_splits_key=kwargs.get("num_splits_key"),
            window_size=kwargs.get("window_size"),
        )

    class CompileCompatibleBackward(cutlass.BwOp):
        OPERATOR = staticmethod(backward_operator)

    def attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, *, scale=None, enable_gqa=False
    ):
        if q.ndim != 4 or dropout_p != 0:
            raise ValueError("Probe supports 4D attention with zero dropout")
        if q.shape[1] != k.shape[1]:
            if not enable_gqa or q.shape[1] % k.shape[1]:
                raise ValueError("Invalid GQA layout")
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        bias = None
        if is_causal:
            if attn_mask is not None:
                raise ValueError("Do not combine causal and explicit masks")
            bias = xo.LowerTriangularMask()
        elif attn_mask is not None:
            bias = attn_mask
            if bias.dtype == torch.bool:
                bias = torch.zeros_like(bias, dtype=q.dtype).masked_fill(~bias, float("-inf"))
            bias = bias.to(dtype=q.dtype).expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])
        out = xo.memory_efficient_attention(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            attn_bias=bias,
            p=0.0,
            scale=scale,
            op=(cutlass.FwOp, CompileCompatibleBackward),
        )
        return out.transpose(1, 2)

    F.scaled_dot_product_attention = attention
    return original
