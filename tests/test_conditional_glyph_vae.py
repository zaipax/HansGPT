import copy
import runpy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hansgpt_research.conditional_glyph_vae import ConditionalGlyphVAE, gaussian_kl, sample_gaussian


def test_bsz_control_retains_reference_order_and_consumed_cursor():
    helper = runpy.run_path("scripts/train_conditional_vae.py")
    lengths = np.random.default_rng(5).integers(900, 1025, size=1031)
    baseline = dict(seed=20260915, batch_size=8, sortish_pool_batches=64)
    control = dict(baseline, batch_size=6, sampler_reference_batch_size=8)
    progress = dict(epoch=0, cursor=0)
    order = list(helper["epoch_sampler"](lengths, baseline, progress))
    assert order == list(helper["epoch_sampler"](lengths, control, progress))
    assert len(set(order)) == len(lengths)
    assert list(helper["epoch_sampler"](lengths, control, dict(epoch=0, cursor=18))) == order[18:]
    unpinned = dict(baseline, batch_size=6)
    assert order != list(helper["epoch_sampler"](lengths, unpinned, progress))


def tiny_config():
    return dict(
        model=dict(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=32,
        ),
        encoder=dict(width=32, layers=1, heads=4),
        decoders=dict(
            width=64,
            semantic_layers=1,
            semantic_slots=4,
            glyph_layers=1,
            part_slots=4,
            heads=4,
            intermediate_size=128,
        ),
        vae=dict(latent_dim=8),
    )


def test_kl_and_reparameterized_gradients():
    mean = torch.randn(3, 8, requires_grad=True)
    lv = torch.zeros_like(mean)
    torch.testing.assert_close(gaussian_kl(mean, lv, mean, lv), torch.zeros(3))
    assert (gaussian_kl(mean, lv, torch.zeros_like(mean), lv) >= 0).all()
    sample_gaussian(mean, lv).sum().backward()
    assert mean.grad.abs().sum() > 0


def test_target_is_posterior_only_and_generation_decodes_once_per_glyph():
    model = ConditionalGlyphVAE(tiny_config()).eval()
    x = torch.randint(2, (1, 3, 1, 32, 32), dtype=torch.uint8)
    hidden = model.forward_hidden(x)[:, -1]
    before = model.prior(hidden)
    model.posterior(hidden, x[:, 0])
    model.posterior(hidden, 1 - x[:, 0])
    after = model.prior(hidden)
    for a, b in zip(before, after, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    calls = []
    handle = model.decoder.register_forward_hook(lambda *_: calls.append(1))
    model.posterior = lambda *_: (_ for _ in ()).throw(AssertionError("Target path in inference"))
    output = model.generate(x, 3, generator=torch.Generator().manual_seed(5))
    assert output.shape == (1, 3, 1, 32, 32) and len(calls) == 3
    handle.remove()


def test_context_causality_cache_and_global_latent_gradients():
    model = ConditionalGlyphVAE(tiny_config()).eval()
    x = torch.randint(2, (1, 4, 1, 32, 32), dtype=torch.uint8)
    full = model.forward_hidden(x)
    changed = x.clone()
    changed[:, 3] = 1 - changed[:, 3]
    torch.testing.assert_close(
        full[:, :3], model.forward_hidden(changed)[:, :3], atol=1e-5, rtol=1e-5
    )
    cache = None
    for i in range(4):
        h, cache = model.forward_hidden(
            x[:, i : i + 1], past_key_values=cache, use_cache=True, return_cache=True
        )
    torch.testing.assert_close(full[:, -1], h[:, -1], atol=1e-5, rtol=1e-5)
    z = torch.randn(1, 8, requires_grad=True)
    model.decode(full[:, -1].detach(), z).square().mean().backward()
    assert z.grad.abs().sum() > 0


def test_han_budget_preserves_preceding_punctuation_and_cuts_at_last_han():
    helper = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/train_conditional_vae.py")
    )
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)
    han = torch.tensor([[0, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)
    actual = helper["trim_han_budget"](mask, han, 2)
    assert actual.tolist() == [[True, True, True, False], [True, False, False, False]]
    assert int((actual & han).sum()) == 2
    assert not helper["trim_han_budget"](mask, han, 0).any()


def test_chunked_posterior_and_context_gradients_match_joint_backward():
    torch.manual_seed(8)
    direct = ConditionalGlyphVAE(tiny_config()).eval()
    chunked = copy.deepcopy(direct)
    x = torch.randint(2, (2, 3, 1, 32, 32), dtype=torch.uint8)
    y = x[:, 1]

    def loss(model, h, target):
        pm, pl = model.prior(h)
        qm, ql = model.posterior(h, target)
        pixels = model.decode(h, qm)
        return (
            F.binary_cross_entropy_with_logits(pixels, target.float())
            + gaussian_kl(qm, ql, pm, pl).mean() / 1024
        )

    loss(direct, direct.forward_hidden(x)[:, -1], y).backward()
    hidden = chunked.forward_hidden(x)[:, -1]
    leaf = hidden.detach().requires_grad_(True)
    for i in range(2):
        (loss(chunked, leaf[i : i + 1], y[i : i + 1]) / 2).backward()
    hidden.backward(leaf.grad)
    for a, b in zip(direct.parameters(), chunked.parameters(), strict=True):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=0.03, atol=2e-5)
