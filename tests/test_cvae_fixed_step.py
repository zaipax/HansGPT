import copy
import os
import runpy

import pytest
import torch

from hansgpt_research.conditional_glyph_vae import ConditionalGlyphVAE
from hansgpt_research.cvae_fixed_step import FixedBackward, install_xformers, prepare_pixels
from hansgpt_research.cvae_training_optimization import head_sums


def test_shared_encoder_loss_mask_and_gradients():
    cfg = runpy.run_path("tests/test_conditional_glyph_vae.py")["tiny_config"]()
    torch.manual_seed(11)
    a = ConditionalGlyphVAE(cfg).train()
    b = copy.deepcopy(a)
    x = torch.randint(2, (2, 4, 1, 32, 32), dtype=torch.uint8)
    y = x.roll(1, 1)
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[0, 2] = False
    batch = dict(glyphs=x, targets=y, attention_mask=torch.ones_like(mask), loss_mask=mask)
    p = prepare_pixels(batch)
    noise = torch.randn(8, 8)
    targets = y.flatten(0, 1)
    h = a.forward_hidden(x).flatten(0, 1)
    rec, kl = head_sums(a, h, targets, a.glyph_features(targets), noise, mask.flatten())
    ((rec + 0.7 * kl) / (mask.sum() * 1024)).backward()
    result = FixedBackward(b, 2, 4, 2, compiled=False)(
        **{k: v for k, v in p.items() if k != "unique"},
        noise=noise,
        scale=torch.tensor(1.0),
        beta=torch.tensor(0.7),
    )
    torch.testing.assert_close(result, torch.stack((rec, kl)).double(), rtol=1e-5, atol=1e-4)
    for x, y in zip(a.parameters(), b.parameters(), strict=True):
        if x.grad is None:
            assert y.grad is None
        else:
            torch.testing.assert_close(x.grad, y.grad, rtol=0.003, atol=3e-5)


@pytest.mark.skipif(
    os.environ.get("HANSGPT_TEST_XFORMERS") != "1", reason="Optional CUDA extension test"
)
def test_xformers_sdpa_outputs_and_gradients():
    import torch.nn.functional as f

    q = torch.randn(2, 4, 16, 32, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    original = f.scaled_dot_product_attention
    for causal in [False, True]:
        a = original(q, k, v, is_causal=causal)
        ga = torch.autograd.grad(a.float().square().sum(), (q, k, v))
        install_xformers()
        try:
            b = f.scaled_dot_product_attention(q, k, v, is_causal=causal)
            gb = torch.autograd.grad(b.float().square().sum(), (q, k, v))
        finally:
            f.scaled_dot_product_attention = original
        torch.testing.assert_close(a, b, rtol=0.02, atol=0.002)
        for x, y in zip(ga, gb, strict=True):
            torch.testing.assert_close(x, y, rtol=0.03, atol=0.02)


@pytest.mark.skipif(os.environ.get("HANSGPT_TEST_GRAPH") != "1", reason="Explicit GPU graph test")
def test_graph_replay_overwrites_gradients_and_accepts_new_noise():
    cfg = runpy.run_path("tests/test_conditional_glyph_vae.py")["tiny_config"]()
    torch.manual_seed(17)
    model = ConditionalGlyphVAE(cfg).cuda().train()
    x = torch.randint(2, (2, 4, 1, 32, 32), dtype=torch.uint8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    p = prepare_pixels(dict(glyphs=x, targets=x.roll(1, 1), attention_mask=mask, loss_mask=mask))
    inputs = {k: v.cuda() for k, v in p.items() if k != "unique"}
    inputs.update(
        noise=torch.randn(8, 8, device="cuda"),
        scale=torch.ones((), device="cuda"),
        beta=torch.ones((), device="cuda"),
    )
    fn = FixedBackward(model, 2, 4, 2, compiled=False)
    fn(**inputs)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            result = fn(**inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    for _ in range(2):
        inputs["noise"].normal_()
        ref = copy.deepcopy(model)
        ref.zero_grad(set_to_none=True)
        expected = FixedBackward(ref, 2, 4, 2, compiled=False)(**inputs)
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(result, expected, rtol=0.01, atol=0.01)
        for a, b in zip(model.parameters(), ref.parameters(), strict=True):
            if a.grad is None:
                assert b.grad is None
            else:
                torch.testing.assert_close(a.grad, b.grad, rtol=0.03, atol=0.003)
