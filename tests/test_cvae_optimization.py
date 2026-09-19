import copy
import os
import runpy

import pytest
import torch
import torch.nn.functional as F

from hansgpt_research.conditional_glyph_vae import ConditionalGlyphVAE, gaussian_kl, sample_gaussian
from hansgpt_research.cvae_training_optimization import make_head_kernel, optimized_backward


@pytest.mark.parametrize("chunk", [2, 3])
def test_optimized_loss_and_all_gradients_match_original_with_tail(chunk):
    cfg = runpy.run_path("tests/test_conditional_glyph_vae.py")["tiny_config"]()
    torch.manual_seed(91)
    original = ConditionalGlyphVAE(cfg).train()
    optimized = copy.deepcopy(original)
    x = torch.randint(2, (5, 3, 1, 32, 32), dtype=torch.uint8)
    y = x[:, 1]
    h = original.forward_hidden(x)[:, -1]
    leaf = h.detach().requires_grad_(True)
    torch.manual_seed(7)
    rec_sum = kl_sum = 0.0
    for start in range(0, 5, chunk):
        a, target = leaf[start : start + chunk], y[start : start + chunk]
        pm, pl = original.prior(a)
        qm, ql = original.posterior(a, target)
        logits = original.decode(a, sample_gaussian(qm, ql))
        rec = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="sum")
        kl = gaussian_kl(qm, ql, pm, pl).sum()
        ((rec + 0.7 * kl) / (5 * 1024)).backward()
        rec_sum += float(rec.detach())
        kl_sum += float(kl.detach())
    h.backward(leaf.grad)
    h2 = optimized.forward_hidden(x)[:, -1]
    torch.manual_seed(7)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    actual = optimized_backward(
        optimized, h2, y, scaler, chunk, 0.7, make_head_kernel(optimized), autocast_enabled=False
    )
    assert actual == pytest.approx((rec_sum, kl_sum), rel=2e-5, abs=2e-4)
    for a, b in zip(original.parameters(), optimized.parameters(), strict=True):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=0.003, atol=2e-5)


def test_nonfinite_statistics_abort_before_backbone_backward():
    cfg = runpy.run_path("tests/test_conditional_glyph_vae.py")["tiny_config"]()
    model = ConditionalGlyphVAE(cfg)
    x = torch.randint(2, (2, 2, 1, 32, 32), dtype=torch.uint8)
    h = model.forward_hidden(x)[:, -1]

    def bad(hidden, *args):
        loss = hidden.sum() * float("nan")
        return loss, loss

    with pytest.raises(FloatingPointError, match="optimizer not updated"):
        optimized_backward(
            model,
            h,
            x[:, 0],
            torch.amp.GradScaler("cuda", enabled=False),
            2,
            1.0,
            bad,
            autocast_enabled=False,
        )
    assert all(p.grad is None for p in model.backbone.parameters())


@pytest.mark.skipif(
    os.environ.get("HANSGPT_TEST_COMPILE") != "1",
    reason="Explicit GPU compilation test; run on the experiment GPU",
)
def test_compiled_cuda_head_gradients_match_eager():
    cfg = runpy.run_path("tests/test_conditional_glyph_vae.py")["tiny_config"]()
    torch.manual_seed(22)
    eager = ConditionalGlyphVAE(cfg).cuda().train()
    compiled = copy.deepcopy(eager)
    h = torch.randn(4, 64, device="cuda", requires_grad=True)
    features = torch.randn(4, 4, 32, device="cuda", requires_grad=True)
    h2 = h.detach().clone().requires_grad_(True)
    f2 = features.detach().clone().requires_grad_(True)
    targets = torch.randint(2, (4, 1, 32, 32), device="cuda", dtype=torch.uint8)
    noise = torch.randn(4, 8, device="cuda")
    valid = torch.tensor([True, True, True, False], device="cuda")
    results = []
    for model, hidden, feats, use_compile in [
        (eager, h, features, False),
        (compiled, h2, f2, True),
    ]:
        kernel = make_head_kernel(model, compiled=use_compile)
        with torch.autocast("cuda", dtype=torch.float16):
            rec, kl = kernel(hidden, targets, feats, noise, valid)
            loss = (rec + 0.7 * kl) / (3 * 1024)
        loss.backward()
        results.append(loss.detach())
    torch.testing.assert_close(results[0], results[1], rtol=0.003, atol=2e-4)
    torch.testing.assert_close(h.grad, h2.grad, rtol=0.03, atol=2e-4)
    torch.testing.assert_close(features.grad, f2.grad, rtol=0.03, atol=2e-4)
    for a, b in zip(eager.parameters(), compiled.parameters(), strict=True):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, rtol=0.03, atol=2e-4)
