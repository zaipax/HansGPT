import copy
import json
import os
import runpy
from pathlib import Path

import pytest
import torch

from hansgpt_research.conditional_glyph_vae import ConditionalGlyphVAE
from hansgpt_research.cvae_fixed_step import FixedBackward, prepare_pixels
from hansgpt_research.cvae_search_training import SharedGlyphCollator, checkpoint_latest
from hansgpt_research.cvae_training_optimization import head_sums


def test_causal_trailing_padding_shared_gradients_match_masked_model():
    cfg = runpy.run_path("tests/test_conditional_glyph_vae.py")["tiny_config"]()
    torch.manual_seed(61)
    a = ConditionalGlyphVAE(cfg).train()
    b = copy.deepcopy(a)
    x = torch.randint(2, (2, 4, 1, 32, 32), dtype=torch.uint8)
    y = x.roll(1, 1)
    attention = torch.tensor([[True, True, False, False], [True, True, True, True]])
    mask = attention.clone()
    mask[1, 1] = False
    batch = dict(glyphs=x, targets=y, attention_mask=attention, loss_mask=mask)
    p = prepare_pixels(batch, allow_trailing_padding=True)
    noise = torch.randn(8, 8)
    h = a.forward_hidden(x, attention).flatten(0, 1)
    targets = y.flatten(0, 1)
    rec, kl = head_sums(a, h, targets, a.glyph_features(targets), noise, mask.flatten())
    ((rec + 0.7 * kl) / (mask.sum() * 1024)).backward()
    sums = FixedBackward(b, 2, 4, 2, compiled=False)(
        **{k: v for k, v in p.items() if k != "unique"},
        noise=noise,
        scale=torch.tensor(1.0),
        beta=torch.tensor(0.7),
    )
    torch.testing.assert_close(sums, torch.stack((rec, kl)).double(), rtol=1e-5, atol=1e-4)
    for u, v in zip(a.parameters(), b.parameters(), strict=True):
        if u.grad is None:
            assert v.grad is None
        else:
            torch.testing.assert_close(u.grad, v.grad, rtol=0.004, atol=4e-5)
    broken = attention.clone()
    broken[0] = torch.tensor([True, False, True, False])
    with pytest.raises(ValueError, match="trailing"):
        prepare_pixels(dict(batch, attention_mask=broken), allow_trailing_padding=True)


def test_collator_pads_partial_batch_without_extra_loss_or_cursor():
    sample = dict(
        glyphs=torch.zeros(4, 1, 32, 32, dtype=torch.uint8),
        targets=torch.ones(4, 1, 32, 32, dtype=torch.uint8),
        attention_mask=torch.ones(4, dtype=torch.bool),
        loss_mask=torch.tensor([True, False, True, True]),
        target_ids=torch.arange(4),
    )
    p = SharedGlyphCollator(2, 4)([sample])
    assert p["source_batch_size"] == 1 and int(p["mask"].sum()) == 3
    assert len(p["targets"]) == 8 and not p["mask"][4:].any()
    assert torch.equal(p["target_ids"][:4], sample["target_ids"])


def test_lr_candidates_only_vary_peak_rate_and_run_identity():
    configs = [
        json.loads(p.read_text())
        for p in sorted(Path("configs/experiments/lr_search_v1").glob("gpu*.json"))
    ]
    assert len(configs) == 8 and {c["gpu"] for c in configs} == set(range(8))
    normalized = []
    for c in configs:
        d = copy.deepcopy(c)
        d.pop("gpu")
        d.pop("experiment")
        d["training"].pop("learning_rate")
        normalized.append(d)
    assert all(c == normalized[0] for c in normalized)


def test_latest_checkpoint_is_resumable_and_final_hardlink_is_immutable(tmp_path):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    model(torch.ones(2, 4)).sum().backward()
    opt.step()
    path = tmp_path / "trial" / "latest.pt"
    path.parent.mkdir()
    checkpoint_latest(path, model, opt, scaler, dict(han=1), dict(seed=7))
    final = path.parent / "final.pt"
    os.link(path, final)
    assert path.stat().st_ino == final.stat().st_ino
    saved = torch.load(final, weights_only=False)
    assert saved["optimizer"]["state"] and saved["rng"] and saved["progress"]["han"] == 1
    checkpoint_latest(path, model, opt, scaler, dict(han=2), dict(seed=7))
    assert torch.load(final, weights_only=False)["progress"]["han"] == 1
    assert torch.load(path, weights_only=False)["progress"]["han"] == 2


@pytest.mark.skipif(
    os.environ.get("HANSGPT_TEST_XFORMERS") != "1", reason="CUDA validation attention path"
)
def test_xformers_float_padding_bias_broadcast():
    import torch.nn.functional as f

    from hansgpt_research.cvae_fixed_step import install_xformers

    q = torch.randn(2, 4, 16, 32, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    bias = torch.zeros(2, 1, 16, 16, device="cuda", dtype=torch.float32)
    bias.masked_fill_(~torch.ones(16, 16, device="cuda", dtype=torch.bool).tril(), float("-inf"))
    bias[0, :, :, 12:] = float("-inf")
    original = f.scaled_dot_product_attention
    a = original(q, k, v, attn_mask=bias)
    ga = torch.autograd.grad(a.float().square().sum(), (q, k, v))
    install_xformers()
    try:
        b = f.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        gb = torch.autograd.grad(b.float().square().sum(), (q, k, v))
    finally:
        f.scaled_dot_product_attention = original
    torch.testing.assert_close(a, b, rtol=0.02, atol=0.002)
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, rtol=0.03, atol=0.02)
