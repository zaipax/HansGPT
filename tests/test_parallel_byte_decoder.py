import pytest
import torch

from hansgpt_research.byte_glyph_decoder import pack_glyph_bytes
from hansgpt_research.byte_training import make_byte_loss
from hansgpt_research.parallel_byte_decoder import ParallelByteDecoder


def decoder():
    return ParallelByteDecoder(
        hidden_size=16, inner_dim=16, layers=1, heads=2, intermediate_size=32
    )


def test_targets_cannot_leak_and_generation_matches_parallel_logits():
    torch.manual_seed(7)
    model = decoder()
    hidden = torch.randn(2, 3, 16)
    zeros = torch.zeros(2, 3, 1, 32, 32, dtype=torch.uint8)
    a = model.teacher_forced_logits(hidden, zeros)
    b = model.teacher_forced_logits(hidden, 1 - zeros)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(pack_glyph_bytes(model.generate(hidden)), a.argmax(-1).byte())
    assert not torch.equal(a, model(hidden + 1))


def test_queries_interact_bidirectionally():
    model = decoder()
    hidden = torch.randn(1, 16)
    (gradient,) = torch.autograd.grad(model(hidden)[0, 0, 0], model.position_embedding.weight)
    assert gradient[-1].abs().sum() > 0  # First byte sees the final query.


def test_outer_model_parallel_factory_gradients_and_raw_cached_generation():
    from hansgpt_research.attention_glyph_lm import AttentionGlyphGPT
    from hansgpt_research.glyph_lm import ModelConfig

    model = AttentionGlyphGPT(
        ModelConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            max_position_embeddings=16,
        ),
        "C",
        {"width": 16, "layers": 1, "heads": 2},
        {"kind": "parallel", "inner_dim": 16, "layers": 1, "heads": 2, "intermediate_size": 32},
    ).eval()
    tiles = torch.randint(0, 2, (1, 3, 1, 32, 32), dtype=torch.uint8)
    hidden = model.forward_hidden(tiles)
    changed = tiles.clone()
    changed[:, -1] = 1 - changed[:, -1]
    torch.testing.assert_close(hidden[:, :-1], model.forward_hidden(changed)[:, :-1])
    model.distribution(hidden).nll(tiles).mean().backward()
    for module in (model.backbone, model.glyph_encoder, model.byte_decoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    cached = model.generate(tiles, 3, use_cache=True)
    uncached = model.generate(tiles, 3, use_cache=False)
    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)


def test_masked_accelerated_loss_and_all_gradients_match_distribution():
    torch.manual_seed(9)
    model = decoder()
    hidden = torch.randn(3, 16, requires_grad=True)
    tiles = torch.randint(0, 2, (3, 1, 32, 32), dtype=torch.uint8)
    mask = torch.tensor([1.0, 0.0, 1.0])
    reference = (model.distribution(hidden).nll(tiles) * mask).sum() * 1024
    parameters = [hidden, *model.parameters()]
    expected = torch.autograd.grad(reference, parameters)
    actual = make_byte_loss(model, compiled=False)(hidden, pack_glyph_bytes(tiles), mask)
    gradients = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual, reference)
    for a, b in zip(gradients, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    assert gradients[0][1].count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_parallel_loss_backward_matches_eager_on_cuda():
    model = decoder().cuda()
    hidden = torch.randn(2, 16, device="cuda", requires_grad=True)
    targets = torch.randint(0, 256, (2, 128), device="cuda", dtype=torch.uint8)
    mask = torch.ones(2, device="cuda")
    parameters = [hidden, *model.parameters()]
    with torch.autocast("cuda", dtype=torch.float16):
        reference = make_byte_loss(model, compiled=False)(hidden, targets, mask)
        actual = make_byte_loss(model, compiled=True)(hidden, targets, mask)
    expected = torch.autograd.grad(reference, parameters)
    gradients = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=0.1)
    for a, b in zip(gradients, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0.02, atol=0.1)
