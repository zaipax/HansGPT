import pytest
import torch

from hansgpt_research.byte_glyph_decoder import ConditionalByteDecoder, pack_glyph_bytes
from hansgpt_research.byte_training import ByteCollator, make_byte_loss


def test_accelerated_loss_matches_joint_autoregressive_likelihood_and_gradients():
    torch.manual_seed(9)
    decoder = ConditionalByteDecoder(
        hidden_size=16, inner_dim=16, layers=1, heads=2, intermediate_size=32
    )
    hidden = torch.randn(3, 16, requires_grad=True)
    tiles = torch.randint(0, 2, (3, 1, 32, 32), dtype=torch.uint8)
    mask = torch.tensor([1.0, 0.0, 1.0])
    reference = (decoder.distribution(hidden).nll(tiles) * mask).sum() * 1024
    parameters = [hidden, *decoder.parameters()]
    expected = torch.autograd.grad(reference, parameters)
    actual = make_byte_loss(decoder, compiled=False)(hidden, pack_glyph_bytes(tiles), mask)
    gradients = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual, reference)
    for a, b in zip(gradients, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_pixel_dedup_and_byte_packing_preserve_order():
    tiles = torch.randint(0, 2, (4, 1, 32, 32), dtype=torch.uint8)
    tiles[3] = tiles[0]
    sample = dict(
        glyphs=tiles,
        targets=tiles.flip(0),
        attention_mask=torch.ones(4, dtype=torch.bool),
        loss_mask=torch.tensor([1, 0, 1, 1], dtype=torch.bool),
        target_ids=torch.arange(4),
    )
    prepared = ByteCollator()([sample])
    torch.testing.assert_close(prepared["tiles"][prepared["indices"]], tiles)
    torch.testing.assert_close(prepared["byte_targets"], pack_glyph_bytes(tiles.flip(0)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_xformers_short_cached_prefix_matches_native_attention():
    pytest.importorskip("xformers")
    import torch.nn.functional as F

    from hansgpt_research.cvae_fixed_step import install_xformers

    original = install_xformers()
    try:
        for length in [2, 3, 7, 9, 17]:
            q = torch.randn(1, 8, 1, 32, device="cuda", dtype=torch.float16)
            k = torch.randn(1, 8, length, 32, device="cuda", dtype=torch.float16)
            v = torch.randn_like(k)
            mask = torch.ones(1, length, device="cuda", dtype=torch.bool)
            mask[:, -1] = False
            expected = original(q, k, v, attn_mask=mask)
            actual = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.005)
    finally:
        F.scaled_dot_product_attention = original
