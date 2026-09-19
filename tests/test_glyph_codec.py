import io
from dataclasses import asdict

import pytest
import torch

from hansgpt_research.attention_glyph_lm import AttentionGlyphEncoder
from hansgpt_research.dual_decoder_glyph_lm import DualDecoderConfig, SpatialGlyphDecoder
from hansgpt_research.glyph_codec import GlyphCodec, load_glyph_codec


def make(arm, slots=4):
    return GlyphCodec(
        arm,
        AttentionGlyphEncoder(1024, width=128, layers=1, heads=4),
        SpatialGlyphDecoder(DualDecoderConfig(glyph_layers=1, semantic_slots=slots)),
    )


@pytest.mark.parametrize("arm", ["fixed", "spatial"])
@pytest.mark.parametrize("slots", [4, 16])
def test_same_latent_contract_and_checkpoint_roundtrip(arm, slots):
    torch.manual_seed(19)
    model = make(arm, slots).eval()
    tiles = torch.randint(2, (2, 1, 32, 32), dtype=torch.uint8)
    with torch.no_grad():
        z = model.encode(tiles)
        actual = model(tiles)
        assert z.shape == (2, slots, 256)
        torch.testing.assert_close(actual, model.decode(z), rtol=0, atol=0)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = make(arm, slots).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(actual, restored(tiles), rtol=0, atol=0)
    with pytest.raises(ValueError):
        model.decode(torch.zeros(2, 1024))


def test_fixed_encoder_cannot_drift_but_adapter_and_decoder_learn():
    model = make("fixed").train()
    tiles = torch.randint(2, (2, 1, 32, 32), dtype=torch.uint8)
    before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    model(tiles).square().mean().backward()
    optimizer.step()
    assert not model.encoder.training
    assert all(p.grad is None for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adapter.parameters())
    for k, v in model.encoder.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)


def test_spatial_queries_and_patch_features_receive_gradients():
    model = make("spatial")
    model(torch.randint(2, (2, 1, 32, 32), dtype=torch.uint8)).square().mean().backward()
    assert model.encoder.queries.grad.abs().sum() > 0
    assert model.encoder.patch_projection.weight.grad.abs().sum() > 0


def test_public_loader_requires_complete_mapping_and_pins_identity(tmp_path):
    model = make("fixed").eval()
    payload = {
        "model": model.state_dict(),
        "metadata": {
            "arm": "fixed",
            "interface": model.interface_version,
            "config": {},
            "base_config": {
                "encoder": {"width": 128, "layers": 1, "heads": 4},
                "decoders": asdict(DualDecoderConfig(glyph_layers=1)),
            },
        },
    }
    path = tmp_path / "codec.pt"
    torch.save(payload, path)
    restored = load_glyph_codec(path)
    tiles = torch.randint(2, (2, 1, 32, 32), dtype=torch.uint8)
    with torch.no_grad():
        original_output, loaded_output = model(tiles), restored(tiles)
        # Freezing all parameters can select a different CPU attention kernel.
        torch.testing.assert_close(original_output, loaded_output, rtol=1e-5, atol=1e-6)
        assert torch.equal(original_output > 0, loaded_output > 0)
    assert not any(p.requires_grad for p in restored.parameters())
    latents = torch.randn(2, 4, 256, requires_grad=True)
    restored.decode(latents).square().mean().backward()
    assert latents.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="identity"):
        load_glyph_codec(path, expected_sha256="0" * 64)
    del payload["model"]["adapter.0.weight"]
    torch.save(payload, path)
    with pytest.raises(RuntimeError):
        load_glyph_codec(path)
