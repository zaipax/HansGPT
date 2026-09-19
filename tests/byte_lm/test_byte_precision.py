import copy

import pytest
import torch

from hansgpt_research.attention_glyph_lm import AttentionGlyphGPT
from hansgpt_research.byte_precision import NonfiniteByteForward, checked_byte_backward
from hansgpt_research.byte_training import ByteBackward, ByteCollator
from hansgpt_research.glyph_lm import ModelConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("kind", ["overflow", "clean", "nan"])
def test_precision_recovery_preserves_batch_gradients_and_rejects_corruption(compiled, kind):
    torch.manual_seed(41)
    model = (
        AttentionGlyphGPT(
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
        )
        .cuda()
        .train()
    )
    # Finite FP32 bias above FP16's range: reproduces NaN loss from a valid model.
    with torch.no_grad():
        model.byte_decoder.condition_projection.bias.fill_(
            {"overflow": 100000, "clean": 0, "nan": float("nan")}[kind]
        )
    tiles = torch.randint(0, 2, (4, 1, 32, 32), dtype=torch.uint8)
    cpu = ByteCollator()(
        [
            dict(
                glyphs=tiles,
                targets=tiles.flip(0),
                attention_mask=torch.ones(4, dtype=torch.bool),
                loss_mask=torch.ones(4),
                target_ids=torch.arange(4),
            )
        ]
    )
    data = {k: cpu[k].cuda() for k in ("tiles", "indices", "byte_targets", "mask")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    backward = ByteBackward(model, 1, 4, chunk=2, compiled=compiled)
    before = copy.deepcopy(model.state_dict())
    reference = copy.deepcopy(model)
    events = []
    if kind == "nan":
        with pytest.raises(NonfiniteByteForward, match="fp32=True"):
            checked_byte_backward(
                backward, optimizer, data, torch.ones((), device="cuda"), on_retry=events.append
            )
        assert len(events) == 1
        assert all(p.grad is None for p in model.parameters())
        assert not optimizer.state
        return
    loss = checked_byte_backward(
        backward, optimizer, data, torch.ones((), device="cuda"), on_retry=events.append
    )
    assert bool(torch.isfinite(loss)), "Finite weights must recover an FP16-only loss overflow"
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    assert len(events) == (1 if kind == "overflow" else 0)
    reference_backward = ByteBackward(reference, 1, 4, chunk=2, compiled=False)
    expected = reference_backward(
        **data, scale=torch.ones((), device="cuda"), fp32=kind == "overflow", checked=True
    )
    torch.testing.assert_close(loss, expected, rtol=1e-3, atol=1e-3)
    for left, right in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=0.02, atol=1e-4)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
