"""Run on the server: lossless bytes, joint likelihood and causal inner decoding."""

import math

import pytest
import torch

from hansgpt_research.byte_glyph_decoder import (
    BYTE_BOS,
    ConditionalByteDecoder,
    byte_grid_nll,
    pack_glyph_bytes,
    unpack_glyph_bytes,
)
from hansgpt_research.glyph_lm import GlyphGPT, ModelConfig


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu_threads():
    # Small 128-step regression models otherwise spend most time starting CPU threads.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def decoder():
    torch.manual_seed(37)
    return ConditionalByteDecoder(
        hidden_size=12, inner_dim=16, layers=2, heads=2, intermediate_size=32
    )


def target_bytes(*leading):
    generator = torch.Generator().manual_seed(29)
    return torch.randint(0, 256, (*leading, 128), generator=generator, dtype=torch.uint8)


def test_pack_bit_order_is_row_major_most_significant_bit_first():
    tiles = torch.zeros(1, 1, 32, 32, dtype=torch.uint8)
    for row, column in ((0, 0), (0, 7), (0, 8), (0, 31), (1, 0), (31, 31)):
        tiles[0, 0, row, column] = 1
    expected = torch.zeros(1, 128, dtype=torch.uint8)
    expected[0, 0], expected[0, 1] = 129, 128
    expected[0, 3], expected[0, 4], expected[0, 127] = 1, 128, 1
    actual = pack_glyph_bytes(tiles)
    assert torch.equal(actual, expected)
    assert torch.equal(unpack_glyph_bytes(actual), tiles)


def test_all_256_byte_values_and_arbitrary_leading_shapes_round_trip():
    values = torch.arange(256, dtype=torch.long).reshape(2, 128)
    assert torch.equal(pack_glyph_bytes(unpack_glyph_bytes(values)), values.to(torch.uint8))
    values = target_bytes(2, 3)
    pixels = unpack_glyph_bytes(values)
    assert pixels.shape == (2, 3, 1, 32, 32)
    assert pixels.dtype == torch.uint8 and bool(((pixels == 0) | (pixels == 1)).all())
    assert torch.equal(pack_glyph_bytes(pixels), values)
    assert unpack_glyph_bytes(values[0, 0]).shape == (1, 32, 32)
    assert torch.equal(pack_glyph_bytes(pixels.float()), values)


@pytest.mark.parametrize("bad_value", [0.5, float("nan"), 2.0])
def test_nonbinary_pixels_are_rejected(bad_value):
    pixels = torch.zeros(1, 1, 32, 32)
    pixels[0, 0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="binary"):
        pack_glyph_bytes(pixels)


@pytest.mark.parametrize("bad_value", [-1, 256])
def test_unpack_rejects_values_outside_full_byte_alphabet(bad_value):
    values = torch.zeros(128, dtype=torch.long)
    values[0] = bad_value
    with pytest.raises(ValueError, match="between"):
        unpack_glyph_bytes(values)


def test_byte_likelihood_is_joint_inside_byte_and_uses_per_pixel_units():
    # First byte is either all zero (.4) or all one (.6), not eight independent bits.
    logits = torch.full((2, 128, 256), -1000.0)
    logits[:, :, 0] = 0
    logits[:, 0, 0] = math.log(0.4)
    logits[:, 0, 255] = math.log(0.6)
    labels = torch.zeros(2, 128, dtype=torch.long)
    labels[1, 0] = 255
    losses = byte_grid_nll(logits, labels)
    torch.testing.assert_close((-1024 * losses).exp(), torch.tensor([0.4, 0.6]))
    assert float((-1024 * losses).exp().sum()) == pytest.approx(1)
    assert float(losses[1]) == pytest.approx(-math.log(0.6) / 1024)
    assert float((-1024 * losses[1]).exp()) != pytest.approx(0.6**8)
    uniform = byte_grid_nll(torch.zeros(2, 128, 256, dtype=torch.float16), labels)
    assert uniform.dtype == torch.float32
    torch.testing.assert_close(uniform, torch.full((2,), math.log(2)))


@pytest.mark.parametrize("position", [0, 31, 64, 126])
def test_teacher_forcing_shift_cannot_see_current_or_future_target_bytes(position):
    model = decoder().eval()
    hidden = torch.randn(1, 12)
    labels = target_bytes(1)
    altered = labels.clone()
    altered[:, position:] = (altered[:, position:].long() + 17).remainder(256).to(torch.uint8)
    with torch.no_grad():
        before = model.teacher_forced_logits(hidden, unpack_glyph_bytes(labels))
        after = model.teacher_forced_logits(hidden, unpack_glyph_bytes(altered))
    torch.testing.assert_close(
        before[:, : position + 1], after[:, : position + 1], atol=1e-7, rtol=1e-6
    )
    assert not torch.allclose(before[:, position + 1], after[:, position + 1], atol=1e-6, rtol=1e-6)


def test_last_target_byte_is_never_an_input_but_still_contributes_to_nll():
    model = decoder().eval()
    hidden = torch.randn(1, 12)
    labels = target_bytes(1)
    changed = labels.clone()
    changed[:, -1] = (changed[:, -1].long() + 1).remainder(256).to(torch.uint8)
    with torch.no_grad():
        before = model.teacher_forced_logits(hidden, unpack_glyph_bytes(labels))
        after = model.teacher_forced_logits(hidden, unpack_glyph_bytes(changed))
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert not torch.equal(byte_grid_nll(before, labels), byte_grid_nll(after, changed))


def test_outer_next_grid_and_inner_next_byte_shifts_are_both_causal():
    torch.manual_seed(71)
    outer = GlyphGPT(
        ModelConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=32,
            max_position_embeddings=4,
        )
    ).eval()
    inner = ConditionalByteDecoder(
        hidden_size=16, inner_dim=16, layers=1, heads=2, intermediate_size=32
    ).eval()
    original_bytes = target_bytes(1, 4)
    changed_bytes = original_bytes.clone()
    changed_bytes[:, 2, 37:] = (changed_bytes[:, 2, 37:].long() + 7).remainder(256).to(torch.uint8)

    def prediction(values):
        grids = unpack_glyph_bytes(values)
        # Outer inputs x0,x1,x2 predict x1,x2,x3. h1 may see x0,x1, never x2.
        encoded = outer.encode_glyphs(grids[:, :-1])
        hidden = outer.backbone(inputs_embeds=encoded, use_cache=False).last_hidden_state[:, 1]
        return hidden, inner.teacher_forced_logits(hidden, grids[:, 2])

    with torch.no_grad():
        before_hidden, before = prediction(original_bytes)
        after_hidden, after = prediction(changed_bytes)
    torch.testing.assert_close(before_hidden, after_hidden, atol=1e-6, rtol=1e-5)
    # Byte 37 is a target at position 37 and only becomes an input at position 38.
    torch.testing.assert_close(before[:, :38], after[:, :38], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(before[:, 38], after[:, 38], atol=1e-6, rtol=1e-5)


def test_cache_matches_all_teacher_forced_logits_including_multi_query_chunks():
    model = decoder().eval()
    hidden = torch.randn(2, 12)
    labels = target_bytes(2).long()
    shifted = torch.cat((torch.full((2, 1), BYTE_BOS), labels[:, :-1]), dim=-1)
    cache = None
    chunks = []
    start = 0
    with torch.no_grad():
        expected = model(hidden, shifted)
        for size in (1, 5, 17, 105):
            current, cache = model(
                hidden, shifted[:, start : start + size], cache=cache, use_cache=True
            )
            chunks.append(current)
            start += size
    assert cache.length == 128
    torch.testing.assert_close(torch.cat(chunks, dim=1), expected, atol=2e-6, rtol=2e-5)
    with pytest.raises(ValueError, match="128"):
        model(hidden, torch.zeros(2, 1, dtype=torch.uint8), cache=cache, use_cache=True)


def test_cache_is_bound_to_condition_and_inner_bos_is_not_a_prediction_class():
    model = decoder().eval()
    hidden = torch.randn(2, 12)
    beginning = torch.full((2, 1), BYTE_BOS)
    with torch.no_grad():
        output, cache = model(hidden, beginning, use_cache=True)
        next_output, next_cache = model(
            hidden, torch.full((2, 1), 255, dtype=torch.uint8), cache=cache, use_cache=True
        )
    assert output.shape == (2, 1, 256)
    assert next_output.shape == (2, 1, 256) and next_cache.length == 2
    assert model.byte_embedding.num_embeddings == 257
    with pytest.raises(ValueError, match="different"):
        model(hidden + 1, torch.zeros(2, 1, dtype=torch.uint8), cache=cache, use_cache=True)
    with pytest.raises(ValueError, match="position zero"):
        model(hidden, beginning, cache=cache, use_cache=True)
    other = decoder().eval()
    with pytest.raises(ValueError, match="different"):
        other(hidden, torch.zeros(2, 1, dtype=torch.uint8), cache=cache, use_cache=True)
    with pytest.raises(ValueError, match="start"):
        model(hidden, torch.zeros(2, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="start"):
        model(hidden, torch.zeros(2, 1, dtype=torch.uint8))


def test_greedy_cached_and_uncached_generation_are_equal_and_return_binary_grids():
    model = decoder()
    hidden = torch.randn(2, 12)
    cached = model.distribution(hidden).decode(strategy="greedy", use_cache=True)
    uncached = model.distribution(hidden).decode(strategy="greedy", use_cache=False)
    assert model.training  # decode restores the caller's mode.
    assert cached.shape == (2, 1, 32, 32) and cached.dtype == torch.uint8
    assert bool(((cached == 0) | (cached == 1)).all())
    assert torch.equal(cached, uncached)


def test_sampling_is_reproducible_and_preserves_each_outer_condition_shape():
    model = decoder().eval()
    hidden = torch.randn(1, 2, 12)
    first = model.distribution(hidden).decode(
        strategy="sample", temperature=0.8, generator=torch.Generator().manual_seed(43)
    )
    repeated = model.distribution(hidden).decode(
        strategy="sample", temperature=0.8, generator=torch.Generator().manual_seed(43)
    )
    different = model.distribution(hidden).decode(
        strategy="sample", temperature=0.8, generator=torch.Generator().manual_seed(47)
    )
    assert first.shape == (1, 2, 1, 32, 32) and first.dtype == torch.uint8
    assert torch.equal(first, repeated) and not torch.equal(first, different)
    assert bool(((first == 0) | (first == 1)).all())


def test_batch_rows_are_independent_in_values_and_condition_gradients():
    model = decoder().eval()
    hidden = torch.randn(3, 12, requires_grad=True)
    targets = unpack_glyph_bytes(target_bytes(3))
    losses = model.distribution(hidden).nll(targets)
    singles = torch.stack(
        [model.distribution(hidden[index]).nll(targets[index]) for index in range(3)]
    )
    torch.testing.assert_close(losses, singles, atol=1e-7, rtol=1e-6)
    gradient = torch.autograd.grad(losses[0], hidden)[0]
    assert bool(gradient[0].abs().sum() > 0)
    assert torch.count_nonzero(gradient[1:]) == 0
    changed_targets = targets.clone()
    changed_targets[1:] = 1 - changed_targets[1:]
    unchanged = model.distribution(hidden).nll(changed_targets)
    torch.testing.assert_close(losses[0], unchanged[0], atol=0, rtol=0)


def test_decoder_api_does_not_silently_reinterpret_pixel_thresholds_or_bad_shapes():
    model = decoder()
    distribution = model.distribution(torch.randn(1, 12))
    with pytest.raises(TypeError):
        distribution.decode(threshold=0.3)
    with pytest.raises(ValueError, match="only"):
        distribution.decode(strategy="greedy", temperature=0.7)
    with pytest.raises(ValueError, match="positive"):
        distribution.decode(strategy="sample", temperature=0)
    with pytest.raises(ValueError, match="strategy"):
        distribution.decode(strategy="mode_threshold")
    with pytest.raises(ValueError, match="match"):
        distribution.nll(torch.zeros(2, 1, 32, 32, dtype=torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA smoke runs on server GPU0")
def test_cuda_fp16_cached_likelihood_and_finite_parameter_update():
    device = torch.device("cuda:0")  # Server runner sets CUDA_VISIBLE_DEVICES=0.
    model = decoder().to(device).eval()
    hidden = torch.randn(2, 12, device=device, dtype=torch.float16, requires_grad=True)
    labels = target_bytes(2).to(device)
    shifted = torch.cat(
        (torch.full((2, 1), BYTE_BOS, dtype=torch.long, device=device), labels[:, :-1].long()),
        dim=-1,
    )
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        full_logits = model(hidden, shifted)
        cache = None
        cached_parts = []
        for position in range(128):
            logits, cache = model(
                hidden, shifted[:, position : position + 1], cache=cache, use_cache=True
            )
            cached_parts.append(logits)
        cached_logits = torch.cat(cached_parts, dim=1)
        full_nll = byte_grid_nll(full_logits, labels)
        cached_nll = byte_grid_nll(cached_logits, labels)
    assert full_logits.dtype == cached_logits.dtype == torch.float16
    assert full_nll.dtype == cached_nll.dtype == torch.float32
    assert cache.length == 128
    # V100 may use different SDPA kernels for full and one-query cached attention.
    # Compare numerical likelihoods, not bitwise argmax at nearly tied byte logits.
    torch.testing.assert_close(cached_logits, full_logits, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(cached_nll, full_nll, atol=1e-4, rtol=2e-4)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
    before = model.byte_head.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model.distribution(hidden).nll(unpack_glyph_bytes(labels)).mean()
    assert loss.dtype == torch.float32 and bool(torch.isfinite(loss))
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(
        gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    assert hidden.grad is not None and bool(torch.isfinite(hidden.grad).all())
    assert bool(hidden.grad.abs().sum() > 0)
    assert bool(model.byte_head.weight.grad.abs().sum() > 0)
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    assert scaler.get_scale() >= previous_scale  # No skipped optimizer update.
    assert optimizer.state
    assert not torch.equal(model.byte_head.weight, before)
