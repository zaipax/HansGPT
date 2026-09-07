"""Numerical, scoring and checkpoint invariants; execute on the training server."""

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hansgpt_research.evaluate_glyph_lm import (
    BinaryMetrics,
    nearest_glyphs,
    select_constant_threshold,
)
from hansgpt_research.train_glyph_lm import (
    EpochSampler,
    SortishEpochSampler,
    learning_rate,
    loss_sum,
    restore_rng,
    save_checkpoint,
    sequence_lengths,
    sha256,
    verify_data_readiness,
)


def test_resume_sampler_uses_consumed_cursor():
    full = list(EpochSampler(31, seed=99, epoch=2))
    assert sorted(full) == list(range(31))
    assert list(EpochSampler(31, seed=99, epoch=2, cursor=13)) == full[13:]
    assert list(EpochSampler(31, seed=99, epoch=3)) != full


def test_accumulated_loss_weights_real_targets_not_padded_microbatches():
    logits = torch.tensor([0.0, 2.0, -4.0]).reshape(1, 3, 1, 1, 1).expand(1, 3, 1, 32, 32)
    targets = torch.ones_like(logits)
    mask = torch.tensor([[True, True, False]])
    total, count = loss_sum(logits, {"targets": targets, "loss_mask": mask})
    expected = torch.nn.functional.softplus(-torch.tensor([0.0, 2.0])).sum()
    assert count == 2
    assert torch.allclose(total, expected)
    left, _ = loss_sum(logits[:, :1], {"targets": targets[:, :1], "loss_mask": mask[:, :1]})
    right, _ = loss_sum(logits[:, 1:], {"targets": targets[:, 1:], "loss_mask": mask[:, 1:]})
    assert torch.allclose((left + right) / count, total / count)


def test_binary_metrics_foreground_and_full_grid_denominators():
    target = torch.zeros(2, 1024, dtype=torch.bool)
    prediction = target.clone()
    target[0, :4] = True
    prediction[0, 2:6] = True
    metrics = BinaryMetrics()
    metrics.add(prediction, target, torch.tensor([1024.0, 512.0]))
    result = metrics.result()
    assert result["foreground_f1"] == pytest.approx(0.5)
    assert result["iou"] == pytest.approx(1 / 3)
    assert result["dice"] == result["foreground_f1"]
    assert result["exact_bitmap_match"] == 0.5
    assert result["hamming_bits_per_grid"] == 2
    assert result["bce_per_pixel"] == 0.75
    assert result["nll_nats_per_grid"] == 768


def test_chunked_hamming_matches_exhaustive_search_including_ties():
    generator = torch.Generator().manual_seed(23)
    gallery = torch.randint(0, 2, (13, 1024), generator=generator).float()
    gallery[8] = gallery[2]
    query = torch.cat((gallery[[8]], torch.randint(0, 2, (6, 1024), generator=generator)))
    ids = torch.arange(4, 17)
    actual_ids, actual_distance = nearest_glyphs(
        query, gallery, ids, query_chunk=2, gallery_chunk=3
    )
    distances = (query[:, None].bool() != gallery[None].bool()).sum(-1)
    indices = (distances.double() + ids.double() * 1e-9).topk(5, largest=False).indices
    assert torch.equal(actual_ids, ids[indices])
    assert torch.equal(actual_distance, distances.gather(1, indices).float())
    assert actual_ids[0, 0].item() == 6


def test_checkpoint_restores_optimizer_rng_and_next_update(tmp_path):
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    def step():
        optimizer.zero_grad()
        value = torch.rand(4, 3) + random.random() + float(np.random.rand())
        model(value).square().mean().backward()
        optimizer.step()

    step()
    checkpoint = tmp_path / "resume.pt"
    save_checkpoint(checkpoint, model, optimizer, scaler, {"cursor": 7}, {"mode": "test"})
    step()
    uninterrupted = {key: value.detach().clone() for key, value in model.state_dict().items()}
    saved = torch.load(checkpoint, weights_only=False)
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    scaler.load_state_dict(saved["scaler"])
    restore_rng(saved["rng"])
    assert saved["progress"]["cursor"] == 7
    step()
    for key, value in model.state_dict().items():
        assert torch.equal(value, uninterrupted[key])


def test_learning_rate_reaches_declared_floor_after_budget():
    cfg = {
        "warmup_tokens": 100,
        "target_tokens": 1000,
        "learning_rate": 0.001,
        "minimum_learning_rate_ratio": 0.1,
    }
    assert learning_rate(50, cfg) == pytest.approx(0.0005)
    assert learning_rate(100, cfg) == pytest.approx(0.001)
    assert learning_rate(1000, cfg) == pytest.approx(0.0001)
    assert learning_rate(2000, cfg) == pytest.approx(0.0001)


def test_baseline_selects_own_validation_threshold_and_excludes_controls():
    bank = torch.zeros(3, 1024)
    bank[0, 0] = 1
    bank[1, 1] = 1
    bank[2] = 1
    probability = torch.zeros(1024)
    probability[0], probability[1] = 0.2, 0.4
    # A frequent all-foreground control must not affect content-F1 threshold choice.
    counts = torch.tensor([9, 1, 100000])
    actual = select_constant_threshold(probability, counts, bank, [2], [0.15, 0.25, 0.5])
    assert actual["selected_threshold"] == 0.15
    assert actual["validation_content_tiles"] == 10
    assert actual["threshold_metrics"]["0.15"]["foreground_f1"] == pytest.approx(2 / 3)
    assert actual["threshold_metrics"]["0.25"]["foreground_f1"] == pytest.approx(0.1)
    assert counts.tolist() == [9, 1, 100000]


def write_ready_fixture(directory, *, full=False):
    """Readiness unit fixture; binary-format validation belongs to the independent verifier."""
    names = ["glyph_bank.npz", "glyph_inventory.json"]
    names += [
        f"{split}.{suffix}"
        for split in ("train", "validation", "test")
        for suffix in ("uint16", "offsets.npy")
    ]
    for name in names:
        (directory / name).write_bytes(name.encode())
    hashes = {name: sha256(directory / name) for name in names}
    manifest = {"status": "bounded_experiment_corpus", "output_sha256": hashes}
    if full:
        files = [
            {"file": f"{index}.parquet", "expected_rows": 3, "scanned_rows": 3, "completed": True}
            for index in range(6)
        ]
        manifest.update(
            status="full_snapshot_experiment_corpus",
            source={
                "provider": "ModelScope",
                "selected_shards": list(range(6)),
                "files": [{"path": file["file"], "expected_rows": 3} for file in files],
            },
            source_scan={
                "all_snapshot_shards_selected": True,
                "all_selected_rows_scanned": True,
                "expected_rows": 18,
                "scanned_rows": 18,
                "files": files,
            },
        )
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    receipt = {
        "passed": True,
        "manifest_sha256": sha256(directory / "manifest.json"),
        "model_consumed_sha256": hashes,
    }
    (directory / "verification.json").write_text(json.dumps(receipt), encoding="utf-8")
    return manifest, receipt


@pytest.mark.parametrize("tamper", ["array", "inventory", "manifest", "failed", "interrupted"])
def test_data_readiness_rejects_tampered_or_interrupted_exports(tmp_path, tamper):
    manifest, receipt = write_ready_fixture(tmp_path)
    assert verify_data_readiness(tmp_path)["verification"]["passed"]
    if tamper in {"array", "inventory"}:
        name = "train.uint16" if tamper == "array" else "glyph_inventory.json"
        (tmp_path / name).write_bytes(b"changed since verification")
    elif tamper == "manifest":
        manifest["status"] = "tampered"
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif tamper == "failed":
        receipt["passed"] = False
        (tmp_path / "verification.json").write_text(json.dumps(receipt), encoding="utf-8")
    else:
        # A verifier removes its old receipt before a rerun; an interrupted rerun is not ready.
        (tmp_path / "verification.json").unlink()
        (tmp_path / "verification.json.tmp").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_data_readiness(tmp_path)


def test_full_training_requires_actual_six_shard_scan_proof(tmp_path):
    write_ready_fixture(tmp_path)
    with pytest.raises(ValueError, match="entire source scan"):
        verify_data_readiness(tmp_path, require_full_snapshot=True)
    manifest, receipt = write_ready_fixture(tmp_path, full=True)
    verify_data_readiness(
        tmp_path, require_full_snapshot=True, expected_provider="ModelScope", expected_shards=6
    )
    # Even a reissued receipt cannot turn a partial scan into a full source experiment.
    manifest["source_scan"]["files"][-1]["scanned_rows"] = 2
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    receipt["manifest_sha256"] = sha256(tmp_path / "manifest.json")
    (tmp_path / "verification.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="entire source scan"):
        verify_data_readiness(
            tmp_path, require_full_snapshot=True, expected_provider="ModelScope", expected_shards=6
        )


def test_data_readiness_rejects_missing_final_manifest(tmp_path):
    write_ready_fixture(tmp_path)
    (tmp_path / "manifest.json").rename(tmp_path / "manifest.json.tmp")
    with pytest.raises(ValueError, match="finalized"):
        verify_data_readiness(tmp_path)


def test_sortish_sampler_preserves_all_chunks_and_exact_resume_order():
    lengths = np.array([3, 17, 128, 7, 1024, 3, 90, 256, 4, 2, 61, 9, 600, 45, 80])
    full = list(SortishEpochSampler(lengths, seed=7, epoch=2, batch_size=4, pool_batches=2))
    assert sorted(full) == list(range(len(lengths)))
    for cursor in (0, 4, 8, 12, 15):
        resumed = list(
            SortishEpochSampler(
                lengths, seed=7, epoch=2, batch_size=4, cursor=cursor, pool_batches=2
            )
        )
        assert resumed == full[cursor:]
    assert list(SortishEpochSampler(lengths, seed=7, epoch=3, batch_size=4, pool_batches=2)) != full
    # No global random state is consumed by constructing the epoch permutation.
    torch.manual_seed(51)
    before = torch.get_rng_state().clone()
    list(SortishEpochSampler(lengths, seed=7, epoch=2, batch_size=4))
    assert torch.equal(before, torch.get_rng_state())


def test_sortish_reduces_padding_and_preserves_document_chunk_lengths():
    class SizedDataset(SimpleNamespace):
        def __len__(self):
            return 4

    dataset = SizedDataset(
        sequence_length=8,
        offsets=np.array([0, 5, 16, 18]),
        chunk_offsets=np.array([0, 1, 3, 4]),
        target_count=15,
    )
    # Four valid target chunks: 4; 8+2; 1. Exact multiples must not create zero-sized chunks.
    assert sequence_lengths(dataset).tolist() == [4, 8, 2, 1]
    lengths = np.tile(np.array([8, 64, 256, 1024]), 16)
    random_order = list(EpochSampler(len(lengths), seed=7, epoch=0))
    sortish_order = list(
        SortishEpochSampler(lengths, seed=7, epoch=0, batch_size=4, pool_batches=64)
    )

    def padded_positions(order):
        return sum(
            int(lengths[order[start : start + 4]].max()) * len(order[start : start + 4])
            for start in range(0, len(order), 4)
        )

    assert padded_positions(sortish_order) == int(lengths.sum())
    assert padded_positions(sortish_order) < padded_positions(random_order)
