import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from hansgpt_research.byte_resume import restore_rank_rng, validate_resume
from hansgpt_research.cvae_distributed_data import select_global_positions


def fixture():
    config = json.loads(
        Path("configs/experiments/hansgpt_qwen3_parallel_byte_gpu3_10m.json").read_text()
    )
    saved = dict(
        metadata={"config": copy.deepcopy(config)},
        rank_rng=[{}],
        progress=dict(
            epoch=0,
            cursor=9828,
            batch_skip=4049,
            steps=2458,
            all_targets=10000000,
            han=8953156,
            overflows=0,
        ),
    )
    config["training"]["target_tokens"] = 100000000
    config["experiment"] = "continuation"
    return saved, config


def test_continuation_preserves_partial_batch_and_cumulative_position():
    saved, config = fixture()
    progress = validate_resume(saved, config, 1)
    mask = torch.ones(4096, dtype=torch.bool)
    consumed = select_global_positions(mask, [4096], 0, 0, progress["batch_skip"])
    remaining = select_global_positions(mask, [4096], 0, progress["batch_skip"], 47)
    assert not (consumed & remaining).any()
    assert (consumed | remaining).all()
    assert progress == saved["progress"] and progress is not saved["progress"]


@pytest.mark.parametrize(
    "key,value",
    [("learning_rate", 1e-3), ("seed", 1), ("batch_size", 2), ("warmup_positions", 2000000)],
)
def test_resume_rejects_training_trajectory_changes(key, value):
    saved, config = fixture()
    config["training"][key] = value
    with pytest.raises(ValueError, match="preserve"):
        validate_resume(saved, config, 1)


def test_resume_rejects_completed_budget_and_world_change():
    saved, config = fixture()
    with pytest.raises(ValueError, match="world size"):
        validate_resume(saved, config, 2)
    config["training"]["target_tokens"] = 10000000
    with pytest.raises(ValueError, match="larger cumulative"):
        validate_resume(saved, config, 1)


def test_resume_rejects_unvalidated_failure_snapshot():
    saved, config = fixture()
    saved["diagnostic_only"] = True
    with pytest.raises(ValueError, match="Diagnostic"):
        validate_resume(saved, config, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_saved_model_optimizer_scaler_and_rng_reproduce_next_update(tmp_path):
    model = torch.nn.Linear(16, 8).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda", init_scale=128)

    def step(model, optimizer, scaler):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(4, 16, device="cuda") + random.random() + float(np.random.rand())
        x = x + torch.rand(1).item()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(x).float().square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    step(model, optimizer, scaler)
    path = tmp_path / "resume.pt"
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            rng=dict(
                python=random.getstate(),
                numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(),
            ),
        ),
        path,
    )
    expected_loss = step(model, optimizer, scaler)
    saved = torch.load(path, weights_only=False, map_location="cpu", mmap=True)
    other = torch.nn.Linear(16, 8).cuda()
    other.load_state_dict(saved["model"])
    opt = torch.optim.AdamW(other.parameters(), lr=2e-4, fused=True)
    opt.load_state_dict(saved["optimizer"])
    amp = torch.amp.GradScaler("cuda")
    amp.load_state_dict(saved["scaler"])
    restore_rank_rng(saved["rng"], torch.device("cuda:0"))
    actual_loss = step(other, opt, amp)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for left, right in zip(model.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert amp.state_dict() == scaler.state_dict()
