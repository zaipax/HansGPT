import pytest

from hansgpt_research.position_schedule import position_learning_rate


def config(stop):
    return dict(
        lr_schedule="global_cosine",
        learning_rate=3e-4,
        minimum_learning_rate_ratio=0.1,
        schedule_total_positions=1089139385,
        warmup_positions=10891394,
        target_tokens=stop,
    )


def test_pilot_and_full_run_use_identical_prefix_schedule():
    pilot, full = config(100000000), config(1089139385)
    for count in [1, 65000, 1000000, 10891394, 50000000, 100000000]:
        assert position_learning_rate(count, pilot) == position_learning_rate(count, full)
    assert position_learning_rate(100000000, pilot) > 2.9e-4
    assert position_learning_rate(1089139385, full) == pytest.approx(3e-5)


def test_global_warmup_reaches_peak_and_is_independent_of_gpu_count():
    cfg = config(100000000)
    assert position_learning_rate(5445697, cfg) == pytest.approx(1.5e-4)
    assert position_learning_rate(10891394, cfg) == pytest.approx(3e-4)
    assert position_learning_rate(1000000, dict(cfg, world_size=4)) == position_learning_rate(
        1000000, dict(cfg, world_size=8)
    )


def test_invalid_horizon_rejected():
    with pytest.raises(ValueError):
        position_learning_rate(1, config(2000000000))
