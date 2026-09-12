import copy
import json
import runpy

import pytest


@pytest.mark.parametrize("mutation", [None, "targets", "learning_rate"])
def test_batch_comparison_checks_control_identity(tmp_path, monkeypatch, mutation):
    main = runpy.run_path("scripts/compare_cvae_batch_runs.py")["main"]
    a = dict(
        data_manifest_sha256="same",
        config=dict(
            model={},
            encoder={},
            decoders={},
            vae={},
            training=dict(batch_size=8, learning_rate=0.0003),
        ),
        progress=dict(overflows=0, all_targets=11000000),
        peak_allocated_gib=29.3,
        validation=dict(negative_elbo_per_pixel=0.3),
        evaluation=dict(posterior_mean_bce=0.2, kl_nats_per_glyph=3),
    )
    b = copy.deepcopy(a)
    b["config"]["training"].update(batch_size=6, sampler_reference_batch_size=8)
    b["peak_allocated_gib"] = 23.4
    if mutation == "targets":
        b["progress"]["all_targets"] -= 1
    if mutation == "learning_rate":
        b["config"]["training"]["learning_rate"] = 0.001
    records = iter([a, b])
    monkeypatch.setitem(main.__globals__, "read_run", lambda _: next(records))
    monkeypatch.chdir(tmp_path)
    if mutation:
        with pytest.raises(ValueError):
            main()
    else:
        main()
        report = json.loads(
            (tmp_path / "artifacts/reports/cvae_10m_batch_comparison/comparison.json").read_text()
        )
        assert report["same_order_han_prefix"]
        assert report["control_minus_baseline"]["peak_allocated_gib"] == pytest.approx(-5.9)
