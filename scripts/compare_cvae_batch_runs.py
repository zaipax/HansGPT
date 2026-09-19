"""Compare completed equal-Han CVAE runs; never select weights or alter training."""

import json
from pathlib import Path


def read_run(name):
    root = Path("artifacts/reports") / (name + "_full")
    logs = Path("artifacts/logs") / (name + "_full")
    complete = json.loads((root / "complete.json").read_text())
    metadata = json.loads((root / "metadata.json").read_text())
    status = json.loads((logs / "status.json").read_text())
    if complete["status"] != "complete" or complete["progress"]["han"] != 10000000:
        raise ValueError("Both runs must complete exactly ten million Han")
    validation = [
        json.loads(x)
        for x in (logs / "training.jsonl").read_text().splitlines()
        if json.loads(x)["kind"] == "validation"
    ]
    return dict(
        name=name,
        progress=complete["progress"],
        config=metadata["config"],
        source_commit=metadata["git_commit"],
        data_manifest_sha256=metadata["data_manifest_sha256"],
        peak_allocated_gib=status["peak_cuda_memory_bytes"] / 2**30,
        elapsed_seconds_including_evaluation=status["seconds"],
        validation=validation[-1]["metrics"],
        evaluation=complete["evaluation"],
    )


def main():
    runs = [
        read_run(name)
        for name in (
            "conditional_vae_24l_10m_ctx1024_v1",
            "conditional_vae_24l_10m_ctx1024_bsz6_v1",
        )
    ]
    a, b = runs
    if a["data_manifest_sha256"] != b["data_manifest_sha256"]:
        raise ValueError("Dataset identity mismatch")
    for key in ("model", "encoder", "decoders", "vae"):
        if a["config"][key] != b["config"][key]:
            raise ValueError(f"Model/objective configuration mismatch: {key}")
    ta, tb = a["config"]["training"], b["config"]["training"]
    allowed = {"batch_size", "sampler_reference_batch_size"}
    if any(ta.get(k) != tb.get(k) for k in set(ta) | set(tb) if k not in allowed):
        raise ValueError("Unexpected training configuration difference")
    if tb.get("sampler_reference_batch_size") != ta["batch_size"]:
        raise ValueError("Control must preserve the baseline sampler's reference batch size")
    if (
        not a["progress"]["overflows"]
        and not b["progress"]["overflows"]
        and a["progress"]["all_targets"] != b["progress"]["all_targets"]
    ):
        raise ValueError("Equal ordered Han prefixes must have equal effective target counts")
    report = dict(
        status="complete",
        runs=runs,
        same_order_han_prefix=(not a["progress"]["overflows"] and not b["progress"]["overflows"]),
        caveats=[
            "Different microbatches change update counts and stochastic latent draws.",
            "Two GPUs and overlapping wall-clock runs are not a hardware throughput benchmark.",
            "Soft nearest-font readings are scoring aids, not generated text or human readability.",
            "If AMP skipped batches, consumed prefixes may differ despite the same sampler order.",
        ],
        control_minus_baseline=dict(
            peak_allocated_gib=b["peak_allocated_gib"] - a["peak_allocated_gib"],
            validation_negative_elbo_per_pixel=b["validation"]["negative_elbo_per_pixel"]
            - a["validation"]["negative_elbo_per_pixel"],
            posterior_reconstruction_bce=b["evaluation"]["posterior_mean_bce"]
            - a["evaluation"]["posterior_mean_bce"],
            kl_nats_per_glyph=b["evaluation"]["kl_nats_per_glyph"]
            - a["evaluation"]["kl_nats_per_glyph"],
        ),
    )
    out = Path("artifacts/reports/cvae_10m_batch_comparison")
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report["control_minus_baseline"]))


if __name__ == "__main__":
    main()
