"""Collect fixed-validation LR-search results without selecting on the test set."""

import argparse
import json
import time
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def collect():
    configs = [
        json.loads(p.read_text())
        for p in sorted(Path("configs/experiments/lr_search_v1").glob("gpu*.json"))
    ]
    rows = []
    for cfg in configs:
        name = cfg["experiment"] + "_full"
        logs = Path("artifacts/logs") / name
        root = Path("artifacts/reports") / name
        s = read_json(logs / "status.json")
        done = read_json(root / "complete.json")
        row = dict(
            gpu=cfg["gpu"],
            peak_lr=cfg["training"]["learning_rate"],
            name=name,
            status=s["status"] if s else "pending",
            han=s["progress"]["han"] if s else 0,
        )
        if s:
            row.update(phase=s["phase"], overflows=s["progress"]["overflows"], error=s.get("error"))
        validations = []
        if (logs / "training.jsonl").exists():
            for line in (logs / "training.jsonl").read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r["kind"] == "validation":
                    validations.append(dict(han=r["progress"]["han"], **r["metrics"]))
        row["validation_curve"] = validations
        if done:
            if done["progress"]["han"] != cfg["training"]["target_han"]:
                raise ValueError("Completed run has incorrect Han budget")
            e = done["evaluation"]
            if e.get("evaluation_split") != "validation":
                raise ValueError("LR selection must use validation generation")
            row.update(
                status="complete",
                progress=done["progress"],
                evaluation=e,
                final_validation=validations[-1],
                checkpoint_sha256=done["final_sha256"],
            )
        rows.append(row)
    terminal = all(r["status"] in ["complete", "failed"] for r in rows)
    complete = [r for r in rows if r["status"] == "complete"]
    ordered = sorted(complete, key=lambda r: r["final_validation"]["negative_elbo_per_pixel"])
    totals = {r["progress"]["all_targets"] for r in complete}
    if len(totals) > 1:
        raise ValueError("Completed trials used unequal target prefixes")
    return dict(
        status="finished" if terminal else "running",
        runs=rows,
        validation_elbo_ranking=[r["peak_lr"] for r in ordered],
        warning=(
            "ELBO ranking is a convergence aid, not a fluent-generation ranking. "
            "Review prior glyphs, coherence, repeats and EOS jointly. "
            "One training seed only; test set remains untouched."
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    out = Path("artifacts/reports/cvae_lr_search_v1")
    out.mkdir(parents=True, exist_ok=True)
    while True:
        report = collect()
        temp = out / "summary.tmp"
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(out / "summary.json")
        print(
            json.dumps(
                dict(
                    status=report["status"],
                    runs=[
                        {k: r[k] for k in ["gpu", "peak_lr", "status", "han"]}
                        for r in report["runs"]
                    ],
                )
            ),
            flush=True,
        )
        if not args.watch or report["status"] == "finished":
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
