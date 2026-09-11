"""Hourly read-only ABC progress check; runs with uv and Python's standard library only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def latest_metrics(path):
    if not path.exists():
        return {}
    result = {}
    # Stream to bound memory for long jobs; partial last writes are ignored.
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") in {"train", "validation", "generation_diagnostic", "overflow"}:
                result[row["kind"]] = row
    return result


def inspect_run(root, run, previous):
    directory = root / "artifacts/logs" / run
    status = read_json(directory / "status.json")
    metadata = read_json(directory / "metadata.json")
    if not status or not metadata:
        return {"run": run, "state": "missing_run", "alert": True}
    progress = status["progress"]
    count = progress["valid_tokens"]
    pid = status.get("pid")
    alive = False
    if pid:
        try:
            command = (Path("/proc") / str(pid) / "cmdline").read_bytes()
            alive = b"train_attention_glyph_lm" in command and run.encode() in command
        except OSError:
            pass
    delta = count - previous.get("valid_tokens", count)
    stale = (datetime.now(UTC) - datetime.fromisoformat(status["time"])).total_seconds()
    state = status["status"]
    receipt = read_json(directory / "training_complete.json")
    if state == "running" and not alive:
        state = "process_missing"
    elif state == "running" and stale > 7200:
        state = "stale_over_two_hours"
    elif state == "complete" and (not receipt or receipt.get("status") != "complete"):
        state = "completion_receipt_missing"
    elapsed = max(1, progress["training_seconds"])
    budget = status["target_budget"]
    rate = count / elapsed
    return {
        "run": run,
        "state": state,
        "phase": status.get("phase"),
        "alert": state not in {"running", "complete"},
        "pid_alive": alive,
        "physical_gpu": metadata["cuda_visible_devices"],
        "git_commit": metadata["git_commit"],
        "valid_tokens": count,
        "target_budget": budget,
        "percent": 100 * count / budget,
        "targets_since_previous_check": delta,
        "status_age_seconds": stale,
        "successful_targets_per_second_including_validation": rate,
        "estimated_remaining_hours": max(0, budget - count) / rate / 3600 if rate else None,
        "overflow_steps": progress["overflow_steps"],
        "best_validation_nll": progress["best_validation_nll"],
        "latest_metrics": latest_metrics(directory / "training.jsonl"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", required=True)
    parser.add_argument("--root", type=Path, default=Path("/root/HansGPT"))
    args = parser.parse_args()
    if not args.round.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid round identifier")
    output = args.root / "artifacts/logs" / (args.round + "_monitor")
    output.mkdir(parents=True, exist_ok=True)
    last = read_json(output / "latest.json") or {}
    previous = {row["run"]: row for row in last.get("runs", [])}
    runs = [args.round + "_" + variant for variant in "abc"]
    report = {
        "time": datetime.now(UTC).isoformat(),
        "runs": [inspect_run(args.root, run, previous.get(run, {})) for run in runs],
    }
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "4,5,6",
            "--query-gpu=index,name,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    report["gpu"] = gpu.stdout.strip().splitlines()
    report["gpu_query_exit_code"] = gpu.returncode
    temporary = output / f"latest.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "latest.json")
    with (output / "hourly.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
