"""Read-only hourly status for a named glyph run, reusing common progress inspection."""

import argparse
import json
import runpy
import subprocess
from datetime import UTC, datetime
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run-name", required=True)
parser.add_argument("--gpu", type=int, required=True)
args = parser.parse_args()
if not args.run_name.replace("_", "").replace("-", "").isalnum() or args.gpu < 0:
    raise ValueError("Invalid run name or GPU")
root = Path("/root/HansGPT")
helpers = runpy.run_path(str(Path(__file__).with_name("check_attention_abc.py")))
output = root / "artifacts/logs" / (args.run_name + "_monitor")
output.mkdir(parents=True, exist_ok=True)
previous = helpers["read_json"](output / "latest.json") or {}
report = {
    "time": datetime.now(UTC).isoformat(),
    "run": helpers["inspect_run"](root, args.run_name, previous.get("run", {})),
}
gpu = subprocess.run(
    [
        "nvidia-smi",
        "-i",
        str(args.gpu),
        "--query-gpu=index,name,memory.used,utilization.gpu",
        "--format=csv,noheader",
    ],
    capture_output=True,
    text=True,
    check=False,
)
report.update(gpu=gpu.stdout.strip().splitlines(), gpu_query_exit_code=gpu.returncode)
evaluation = root / "artifacts/reports" / (args.run_name + "_evaluation")
report["evaluation_complete"] = (evaluation / "evaluation_complete.json").is_file()
temporary = output / "latest.tmp"
temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
temporary.replace(output / "latest.json")
with (output / "hourly.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(report) + "\n")
print(json.dumps(report, indent=2))
