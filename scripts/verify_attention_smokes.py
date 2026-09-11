"""Require successful, current-source, same-data ABC smoke receipts before full runs."""

import argparse
import json
import subprocess
from pathlib import Path

from hansgpt_research.train_glyph_lm import sha256
from hansgpt_research.train_structured_glyph_lm import effective_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--round", required=True)
args = parser.parse_args()
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
identities = []
for variant in "abc":
    directory = Path("artifacts/logs") / f"{args.round}_{variant}_smoke"
    receipt = json.loads((directory / "training_complete.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    config = json.loads(Path(f"configs/experiments/hansgpt_attention_{variant}.json").read_text())
    expected = effective_config(config, mode="smoke", smoke_tokens=4096)
    if (
        receipt["status"] != "complete"
        or receipt["mode"] != "smoke"
        or receipt["progress"]["optimizer_steps"] < 1
        or receipt["metadata_sha256"] != sha256(directory / "metadata.json")
        or metadata["git_commit"] != commit
        or metadata["config"] != expected
    ):
        raise ValueError(f"Variant {variant} lacks a successful current smoke")
    identities.append(
        (
            metadata["data_sha256"],
            metadata["data_manifest_sha256"],
            metadata["validation_selection"],
        )
    )
if not all(value == identities[0] for value in identities):
    raise ValueError("ABC smoke datasets or validation subsets differ")
print("All three current-source smoke tests passed on matched data and validation samples")
