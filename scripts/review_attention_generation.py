"""Add cycle-aware summaries without altering original evaluation artifacts or receipts."""

import argparse
import json
import runpy
from pathlib import Path

import numpy as np
import torch

from hansgpt_research.glyph_lm import GlyphSequenceDataset
from hansgpt_research.train_glyph_lm import runtime_metadata, sha256, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
receipt = json.loads((args.output / "evaluation_complete.json").read_text())
for name in ("raw_generation.npz", "generation.json", "metadata.json"):
    if sha256(args.output / name) != receipt["outputs"][name]:
        raise ValueError(f"Original artifact checksum mismatch: {name}")
metadata = json.loads((args.output / "metadata.json").read_text())
config = metadata["training"]["config"]
data = Path("data/processed/modelscope_zhwiki_full_v1")
review_metadata = runtime_metadata(config, data, torch.device("cpu"), mode="full")
if review_metadata["data_sha256"] != metadata["training"]["data_sha256"]:
    raise ValueError("Review corpus identity mismatch")
dataset = GlyphSequenceDataset(data, "test", config["training"]["sequence_length"])
helpers = runpy.run_path(str(Path(__file__).with_name("evaluate_attention_abc.py")))
labels, controls = helpers["label_lookup"](dataset)
with np.load(args.output / "raw_generation.npz") as arrays:
    result = helpers["generation_summary"](arrays["prompts"], arrays["generated"], labels, controls)
original = json.loads((args.output / "generation.json").read_text())
result.update(
    documents=original["documents"],
    source_pages=original["source_pages"],
    review_git_commit=review_metadata["git_commit"],
    raw_generation_sha256=receipt["outputs"]["raw_generation.npz"],
)
write_json(args.output / "generation_review.json", result)
print(json.dumps(result["summary"]))
