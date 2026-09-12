"""Tune real GPU5 workloads before selecting a dual-decoder training configuration."""

import argparse
import gc
import json
import runpy
import subprocess
from pathlib import Path

import torch

from hansgpt_research.glyph_lm import GlyphSequenceDataset
from hansgpt_research.train_attention_glyph_lm import check_gpu
from hansgpt_research.train_glyph_lm import sequence_lengths, verify_data_readiness, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=3)
args = parser.parse_args()
if args.output.exists() or args.steps < 1:
    raise ValueError("Use a fresh output and a positive measurement count")
if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
    raise RuntimeError("Commit source before benchmarking")
device = torch.device("cuda:0")
check_gpu("D", device)
config = json.loads(Path("configs/experiments/hansgpt_dual_decoder.json").read_text())
data = Path("data/processed/modelscope_zhwiki_full_v1")
identity = verify_data_readiness(
    data, require_full_snapshot=True, expected_provider="ModelScope", expected_shards=6
)
if identity["manifest_sha256"] != config["data_requirements"]["manifest_sha256"]:
    raise ValueError("Benchmark corpus differs from the declared dataset")
dataset = GlyphSequenceDataset(data, "train", 256)
indices = (sequence_lengths(dataset) == 256).nonzero()[0][:64]
samples = [dataset[int(i)] for i in indices]
if len(samples) != 64:
    raise ValueError("Need 64 real full-length chunks")
run_case = runpy.run_path(str(Path(__file__).with_name("benchmark_attention_abc.py")))["run_case"]
cases = [
    {"name": "batch16_head128", "batch_size": 16, "head_chunk_size": 128, "checkpointing": False},
    {"name": "batch32_head128", "batch_size": 32, "head_chunk_size": 128, "checkpointing": False},
    {"name": "batch32_head256", "batch_size": 32, "head_chunk_size": 256, "checkpointing": False},
    {"name": "batch32_head512", "batch_size": 32, "head_chunk_size": 512, "checkpointing": False},
    {
        "name": "batch32_fused",
        "batch_size": 32,
        "head_chunk_size": 256,
        "checkpointing": False,
        "fused_adamw": True,
    },
    {
        "name": "batch48",
        "batch_size": 48,
        "head_chunk_size": 256,
        "checkpointing": False,
        "fused_adamw": True,
    },
    {
        "name": "batch64_recompute",
        "batch_size": 64,
        "head_chunk_size": 256,
        "checkpointing": True,
        "fused_adamw": True,
    },
]
report = {
    "config": config,
    "gpu": torch.cuda.get_device_name(device),
    "pytorch": torch.__version__,
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "data_manifest_sha256": identity["manifest_sha256"],
    "sample_indices": indices.tolist(),
    "protocol": "64 real 256-grid chunks; 2 warmup updates; includes transfer/backward/update",
    "cases": [],
}
for case in cases:
    try:
        result = run_case(config, samples, case, device, args.steps)
    except torch.OutOfMemoryError:
        result = {"case": case, "error": "cuda_out_of_memory"}
    report["cases"].append(result)
    write_json(args.output, report)
    print(json.dumps(result), flush=True)
    gc.collect()
    torch.cuda.empty_cache()
