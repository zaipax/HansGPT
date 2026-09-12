"""Report paragraph and actual isolated-chunk lengths after corpus verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def summarize(lengths):
    edges = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    bins, low = [], 1
    for high in [*edges, max(4097, int(lengths.max()))]:
        selected = (lengths >= low) & (lengths <= high)
        bins.append({"min": low, "max": high, "count": int(selected.sum()),
                     "percent": float(selected.mean() * 100)})
        low = high + 1
    return {
        "count": len(lengths), "min": int(lengths.min()), "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "percentiles": dict(zip(["50", "75", "90", "95", "99"],
                                np.percentile(lengths, [50, 75, 90, 95, 99]).tolist())),
        "bins": bins,
    }


def corpus_report(root, contexts):
    receipt = json.loads((root / "verification.json").read_text())
    if not receipt.get("passed"):
        raise ValueError(f"Corpus verification did not pass: {root}")
    manifest_hash = digest(root / "manifest.json")
    if receipt.get("manifest_sha256") != manifest_hash:
        raise ValueError(f"Verification does not match manifest: {root}")
    result = {"path": str(root), "manifest_sha256": manifest_hash, "splits": {}}
    for split in ["train", "validation", "test"]:
        path = root / f"{split}.offsets.npy"
        content = np.diff(np.load(path, allow_pickle=False)) - 2
        if not len(content) or np.any(content < 1):
            raise ValueError("Expected nonempty BOS/content/EOS paragraphs")
        entry = {"offsets_sha256": digest(path), "paragraphs": summarize(content), "contexts": {}}
        for ctx in contexts:
            counts = content // ctx + 1
            lengths = np.full(int(counts.sum()), ctx, dtype=np.int64)
            lengths[np.cumsum(counts) - 1] = content % ctx + 1
            targets = int((content + 1).sum())
            assert int(lengths.sum()) == targets
            entry["contexts"][str(ctx)] = {
                **summarize(lengths), "valid_targets": targets,
                "full_chunks": int((lengths == ctx).sum()),
                "full_chunk_percent": float((lengths == ctx).mean() * 100),
                "full_chunk_target_percent": float((lengths == ctx).sum() * ctx / targets * 100),
            }
        result["splits"][split] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-verification", action="store_true")
    args = parser.parse_args()
    while not (args.data / "verification.json").exists():
        if not args.wait_for_verification:
            raise FileNotFoundError("Corpus has not completed verification")
        print(datetime.now(UTC).isoformat(), "Waiting for verified corpus", flush=True)
        time.sleep(60)
    contexts = [256, 512, 1024, 2048]
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "definitions": {
            "paragraph_length": "Han plus punctuation; excludes BOS/EOS",
            "chunk_length": "Valid next-grid targets including EOS; excludes padding",
            "contexts": "Independent simulations; paragraphs are not packed together",
        },
        "corpus": corpus_report(args.data, contexts),
    }
    if args.compare:
        report["comparison"] = corpus_report(args.compare, contexts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_suffix(".tmp")
    pending.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    pending.replace(args.output)
    print("Completed length report:", args.output, flush=True)


if __name__ == "__main__":
    main()
