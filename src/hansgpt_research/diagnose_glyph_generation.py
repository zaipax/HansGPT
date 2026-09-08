"""Fixed-page D0/D1 diagnostics for raw binary glyph generation; inference only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from hansgpt_research.evaluate_glyph_lm import (
    BinaryMetrics,
    category_bitmap_sets,
    contact_sheet,
    exact_category_memberships,
    glyph_categories,
    nearest_glyphs,
)
from hansgpt_research.glyph_lm import GlyphGPT, GlyphSequenceDataset, ModelConfig
from hansgpt_research.train_glyph_lm import autocast_context, runtime_metadata, sha256, write_json


def stable_hash(seed: int, kind: str, identifier: str) -> str:
    return hashlib.sha256(f"{seed}:{kind}:{identifier}".encode()).hexdigest()


def select_prompt_records(
    records,
    offsets: np.ndarray,
    *,
    count: int,
    max_prompt_length: int,
    seed: int,
    split: str,
    exclude_pages: set[str] | None = None,
) -> tuple[list[dict], dict]:
    """Select at most one eligible paragraph per source page by two fixed hash rankings."""
    excluded = set(exclude_pages or ())
    candidates = {}
    rows = 0
    for index, record in enumerate(records):
        rows += 1
        page_id = str(record["source_page_id"])
        # The three old v1 test prompts all lie within its first three stored paragraphs.
        # Excluding their entire pages prevents a later paragraph of that page reentering.
        if split == "test" and index < 3:
            excluded.add(page_id)
        length = int(offsets[index + 1] - offsets[index])
        if length < max_prompt_length + 8:
            continue
        sample_id = str(record["sample_id"])
        candidate = {
            "document_index": index,
            "source_page_id": page_id,
            "sample_id": sample_id,
            "stored_tiles_including_bos_eos": length,
            "text_sha256": record["text_sha256"],
            "paragraph_rank": stable_hash(seed, "paragraph", sample_id),
            "page_rank": stable_hash(seed, "page", page_id),
        }
        if (
            page_id not in candidates
            or candidate["paragraph_rank"] < candidates[page_id]["paragraph_rank"]
        ):
            candidates[page_id] = candidate
    if rows != len(offsets) - 1:
        raise ValueError("Parquet rows do not align with document offsets")
    eligible = sorted(
        (record for page, record in candidates.items() if page not in excluded),
        key=lambda record: (record["page_rank"], record["source_page_id"]),
    )
    if count <= 0 or len(eligible) < count:
        raise ValueError(f"Requested {count} pages; only {len(eligible)} eligible pages remain")
    selected = eligible[:count]
    for index, record in enumerate(selected):
        record["prompt_index"] = index
        record["cohort"] = (
            "final" if split == "test" else "development" if index < count // 2 else "audit"
        )
    return selected, {
        "scanned_paragraphs": rows,
        "eligible_pages": len(eligible),
        "excluded_page_ids": sorted(excluded),
        "eligibility": f"stored length >= {max_prompt_length + 8}; eight valid future targets",
        "selection": "one paragraph per page; minimum paragraph hash then page hash",
        "split": split,
        "test_policy": "test is final only, never a decoding development cohort",
    }


def load_model(checkpoint: dict, device: torch.device):
    metadata = checkpoint["metadata"]
    config = metadata["config"]
    if metadata.get("model_family") == "structured_glyph_gpt" or "head" in metadata:
        from hansgpt_research.structured_glyph_lm import StructuredGlyphGPT

        head = metadata.get("head", config.get("head", {}))
        model = StructuredGlyphGPT(
            ModelConfig.from_dict(config["model"]), components=head.get("components", 1)
        )
    else:
        model = GlyphGPT(ModelConfig.from_dict(config["model"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


class IndependentPrediction:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits.float()
        if not bool(torch.isfinite(self.logits).all()):
            raise FloatingPointError("Nonfinite generation logits")

    def decode(self, *, threshold: float, strategy: str, generator=None) -> torch.Tensor:
        probability = self.logits.sigmoid()
        if strategy == "sample_pixels":
            return torch.bernoulli(probability, generator=generator).to(torch.uint8)
        # A single Bernoulli component has no mixture component choice to sample.
        return (probability >= threshold).to(torch.uint8)

    def nll(self, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            self.logits, targets.float(), reduction="none"
        ).mean(dim=(-3, -2, -1))


def next_prediction(model, glyphs: torch.Tensor, cache):
    if hasattr(model, "forward_hidden") and hasattr(model, "distribution"):
        hidden, cache = model.forward_hidden(
            glyphs, past_key_values=cache, use_cache=True, return_cache=True
        )
        return model.distribution(hidden[:, -1:]), cache
    logits, cache = model(glyphs, past_key_values=cache, use_cache=True, return_cache=True)
    return IndependentPrediction(logits[:, -1:]), cache


def assert_binary(glyphs: torch.Tensor) -> None:
    if (
        glyphs.ndim != 5
        or tuple(glyphs.shape[-3:]) != (1, 32, 32)
        or glyphs.dtype != torch.uint8
        or not bool(((glyphs == 0) | (glyphs == 1)).all())
    ):
        raise ValueError("Expected uint8 [batch,sequence,1,32,32] strict binary glyphs")


@torch.inference_mode()
def rollout(
    model,
    prompt: torch.Tensor,
    reference: torch.Tensor,
    *,
    steps: int,
    threshold: float,
    strategy: str,
    seed: int,
    precision: str,
    eos_glyph: torch.Tensor | None = None,
    teacher_forcing: bool = False,
) -> dict:
    """D0 raw feedback and D1 teacher forcing share the exact decoding implementation."""
    assert_binary(prompt)
    assert_binary(reference)
    if prompt.shape[0] != 1 or reference.shape[0] != 1:
        raise ValueError("One prompt per rollout keeps stopping and denominators unambiguous")
    if teacher_forcing and steps > reference.shape[1]:
        raise ValueError("Teacher forcing requires real future targets, never padded references")
    generator = torch.Generator(device=prompt.device).manual_seed(seed)
    context, model_input, cache = prompt, prompt, None
    tiles, reference_nll = [], []
    eos_step = None
    for index in range(steps):
        with autocast_context(prompt.device, precision):
            distribution, next_cache = next_prediction(model, model_input, cache)
            if index < min(8, reference.shape[1]):
                nll = distribution.nll(reference[:, index : index + 1])
                if not bool(torch.isfinite(nll).all()):
                    raise FloatingPointError("Nonfinite paired conditional NLL")
                reference_nll.append(float(nll.mean()))
            tile = distribution.decode(threshold=threshold, strategy=strategy, generator=generator)
        assert_binary(tile)
        tiles.append(tile)
        if eos_glyph is not None and bool((tile == eos_glyph).all()):
            eos_step = index + 1
            break
        feedback = reference[:, index : index + 1] if teacher_forcing else tile
        context = torch.cat((context, feedback), dim=1)
        if context.shape[1] <= model.config.max_position_embeddings:
            cache, model_input = next_cache, feedback
        else:
            cache = None
            model_input = context[:, -model.config.max_position_embeddings :]
    generated = torch.cat(tiles, dim=1) if tiles else prompt[:, :0]
    return {
        "tiles": generated,
        "reference_nll_per_pixel": reference_nll,
        "eos_step": eos_step,
        "terminated_by_eos": eos_step is not None,
        "teacher_forcing": teacher_forcing,
    }


def longest_true_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def repetition_metrics(bitmaps: np.ndarray, *, near_hamming: int = 4, max_period: int = 4) -> dict:
    flat = bitmaps.reshape(len(bitmaps), 1024)
    if not len(flat):
        return {
            "adjacent_exact_repeat_rate": None,
            "adjacent_near_repeat_rate": None,
            "max_exact_run": 0,
            "unique_bitmaps": 0,
            "has_short_cycle": False,
            "longest_short_cycle_tiles": 0,
        }
    adjacent = np.count_nonzero(flat[1:] != flat[:-1], axis=1)
    longest_cycle = 0
    for period in range(1, min(max_period, len(flat) - 1) + 1):
        distances = np.count_nonzero(flat[period:] != flat[:-period], axis=1)
        matched = longest_true_run(distances <= near_hamming)
        if matched:
            longest_cycle = max(longest_cycle, matched + period)
    return {
        "adjacent_exact_repeat_rate": float((adjacent == 0).mean()) if len(adjacent) else 0.0,
        "adjacent_near_repeat_rate": float((adjacent <= near_hamming).mean())
        if len(adjacent)
        else 0.0,
        "near_hamming_bits": near_hamming,
        "max_cycle_period": max_period,
        "max_exact_run": longest_true_run(adjacent == 0) + 1,
        "unique_bitmaps": len(np.unique(np.packbits(flat, axis=1), axis=0)),
        "has_short_cycle": longest_cycle >= 32,
        "longest_short_cycle_tiles": longest_cycle,
        "cycle_definition": (
            "period 1..configured maximum, per-tile Hamming <= configured bound, run >=32 tiles"
        ),
    }


def trim_at_eos(
    generated: torch.Tensor, eos_glyph: torch.Tensor
) -> tuple[torch.Tensor, int | None]:
    """Accept future batched generators' trailing padding without ever scoring it."""
    assert_binary(generated)
    if generated.shape[0] != 1:
        raise ValueError("Stopping audit requires one row")
    matches = (generated[0] == eos_glyph.reshape(1, 1, 32, 32)).flatten(1).all(-1)
    locations = matches.nonzero().flatten()
    stop = int(locations[0]) + 1 if len(locations) else None
    return generated[:, :stop] if stop is not None else generated, stop


class GlyphScorer:
    def __init__(
        self, dataset, device: torch.device, near_hamming: int = 4, max_cycle_period: int = 4
    ):
        self.bank = dataset.glyph_bank.to(device)
        self.gallery_ids = torch.tensor(dataset.gallery_ids, device=device, dtype=torch.long)
        self.gallery = self.bank[self.gallery_ids]
        self.categories = glyph_categories(dataset.inventory)
        self.categories["control"] = list(dataset.control_ids.values())
        self.categories["sentence_end"] = [
            int(identifier)
            for character, identifier in dataset.inventory["characters"].items()
            if character in "。！？"
        ]
        self.categories.update(
            {name.lower(): [identifier] for name, identifier in dataset.control_ids.items()}
        )
        self.bitmap_sets = category_bitmap_sets(dataset.glyph_bank, self.categories)
        self.labels = {
            int(identifier): character
            for character, identifier in dataset.inventory["characters"].items()
        }
        self.control_labels = {
            identifier: f"<{name}>" for name, identifier in dataset.control_ids.items()
        }
        packed = np.packbits(
            dataset.glyph_bank.numpy().reshape(len(dataset.glyph_bank), 1024), axis=1
        )
        self.exact_labels = defaultdict(list)
        for index, bitmap in enumerate(packed):
            self.exact_labels[bitmap.tobytes()].append(
                self.labels.get(index, self.control_labels.get(index))
            )
        self.near_hamming = near_hamming
        self.max_cycle_period = max_cycle_period

    @torch.inference_mode()
    def score(self, tiles: torch.Tensor, *, requested_horizon: int) -> dict:
        assert_binary(tiles)
        generated = tiles[0]
        count = len(generated)
        memberships = exact_category_memberships(generated, self.bitmap_sets)
        ids, distances = nearest_glyphs(generated, self.gallery, self.gallery_ids)
        content = memberships["han_only"] | memberships["punctuation_only"]
        any_legal = content | memberships["control"]
        terminal_eos = bool(count and memberships["eos"][-1])
        body_count = count - int(terminal_eos)
        illegal_positions = np.flatnonzero(~content[:body_count])
        legal_prefix = int(illegal_positions[0]) if len(illegal_positions) else body_count
        sentence_ends = np.flatnonzero(memberships["sentence_end"][:legal_prefix])
        legal_sentence_prefix = int(sentence_ends[-1]) + 1 if len(sentence_ends) else 0
        array = generated.cpu().numpy()
        packed = np.packbits(array.reshape(count, 1024), axis=1)
        exact_text = []
        for bitmap in packed:
            labels = self.exact_labels.get(bitmap.tobytes(), [])
            exact_text.append(
                labels[0] if len(labels) == 1 else "{" + "|".join(labels) + "}" if labels else "□"
            )
        result = {
            "requested_horizon": requested_horizon,
            "observed_tiles": count,
            "reached_horizon": count >= requested_horizon,
            "exact_content_legal_rate": float(content.mean()) if count else None,
            "exact_any_library_rate": float(any_legal.mean()) if count else None,
            "entire_horizon_content_legal": bool(count == requested_horizon and content.all()),
            "content_positions_before_terminal_eos": body_count,
            "content_legal_rate_before_terminal_eos": float(content[:body_count].mean())
            if body_count
            else None,
            "all_pre_eos_content_legal": bool(body_count and content[:body_count].all()),
            "terminal_eos": terminal_eos,
            "initial_legal_content_prefix_tiles": legal_prefix,
            "first_illegal_body_position": legal_prefix + 1 if legal_prefix < body_count else None,
            "legal_sentence_prefix_tiles": legal_sentence_prefix,
            "has_complete_legal_sentence_prefix": bool(legal_sentence_prefix),
            "complete_legal_sentence_before_eos": bool(
                terminal_eos and body_count >= 8 and legal_sentence_prefix == body_count
            ),
            "sentence_scope": (
                "exact content bitmaps and ending punctuation only; not semantic fluency"
            ),
            "mean_nearest_hamming_bits": float(distances[:, 0].float().mean()) if count else None,
            "median_nearest_hamming_bits": float(distances[:, 0].float().median())
            if count
            else None,
            "p95_nearest_hamming_bits": float(torch.quantile(distances[:, 0].float(), 0.95))
            if count
            else None,
            "foreground_rate": float(generated.float().mean()) if count else None,
            "ends_with_exact_sentence_punctuation": bool(
                memberships["sentence_end"][body_count - 1]
            )
            if body_count
            else False,
            "exact_transcription_diagnostic_only": "".join(exact_text),
            "nearest_transcription_diagnostic_only": "".join(
                self.labels[int(identifier)] for identifier in ids[:, 0].cpu()
            ),
            "gallery_size": len(self.gallery_ids),
            "gallery_scope": "all content glyphs; controls excluded; no projection enters feedback",
            **repetition_metrics(
                array, near_hamming=self.near_hamming, max_period=self.max_cycle_period
            ),
        }
        for name, mask in memberships.items():
            result[f"exact_{name}_count"] = int(mask.sum())
            result[f"exact_{name}_rate"] = float(mask.mean()) if count else None
        return result


@torch.inference_mode()
def paired_metrics(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    nll: list[float],
    scorer: GlyphScorer,
    reference_ids: torch.Tensor,
) -> dict:
    count = min(prediction.shape[1], reference.shape[1], 8)
    metrics = BinaryMetrics()
    losses = torch.tensor(nll[:count], device=prediction.device) * 1024
    metrics.add(prediction[0, :count], reference[0, :count], losses)
    controls = torch.tensor(list(scorer.control_labels), device=prediction.device)
    content = ~torch.isin(reference_ids[:count], controls)
    nearest, _ = nearest_glyphs(prediction[0, :count][content], scorer.gallery, scorer.gallery_ids)
    actual = reference_ids[:count][content]
    values = metrics.result()
    values["nll_per_pixel"] = values.pop("bce_per_pixel")
    return {
        **values,
        "content_retrieval_targets": len(actual),
        "content_top1_count": int((nearest[:, 0] == actual).sum()),
        "content_top5_count": int((nearest == actual[:, None]).any(-1).sum()),
        "per_position_nll_per_pixel": nll[:count],
        "scope": (
            "same-reference diagnostic at first eight positions; "
            "not semantic correctness after divergence"
        ),
    }


def bootstrap_mean(values: list[float], seed: int, *, samples: int = 1000) -> dict:
    if not values:
        return {"prompts": 0, "mean": None, "ci95": None}
    array = np.array(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.array(
        [generator.choice(array, size=len(array), replace=True).mean() for _ in range(samples)]
    )
    return {
        "prompts": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "confidence_unit": (
            "source-page prompt; percentile bootstrap, descriptive for small cohorts"
        ),
        "constant_observations": bool(np.all(array == array[0])),
        "ci_note": (
            "A degenerate interval from constant observed prompts does not establish "
            "zero population uncertainty."
        ),
    }


def summarize_examples(examples: list[dict], seed: int) -> list[dict]:
    groups = defaultdict(list)
    for example in examples:
        for horizon, metrics in example["horizons"].items():
            key = (
                example["cohort"],
                example["prompt_length"],
                example["threshold"],
                example["strategy"],
                example["natural_stop"],
                horizon,
            )
            groups[key].append((example, metrics))
    summaries = []
    names = (
        "exact_content_legal_rate",
        "exact_han_only_rate",
        "exact_punctuation_only_rate",
        "exact_control_rate",
        "content_legal_rate_before_terminal_eos",
        "initial_legal_content_prefix_tiles",
        "legal_sentence_prefix_tiles",
        "has_complete_legal_sentence_prefix",
        "complete_legal_sentence_before_eos",
        "mean_nearest_hamming_bits",
        "adjacent_exact_repeat_rate",
        "adjacent_near_repeat_rate",
        "entire_horizon_content_legal",
        "has_short_cycle",
        "reached_horizon",
        "ends_with_exact_sentence_punctuation",
    )
    for key, entries in sorted(groups.items()):
        cohort, prompt_length, threshold, strategy, natural_stop, horizon = key
        summaries.append(
            {
                "cohort": cohort,
                "prompt_length": prompt_length,
                "threshold": threshold,
                "strategy": strategy,
                "natural_stop": natural_stop,
                "horizon": int(horizon),
                "metrics": {
                    name: bootstrap_mean(
                        [float(m[name]) for _, m in entries if m.get(name) is not None], seed
                    )
                    for name in names
                },
                "natural_eos_rate": sum(e["terminated_by_eos"] for e, _ in entries) / len(entries),
                "early_eos_before_eight_rate": sum(e["early_eos_before_eight"] for e, _ in entries)
                / len(entries),
                "hit_length_cap_rate": sum(e["hit_length_cap"] for e, _ in entries) / len(entries),
                "stopping_aggregation_scope": {
                    "natural_eos_rate": "whole rollout; not EOS incidence by this horizon",
                    "early_eos_before_eight_rate": "whole rollout; EOS before generated position 8",
                    "hit_length_cap_rate": "whole rollout; no EOS before the requested maximum",
                },
                "paired": {
                    name: bootstrap_mean(
                        [
                            e["paired"][horizon]["raw_feedback"][name]
                            - e["paired"][horizon]["teacher_forced"][name]
                            for e, _ in entries
                        ],
                        seed,
                    )
                    for name in ("nll_per_pixel", "foreground_f1", "exact_bitmap_match")
                }
                if horizon in {"1", "2", "4", "8"}
                else {},
                "paired_difference_direction": "raw feedback minus teacher forcing",
            }
        )
    return summaries


def parse_csv(value: str, conversion):
    return sorted(set(conversion(item.strip()) for item in value.split(",") if item.strip()))


def diagnose(args: argparse.Namespace) -> None:
    prompt_lengths = parse_csv(args.prompt_lengths, int)
    thresholds = parse_csv(args.thresholds, float)
    if (
        not prompt_lengths
        or min(prompt_lengths) < 1
        or not thresholds
        or any(not 0 < threshold < 1 for threshold in thresholds)
        or args.max_new < 8
        or args.prompt_count < 2
        or not 0 <= args.near_hamming <= 1024
        or args.max_cycle_period < 1
    ):
        raise ValueError(
            "Need >=2 prompts, positive prompt lengths, max-new>=8 and thresholds in (0,1)"
        )
    output, data = Path(args.output), Path(args.data)
    if output.exists():
        raise FileExistsError("Choose a new versioned diagnostic output directory")
    device = torch.device(args.device)
    if device.type == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES=0 for the authorized GPU")
    torch.set_float32_matmul_precision("highest")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["metadata"]["config"]
    metadata = runtime_metadata(config, data, device, mode="diagnostic")
    for key in ("data_sha256", "data_manifest_sha256"):
        if checkpoint["metadata"].get(key) != metadata[key]:
            raise ValueError(f"Diagnostic dataset differs from checkpoint: {key}")
    parquet = data / f"{args.split}.parquet"
    parquet_hash = sha256(parquet)
    if (
        metadata["data_manifests"]["manifest.json"]["content"]["output_sha256"].get(parquet.name)
        != parquet_hash
    ):
        raise ValueError("Prompt source Parquet differs from finalized dataset manifest")
    dataset = GlyphSequenceDataset(data, args.split, config["training"]["sequence_length"])
    columns = ["source_page_id", "sample_id", "text_sha256"]
    records = (
        row
        for batch in pq.ParquetFile(parquet).iter_batches(columns=columns, batch_size=1024)
        for row in batch.to_pylist()
    )
    excluded = set()
    if args.exclude_prompt_manifest:
        previous = json.loads(Path(args.exclude_prompt_manifest).read_text("utf-8"))
        excluded.update(record["source_page_id"] for record in previous["prompts"])
    prompts, selection = select_prompt_records(
        records,
        dataset.offsets,
        count=args.prompt_count,
        max_prompt_length=max(prompt_lengths),
        seed=args.seed,
        split=args.split,
        exclude_pages=excluded,
    )
    output.mkdir(parents=True)
    manifest = {
        "prompts": prompts,
        "selection": selection,
        "seed": args.seed,
        "data_manifest_sha256": metadata["data_manifest_sha256"],
        "source_parquet_sha256": parquet_hash,
        "prompt_lengths": prompt_lengths,
    }
    write_json(output / "prompt_manifest.json", manifest)
    metadata.update(
        checkpoint_sha256=sha256(Path(args.checkpoint)),
        checkpoint_git_commit=checkpoint["metadata"]["git_commit"],
        checkpoint_model_family=checkpoint["metadata"].get("model_family", "glyph_gpt_v1"),
        arguments=vars(args),
        prompt_manifest_sha256=sha256(output / "prompt_manifest.json"),
        inference_only=True,
        metric_scope=(
            "raw binary grids; nearest text is diagnostic; no semantic acceptance inferred"
        ),
        seed=args.seed,
        checkpoint_training_seed=config["training"]["seed"],
    )
    metadata["model_revision"] = metadata["checkpoint_sha256"]
    write_json(output / "metadata.json", metadata)
    model = load_model(checkpoint, device)
    scorer = GlyphScorer(dataset, device, args.near_hamming, args.max_cycle_period)
    eos = dataset.glyph_bank[dataset.control_ids["EOS"]].to(device).reshape(1, 1, 1, 32, 32)
    horizons = sorted({h for h in (1, 2, 4, 8, 32, 128, args.max_new) if h <= args.max_new})
    examples = []
    try:
        for record in prompts:
            document = record["document_index"]
            identifiers = torch.tensor(
                np.array(
                    dataset.tokens[dataset.offsets[document] : dataset.offsets[document + 1]],
                    dtype=np.int64,
                ),
                device=device,
            )
            glyphs = scorer.bank[identifiers].unsqueeze(0)
            for prompt_length in prompt_lengths:
                prompt, reference = glyphs[:, :prompt_length], glyphs[:, prompt_length:]
                for threshold in thresholds:
                    example_seed = int(
                        stable_hash(args.seed, f"draw:{prompt_length}", record["sample_id"])[:15],
                        16,
                    )
                    raw = rollout(
                        model,
                        prompt,
                        reference,
                        steps=args.max_new,
                        threshold=threshold,
                        strategy=args.strategy,
                        seed=example_seed,
                        precision=config["training"]["precision"],
                        eos_glyph=eos if args.natural_stop else None,
                    )
                    teacher = rollout(
                        model,
                        prompt,
                        reference,
                        steps=8,
                        threshold=threshold,
                        strategy=args.strategy,
                        seed=example_seed,
                        precision=config["training"]["precision"],
                        teacher_forcing=True,
                    )
                    name = (
                        f"p{record['prompt_index']:04d}_l{prompt_length}_"
                        f"t{threshold:g}_{args.strategy}"
                    )
                    generated, stop = (
                        trim_at_eos(raw["tiles"], eos)
                        if args.natural_stop
                        else (raw["tiles"], None)
                    )
                    example = {
                        **record,
                        "prompt_length": prompt_length,
                        "threshold": threshold,
                        "strategy": args.strategy,
                        "natural_stop": args.natural_stop,
                        "seed": example_seed,
                        "terminated_by_eos": stop is not None,
                        "eos_step": stop,
                        "reference_eos_step": reference.shape[1],
                        "eos_before_reference_end": stop is not None and stop < reference.shape[1],
                        "reference_stop_note": "single-reference timing, not semantic correctness",
                        "early_eos_before_eight": stop is not None and stop < 8,
                        "hit_length_cap": stop is None and generated.shape[1] == args.max_new,
                        "npz": name + ".npz",
                        "contact_sheet": name + ".png",
                        "teacher_contact_sheet": name + "_teacher.png",
                        "horizons": {},
                        "paired": {},
                        "feedback": "raw binary prediction, never projected",
                    }
                    for horizon in horizons:
                        example["horizons"][str(horizon)] = scorer.score(
                            generated[:, :horizon], requested_horizon=horizon
                        )
                    for horizon in (1, 2, 4, 8):
                        valid = min(horizon, generated.shape[1])
                        reference_ids = identifiers[prompt_length : prompt_length + valid]
                        example["paired"][str(horizon)] = {
                            "aligned_positions": valid,
                            "teacher_forced": paired_metrics(
                                teacher["tiles"][:, :valid],
                                reference[:, :valid],
                                teacher["reference_nll_per_pixel"][:valid],
                                scorer,
                                reference_ids,
                            ),
                            "raw_feedback": paired_metrics(
                                generated[:, :valid],
                                reference[:, :valid],
                                raw["reference_nll_per_pixel"][:valid],
                                scorer,
                                reference_ids,
                            ),
                        }
                    np.savez_compressed(
                        output / example["npz"],
                        prompt=prompt.cpu().numpy(),
                        generated=generated.cpu().numpy(),
                        teacher_forced=teacher["tiles"].cpu().numpy(),
                        reference_next_eight=reference[:, :8].cpu().numpy(),
                    )
                    contact_sheet(output / example["contact_sheet"], prompt[0], generated[0])
                    contact_sheet(
                        output / example["teacher_contact_sheet"], prompt[0], teacher["tiles"][0]
                    )
                    write_json(output / (name + ".json"), example)
                    examples.append(example)
                    print(
                        json.dumps(
                            {
                                "completed_examples": len(examples),
                                "cohort": record["cohort"],
                                "prompt": record["prompt_index"],
                                "threshold": threshold,
                            }
                        ),
                        flush=True,
                    )
        summaries = summarize_examples(examples, args.seed)
        write_json(
            output / "summary.json",
            {
                "status": "complete",
                "examples": len(examples),
                "cohorts": summaries,
                "semantic_evaluation": "not performed; raw images require blind review",
                "prompt_count": len(prompts),
                "prompt_manifest_sha256": sha256(output / "prompt_manifest.json"),
            },
        )
        failures = [
            {
                "prompt": example["prompt_index"],
                "cohort": example["cohort"],
                "prompt_length": example["prompt_length"],
                "threshold": example["threshold"],
                "json": example["npz"].replace(".npz", ".json"),
                "image": example["contact_sheet"],
            }
            for example in examples
            if flagged_example(example, args.max_new)
        ]
        write_json(
            output / "failure_index.json",
            {
                "failures": failures,
                "all_examples_retained": True,
                "definition": (
                    "fixed: incomplete/noncontent full horizon; natural: missing/early EOS "
                    "or illegal body; either: >=32-tile short cycle; not semantic judgment"
                ),
            },
        )
        write_json(
            output / "diagnostics_complete.json",
            {
                "status": "complete",
                "metadata_sha256": sha256(output / "metadata.json"),
                "summary_sha256": sha256(output / "summary.json"),
                "failure_index_sha256": sha256(output / "failure_index.json"),
                "examples": len(examples),
            },
        )
    except BaseException as error:
        write_json(
            output / "diagnostics_failed.json",
            {
                "status": "failed",
                "completed_examples": len(examples),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def flagged_example(example: dict, horizon: int) -> bool:
    metrics = example["horizons"][str(horizon)]
    if metrics["has_short_cycle"]:
        return True
    if example["natural_stop"]:
        return (
            not example["terminated_by_eos"]
            or example["early_eos_before_eight"]
            or not metrics["all_pre_eos_content_legal"]
        )
    return not metrics["entire_horizon_content_legal"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--prompt-count", type=int, default=64)
    parser.add_argument("--prompt-lengths", default="16,64")
    parser.add_argument("--thresholds", default="0.3,0.45,0.5")
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument(
        "--strategy",
        choices=("mode_threshold", "sample_threshold", "sample_pixels"),
        default="mode_threshold",
    )
    parser.add_argument("--natural-stop", action="store_true")
    parser.add_argument("--exclude-prompt-manifest")
    parser.add_argument("--near-hamming", type=int, default=4)
    parser.add_argument("--max-cycle-period", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    diagnose(parser.parse_args())


if __name__ == "__main__":
    main()
