"""Replay the failed data window with the last saved weights, without updating them."""

import argparse
import itertools
import json
import os
from pathlib import Path

import torch

from hansgpt_research.byte_training import ByteCollator, make_byte_loss
from hansgpt_research.cvae_fixed_step import install_xformers
from hansgpt_research.packed_glyph_data import PackedGlyphSequenceDataset
from hansgpt_research.train_attention_glyph_lm import model_from_config
from hansgpt_research.train_glyph_lm import SortishEpochSampler, sequence_lengths, write_json


def stats(value):
    finite = torch.isfinite(value)
    return dict(
        shape=list(value.shape),
        dtype=str(value.dtype),
        nonfinite=int((~finite).sum()),
        max_abs=float(value.float().abs().max()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cursor", type=int, default=90040)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("Replay is restricted to physical GPU3")
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    config = saved["metadata"]["config"]
    cfg = config["training"]
    model = model_from_config(config).cuda().eval()
    model.load_state_dict(saved["model"])
    report = dict(
        checkpoint=str(args.checkpoint),
        saved_progress=saved["progress"],
        cursor=args.cursor,
        limitation="Failed 91.57M weights were not saved",
        cases=[],
    )
    del saved
    report["nonfinite_parameters"] = [
        name for name, p in model.named_parameters() if not bool(torch.isfinite(p).all())
    ]
    ds = PackedGlyphSequenceDataset(config["data"], "train", cfg["sequence_length"])
    sampler = SortishEpochSampler(
        sequence_lengths(ds),
        cfg["seed"],
        0,
        cfg["sampler_reference_batch_size"],
        args.cursor,
        cfg["sortish_pool_batches"],
    )
    indices = list(itertools.islice(sampler, cfg["batch_size"]))
    report["dataset_indices"] = indices
    cpu = ByteCollator()([ds[index] for index in indices])
    data = {key: value.cuda() for key, value in cpu.items()}
    report["valid_targets"] = int(data["mask"].sum())
    install_xformers()
    for fp16, compiled in [(True, False), (True, True), (False, False)]:
        case = dict(fp16=fp16, compiled_head=compiled, layers=[])
        handles = []
        for index, layer in enumerate(model.backbone.layers):

            def hook(module, inputs, output, index=index, case=case):
                value = output[0] if isinstance(output, tuple) else output
                case["layers"].append(dict(layer=index, **stats(value)))

            handles.append(layer.register_forward_hook(hook))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            encoded = torch.cat([model.glyph_encoder(part) for part in data["tiles"].split(256)])
            case["encoded"] = stats(encoded)
            embeddings = encoded[data["indices"]].reshape(
                cfg["batch_size"], cfg["sequence_length"], -1
            )
            hidden = model.backbone(
                inputs_embeds=embeddings, use_cache=False, return_dict=True
            ).last_hidden_state.flatten(0, 1)
            case["hidden"] = stats(hidden)
            loss = make_byte_loss(model.byte_decoder, compiled=compiled)
            values = []
            for start in range(0, len(hidden), cfg["head_chunk_size"]):
                end = start + cfg["head_chunk_size"]
                values.append(
                    float(
                        loss(
                            hidden[start:end],
                            data["byte_targets"][start:end],
                            data["mask"][start:end],
                        )
                    )
                )
            case["chunk_losses"] = values
            case["nll_per_pixel"] = sum(values) / (int(data["mask"].sum()) * 1024)
        for handle in handles:
            handle.remove()
        report["cases"].append(case)
        write_json(args.output, report)
        print(json.dumps({k: v for k, v in case.items() if k != "layers"}), flush=True)


if __name__ == "__main__":
    main()

