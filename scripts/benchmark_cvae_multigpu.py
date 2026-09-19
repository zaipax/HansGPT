"""Synchronous data-parallel CVAE throughput including CPU prefetch and NCCL.

Uses explicit post-backward FP32 gradient buckets because FixedBackward performs
several detached/chunked backward passes. This is not an overlapped DDP benchmark.
"""

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import regex
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from hansgpt_research.conditional_glyph_vae import ConditionalGlyphVAE
from hansgpt_research.cvae_fixed_step import FixedBackward, install_xformers
from hansgpt_research.cvae_search_training import SharedGlyphCollator
from hansgpt_research.packed_glyph_data import PackedGlyphSequenceDataset
from hansgpt_research.train_glyph_lm import (
    SortishEpochSampler, runtime_metadata, sequence_lengths, write_json,
)
from hansgpt_research.train_structured_glyph_lm import complete_optimizer_step, optimizer_for


def sync_gradients(parameters, buffer):
    """SUM globally weighted scaled gradients; bounded FP32 scratch space."""
    views, used = [], 0

    def flush():
        if not views:
            return
        dist.all_reduce(buffer[:used])
        for grad, start, size in views:
            grad.copy_(buffer[start:start + size])

    for parameter in parameters:
        if parameter.grad is None:
            raise RuntimeError("Missing gradient: cannot silently change collective order")
        grad = parameter.grad.view(-1)
        for start in range(0, grad.numel(), buffer.numel()):
            piece = grad[start:start + buffer.numel()]
            if used + piece.numel() > buffer.numel():
                flush()
                views, used = [], 0
            buffer[used:used + piece.numel()].copy_(piece)
            views.append((piece, used, piece.numel()))
            used += piece.numel()
    flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 2:
        raise ValueError('Require measured steps and at least two warmup steps')
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    torch.set_num_threads(4)
    config = json.loads(Path('configs/experiments/lr_search_v1/gpu4.json').read_text())
    cfg = config['training']
    batch, ctx = cfg['batch_size'], cfg['sequence_length']
    torch.manual_seed(cfg['seed'])
    install_xformers()
    model = ConditionalGlyphVAE(config).to(device).train()
    params = list(model.parameters())
    if world > 1:
        for p in params:
            dist.broadcast(p.data, 0)
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        meta = runtime_metadata(config, Path(config['data']), device)
        meta.update(world_size=world, physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    nccl_environment={k:v for k,v in os.environ.items() if k.startswith('NCCL_')},
                    global_batch=batch * world, benchmark_lr=3e-4, benchmark_beta=1,
                    scope='Fresh model; post-backward NCCL SUM; no checkpoints/evaluation')
        write_json(args.output / 'metadata.json', meta)
    ds = PackedGlyphSequenceDataset(config['data'], 'train', ctx)
    sampler = SortishEpochSampler(sequence_lengths(ds), cfg['seed'], 0,
                                 cfg['sampler_reference_batch_size'], 0, 64)
    total = args.steps + args.warmup
    order = list(itertools.islice(iter(sampler), total * world * batch))
    if len(order) != total * world * batch:
        raise ValueError('Insufficient data')
    local_order = [index for step in range(total) for index in
                   order[(step * world + rank) * batch:(step * world + rank + 1) * batch]]
    han_lookup = torch.zeros(len(ds.glyph_bank), dtype=torch.bool)
    for char, index in ds.inventory['characters'].items():
        if regex.fullmatch(r'[\p{Unified_Ideograph}〇]', char):
            han_lookup[int(index)] = True
    loader = DataLoader(ds, batch_size=batch, sampler=local_order, num_workers=2,
                        pin_memory=True, collate_fn=SharedGlyphCollator(batch, ctx),
                        multiprocessing_context='spawn',
                        generator=torch.Generator().manual_seed(cfg['seed']))
    iterator = iter(loader)
    optimizer = optimizer_for(model, cfg)
    scaler = torch.amp.GradScaler('cuda')
    scaler.scale(torch.ones((), device=device))
    backward = FixedBackward(model, batch, ctx, cfg['head_chunk_size'], compiled=True)
    buffer = torch.empty(8 * 1024 * 1024, device=device) if world > 1 else None
    generator = torch.Generator(device=device).manual_seed(cfg['seed'] + 100 + rank)
    beta = torch.ones((), device=device)
    rows = []
    start = None
    for i in range(total):
        if i == args.warmup:
            torch.cuda.synchronize()
            if world > 1:
                dist.barrier()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
        tick = time.perf_counter()
        cpu = next(iterator)
        counts = torch.tensor([int(cpu['mask'].sum()),
                               int(han_lookup[cpu['target_ids']][cpu['mask'].bool()].sum())],
                              device=device, dtype=torch.long)
        local_targets = counts[0].clone()
        if world > 1:
            dist.all_reduce(counts)
        data = {k: cpu[k].to(device, non_blocking=True)
                for k in ['tiles', 'x_index', 'y_index', 'targets', 'mask']}
        # FixedBackward divides by local targets. Weight before global SUM so
        # unequal masks still match the global per-valid-pixel objective.
        scale = scaler._get_scale_async() * local_targets / counts[0]
        noise = torch.randn(batch * ctx, config['vae']['latent_dim'], device=device,
                            generator=generator)
        optimizer.zero_grad(set_to_none=True)
        losses = backward(**data, noise=noise, scale=scale, beta=beta)
        torch.cuda.synchronize()
        comm_start = time.perf_counter()
        if world > 1:
            sync_gradients(params, buffer)
        torch.cuda.synchronize()
        comm_seconds = time.perf_counter() - comm_start
        update = complete_optimizer_step(optimizer, scaler, params, 1.0)
        if not update['succeeded'] or not bool(torch.isfinite(losses).all()):
            raise FloatingPointError('Invalid update; benchmark aborted')
        torch.cuda.synchronize()
        row = dict(step=i, seconds=time.perf_counter() - tick, comm_seconds=comm_seconds,
                   targets=int(counts[0]), han=int(counts[1]))
        if i >= args.warmup:
            rows.append(row)
        if rank == 0 and (i % 10 == 0 or i == total - 1):
            print(json.dumps(row), flush=True)
    elapsed = torch.tensor(time.perf_counter() - start, dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    # Full elementwise agreement, one parameter at a time, outside timing.
    max_diff = torch.zeros((), device=device)
    if world > 1:
        for p in params:
            reference = p.detach().clone()
            dist.broadcast(reference, 0)
            max_diff = torch.maximum(max_diff, (p.detach() - reference).abs().max())
        dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
        if max_diff.item() != 0:
            raise AssertionError('Replicas diverged')
    result = dict(status='passed', rank=rank, world_size=world, steps=args.steps,
                  global_batch=batch * world, seconds=float(elapsed),
                  targets_per_second=sum(r['targets'] for r in rows) / float(elapsed),
                  han_per_second=sum(r['han'] for r in rows) / float(elapsed),
                  mean_communication_seconds=sum(r['comm_seconds'] for r in rows)/len(rows),
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  replica_max_difference=float(max_diff), rows=rows)
    write_json(args.output / f'rank{rank}.json', result)
    if rank == 0:
        print(json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
