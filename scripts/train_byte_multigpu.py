"""Synchronous byte training with separate global LR horizon and stopping budget."""

import argparse
import json
import os
import random
import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import regex
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from hansgpt_research.train_attention_glyph_lm import model_from_config, validate_nll, generation_diagnostic
from hansgpt_research.byte_training import ByteBackward, ByteCollator, DEFAULT_ACCELERATION
from hansgpt_research.byte_resume import validate_resume, restore_rank_rng
from hansgpt_research.byte_precision import checked_byte_backward, NonfiniteByteForward
from hansgpt_research.position_schedule import position_learning_rate
from hansgpt_research.checkpoint_retention import prune_run_checkpoints
from hansgpt_research.cvae_distributed_data import PaddedDataset, RankBatches, select_global_positions
from hansgpt_research.cvae_fixed_step import install_xformers
from hansgpt_research.distributed_sync import (
    DEFAULT_GRADIENT_REDUCE_BUCKET_MIB,
    allocate_gradient_reduce_buffer,
    sync_gradients,
)
from hansgpt_research.glyph_lm import GlyphSequenceDataset
from hansgpt_research.packed_glyph_data import PackedGlyphSequenceDataset
from hansgpt_research.train_glyph_lm import (
    SortishEpochSampler, learning_rate, runtime_metadata, sequence_lengths, sha256, write_json,
)
from hansgpt_research.train_structured_glyph_lm import (
    complete_optimizer_step, optimizer_for, validation_subset,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', type=Path, help='Continue in a fresh run directory')
    parser.add_argument('--smoke-fp32', action='store_true', help='Exercise FP32 recovery memory in smoke only')
    args = parser.parse_args()
    if args.smoke_fp32 and not args.smoke:
        raise ValueError('--smoke-fp32 requires --smoke')
    config = json.loads(args.config.read_text())
    cfg = config['training']
    bucket_mib = int(cfg.get('gradient_reduce_bucket_mib', DEFAULT_GRADIENT_REDUCE_BUCKET_MIB))
    for key, value in DEFAULT_ACCELERATION.items():
        cfg.setdefault(key, value)
    if any(cfg[k] != v for k,v in DEFAULT_ACCELERATION.items()) or config['variant'] != 'C':
        raise ValueError('Requires pure C architecture and default acceleration')
    rank, local, world = [int(os.environ[k]) for k in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE']]
    if world != cfg.get('world_size',4) or cfg['gradient_accumulation_steps'] != 1:
        raise ValueError('World size must match the declared configuration; accumulation must be one')
    if cfg.get('nonfinite_fp32_retry') and world != 1:
        raise ValueError('Precision retry is currently validated only for single-rank training')
    position_learning_rate(1,cfg)
    saved = None
    resumed_progress = None
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', mmap=True, weights_only=False)
        resumed_progress = validate_resume(saved, config, world)
    initial_positions = resumed_progress['all_targets'] if resumed_progress else 0
    initial_steps = resumed_progress['steps'] if resumed_progress else 0
    if args.smoke:
        cfg.update(target_tokens=initial_positions+65536, checkpoint_every_positions=32768, validation_samples=8)
        if args.resume:
            cfg['checkpoint_every_positions'] = cfg['target_tokens']
    name = config['experiment'] + ('_smoke' if args.smoke else '_full')
    output = Path('artifacts/reports') / name
    logs = Path('artifacts/logs') / name
    checkpoints = Path('artifacts/checkpoints') / name
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])
    progress = dict(epoch=0, cursor=0, batch_skip=0, steps=0, all_targets=0, han=0, overflows=0)
    if resumed_progress:
        progress.update(resumed_progress)
    if cfg.get('nonfinite_fp32_retry'):
        progress.setdefault('fp32_retries', 0)
    started = time.monotonic()
    measurement_start = None
    measurement_targets = 0
    measurement_seconds = 0.0

    def status(phase, state='running', **extra):
        if rank == 0:
            write_json(logs / 'status.json', dict(status=state, phase=phase,
                time=datetime.now(UTC).isoformat(), progress=dict(progress),
                target_positions=cfg['target_tokens'], seconds=time.monotonic()-started,
                physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'), **extra))

    def log(kind, **extra):
        if rank == 0:
            row = dict(kind=kind, time=datetime.now(UTC).isoformat(),
                       progress=dict(progress), seconds=time.monotonic()-started, **extra)
            with (logs / 'training.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)

    try:
        install_xformers()
        model = model_from_config(config).to(device).train()
        if saved is not None:
            model.load_state_dict(saved['model'], strict=True)
        if cfg['gradient_checkpointing']:
            model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        parameters = list(model.parameters())
        dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=30))
        if rank == 0:
            for directory in [output, logs, checkpoints]:
                directory.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        status('initializing')
        for p in parameters:
            dist.broadcast(p.data, 0)
        metadata = None
        if rank == 0:
            metadata = runtime_metadata(config, Path(config['data']), device)
            metadata.update(parameters=sum(p.numel() for p in parameters), world_size=world,
                smoke_forced_fp32=args.smoke_fp32,
                precision_recovery='same-batch eager FP32' if cfg.get('nonfinite_fp32_retry') else None,
                global_batch=world*cfg['batch_size'], initialization='all weights random; no checkpoint',
                budget_unit='global successful valid prediction positions, including punctuation/controls',
                schedule_unit='global successful valid prediction positions',
                synchronization='FP32 NCCL SUM after full chunked backward, weighted by valid targets',
                gradient_reduce_bucket_mib=bucket_mib,
                nccl_environment={k:v for k,v in os.environ.items() if k.startswith('NCCL_')})
            if saved is not None:
                if metadata['data_manifest_sha256'] != saved['metadata']['data_manifest_sha256']:
                    raise ValueError('Resume dataset manifest changed')
                metadata.update(initialization='resumed model/optimizer/scaler/data cursor/RNG',
                    resume_checkpoint=str(args.resume), resume_sha256=sha256(args.resume),
                    resume_git_commit=saved['metadata']['git_commit'],
                    resume_progress=dict(progress))
            write_json(output / 'metadata.json', metadata)
        ds = PackedGlyphSequenceDataset(config['data'], 'train', cfg['sequence_length'])
        if cfg.get('lr_schedule') == 'global_cosine' and cfg['schedule_total_positions'] != ds.target_count:
            raise ValueError('Full-data LR horizon must equal the verified training target count')
        lengths = sequence_lengths(ds)
        han_lookup = torch.zeros(len(ds.glyph_bank), dtype=torch.bool)
        for char, index in ds.inventory['characters'].items():
            if regex.fullmatch(r'[\p{Unified_Ideograph}〇]', char):
                han_lookup[int(index)] = True
        subset = selection = validation = None
        if rank == 0:
            validation = GlyphSequenceDataset(config['data'], 'validation', cfg['sequence_length'])
            subset, selection = validation_subset(validation, cfg)
        optimizer = optimizer_for(model, cfg)
        scaler = torch.amp.GradScaler('cuda')
        scaler.scale(torch.ones((), device=device))
        resumed_rng = None
        if saved is not None:
            optimizer.load_state_dict(saved['optimizer'])
            scaler.load_state_dict(saved['scaler'])
            resumed_rng = saved['rank_rng'][rank]
            del saved
            saved = None
        backward = ByteBackward(model, cfg['batch_size'], cfg['sequence_length'],
                                 cfg['head_chunk_size'], compiled=True)
        buffer = allocate_gradient_reduce_buffer(device, bucket_mib)

        def validate():
            dist.barrier()
            if rank == 0:
                status('validation')
                log('validation', metrics=validate_nll(model, subset, selection, dict(cfg,num_workers=0), device))
            dist.barrier()
            model.train()

        def checkpoint(diagnostic=False):
            status('checkpoint')
            # Preserve each rank's RNG with the global cursor and partial mask.
            state = dict(torch=torch.get_rng_state(),
                         cuda=torch.cuda.get_rng_state(device), numpy=np.random.get_state(),
                         python=random.getstate())
            states = [None] * world
            dist.all_gather_object(states, state)
            # Verify all weights agree before publishing a shared checkpoint.
            difference = torch.zeros((), device=device)
            for parameter in ([] if diagnostic else parameters):
                reference = parameter.detach().clone()
                dist.broadcast(reference, 0)
                difference = torch.maximum(difference, (parameter.detach()-reference).abs().max())
            dist.all_reduce(difference, op=dist.ReduceOp.MAX)
            if float(difference) != 0:
                raise RuntimeError('Model replicas diverged')
            if rank == 0:
                prefix = 'diagnostic_positions' if diagnostic else 'positions'
                path = checkpoints / f"{prefix}_{progress['all_targets']:09d}.pt"
                if path.exists():
                    raise FileExistsError(path)
                required = sum(p.numel()*p.element_size() for p in parameters)*3 + 2**30
                if shutil.disk_usage(checkpoints).free < required:
                    raise OSError('Insufficient space for retained checkpoint')
                temporary = path.with_suffix('.pt.tmp')
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(), progress=dict(progress), rank_rng=states,
                    metadata=metadata, diagnostic_only=diagnostic), temporary)
                temporary.replace(path)
                log('checkpoint', path=str(path), sha256=sha256(path),
                    diagnostic_only=diagnostic, replica_max_difference=None if diagnostic else 0)
                if cfg.get('keep_recent_checkpoints') and not diagnostic:
                    deleted=prune_run_checkpoints(checkpoints,
                        keep_recent=cfg['keep_recent_checkpoints'],
                        retain_every=cfg['retain_every_positions'],final_position=cfg['target_tokens'])
                    if deleted:
                        log('checkpoint_retention', removed=deleted)
            dist.barrier()

        validate()
        if resumed_rng is not None:
            restore_rank_rng(resumed_rng, device)
            log('resumed', checkpoint=str(args.resume))
        while progress['all_targets'] < cfg['target_tokens']:
            sampler = SortishEpochSampler(lengths, cfg['seed'], progress['epoch'],
                cfg['sampler_reference_batch_size'], progress['cursor'], cfg['sortish_pool_batches'])
            loader = DataLoader(PaddedDataset(ds),
                batch_sampler=RankBatches(sampler, cfg['batch_size'], rank, world),
                num_workers=cfg['num_workers'], multiprocessing_context='spawn', pin_memory=True,
                collate_fn=ByteCollator(),
                generator=torch.Generator().manual_seed(cfg['seed']+progress['epoch']))
            for cpu in loader:
                if progress['steps'] >= initial_steps+10 and measurement_start is None:
                    torch.cuda.synchronize()
                    dist.barrier()
                    measurement_start = time.monotonic()
                    measurement_targets = progress['all_targets']
                local_count = torch.tensor(int(cpu['mask'].sum()), device=device, dtype=torch.long)
                gathered = [torch.zeros_like(local_count) for _ in range(world)]
                dist.all_gather(gathered, local_count)
                counts = [int(x) for x in gathered]
                total_batch = sum(counts)
                # Split a batch at an exact checkpoint boundary, then consume its
                # remaining targets on the next update; no data skipped or doubled.
                while progress['batch_skip'] < total_batch and progress['all_targets'] < cfg['target_tokens']:
                    milestone = min(cfg['target_tokens'],
                        (progress['all_targets']//cfg['checkpoint_every_positions']+1)*cfg['checkpoint_every_positions'])
                    take = min(total_batch-progress['batch_skip'], milestone-progress['all_targets'])
                    mask = select_global_positions(cpu['mask'], counts, rank, progress['batch_skip'], take)
                    local_valid = int(mask.sum())
                    han = torch.tensor(int(han_lookup[cpu['target_ids']][mask].sum()), device=device)
                    dist.all_reduce(han)
                    positions = progress['all_targets'] + take
                    lr = position_learning_rate(positions,cfg)
                    for group in optimizer.param_groups:
                        group['lr'] = lr
                    data = {k: cpu[k].to(device, non_blocking=True)
                            for k in ['tiles','indices','byte_targets']}
                    data['mask'] = mask.to(device, non_blocking=True)
                    for attempt in range(20):
                        optimizer.zero_grad(set_to_none=True)
                        scale = scaler._get_scale_async()*(local_valid/take)
                        if args.smoke_fp32:
                            sums = backward(**data, scale=scale, fp32=True, checked=True)
                        elif cfg.get('nonfinite_fp32_retry'):
                            def on_retry(stage):
                                progress['fp32_retries'] += 1
                                log('nonfinite_retry_same_batch_fp32', stage=stage,
                                    dataset_cursor=progress['cursor'], loss_scale=float(scale))
                            try:
                                sums = checked_byte_backward(backward, optimizer, data, scale,
                                                             on_retry=on_retry)
                            except NonfiniteByteForward as error:
                                # No update was made: retain the exact next batch and last
                                # successful model/optimizer state for deterministic replay.
                                torch.save(dict(batch=cpu, selected_mask=mask, progress=dict(progress),
                                                error=str(error)), logs/'nonfinite_batch.pt')
                                checkpoint(diagnostic=True)
                                raise
                        else:
                            sums = backward(**data, scale=scale)
                        dist.all_reduce(sums)
                        if not bool(torch.isfinite(sums).all()):
                            raise FloatingPointError('Nonfinite global loss')
                        sync_gradients(parameters, buffer)
                        update = complete_optimizer_step(optimizer, scaler, parameters, cfg['max_grad_norm'])
                        success = torch.tensor(int(update['succeeded']), device=device)
                        dist.all_reduce(success)
                        if int(success) == world:
                            break
                        if int(success) != 0:
                            raise RuntimeError('Inconsistent AMP update across ranks')
                        progress['overflows'] += 1
                        log('amp_retry_same_batch', optimizer=update)
                    else:
                        raise FloatingPointError('Repeated AMP overflow')
                    progress['all_targets'] += take
                    progress['han'] += int(han)
                    progress['batch_skip'] += take
                    progress['steps'] += 1
                    if progress['steps'] % 10 == 0 or progress['all_targets'] == milestone:
                        log('train', nll_per_pixel=float(sums)/(take*1024), learning_rate=lr,
                            optimizer=update, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                    status('training')
                    optimizer.zero_grad(set_to_none=True)
                    del data, sums
                    if progress['all_targets'] == milestone:
                        if progress['all_targets'] == cfg['target_tokens'] and measurement_start is not None:
                            torch.cuda.synchronize()
                            measurement_seconds = time.monotonic()-measurement_start
                        checkpoint()
                        validate()
                if progress['batch_skip'] == total_batch:
                    progress['cursor'] = min(len(ds), progress['cursor']+world*cfg['batch_size'])
                    progress['batch_skip'] = 0
                if progress['all_targets'] == cfg['target_tokens']:
                    break
            if progress['all_targets'] < cfg['target_tokens']:
                progress['epoch'] += 1
                progress['cursor'] = 0
                progress['batch_skip'] = 0
        dist.barrier()
        elapsed = torch.tensor(measurement_seconds, device=device, dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        memory = torch.tensor([torch.cuda.max_memory_allocated()/2**30,
                               torch.cuda.max_memory_reserved()/2**30],device=device)
        memories = [torch.zeros_like(memory) for _ in range(world)]
        dist.all_gather(memories,memory)
        if rank == 0:
            status('generation_evaluation')
            evidence = generation_diagnostic(model, subset, cfg, device, output, progress['steps'],
                                             'smoke' if args.smoke else 'full')
            write_json(output / 'complete.json', dict(status='complete', progress=dict(progress),
                       evaluation=evidence, initialization=metadata['initialization'],
                       initial_positions=initial_positions, new_positions=progress['all_targets']-initial_positions,
                       measured_seconds=float(elapsed),
                       measured_targets=progress['all_targets']-measurement_targets if measurement_start else 0,
                       targets_per_second=(progress['all_targets']-measurement_targets)/float(elapsed) if float(elapsed)>0 else None,
                       per_rank_peak_memory_gib=[x.tolist() for x in memories],
                       timing_scope='first ten updates excluded; final checkpoint/validation/generation excluded',
                       checkpoint=str(checkpoints / f"positions_{progress['all_targets']:09d}.pt")))
            status('finished', 'complete')
        dist.barrier()
    except BaseException as error:
        if logs.exists():
            write_json(logs / f'failure_rank{rank}.json', dict(error=str(error),progress=progress))
            status('failed', 'failed', error=str(error))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
