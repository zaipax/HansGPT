"""Train pure-Transformer C byte model; default xFormers + compile + fused AdamW."""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import regex
import torch
from torch.utils.data import DataLoader

from hansgpt_research.byte_training import DEFAULT_ACCELERATION, ByteBackward, ByteCollator
from hansgpt_research.cvae_fixed_step import install_xformers
from hansgpt_research.packed_glyph_data import PackedGlyphSequenceDataset
from hansgpt_research.train_attention_glyph_lm import model_from_config
from hansgpt_research.train_glyph_lm import SortishEpochSampler, sequence_lengths, runtime_metadata, write_json, save_checkpoint
from hansgpt_research.train_structured_glyph_lm import optimizer_for, complete_optimizer_step


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/experiments/hansgpt_attention_c_ctx1024.json'))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    cfg=config['training']
    for key,value in DEFAULT_ACCELERATION.items(): cfg.setdefault(key,value)
    if any(cfg[k]!=v for k,v in DEFAULT_ACCELERATION.items()):
        raise ValueError('This runner requires the default accelerated stack')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='7' or config['variant']!='C':
        raise ValueError('This experiment requires variant C on physical GPU7')
    steps=3 if args.smoke else cfg['max_steps']
    name=config['experiment']+('_smoke' if args.smoke else '_full')
    output=Path('artifacts/reports')/name
    logs=Path('artifacts/logs')/name
    checkpoints=Path('artifacts/checkpoints')/name
    for p in [output,logs,checkpoints]: p.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda:0')
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    start=time.monotonic()
    progress=dict(steps=0,all_targets=0,han=0,overflows=0,epoch=0,cursor=0)
    rows=[]
    measured_start=None

    def status(state='running',**extra):
        write_json(logs/'status.json',dict(status=state,time=datetime.now(UTC).isoformat(),
            progress=dict(progress),target_steps=steps,seconds=time.monotonic()-start,**extra))

    try:
        status(phase='initializing')
        install_xformers()
        metadata=runtime_metadata(config,Path(config['data']),device)
        model=model_from_config(config).to(device).train()
        if cfg['gradient_checkpointing']:
            model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        metadata.update(parameters=sum(p.numel() for p in model.parameters()),
            initialization='random; no old weights loaded',compile_scope='full byte-head loss graph',
            timing_scope='data prefetch/transfers/backward/optimizer; first 10 updates excluded; no checkpoint IO')
        write_json(output/'metadata.json',metadata)
        ds=PackedGlyphSequenceDataset(config['data'],'train',cfg['sequence_length'])
        lengths=sequence_lengths(ds)
        han_lookup=torch.zeros(len(ds.glyph_bank),dtype=torch.bool)
        for char,index in ds.inventory['characters'].items():
            if regex.fullmatch(r'[\p{Unified_Ideograph}〇]',char): han_lookup[int(index)]=True
        optimizer=optimizer_for(model,cfg)
        scaler=torch.amp.GradScaler('cuda')
        scaler.scale(torch.ones((),device=device))
        backward=ByteBackward(model,cfg['batch_size'],cfg['sequence_length'],cfg['head_chunk_size'],compiled=True)
        warmup_steps=0 if args.smoke else 10
        while progress['steps']<steps:
            sampler=SortishEpochSampler(lengths,cfg['seed'],progress['epoch'],cfg['batch_size'],0,64)
            loader=DataLoader(ds,batch_size=cfg['batch_size'],sampler=sampler,drop_last=True,
                num_workers=2,pin_memory=True,collate_fn=ByteCollator(),
                generator=torch.Generator().manual_seed(cfg['seed']+progress['epoch']))
            iterator=iter(loader)
            for _ in range(len(loader)):
                if progress['steps']==warmup_steps and measured_start is None:
                    torch.cuda.synchronize(); measured_start=time.monotonic()
                tick=time.monotonic()
                cpu=next(iterator)
                targets=int(cpu['mask'].sum())
                if not targets: continue
                data={k:cpu[k].to(device,non_blocking=True) for k in ['tiles','indices','byte_targets','mask']}
                for retry in range(20):
                    optimizer.zero_grad(set_to_none=True)
                    loss=backward(**data,scale=scaler._get_scale_async())
                    if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite byte NLL')
                    update=complete_optimizer_step(optimizer,scaler,model.parameters(),1.0)
                    if update['succeeded']: break
                    progress['overflows']+=1
                else: raise FloatingPointError('Repeated AMP overflow')
                torch.cuda.synchronize()
                seconds=time.monotonic()-tick
                progress['steps']+=1; progress['all_targets']+=targets
                progress['han']+=int(han_lookup[cpu['target_ids']][cpu['mask'].bool()].sum())
                progress['cursor']+=cfg['batch_size']
                row=dict(step=progress['steps'],seconds=seconds,targets=targets,
                         nll_per_pixel=float(loss)/(targets*1024),grad_norm=update['grad_norm'])
                if progress['steps']>warmup_steps: rows.append(row)
                if progress['steps']%10==0 or progress['steps']==steps:
                    with (logs/'training.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                    print(json.dumps(row),flush=True)
                status(phase='training',latest=row,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                del data,loss
                if progress['steps']==steps: break
            if progress['steps']<steps: progress['epoch']+=1;progress['cursor']=0
        elapsed=time.monotonic()-measured_start
        result=dict(status='complete',progress=progress,measured_steps=len(rows),measured_seconds=elapsed,
                    targets_per_second=sum(r['targets'] for r in rows)/elapsed,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,rows=rows)
        status(phase='saving')
        save_checkpoint(checkpoints/'final.pt',model,optimizer,scaler,dict(progress),metadata)
        write_json(output/'result.json',result)
        status('complete',phase='finished',targets_per_second=result['targets_per_second'])
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    except BaseException as error:
        status('failed',error=str(error));raise


if __name__=='__main__': main()
