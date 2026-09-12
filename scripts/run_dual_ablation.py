"""A/B/C continuation and glyph reconstruction controls with measured time matching."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import runpy
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from hansgpt_research.glyph_lm import GlyphSequenceDataset, collate_glyph_sequences
from hansgpt_research.train_attention_glyph_lm import model_from_config, validate_nll
from hansgpt_research.train_glyph_lm import (
    SortishEpochSampler, autocast_context, learning_rate, runtime_metadata,
    save_checkpoint, sequence_lengths, sha256, write_json,
)
from hansgpt_research.train_structured_glyph_lm import (
    advance_progress, complete_optimizer_step, generator_backward, initial_progress,
    optimizer_for, validation_subset,
)


def glyph_partition(bank, eligible, seed, fraction):
    """Keep identical raster aliases in one reconstruction split."""
    groups = {}
    for index in eligible:
        key = np.packbits(bank[index].numpy().reshape(-1)).tobytes()
        groups.setdefault(key, int(index))
    ordered = sorted(groups, key=lambda b: hashlib.sha256(str(seed).encode()+b).digest())
    n = max(1, int(len(ordered)*fraction))
    if n >= len(ordered):
        raise ValueError("Need both reconstruction splits")
    return [groups[k] for k in ordered[n:]], [groups[k] for k in ordered[:n]]


def budget_decision(spent, b_status):
    if b_status['status'] == 'failed':
        raise RuntimeError('B failed; matched-time control cannot continue')
    budget = b_status['compute_seconds']
    if b_status.get('training_budget_final') and spent >= budget:
        return 'stop'
    return 'train' if spent < budget else 'wait'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/dual_ablation_pilot.json')
    parser.add_argument('--arm', choices=['A','B','C'], required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-tag', default='smoke')
    args = parser.parse_args()
    if not args.smoke_tag.replace('_','').isalnum():raise ValueError('Invalid smoke tag')
    protocol = json.loads(Path(args.config).read_text())
    if args.smoke:
        protocol.update(language_targets=16384, glyph_steps=3, validate_every_targets=8192,
                        generation_samples=2, generation_length=4)
    gpu = protocol['gpus'][args.arm]
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu) or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise RuntimeError('Wrong physical GPU mapping')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('Committed clean source required')
    name = protocol['name'] + ('_'+args.smoke_tag if args.smoke else '')
    run = name+'_'+args.arm
    logs = Path('artifacts/logs')/run
    checkpoints = Path('artifacts/checkpoints')/run
    reports = Path('artifacts/reports')/run
    for path in [logs,checkpoints,reports]:
        if path.exists():raise FileExistsError(path)
    for path in [logs,checkpoints,reports]:path.mkdir(parents=True)
    seed = protocol['seed']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4)
    device=torch.device('cuda:0')
    init=Path(protocol['initial_checkpoint'])
    if sha256(init)!=protocol['initial_sha256']:raise ValueError('Initial checkpoint changed')
    saved=torch.load(init,map_location='cpu',weights_only=False)
    config=saved['metadata']['config']
    cfg=config['training']
    cfg.update(seed=seed,learning_rate=protocol['language_learning_rate'],
               minimum_learning_rate_ratio=protocol['minimum_learning_rate_ratio'],
               warmup_tokens=protocol['language_warmup_targets'],target_tokens=protocol['language_targets'],
               validate_every_tokens=protocol['validate_every_targets'])
    metadata=runtime_metadata(config,Path(protocol['data']),device,mode='smoke' if args.smoke else 'full')
    for key in ['data_sha256','data_manifest_sha256','data_verification_sha256']:
        if metadata[key]!=saved['metadata'][key]:raise ValueError('Initial corpus mismatch: '+key)
    model=model_from_config(config).cuda()
    model.load_state_dict(saved['model'],strict=True);del saved
    datasets={s:GlyphSequenceDataset(protocol['data'],s,256) for s in ['train','validation','test']}
    dataset=datasets['train'];lengths=sequence_lengths(dataset)
    subset,selection=validation_subset(datasets['validation'],cfg)
    protocol_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    metadata.update(protocol=protocol,protocol_sha256=protocol_hash,arm=args.arm,
                    initial_checkpoint_sha256=protocol['initial_sha256'],
                    optimizer_policy='fresh AdamW for all language arms; same token LR schedule; C holds floor after target',
                    compute_budget='synchronized optimizer-update wall time, including transfers, excluding loading/eval/checkpoint/waits',
                    script_sha256=sha256(Path(__file__)))
    write_json(logs/'metadata.json',metadata)
    progress=initial_progress();compute=0.;final_budget=False;started=time.monotonic()
    optimizer=None;scaler=None

    def status(phase, state='running', **extra):
        write_json(logs/'status.json',dict(status=state,phase=phase,pid=os.getpid(),
            time=datetime.now(UTC).isoformat(),arm=args.arm,gpu=gpu,protocol_sha256=protocol_hash,
            compute_seconds=compute,training_budget_final=final_budget,progress=progress,
            wall_seconds=time.monotonic()-started,**extra))

    def log(kind, **extra):
        row=dict(kind=kind,time=datetime.now(UTC).isoformat(),progress=dict(progress),
                 compute_seconds=compute,**extra)
        with (logs/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

    def save(filename):
        if filename=='best.pt':
            path=checkpoints/filename;pending=path.with_suffix('.tmp')
            torch.save(dict(model=model.state_dict(),metadata=metadata,progress=dict(progress)),pending)
            pending.replace(path)
            return
        save_checkpoint(checkpoints/filename,model,optimizer,scaler,dict(progress),metadata)

    helpers=runpy.run_path('scripts/evaluate_attention_abc.py')
    labels,controls=helpers['label_lookup'](dataset)

    @torch.inference_mode()
    def generation(final=False):
        ds=datasets['test' if final else 'validation']
        # Periodic diagnostic uses fixed validation chunks, final uses independent test pages.
        if final:
            documents,_=helpers['select_documents'](ds,protocol['generation_samples'],seed)
            ids=np.stack([ds.tokens[int(ds.offsets[i]):int(ds.offsets[i])+16] for i in documents]).astype(np.int64)
            prompts=ds.glyph_bank[ids].to(device)
        else:
            prompts=torch.stack([subset[i]['glyphs'][:16] for i in range(min(4,len(subset)))]).to(device)
        with autocast_context(device,'fp16'):
            output=model.generate(prompts,protocol['generation_length'] if final else 32,
                                  threshold=.5,eos_glyph=ds.glyph_bank[ds.control_ids['EOS']])
        p,g=prompts.cpu().numpy(),output.cpu().numpy()
        result=helpers['generation_summary'](p,g,labels,controls)
        result['protocol']='fixed validation chunks' if not final else 'independent test pages; raw output; 0.5 threshold'
        tag='final' if final else str(progress['valid_tokens'])
        write_json(reports/f'generation_{tag}.json',result)
        np.savez_compressed(reports/f'generation_{tag}.npz',prompts=p,generated=g)
        if final:
            for start in range(0,len(p),4):
                helpers['draw_samples'](reports/f'samples_{start:02d}.png',p[start:start+4],g[start:start+4],run)
        return result['summary']

    def validate():
        status('validation');model.eval()
        result=validate_nll(model,subset,selection,cfg,device)
        progress['last_validation_tokens']=progress['valid_tokens']
        if progress['best_validation_nll'] is None or result['nll_per_pixel']<progress['best_validation_nll']:
            progress['best_validation_nll']=result['nll_per_pixel'];save('best.pt')
        log('validation',metrics=result,generation=generation())
        model.train();status('language')

    try:
        status('initializing')
        if args.arm=='B':
            freq=np.bincount(dataset.tokens.astype(np.int64),minlength=len(dataset.glyph_bank))
            eligible=[i for i in np.flatnonzero(freq) if i not in dataset.control_ids.values()]
            train_ids,val_ids=glyph_partition(dataset.glyph_bank,eligible,seed,protocol['glyph_validation_fraction'])
            write_json(reports/'glyph_split.json',dict(train_ids=train_ids,validation_ids=val_ids,
                scope='held out from reconstruction only; base LM may already have seen these glyphs'))
            adapter=nn.Linear(1024,1024).cuda()
            params=list(model.glyph_encoder.parameters())+list(model.glyph_decoder.parameters())+list(adapter.parameters())
            opt=torch.optim.AdamW(params,lr=protocol['glyph_learning_rate'],betas=(.9,.95),weight_decay=0)
            scale=torch.amp.GradScaler('cuda');rng=np.random.default_rng(seed)
            train_bank=dataset.glyph_bank[train_ids].cuda();val_bank=dataset.glyph_bank[val_ids].cuda()

            @torch.inference_mode()
            def reconstruction(bank):
                loss=exact=total=0;model.eval()
                for tiles in bank.split(256):
                    with autocast_context(device,'fp16'):
                        logits=model.glyph_decoder(adapter(model.glyph_encoder(tiles)))
                    loss+=float(F.binary_cross_entropy_with_logits(logits.float(),tiles.float(),reduction='sum'))
                    exact+=int(((logits>0)==tiles.bool()).flatten(1).all(1).sum());total+=len(tiles)
                return dict(nll=loss/(total*1024),exact=exact/total,glyphs=total)

            overflow_streak=0
            for step in range(protocol['glyph_steps']+1):
                if step%200==0 or step==protocol['glyph_steps']:
                    log('reconstruction',step=step,train=reconstruction(train_bank),validation=reconstruction(val_bank))
                    status('glyph_pretraining',glyph_step=step)
                if step==protocol['glyph_steps']:break
                model.train();torch.cuda.synchronize();tick=time.monotonic();opt.zero_grad(set_to_none=True)
                tiles=train_bank[rng.integers(len(train_bank),size=protocol['glyph_batch_size'])]
                with autocast_context(device,'fp16'):
                    logits=model.glyph_decoder(adapter(model.glyph_encoder(tiles)))
                    loss=F.binary_cross_entropy_with_logits(logits.float(),tiles.float())
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Reconstruction loss')
                scale.scale(loss).backward()
                update=complete_optimizer_step(opt,scale,params,1)
                overflow_streak=0 if update['succeeded'] else overflow_streak+1
                if not update['succeeded']:log('reconstruction_overflow',step=step,optimizer=update)
                if overflow_streak>=20:raise FloatingPointError('Repeated reconstruction overflow')
                torch.cuda.synchronize();compute+=time.monotonic()-tick
                if step%10==0:status('glyph_pretraining',glyph_step=step+1)
            del adapter,opt,scale,params,train_bank,val_bank
        # Identical language optimizer initialization and data RNG for all three arms.
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        optimizer=optimizer_for(model,cfg);scaler=torch.amp.GradScaler('cuda')
        validate()
        stopped=False;last_checkpoint_tokens=0
        while not stopped:
            sampler=SortishEpochSampler(lengths,seed,progress['epoch'],32,progress['cursor'],64)
            loader=DataLoader(dataset,batch_size=32,sampler=sampler,num_workers=2,pin_memory=True,
                collate_fn=collate_glyph_sequences,generator=torch.Generator().manual_seed(seed+progress['epoch']))
            for batch in loader:
                if args.arm in ['A','B']:
                    if progress['valid_tokens']>=protocol['language_targets']:stopped=True;break
                else:
                    b_path=Path('artifacts/logs')/(name+'_B')/'status.json'
                    while True:
                        if time.monotonic()-started>86400:raise TimeoutError('B budget wait exceeded one day')
                        if not b_path.exists():status('waiting_for_B');time.sleep(10);continue
                        b=json.loads(b_path.read_text())
                        if b['protocol_sha256']!=protocol_hash:raise ValueError('B protocol mismatch')
                        action=budget_decision(compute,b)
                        if action=='wait':status('waiting_for_B');time.sleep(10);continue
                        stopped=action=='stop';break
                    if stopped:break
                tokens=int(batch['loss_mask'].sum());optimizer.zero_grad(set_to_none=True)
                lr=learning_rate(progress['valid_tokens']+tokens,cfg)
                for group in optimizer.param_groups:group['lr']=lr
                torch.cuda.synchronize();tick=time.monotonic()
                result=generator_backward(model,None,[batch],cfg,device,scaler,tokens,frozen=False)
                update=complete_optimizer_step(optimizer,scaler,model.parameters(),1)
                torch.cuda.synchronize();compute+=time.monotonic()-tick
                advance_progress(progress,tokens=tokens,samples=len(batch['glyphs']),
                                 generator_ok=update['succeeded'],discriminator_ok=None)
                if progress['overflow_streak']>=20:raise FloatingPointError('Repeated AMP overflow')
                if progress['optimizer_steps']%10==0:
                    log('train',nll=result['nll_sum']/tokens,lr=lr,optimizer=update);status('language')
                if progress['valid_tokens']-progress['last_validation_tokens']>=protocol['validate_every_targets']:
                    validate()
                if progress['valid_tokens']-last_checkpoint_tokens>=5000000:
                    save('latest.pt');last_checkpoint_tokens=progress['valid_tokens']
            if not stopped:
                progress['epoch']+=1;progress['cursor']=0
        final_budget=True;status('final_validation')
        if progress['last_validation_tokens']!=progress['valid_tokens']:validate()
        save('final.pt');model.eval();status('final_evaluation')
        test=datasets['test'];test_selection=dict(scope='full_test',expected_targets=test.target_count)
        # Smoke keeps evaluation small while exercising the final report path.
        if args.smoke:test,test_selection=validation_subset(test,cfg)
        full=validate_nll(model,test,test_selection,cfg,device)
        generated=generation(final=True)
        receipt=dict(status='complete',arm=args.arm,protocol_sha256=protocol_hash,
            progress=progress,compute_seconds=compute,full_test=full,generation=generated,
            final_checkpoint_sha256=sha256(checkpoints/'final.pt'))
        if args.arm=='C':receipt['matched_B_seconds']=json.loads(b_path.read_text())['compute_seconds']
        write_json(reports/'complete.json',receipt);status('finished','complete')
    except BaseException as error:
        status('failed','failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__':main()
