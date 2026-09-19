"""Replay glyph warm-up, isolate interface drift, and extend reconstruction only."""

import argparse
import copy
import json
import os
import runpy
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from hansgpt_research.glyph_lm import GlyphSequenceDataset
from hansgpt_research.train_attention_glyph_lm import model_from_config, validate_nll
from hansgpt_research.train_glyph_lm import autocast_context, sha256, write_json
from hansgpt_research.train_structured_glyph_lm import complete_optimizer_step, validation_subset


def cpu_state(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=10000)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='5':raise RuntimeError('Use GPU5')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('Clean committed source required')
    args.output.mkdir(parents=True,exist_ok=False)
    p=json.loads(Path('configs/experiments/dual_ablation_pilot.json').read_text())
    seed=p['seed'];torch.manual_seed(seed);torch.set_num_threads(4)
    device=torch.device('cuda:0')
    assert sha256(Path(p['initial_checkpoint']))==p['initial_sha256']
    saved=torch.load(p['initial_checkpoint'],map_location='cpu',weights_only=False)
    cfg=saved['metadata']['config'];model=model_from_config(cfg).cuda()
    model.load_state_dict(saved['model'],strict=True);del saved
    ds=GlyphSequenceDataset(p['data'],'train',256)
    validation=GlyphSequenceDataset(p['data'],'validation',256)
    subset,selection=validation_subset(validation,cfg['training'])
    helpers=runpy.run_path('scripts/run_dual_ablation.py')
    freq=np.bincount(ds.tokens.astype(np.int64),minlength=len(ds.glyph_bank))
    eligible=[i for i in np.flatnonzero(freq) if i not in ds.control_ids.values()]
    train_ids,val_ids=helpers['glyph_partition'](ds.glyph_bank,eligible,seed,.1)
    adapter=nn.Linear(1024,1024).cuda()
    params=list(model.glyph_encoder.parameters())+list(model.glyph_decoder.parameters())+list(adapter.parameters())
    opt=torch.optim.AdamW(params,lr=3e-4,betas=(.9,.95),weight_decay=0)
    scaler=torch.amp.GradScaler('cuda');rng=np.random.default_rng(seed)
    train=ds.glyph_bank[train_ids].cuda();heldout=ds.glyph_bank[val_ids].cuda()
    old_encoder=cpu_state(model.glyph_encoder);old_decoder=cpu_state(model.glyph_decoder)
    report=dict(initial_sha256=p['initial_sha256'],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol='replay fixed glyph warm-up; constant LR 3e-4; all language modules unchanged; validation-only diagnosis',
        train_glyphs=len(train),heldout_glyphs=len(heldout),curve=[],swaps={})
    started=time.monotonic()

    def publish():
        report['seconds']=time.monotonic()-started
        write_json(args.output/'report.json',report)

    @torch.inference_mode()
    def reconstruct(bank):
        model.eval();nll=exact=tp=fp=fn=errors=count=0
        for tiles in bank.split(256):
            with autocast_context(device,'fp16'):
                logits=model.glyph_decoder(adapter(model.glyph_encoder(tiles)))
            pred=logits>0;true=tiles.bool();difference=(pred!=true).flatten(1).sum(1)
            count+=len(tiles);exact+=int((difference==0).sum());errors+=int(difference.sum())
            tp+=int((pred&true).sum());fp+=int((pred&~true).sum());fn+=int((~pred&true).sum())
            nll+=float(F.binary_cross_entropy_with_logits(logits.float(),tiles.float(),reduction='sum'))
        return dict(nll=nll/(count*1024),exact=exact/count,exact_count=exact,
                    hamming=errors/count,f1=2*tp/max(1,2*tp+fp+fn))

    @torch.inference_mode()
    def swaps(step):
        new_encoder=cpu_state(model.glyph_encoder);new_decoder=cpu_state(model.glyph_decoder)
        result={}
        for name,e,d in [('original',old_encoder,old_decoder),
                         ('changed_encoder_only',new_encoder,old_decoder),
                         ('changed_decoder_only',old_encoder,new_decoder),
                         ('both_changed',new_encoder,new_decoder)]:
            model.glyph_encoder.load_state_dict(e);model.glyph_decoder.load_state_dict(d)
            result[name]=validate_nll(model,subset,selection,cfg['training'],device)['nll_per_pixel']
        model.glyph_encoder.load_state_dict(new_encoder);model.glyph_decoder.load_state_dict(new_decoder)
        report['swaps'][str(step)]=result
        torch.save(dict(encoder=new_encoder,decoder=new_decoder,adapter=cpu_state(adapter),
                        optimizer=opt.state_dict(),step=step,initial_sha256=p['initial_sha256']),
                   args.output/f'reconstruction_step{step}.pt')
        print('INTERFACE',step,json.dumps(result),flush=True)

    overflow_streak=0
    for step in range(args.steps+1):
        if step%1000==0 or step==args.steps:
            row=dict(step=step,train=reconstruct(train),heldout=reconstruct(heldout))
            report['curve'].append(row);print('RECON',json.dumps(row),flush=True);publish()
        if step in {2000,args.steps} and step>0:
            swaps(step);publish()
        if step==args.steps:break
        model.train();opt.zero_grad(set_to_none=True)
        tiles=train[rng.integers(len(train),size=256)]
        with autocast_context(device,'fp16'):
            logits=model.glyph_decoder(adapter(model.glyph_encoder(tiles)))
            loss=F.binary_cross_entropy_with_logits(logits.float(),tiles.float())
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite reconstruction loss')
        scaler.scale(loss).backward();result=complete_optimizer_step(opt,scaler,params,1)
        overflow_streak=0 if result['succeeded'] else overflow_streak+1
        if overflow_streak>=20:raise FloatingPointError('Repeated overflow')
    report['complete']=True;publish()


if __name__=='__main__':main()
