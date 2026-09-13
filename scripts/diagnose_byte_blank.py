"""Differential blank-output diagnosis on one fixed trained byte-model checkpoint."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hansgpt_research.byte_training import ByteBackward, ByteCollator
from hansgpt_research.byte_glyph_decoder import pack_glyph_bytes
from hansgpt_research.cvae_fixed_step import install_xformers
from hansgpt_research.glyph_lm import GlyphSequenceDataset, collate_glyph_sequences
from hansgpt_research.packed_glyph_data import PackedGlyphSequenceDataset
from hansgpt_research.train_attention_glyph_lm import model_from_config
from hansgpt_research.train_structured_glyph_lm import validation_subset
from hansgpt_research.train_glyph_lm import write_json


def main():
    torch.set_num_threads(4)
    checkpoint=Path('artifacts/checkpoints/hansgpt_byte_c_four_gpu_b8_h2048_10m_v1_full/positions_010000000.pt')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    config=saved['metadata']['config']
    model=model_from_config(config).cuda().eval()
    model.load_state_dict(saved['model'])
    validation=GlyphSequenceDataset(config['data'],'validation',1024)
    subset,_=validation_subset(validation,config['training'])
    records=[subset[i] for i in range(len(subset)) if int(subset[i]['attention_mask'].sum())>=16][:4]
    prompts=torch.stack([r['glyphs'][:16] for r in records]).cuda()
    original_report=Path('artifacts/reports/hansgpt_byte_c_four_gpu_b8_h2048_10m_v1_full')
    artifact=np.load(next(original_report.glob('generation_step*.npz')))
    assert np.array_equal(prompts[:1].cpu().numpy(),artifact['prompt'])
    report=dict(checkpoint=str(checkpoint),progress=saved['progress'],hypotheses=[
        'Insufficient optimization versus old 30494-update run',
        'Greedy zero-byte fixed point', 'Accelerated implementation mismatch', 'Corrupt targets/masks'])
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
        h_native=model.forward_hidden(prompts)[:,-1]
        greedy_native=model.byte_decoder.generate(h_native)
        greedy_uncached=model.byte_decoder.generate(h_native,use_cache=False)
    original=install_xformers()
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            h=model.forward_hidden(prompts)[:,-1]
            greedy=model.byte_decoder.generate(h)
            generated=model.generate(prompts,2)
            sampled=model.byte_decoder.generate(h,strategy='sample',generator=torch.Generator(device='cuda').manual_seed(20260913))
            zero=model.byte_decoder.teacher_forced_logits(h,torch.zeros(4,1,32,32,device='cuda',dtype=torch.uint8))
            p0=zero.float().softmax(-1)[...,0]
        report['inference']=dict(prompt_count=4,
            native_black_pixels=greedy_native.flatten(1).sum(1).tolist(),
            xformers_black_pixels=greedy.flatten(1).sum(1).tolist(),
            cached_matches_uncached=bool(torch.equal(greedy_native,greedy_uncached)),
            native_matches_xformers=bool(torch.equal(greedy_native,greedy)),
            replay_matches_saved_first_two=bool(np.array_equal(generated[:1].cpu().numpy(),artifact['generated'][:,:2])),
            hidden_relative_l2=float((h.float()-h_native.float()).norm()/h_native.float().norm()),
            sampled_black_pixels=sampled.flatten(1).sum(1).tolist(),
            zero_prefix_argmax_zero_counts=(zero.argmax(-1)==0).sum(1).tolist(),
            zero_probability_min_mean_max=[p0.min(1).values.tolist(),p0.mean(1).tolist(),p0.max(1).values.tolist()])
        print(json.dumps({'inference':report['inference']}),flush=True)

        # End-to-end gradient comparison, including input encoder, backbone,
        # masked holes, right padding and byte loss. Same xFormers backend.
        ds=PackedGlyphSequenceDataset(config['data'],'train',16)
        sample=ds[5]
        sample['attention_mask'][-4:]=False
        sample['loss_mask'][-4:]=False
        sample['glyphs'][-4:]=0
        cpu=collate_glyph_sequences([sample])
        prepared=ByteCollator()([sample])
        batch={k:v.cuda() for k,v in cpu.items()}
        params=list(model.named_parameters())
        model.train();model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            hidden=model.forward_hidden(batch['glyphs'],batch['attention_mask'])
            mask=batch['loss_mask'].bool()
            reference=model.distribution(hidden[mask]).nll(batch['targets'][mask]).mean()
        (reference*1024).backward()
        gradients={name:p.grad.detach().clone() for name,p in params}
        model.zero_grad(set_to_none=True)
        runner=ByteBackward(model,1,16,chunk=8,compiled=True)
        data={k:prepared[k].cuda() for k in ['tiles','indices','byte_targets','mask']}
        actual=runner(**data,scale=torch.tensor(1024.,device='cuda'))/(data['mask'].sum()*1024)
        statistics={}
        for prefix in ['glyph_encoder','backbone','byte_decoder']:
            dot=norm_a=norm_b=diff=0.0
            for name,p in params:
                if not name.startswith(prefix): continue
                a=gradients[name].float();b=p.grad.float()
                dot+=float((a*b).double().sum());norm_a+=float(a.double().square().sum())
                norm_b+=float(b.double().square().sum());diff+=float((a-b).double().square().sum())
            statistics[prefix]=dict(relative_l2=(diff/max(norm_a,1e-30))**.5,
                                    cosine=dot/max((norm_a*norm_b)**.5,1e-30))
        report['training_differential']=dict(original_nll=float(reference),compiled_nll=float(actual),gradients=statistics)
        assert abs(float(reference)-float(actual))<1e-3
        assert all(s['cosine']>0.999 and s['relative_l2']<0.03 for s in statistics.values())
        print(json.dumps({'training_differential':report['training_differential']}),flush=True)
        del gradients,runner,data,batch
        model.zero_grad(set_to_none=True)
        masks=valid=zero_bytes=all_blank=0
        for i in range(100):
            row=ds[i];m=row['loss_mask'].bool();y=row['targets'][m]
            values=pack_glyph_bytes(y)
            valid+=len(y);zero_bytes+=int((values==0).sum());all_blank+=int((y.flatten(1).sum(1)==0).sum())
            masks+=int((row['loss_mask'].bool() & ~row['attention_mask'].bool()).sum())
        report['target_audit']=dict(scope='first 100 training windows at context16',
            valid_targets=valid,blank_target_grids=all_blank,zero_byte_fraction=zero_bytes/(valid*128),
            loss_on_padding=masks)
    finally:
        F.scaled_dot_product_attention=original
    out=Path('artifacts/reports/byte_blank_diagnosis')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'result.json',report)
    print(json.dumps(report['target_audit']),flush=True)


if __name__=='__main__':main()
