"""Controlled probes of context length, conditioning and greedy byte collapse."""

import collections
import json
import runpy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hansgpt_research.byte_glyph_decoder import pack_glyph_bytes
from hansgpt_research.cvae_fixed_step import install_xformers
from hansgpt_research.packed_glyph_data import PackedGlyphSequenceDataset
from hansgpt_research.train_attention_glyph_lm import model_from_config
from hansgpt_research.train_glyph_lm import write_json


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(20260913)
    root=Path('artifacts/reports/hansgpt_byte_c_eight_gpu_b8_h2048_global_lr_100m_v1_full')
    checkpoint=Path('artifacts/checkpoints')/root.name/'positions_100000000.pt'
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    model=model_from_config(saved['metadata']['config']).cuda().eval()
    model.load_state_dict(saved['model'])
    ds=PackedGlyphSequenceDataset(saved['metadata']['config']['data'],'validation',1024)
    helpers=runpy.run_path('scripts/evaluate_attention_abc.py')
    labels,controls=helpers['label_lookup'](ds)
    out=Path('artifacts/reports/byte_comma_diagnosis');out.mkdir(exist_ok=True)
    original=install_xformers()
    report=dict(checkpoint=str(checkpoint),checkpoint_progress=saved['progress'],lengths={},inference={},conditioning={})

    def summarize(grids):
        raw=grids.cpu().numpy()
        keys=[helpers['bitmap_key'](x) for x in raw]
        exact=[key in labels for key in keys]
        counts=collections.Counter(helpers['transcribe'](x[None],labels,controls) for x in raw)
        return dict(count=len(raw),exact_content=sum(exact),unique_bitmaps=len(set(keys)),
                    blank=int((raw.reshape(len(raw),-1).sum(1)==0).sum()),labels=dict(counts))

    try:
        # Reproduce saved short-prompt symptom under multiple inference backends.
        prior=np.load(root/'review8/generation.npz')
        prompts=torch.from_numpy(prior['prompts']).cuda()
        with torch.autocast('cuda',dtype=torch.float16):
            h=model.forward_hidden(prompts)[:,-1]
            greedy=model.byte_decoder.generate(h)
        assert np.array_equal(greedy.cpu().numpy(),prior['generated'][:,0])
        F.scaled_dot_product_attention=original
        with torch.autocast('cuda',dtype=torch.float16):
            hn=model.forward_hidden(prompts)[:,-1]
            native=model.byte_decoder.generate(hn)
            uncached=model.byte_decoder.generate(hn[:2],use_cache=False)
        install_xformers()
        report['inference']=dict(reproduced_saved=True,native_matches_xformers=bool(torch.equal(native,greedy)),
                                 cached_matches_uncached=bool(torch.equal(native[:2],uncached)),
                                 greedy=summarize(greedy))
        print(json.dumps(report['inference'],ensure_ascii=False),flush=True)

        # Each length ends at the same held-out target; only left context changes.
        indices=np.random.default_rng(20260913).choice(len(ds)-1,32,replace=False)
        records=[ds[int(i)] for i in indices]
        targets=torch.stack([r['targets'][-1] for r in records]).cuda()
        hidden_by_length={}
        for length in [16,64,256,1024]:
            hs=[]
            for start in range(0,len(records),4):
                x=torch.stack([r['glyphs'][-length:] for r in records[start:start+4]]).cuda()
                with torch.autocast('cuda',dtype=torch.float16):
                    hs.append(model.forward_hidden(x)[:,-1])
            hidden=torch.cat(hs);hidden_by_length[length]=hidden
            with torch.autocast('cuda',dtype=torch.float16):
                raw=model.byte_decoder.generate(hidden)
                nll=model.distribution(hidden).nll(targets)
            report['lengths'][str(length)]=dict(greedy=summarize(raw),paired_target_nll=float(nll.mean()))
            print(json.dumps({'length':length,**report['lengths'][str(length)]},ensure_ascii=False),flush=True)

        hidden=hidden_by_length[1024]
        with torch.autocast('cuda',dtype=torch.float16):
            real=model.distribution(hidden).nll(targets)
            shuffled=model.distribution(hidden.roll(1,0)).nll(targets)
            zero=model.distribution(torch.zeros_like(hidden)).nll(targets)
        report['conditioning']=dict(targets=len(targets),real_nll=float(real.mean()),
            shuffled_nll=float(shuffled.mean()),zero_nll=float(zero.mean()),
            shuffle_increases_nll=int((shuffled>real).sum()),
            per_target_real=real.tolist(),per_target_shuffled=shuffled.tolist())

        # Same contexts, changing only categorical temperature. These are raw
        # samples; nearest-font scoring never replaces generated feedback.
        report['sampling']={}
        sampled_for_image=[]
        for temperature in [0.7,1.0,1.2]:
            with torch.autocast('cuda',dtype=torch.float16):
                sampled=model.byte_decoder.generate(hidden,strategy='sample',temperature=temperature,
                    generator=torch.Generator(device='cuda').manual_seed(20260913))
            report['sampling'][str(temperature)]=summarize(sampled)
            sampled_for_image.append(sampled.cpu().numpy())
        with torch.autocast('cuda',dtype=torch.float16):
            same=model.byte_decoder.generate(hidden[:1].expand(64,-1),strategy='sample',
                generator=torch.Generator(device='cuda').manual_seed(20260913))
        report['same_context_64_draws']=summarize(same)

        # Check whether local byte argmax is also best among selected whole glyphs.
        candidates=['，','。','的','一','是','不','中','国','人','在']
        gallery=torch.stack([ds.glyph_bank[ds.inventory['characters'][c]] for c in candidates]).cuda()
        expanded_h=hidden[:8,None].expand(-1,len(candidates),-1).reshape(-1,hidden.shape[-1])
        expanded_y=gallery[None].expand(8,-1,-1,-1,-1).reshape(-1,1,32,32)
        with torch.autocast('cuda',dtype=torch.float16):
            scores=model.distribution(expanded_h).nll(expanded_y).reshape(8,-1)*1024
        report['candidate_joint_nll']=dict(candidates=candidates,per_context=scores.tolist(),
            selected=[candidates[i] for i in scores.argmin(1).tolist()],
            warning='Diagnostic candidate scores only; no vocabulary constraint in generation')

        # Free-running categorical generation for eight original prompts.
        generator=torch.Generator(device='cuda').manual_seed(20260913)
        current=prompts;cache=None;generated=[]
        for _ in range(32):
            with torch.autocast('cuda',dtype=torch.float16):
                hs,cache=model.forward_hidden(current,past_key_values=cache,use_cache=True,return_cache=True)
                tile=model.byte_decoder.generate(hs[:,-1:],strategy='sample',generator=generator)
            generated.append(tile);current=tile
        raw=torch.cat(generated,1).cpu().numpy();p=prompts.cpu().numpy()
        summary=helpers['generation_summary'](p,raw,labels,controls)
        summary['protocol']='8 fixed validation prompts; 32 raw categorical-sampled glyphs; temperature1; fixed-length diagnostic'
        write_json(out/'sampled_generation.json',summary)
        np.savez_compressed(out/'sampled_generation.npz',prompts=p,generated=raw)
        for start in [0,4]:helpers['draw_samples'](out/f'sampled_{start:02d}.png',p[start:start+4],raw[start:start+4],
            '100M byte model: blue=input; raw categorical sampling, temperature1')
        report['free_sampling']=summary['summary']
        report['validation_windows']=indices.tolist()
        write_json(out/'result.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ['lengths','candidate_joint_nll','inference','conditioning']},ensure_ascii=False),flush=True)
    finally:
        F.scaled_dot_product_attention=original


if __name__=='__main__':main()
