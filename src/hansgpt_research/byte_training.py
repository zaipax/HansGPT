"""Default accelerated training path for the pure-Transformer byte glyph model."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

DEFAULT_ACCELERATION = dict(attention_backend='xformers', compile_head=True, fused_adamw=True)


@dataclass
class ByteCollator:
    def __call__(self, samples):
        batch = default_collate(samples)
        attention = batch['attention_mask'].bool()
        if ((~attention[:, :-1]) & attention[:, 1:]).any():
            raise ValueError('Only trailing padding supported')
        if (batch['loss_mask'].bool() & ~attention).any():
            raise ValueError('Padding must have zero loss')
        x = batch['glyphs'].numpy().reshape(-1, 1024)
        y = batch['targets'].numpy().reshape(-1, 1024)
        if not np.isin(x, [0,1]).all() or not np.isin(y,[0,1]).all():
            raise ValueError('Binary glyphs required')
        unique, inverse = np.unique(np.packbits(x, axis=1), axis=0, return_inverse=True)
        return dict(tiles=torch.from_numpy(np.unpackbits(unique,axis=1).reshape(-1,1,32,32)),
                    indices=torch.from_numpy(inverse),
                    byte_targets=torch.from_numpy(np.packbits(y,axis=1)),
                    mask=batch['loss_mask'].flatten(), target_ids=batch['target_ids'].flatten())


def make_byte_loss(decoder, compiled=True):
    def loss(hidden, targets, mask):
        start = torch.full((len(targets),1),256,dtype=torch.long,device=targets.device)
        shifted = torch.cat((start, targets[:,:-1].long()),dim=1)
        position = torch.arange(128,device=targets.device)
        condition = decoder.condition_projection(hidden.to(decoder.condition_projection.weight.dtype))
        x = decoder.byte_embedding(shifted)+decoder.position_embedding(position)+condition[:,None]
        for block in decoder.blocks:
            x = block(x)[0]
        logits = decoder.byte_head(decoder.final_norm(x))
        ce = F.cross_entropy(logits.float().reshape(-1,256),targets.long().reshape(-1),reduction='none')
        return (ce.reshape(-1,128).sum(-1)*mask).sum()
    return torch.compile(loss, fullgraph=True, dynamic=False) if compiled else loss


class ByteBackward:
    def __init__(self, model, batch, context, chunk=256, compiled=True):
        if batch*context % chunk:
            raise ValueError('Head chunk must divide the fixed positional shape')
        self.model,self.batch,self.context,self.chunk = model,batch,context,chunk
        self.loss = make_byte_loss(model.byte_decoder,compiled)

    def __call__(self, tiles, indices, byte_targets, mask, scale):
        model = self.model
        with torch.autocast(tiles.device.type,dtype=torch.float16,enabled=tiles.is_cuda):
            encoded = torch.cat([model.glyph_encoder(part) for part in tiles.split(256)])
            embeddings = encoded[indices].reshape(self.batch,self.context,-1)
            hidden = model.backbone(inputs_embeds=embeddings,attention_mask=None,
                                    use_cache=False,return_dict=True).last_hidden_state.reshape(self.batch*self.context,-1)
        leaf = hidden.detach().requires_grad_(True)
        total = torch.zeros((),dtype=torch.float64,device=tiles.device)
        denominator = mask.sum().clamp_min(1)*1024
        for start in range(0,len(mask),self.chunk):
            stop=start+self.chunk
            with torch.autocast(tiles.device.type,dtype=torch.float16,enabled=tiles.is_cuda):
                value=self.loss(leaf[start:stop],byte_targets[start:stop],mask[start:stop])
            (value/denominator*scale).backward()
            total.add_(value.detach().double())
        hidden.backward(leaf.grad)
        return total
