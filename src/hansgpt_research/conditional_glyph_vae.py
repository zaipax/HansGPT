"""One shared continuous latent per glyph; posterior targets never enter generation."""

import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel

from hansgpt_research.attention_glyph_lm import AttentionGlyphEncoder
from hansgpt_research.dual_decoder_glyph_lm import (
    DualDecoderConfig,
    SemanticDecoder,
    SpatialGlyphDecoder,
)
from hansgpt_research.glyph_codec import SpatialQueryEncoder
from hansgpt_research.glyph_lm import ModelConfig, _unique_binary_pixels, validate_binary_tiles


def gaussian_kl(q_mean, q_logvar, p_mean, p_logvar):
    """Analytic KL(q||p), in nats per whole glyph, not per pixel."""
    qm, ql, pm, pl = (x.float() for x in (q_mean, q_logvar, p_mean, p_logvar))
    return 0.5 * (pl - ql + (ql.exp() + (qm - pm).square()) * torch.exp(-pl) - 1).sum(-1)


def gaussian_log_prob(z, mean, logvar):
    return -0.5 * (
        torch.log(torch.tensor(2 * torch.pi, device=z.device))
        + logvar.float()
        + (z.float() - mean.float()).square() * torch.exp(-logvar.float())
    ).sum(-1)


def sample_gaussian(mean, logvar, generator=None):
    noise = torch.randn(mean.shape, device=mean.device, dtype=torch.float32, generator=generator)
    return mean.float() + torch.exp(0.5 * logvar.float()) * noise


class ConditionalGlyphVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spec = config
        self.config = ModelConfig.from_dict(config["model"])
        enc = config["encoder"]
        dec = DualDecoderConfig(**config["decoders"])
        self.slots = dec.semantic_slots
        self.width = dec.width
        original = AttentionGlyphEncoder(self.config.hidden_size, **enc)
        self.encoder = SpatialQueryEncoder(original, slots=self.slots, width=enc["width"])
        self.input_projection = nn.Linear(
            self.slots * enc["width"], self.config.hidden_size, bias=False
        )
        self.input_norm = nn.RMSNorm(self.config.hidden_size)
        m = self.config
        lc = LlamaConfig(
            vocab_size=1,
            hidden_size=m.hidden_size,
            num_hidden_layers=m.num_hidden_layers,
            num_attention_heads=m.num_attention_heads,
            num_key_value_heads=m.num_key_value_heads,
            intermediate_size=m.intermediate_size,
            head_dim=m.hidden_size // m.num_attention_heads,
            max_position_embeddings=m.max_position_embeddings,
            rope_theta=m.rope_theta,
            attention_dropout=0.0,
            use_cache=False,
            _attn_implementation="sdpa",
        )
        self.backbone = LlamaModel(lc)
        self.backbone.embed_tokens = None
        self.backbone.main_input_name = "inputs_embeds"
        self.semantic = SemanticDecoder(m, dec)
        self.context_projection = nn.Linear(m.hidden_size, self.slots * self.width)
        self.target_projection = nn.Linear(enc["width"], self.width)
        self.prior_query = nn.Parameter(torch.randn(1, 1, self.width) * 0.02)
        self.posterior_query = nn.Parameter(torch.randn(1, 1, self.width) * 0.02)

        def transformer():
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    self.width,
                    dec.heads,
                    dec.intermediate_size,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=1,
                enable_nested_tensor=False,
            )

        self.prior_transformer = transformer()
        self.posterior_transformer = transformer()
        latent = config["vae"]["latent_dim"]
        self.prior_head = nn.Linear(self.width, latent * 2)
        self.posterior_head = nn.Linear(self.width, latent * 2)
        self.latent_projection = nn.Linear(latent, self.slots * self.width)
        self.decoder = SpatialGlyphDecoder(dec)

    def glyph_features(self, tiles):
        flat = tiles.reshape(-1, 1, 32, 32)
        validate_binary_tiles(flat.unsqueeze(1))
        unique, inverse = _unique_binary_pixels(flat.flatten(1).to(torch.uint8), "packed")
        encoded = self.encoder(unique.reshape(-1, 1, 32, 32))
        return encoded[inverse]

    def forward_hidden(
        self, glyphs, attention_mask=None, past_key_values=None, use_cache=False, return_cache=False
    ):
        validate_binary_tiles(glyphs)
        features = self.glyph_features(glyphs).flatten(1)
        inputs = self.input_norm(self.input_projection(features)).reshape(*glyphs.shape[:2], -1)
        outer_cache, semantic_cache = past_key_values or (None, None)
        output = self.backbone(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            past_key_values=outer_cache,
            use_cache=use_cache,
            return_dict=True,
        )
        hidden, semantic_cache = self.semantic(
            output.last_hidden_state, attention_mask, semantic_cache, use_cache
        )
        return (hidden, (output.past_key_values, semantic_cache)) if return_cache else hidden

    def context_tokens(self, hidden):
        return self.context_projection(hidden).reshape(-1, self.slots, self.width)

    def prior(self, hidden):
        context = self.context_tokens(hidden)
        tokens = torch.cat((self.prior_query.expand(len(context), -1, -1), context), 1)
        mean, logvar = self.prior_head(self.prior_transformer(tokens)[:, 0]).chunk(2, -1)
        return mean.float(), logvar.float().clamp(-6, 2)

    def posterior(self, hidden, targets):
        context = self.context_tokens(hidden)
        target = self.target_projection(self.glyph_features(targets))
        tokens = torch.cat((self.posterior_query.expand(len(context), -1, -1), context, target), 1)
        mean, logvar = self.posterior_head(self.posterior_transformer(tokens)[:, 0]).chunk(2, -1)
        return mean.float(), logvar.float().clamp(-6, 2)

    def decode(self, hidden, z):
        # One global z generates all spatial conditions; pixels never sample independently.
        memory = self.context_tokens(hidden) + self.latent_projection(z).reshape(
            -1, self.slots, self.width
        )
        return self.decoder(memory.flatten(1))

    @torch.inference_mode()
    def generate(self, prompt, max_new, *, generator=None, eos_glyph=None, use_cache=True):
        validate_binary_tiles(prompt)
        self.eval()
        context = prompt
        cache = None
        tiles = []
        finished = torch.zeros(len(prompt), dtype=torch.bool, device=prompt.device)
        model_input = context[:, -self.config.max_position_embeddings :]
        for _ in range(max_new):
            hidden, cache = self.forward_hidden(
                model_input, past_key_values=cache, use_cache=use_cache, return_cache=True
            )
            hidden = hidden[:, -1]
            mean, logvar = self.prior(hidden)
            z = sample_gaussian(mean, logvar, generator)
            tile = (self.decode(hidden, z) >= 0).to(torch.uint8).unsqueeze(1)
            tile[finished] = 0
            tiles.append(tile)
            if eos_glyph is not None:
                finished |= (
                    (tile == eos_glyph.to(tile.device).reshape(1, 1, 1, 32, 32)).flatten(1).all(1)
                )
                if bool(finished.all()):
                    break
            context = torch.cat((context, tile), 1)
            if use_cache and context.shape[1] <= self.config.max_position_embeddings:
                model_input = tile
            else:
                cache = None
                model_input = context[:, -self.config.max_position_embeddings :]
        return torch.cat(tiles, 1) if tiles else prompt[:, :0]
