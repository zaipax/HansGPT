"""Whole-tile latent mixtures and conditional adversarial tools for binary glyphs.

No character IDs or glyph gallery enter these models. Mixture components are
context-dependent image distributions. A sampled component is shared by all
1024 pixels of one outer language position, and generated pixels feed the CNN.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from hansgpt_research.glyph_lm import GlyphGPT, ModelConfig, validate_binary_tiles

DECODE_STRATEGIES = frozenset({"mode_threshold", "sample_threshold", "sample_pixels"})


def _check_threshold(threshold: float) -> None:
    if not 0 < threshold < 1:
        raise ValueError("The binary threshold must be in (0, 1)")


def _check_binary_images(images: Tensor, leading_shape: tuple[int, ...], name: str) -> None:
    if tuple(images.shape) != (*leading_shape, 1, 32, 32):
        raise ValueError(f"{name} must have shape {(*leading_shape, 1, 32, 32)}")
    if not bool(((images == 0) | (images == 1)).all()):
        raise ValueError(f"{name} must contain only binary 0/1 pixels")


@dataclass(frozen=True)
class GlyphDistribution:
    """A normalized mixture of whole-grid product Bernoulli distributions.

    pixel_logits: [..., components, 1024]
    component_logits: [..., components]
    The leading dimensions can be [batch, sequence], valid positions only, or
    empty for one grid. Likelihood computations use float32 under AMP.
    """

    pixel_logits: Tensor
    component_logits: Tensor

    def __post_init__(self) -> None:
        if self.pixel_logits.ndim < 2 or self.pixel_logits.shape[-1] != 1024:
            raise ValueError("pixel_logits must have shape [..., components, 1024]")
        if self.pixel_logits.shape[-2] < 1:
            raise ValueError("At least one mixture component is required")
        if self.component_logits.shape != self.pixel_logits.shape[:-1]:
            raise ValueError("Component logits must match all leading and component dimensions")
        if self.pixel_logits.device != self.component_logits.device:
            raise ValueError("Pixel and component logits must share a device")
        if (
            not self.pixel_logits.is_floating_point()
            or not self.component_logits.is_floating_point()
        ):
            raise ValueError("Distribution logits must be floating point")

    @property
    def components(self) -> int:
        return self.pixel_logits.shape[-2]

    @property
    def leading_shape(self) -> tuple[int, ...]:
        return tuple(self.pixel_logits.shape[:-2])

    def _component_log_prob(self, targets: Tensor) -> Tensor:
        _check_binary_images(targets, self.leading_shape, "targets")
        target_pixels = targets.float().reshape(*self.leading_shape, 1, 1024)
        return -F.binary_cross_entropy_with_logits(
            self.pixel_logits.float(), target_pixels.expand_as(self.pixel_logits), reduction="none"
        ).sum(dim=-1)

    def nll(self, targets: Tensor) -> Tensor:
        """Return [...]: exact whole-grid mixture NLL divided by 1024, nats/pixel.

        Targets are already shifted next-grid labels. No target image is passed
        to the CNN here. Pixel log-probabilities are summed before adding log pi
        and marginalizing the single whole-grid component with logsumexp.
        """
        component_log_prob = self._component_log_prob(targets)
        if self.components == 1:
            return -component_log_prob.squeeze(-1) / 1024
        weighted_log_prob = (
            F.log_softmax(self.component_logits.float(), dim=-1) + component_log_prob
        )
        return -torch.logsumexp(weighted_log_prob, dim=-1) / 1024

    def responsibilities(self, targets: Tensor) -> Tensor:
        """Posterior component usage for training diagnostics, never input features."""
        scores = F.log_softmax(self.component_logits.float(), dim=-1)
        return (scores + self._component_log_prob(targets)).softmax(dim=-1)

    def selected_pixel_logits(
        self,
        strategy: str = "mode_threshold",
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Select one component and return differentiable logits [...,1,32,32].

        mode_threshold selects argmax pi separately for every tile (not a global
        fixed component, and not necessarily the grid MAP). The other strategies
        sample k ~ pi. Discrete selection does not supply a gradient to pi;
        mixture NLL trains those weights. This method permits ST adversarial use.
        """
        if strategy not in DECODE_STRATEGIES:
            raise ValueError(f"Unknown glyph decode strategy: {strategy}")
        if not bool(torch.isfinite(self.pixel_logits).all()) or not bool(
            torch.isfinite(self.component_logits).all()
        ):
            raise FloatingPointError("Cannot decode a glyph from nonfinite logits")
        if self.components == 1:
            selected = self.pixel_logits[..., 0, :]
        else:
            weights = self.component_logits.float().reshape(-1, self.components)
            if strategy == "mode_threshold":
                indices = weights.argmax(dim=-1, keepdim=True)
            else:
                indices = torch.multinomial(weights.softmax(-1), 1, generator=generator)
            pixels = self.pixel_logits.reshape(-1, self.components, 1024)
            selected = pixels.gather(1, indices.unsqueeze(-1).expand(-1, 1, 1024)).squeeze(1)
        return selected.reshape(*self.leading_shape, 1, 32, 32)

    @torch.no_grad()
    def decode(
        self,
        threshold: float = 0.5,
        strategy: str = "mode_threshold",
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Return strict uint8 images; never average component probability maps.

        sample_pixels samples the learned mixture: one k per grid, then its
        Bernoulli pixels. sample_threshold samples only k and thresholds its
        probabilities, so it is a distinct deterministic-per-component decoder.
        """
        _check_threshold(threshold)
        probabilities = self.selected_pixel_logits(strategy, generator).float().sigmoid()
        if strategy == "sample_pixels":
            random_values = torch.rand(
                probabilities.shape,
                device=probabilities.device,
                dtype=probabilities.dtype,
                generator=generator,
            )
            return (random_values < probabilities).to(torch.uint8)
        return (probabilities >= threshold).to(torch.uint8)


class StructuredGlyphGPT(GlyphGPT):
    """The v1 CNN and causal backbone with a replaceable whole-tile mixture head."""

    def __init__(self, config: ModelConfig, components: int = 1, init_noise: float = 0.001):
        if isinstance(components, bool) or not isinstance(components, int) or components < 1:
            raise ValueError("components must be a positive integer")
        if not math.isfinite(init_noise) or init_noise < 0:
            raise ValueError("init_noise must be a finite nonnegative standard deviation")
        super().__init__(config)
        self.components = components
        self.init_noise = init_noise
        self.component_head: nn.Linear | None = None
        if components > 1:
            old_weight, old_bias = self.pixel_head.weight.detach(), self.pixel_head.bias.detach()
            self.pixel_head = nn.Linear(config.hidden_size, components * 1024, bias=True)
            self.component_head = nn.Linear(config.hidden_size, components, bias=True)
            with torch.no_grad():
                self.pixel_head.weight.copy_(old_weight.repeat(components, 1))
                self.pixel_head.bias.copy_(old_bias.repeat(components))
                if init_noise:
                    self.pixel_head.weight.add_(
                        torch.randn_like(self.pixel_head.weight) * init_noise
                    )
                    self.pixel_head.bias.add_(torch.randn_like(self.pixel_head.bias) * init_noise)
                self.component_head.weight.zero_()
                self.component_head.bias.zero_()

    def load_v1_state_dict(self, state: Mapping[str, Tensor]) -> Any:
        """Strictly migrate all v1 weights; K1 is exact, K>1 copies and perturbs heads.

        Source tensors are never modified. No source character embeddings or
        already-structured checkpoint keys are silently accepted or discarded.
        """
        expected = set(self.state_dict()) - {"component_head.weight", "component_head.bias"}
        if set(state) != expected:
            raise ValueError(
                "V1 checkpoint keys differ: "
                f"missing={sorted(expected - set(state))}, "
                f"unexpected={sorted(set(state) - expected)}"
            )
        if state["pixel_head.weight"].shape != (1024, self.config.hidden_size):
            raise ValueError("V1 pixel head weight does not match the configured hidden dimension")
        if state["pixel_head.bias"].shape != (1024,):
            raise ValueError("V1 pixel head must predict exactly 1024 binary pixels")
        if self.components == 1:
            return self.load_state_dict(state, strict=True)
        migrated = dict(state)
        migrated["pixel_head.weight"] = state["pixel_head.weight"].repeat(self.components, 1)
        migrated["pixel_head.bias"] = state["pixel_head.bias"].repeat(self.components)
        if self.init_noise:
            for key in ("pixel_head.weight", "pixel_head.bias"):
                migrated[key] = migrated[key] + torch.randn_like(migrated[key]) * self.init_noise
        migrated["component_head.weight"] = torch.zeros_like(self.component_head.weight)
        migrated["component_head.bias"] = torch.zeros_like(self.component_head.bias)
        return self.load_state_dict(migrated, strict=True)

    def forward_hidden(
        self,
        glyphs: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = False,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, Any]:
        embeddings = self.encode_glyphs(glyphs)
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != glyphs.shape[0]:
                raise ValueError("attention_mask must be [batch, total_context_length]")
            if past_key_values is None and attention_mask.shape[1] != glyphs.shape[1]:
                raise ValueError("attention_mask must match the input sequence length")
            if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
                raise ValueError("attention_mask must be binary")
        outputs = self.backbone(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        if return_cache:
            return outputs.last_hidden_state, outputs.past_key_values
        return outputs.last_hidden_state

    def distribution(self, hidden: Tensor) -> GlyphDistribution:
        if hidden.shape[-1] != self.config.hidden_size:
            raise ValueError("Hidden states must end with the configured hidden dimension")
        leading_shape = hidden.shape[:-1]
        pixel_logits = self.pixel_head(hidden).reshape(*leading_shape, self.components, 1024)
        component_logits = (
            self.component_head(hidden)
            if self.component_head is not None
            else pixel_logits.new_zeros((*leading_shape, 1))
        )
        return GlyphDistribution(pixel_logits, component_logits)

    def head_nll(
        self,
        hidden: Tensor,
        targets: Tensor,
        chunk_size: int = 256,
        checkpoint_chunks: bool = False,
    ) -> Tensor:
        """Score selected valid hidden states in bounded chunks, returning [...].

        With checkpoint_chunks=True, recompute each head/loss chunk on backward
        rather than retaining its expanded mixture logits. Nonreentrant checkpoint
        supports frozen hidden states while still training the head parameters.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if hidden.shape[-1] != self.config.hidden_size:
            raise ValueError("Hidden states must end with the configured hidden dimension")
        leading_shape = tuple(hidden.shape[:-1])
        _check_binary_images(targets, leading_shape, "targets")
        flat_hidden = hidden.reshape(-1, self.config.hidden_size)
        flat_targets = targets.reshape(-1, 1, 32, 32)
        if not len(flat_hidden):
            raise ValueError("At least one valid target is required")

        def score(chunk_hidden: Tensor, chunk_targets: Tensor) -> Tensor:
            return self.distribution(chunk_hidden).nll(chunk_targets)

        losses = []
        for start in range(0, len(flat_hidden), chunk_size):
            chunk_hidden = flat_hidden[start : start + chunk_size]
            chunk_targets = flat_targets[start : start + chunk_size]
            if checkpoint_chunks and torch.is_grad_enabled():
                losses.append(checkpoint(score, chunk_hidden, chunk_targets, use_reentrant=False))
            else:
                losses.append(score(chunk_hidden, chunk_targets))
        return torch.cat(losses).reshape(leading_shape)

    def forward(
        self,
        glyphs: Tensor,
        attention_mask: Tensor | None = None,
        *,
        past_key_values: Any = None,
        use_cache: bool = False,
        return_cache: bool = False,
    ) -> GlyphDistribution | tuple[GlyphDistribution, Any]:
        hidden = self.forward_hidden(
            glyphs, attention_mask, past_key_values, use_cache, return_cache
        )
        if return_cache:
            values, cache = hidden
            return self.distribution(values), cache
        return self.distribution(hidden)

    @torch.no_grad()
    def generate(
        self,
        prompt_glyphs: Tensor,
        max_new_tokens: int,
        *,
        threshold: float = 0.5,
        eos_glyph: Tensor | None = None,
        use_cache: bool = True,
        strategy: str = "mode_threshold",
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate new uint8 grids, using the same CNN and no glyph-bank feedback.

        Prompts are unpadded. Exact EOS equality stops a row; completed rows have
        zero padding after their first EOS. Returns [batch,new_positions,1,32,32].
        Long contexts fall back to a correctly recomputed sliding window.
        """
        validate_binary_tiles(prompt_glyphs, name="prompt_glyphs")
        _check_threshold(threshold)
        if max_new_tokens < 0 or strategy not in DECODE_STRATEGIES:
            raise ValueError("Need nonnegative max_new_tokens and a supported decode strategy")
        if eos_glyph is not None:
            eos_glyph = eos_glyph.to(prompt_glyphs.device).reshape(1, 1, 1, 32, 32)
            validate_binary_tiles(eos_glyph, name="eos_glyph")
        was_training = self.training
        self.eval()
        try:
            context = prompt_glyphs.to(torch.uint8)
            generated: list[Tensor] = []
            finished = torch.zeros(context.shape[0], dtype=torch.bool, device=context.device)
            cache = None
            model_input = context[:, -self.config.max_position_embeddings :]
            for _ in range(max_new_tokens):
                hidden, next_cache = self.forward_hidden(
                    model_input, past_key_values=cache, use_cache=use_cache, return_cache=True
                )
                tile = self.distribution(hidden[:, -1:]).decode(threshold, strategy, generator)
                tile[finished] = 0
                generated.append(tile)
                if eos_glyph is not None:
                    finished |= (tile == eos_glyph).flatten(1).all(1)
                    if bool(finished.all()):
                        break
                context = torch.cat((context, tile), dim=1)
                if use_cache and context.shape[1] <= self.config.max_position_embeddings:
                    cache, model_input = next_cache, tile
                else:
                    cache = None
                    model_input = context[:, -self.config.max_position_embeddings :]
            return torch.cat(generated, dim=1) if generated else context[:, :0]
        finally:
            self.train(was_training)


def hard_binary_st(logits: Tensor, threshold: float = 0.5) -> Tensor:
    """Hard 0/1 float32 forward, sigmoid-surrogate backward (biased estimator).

    D sees the same binary domain for real and fake images. This float tensor
    exists only for gradient flow; use uint8 decode outputs for actual feedback.
    """
    _check_threshold(threshold)
    if not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
        raise ValueError("ST binarization needs finite floating-point logits")
    probabilities = logits.float().sigmoid()
    hard = (probabilities >= threshold).to(probabilities.dtype)
    return hard + (probabilities - probabilities.detach())


class ConditionalGlyphDiscriminator(nn.Module):
    """Per-image CNN and context projection, returning one unbounded score per grid.

    Call D(fake.detach(), h.detach()) for discriminator updates. For generator
    updates freeze D parameters and call D(hard_binary_st(logits), h.detach()).
    No inputs are implicitly detached here, so the image path retains ST grads.
    GroupNorm has no cross-image statistics or running-state updates.
    """

    def __init__(self, hidden_size: int = 768, channels: int = 32):
        super().__init__()
        if hidden_size < 1 or channels < 8 or channels % 8:
            raise ValueError("Need positive hidden_size and channels divisible by eight")
        self.hidden_size = hidden_size
        widths = (channels, channels * 2, channels * 4, channels * 4)
        layers: list[nn.Module] = []
        input_channels = 1
        for width in widths:
            layers.extend(
                [
                    nn.Conv2d(input_channels, width, 3, stride=2, padding=1),
                    nn.GroupNorm(8, width),
                    nn.LeakyReLU(0.2),
                ]
            )
            input_channels = width
        self.convolutions = nn.Sequential(*layers)
        self.image_projection = nn.Linear(widths[-1] * 2 * 2, widths[-1])
        self.context_projection = nn.Linear(hidden_size, widths[-1], bias=False)
        self.unconditional = nn.Linear(widths[-1], 1)
        self.projection_scale = math.sqrt(widths[-1])

    def forward(self, hard_tiles: Tensor, hidden: Tensor) -> Tensor:
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError("Discriminator context has the wrong hidden dimension")
        leading_shape = tuple(hidden.shape[:-1])
        _check_binary_images(hard_tiles, leading_shape, "discriminator images")
        images = hard_tiles.reshape(-1, 1, 32, 32).to(self.image_projection.weight.dtype)
        features = self.convolutions(images).flatten(1)
        features = F.leaky_relu(self.image_projection(features), negative_slope=0.2)
        condition = self.context_projection(
            hidden.reshape(-1, self.hidden_size).to(self.context_projection.weight.dtype)
        )
        score = self.unconditional(features).squeeze(-1)
        score = score + (features * condition).sum(-1) / self.projection_scale
        return score.reshape(leading_shape)
