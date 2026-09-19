"""Shared pytest fixtures and utilities for HansGPT test suites."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def tiny_cvae_config() -> dict:
    """Minimal CVAE configuration for unit testing without downloading models or weights."""
    return dict(
        model=dict(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=32,
        ),
        encoder=dict(width=32, layers=1, heads=4),
        decoders=dict(
            width=64,
            semantic_layers=1,
            semantic_slots=4,
            glyph_layers=1,
            part_slots=4,
            heads=4,
            intermediate_size=128,
        ),
        vae=dict(latent_dim=8),
    )
