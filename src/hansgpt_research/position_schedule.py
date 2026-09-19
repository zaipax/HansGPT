"""Learning-rate horizon independent of a pilot run's stopping budget."""

from hansgpt_research.train_glyph_lm import learning_rate


def position_learning_rate(positions, cfg):
    mode = cfg.get('lr_schedule', 'constant')
    if mode == 'constant':
        return cfg['learning_rate']
    if mode != 'global_cosine':
        raise ValueError('Unknown learning-rate schedule')
    total = cfg['schedule_total_positions']
    warmup = cfg['warmup_positions']
    if not 0 < warmup < total or not 0 < cfg['target_tokens'] <= total:
        raise ValueError('Invalid global schedule or pilot budget')
    return learning_rate(positions, dict(cfg, warmup_tokens=warmup, target_tokens=total))
