"""Warmup-stable-decay learning-rate schedule (linear warmup, flat, linear decay)."""


def wsd_lr(step: int, total_steps: int, peak_lr: float, warmup_frac: float = 0.02,
           decay_frac: float = 0.2, final_ratio: float = 0.0) -> float:
    warm = max(1, int(round(total_steps * warmup_frac))) if warmup_frac > 0 else 0
    decay_start = total_steps - int(round(total_steps * decay_frac))
    if step < warm:
        return peak_lr * step / warm
    if step < decay_start:
        return peak_lr
    frac = (step - decay_start) / max(1, total_steps - decay_start)
    frac = min(1.0, frac)
    return peak_lr * (1.0 - frac * (1.0 - final_ratio))
