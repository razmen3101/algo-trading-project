"""ResidualFeatureBuilder — mispricing / anomaly features.

All normalization uses a *shifted* EWM (or rolling) statistic so that the
statistic at time t never includes the observation at time t:

    ewm_mean = s.ewm(span=K).mean().shift(1)
    ewm_std  = s.ewm(span=K).std().shift(1)
    z        = (s - ewm_mean) / ewm_std

`price_residual` and `shadow_price` are observable at t (the shadow price is a
fit on past data evaluated on same-day predictor prices), so they may be
features. `return_residual` requires the realized future return and is produced
for evaluation only — it is tagged in ``EVAL_ONLY`` and excluded from features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVAL_ONLY = ("return_residual", "fwd_return")


def ewm_z(s: pd.Series, span: int):
    mean = s.ewm(span=span).mean().shift(1)
    std = s.ewm(span=span).std().shift(1)
    z = (s - mean) / std.replace(0, np.nan)
    return mean, std, z


class ResidualFeatureBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, price: pd.Series, shadow_price: pd.Series,
              predicted_return: pd.Series,
              fwd_return: pd.Series | None = None) -> pd.DataFrame:
        span = self.cfg.ewm_span
        win = self.cfg.resid_roll_win

        residual = (price - shadow_price).rename("price_residual")
        gap = (price / shadow_price - 1.0).rename("shadow_price_gap_pct")

        r_mean, r_std, r_z = ewm_z(residual, span)
        # rolling stats (shifted) as a complement to EWM
        roll_mean = residual.rolling(win).mean().shift(1)
        roll_std = residual.rolling(win).std().shift(1)
        # rolling percentile rank of the current residual within the window
        roll_rank = residual.rolling(win).apply(
            lambda w: (w[:-1] < w[-1]).mean() if len(w) > 1 else np.nan, raw=True
        ).shift(1)

        out = pd.DataFrame({
            "price_residual":        residual,
            "price_residual_z":      r_z,
            "shadow_price":          shadow_price,
            "shadow_price_gap_pct":  gap,
            "predicted_return":      predicted_return,
            "residual_ewm_mean":     r_mean,
            "residual_ewm_std":      r_std,
            "residual_ewm_z":        r_z,
            "residual_roll_mean":    roll_mean,
            "residual_roll_std":     roll_std,
            "residual_rank":         roll_rank,
        })

        # evaluation-only columns (NOT fed to the classifier)
        if fwd_return is not None:
            out["fwd_return"] = fwd_return
            out["return_residual"] = fwd_return - predicted_return
        return out

    @staticmethod
    def feature_columns(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in EVAL_ONLY]
