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


def rolling_percentile_shifted(s: pd.Series, win: int) -> pd.Series:
    return s.rolling(win).apply(
        lambda w: (w[:-1] < w[-1]).mean() if len(w) > 1 else np.nan, raw=True
    ).shift(1)


def lag1_autocorr_shifted(s: pd.Series, win: int) -> pd.Series:
    return s.rolling(win).apply(
        lambda w: pd.Series(w).autocorr(lag=1) if len(w) > 2 else np.nan,
        raw=False,
    ).shift(1)


def rolling_signed_rank_shifted(s: pd.Series, win: int) -> pd.Series:
    return s.rolling(win).apply(
        lambda w: float(pd.Series(w).rank(pct=True).iloc[-1]) if len(w) > 1 else np.nan,
        raw=False,
    ).shift(1)


def rolling_abs_percentile_shifted(s: pd.Series, win: int) -> pd.Series:
    return s.abs().rolling(win).apply(
        lambda w: float(pd.Series(w).rank(pct=True).iloc[-1]) if len(w) > 1 else np.nan,
        raw=False,
    ).shift(1)


def consecutive_days_above(series: pd.Series, threshold: float) -> pd.Series:
    out = []
    run = 0
    for value in series.values:
        if pd.notna(value) and abs(value) > threshold:
            run += 1
        else:
            run = 0
        out.append(run)
    return pd.Series(out, index=series.index, dtype=float)


def distance_from_peak(abs_series: pd.Series, reset_threshold: float = 1.0) -> pd.Series:
    out = []
    peak = np.nan
    active = False
    for value in abs_series.values:
        if pd.isna(value):
            out.append(np.nan)
            continue
        if value > reset_threshold:
            if not active or pd.isna(peak):
                peak = value
                active = True
            else:
                peak = max(peak, value)
            out.append(0.0 if peak <= 0 else max(0.0, 1.0 - value / peak))
        else:
            active = False
            peak = np.nan
            out.append(0.0)
    return pd.Series(out, index=abs_series.index, dtype=float)


def residual_regime(abs_z: pd.Series) -> pd.DataFrame:
    bins = pd.cut(abs_z, bins=[-np.inf, 1.0, 1.5, 2.0, np.inf], labels=["calm", "normal", "elevated", "extreme"])
    return pd.get_dummies(bins, prefix="residual_regime")


def fibonacci_retrace_features(abs_z: pd.Series, distance_pct: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({
        "fib_retrace_pct": distance_pct,
        "fib_23_hit": (distance_pct >= 0.236).astype(float),
        "fib_38_hit": (distance_pct >= 0.382).astype(float),
        "fib_50_hit": (distance_pct >= 0.500).astype(float),
        "fib_61_hit": (distance_pct >= 0.618).astype(float),
        "fib_78_hit": (distance_pct >= 0.786).astype(float),
    }, index=abs_z.index)
    return out


def family_feature_block(series: pd.Series, prefix: str, span: int, win: int) -> pd.DataFrame:
    mean, std, z = ewm_z(series, span)
    rank = rolling_signed_rank_shifted(series, win)
    percentile = rolling_abs_percentile_shifted(series, win)
    velocity = series.diff()
    acceleration = velocity.diff()
    abs_z = z.abs()
    days_1 = consecutive_days_above(z, 1.0)
    days_15 = consecutive_days_above(z, 1.5)
    days_2 = consecutive_days_above(z, 2.0)
    days_3 = consecutive_days_above(z, 3.0)
    dist_peak = distance_from_peak(abs_z)
    reversion_speed = abs_z.shift(1) - abs_z
    fib = fibonacci_retrace_features(abs_z, dist_peak)
    regime = residual_regime(abs_z)

    out = pd.DataFrame({
        f"{prefix}_residual": series,
        f"{prefix}_residual_z": z,
        f"{prefix}_residual_rank": rank,
        f"{prefix}_residual_percentile": percentile,
        f"{prefix}_velocity": velocity,
        f"{prefix}_acceleration": acceleration,
        f"{prefix}_days_above_1_sigma": days_1,
        f"{prefix}_days_above_1_5_sigma": days_15,
        f"{prefix}_days_above_2_sigma": days_2,
        f"{prefix}_days_above_3_sigma": days_3,
        f"{prefix}_distance_from_peak": dist_peak,
        f"{prefix}_reversion_speed": reversion_speed,
    }, index=series.index)
    out = out.join(fib.add_prefix(f"{prefix}_"))
    out = out.join(regime.add_prefix(f"{prefix}_"))
    out[f"{prefix}_residual_ewm_mean"] = mean
    out[f"{prefix}_residual_ewm_std"] = std
    out[f"{prefix}_residual_ewm_z"] = z
    out[f"{prefix}_residual_abs_z"] = abs_z
    out[f"{prefix}_residual_sign"] = np.sign(series)
    out[f"{prefix}_residual_distance_from_zero"] = series.abs()
    return out


class ResidualFeatureBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def _residual_series(self, price: pd.Series, shadow_price: pd.Series) -> pd.DataFrame:
        shadow_safe = shadow_price.replace(0, np.nan)
        raw_residual = (price - shadow_price).rename("raw_residual")
        pct_residual = ((price - shadow_price) / shadow_safe).rename("percent_residual")
        log_residual = np.log(price / shadow_safe).rename("log_residual")

        residual_type = getattr(self.cfg, "residual_type", "raw")
        if residual_type == "percent":
            selected = pct_residual.rename("price_residual")
        elif residual_type == "log":
            selected = log_residual.rename("price_residual")
        else:
            selected = raw_residual.rename("price_residual")

        return pd.concat([selected, raw_residual, pct_residual, log_residual], axis=1)

    def build(self, price: pd.Series, shadow_price: pd.Series,
              predicted_return: pd.Series,
              fwd_return: pd.Series | None = None) -> pd.DataFrame:
        span = self.cfg.ewm_span
        win = self.cfg.resid_roll_win

        residual_df = self._residual_series(price, shadow_price)
        residual = residual_df["price_residual"]
        gap = (price / shadow_price - 1.0).rename("shadow_price_gap_pct")

        r_mean, r_std, r_z = ewm_z(residual, span)
        # rolling stats (shifted) as a complement to EWM
        roll_mean = residual.rolling(win).mean().shift(1)
        roll_std = residual.rolling(win).std().shift(1)
        roll_rank = rolling_percentile_shifted(residual, win)

        # live-safe return-derived features (all causal normalization)
        _, _, pred_ret_z = ewm_z(predicted_return, span)
        pred_ret_pct = rolling_abs_percentile_shifted(predicted_return, win)
        pred_ret_rank = rolling_signed_rank_shifted(predicted_return, win)
        pred_ret_ewm_z = pred_ret_z.copy()

        residual_abs_z = r_z.abs().rename("residual_abs_z")
        residual_sign = np.sign(residual).rename("residual_sign")
        residual_dist0 = residual.abs().rename("residual_distance_from_zero")
        residual_pct = roll_rank.rename("residual_percentile")
        residual_ewm_slope = r_z.diff().rename("residual_ewm_slope")
        lag1 = lag1_autocorr_shifted(residual, win)
        residual_half_life_proxy = pd.Series(
            np.where((lag1 > 0) & (lag1 < 1), -np.log(2.0) / np.log(lag1), np.nan),
            index=residual.index,
            name="residual_half_life_proxy",
        )
        residual_excursion_bucket = pd.cut(
            residual_abs_z,
            bins=[-np.inf, 1.0, 1.5, 2.0, 3.0, np.inf],
            labels=[0, 1, 2, 3, 4],
        ).astype(float).rename("residual_excursion_bucket")

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
            "residual_percentile":   residual_pct,
            "residual_ewm_slope":    residual_ewm_slope,
            "residual_half_life_proxy": residual_half_life_proxy,
            "residual_abs_z":        residual_abs_z,
            "residual_sign":         residual_sign,
            "residual_distance_from_zero": residual_dist0,
            "residual_excursion_bucket": residual_excursion_bucket,
        })

        if getattr(self.cfg, "enable_multi_residual_engine", False):
            family_frames = []
            for prefix, series in (("raw", residual_df["raw_residual"]),
                                   ("percent", residual_df["percent_residual"]),
                                   ("log", residual_df["log_residual"])):
                family_frames.append(family_feature_block(series, prefix, span, win))
            family = pd.concat(family_frames, axis=1)
            out = out.join(family)

        if getattr(self.cfg, "enable_return_feature_expansion", False):
            pred_days_pos = consecutive_days_above(predicted_return, 0.0)
            pred_days_neg = consecutive_days_above(-predicted_return, 0.0)
            pred_abs = predicted_return.abs()
            pred_peak = distance_from_peak(pred_abs, reset_threshold=0.0)
            pred_regime = pd.cut(
                pred_ret_z, bins=[-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf],
                labels=["strong_negative", "weak_negative", "neutral", "weak_positive", "strong_positive"],
            )
            pred_regime_oh = pd.get_dummies(pred_regime, prefix="predicted_return_regime")
            out = out.join(pd.DataFrame({
                "predicted_return_z": pred_ret_z,
                "predicted_return_rank": pred_ret_rank,
                "predicted_return_percentile": pred_ret_pct,
                "predicted_return_direction": np.sign(predicted_return),
                "predicted_return_ewm_z": pred_ret_ewm_z,
                "predicted_return_velocity": predicted_return.diff(),
                "predicted_return_acceleration": predicted_return.diff().diff(),
                "predicted_return_distance_from_extreme": pred_peak,
                "predicted_return_days_positive": pred_days_pos,
                "predicted_return_days_negative": pred_days_neg,
                "predicted_return_regime": pred_regime.astype(object),
            }, index=predicted_return.index))
            out = out.join(pred_regime_oh)

        # evaluation-only columns (NOT fed to the classifier)
        if fwd_return is not None:
            out["fwd_return"] = fwd_return
            out["return_residual"] = fwd_return - predicted_return
        return out

    @staticmethod
    def feature_columns(df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in EVAL_ONLY]
