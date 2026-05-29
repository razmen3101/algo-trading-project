"""TechnicalRuleFeatureBuilder — indicators + binary rules per ticker.

Reuses the project's existing ``analysis.indicators.Indicators`` for RSI, MACD,
ATR, OBV, PFE, SMA and EMA (computed on single-column frames).

Leakage policy:
  * Indicator values are point-in-time observable at the close of day t, so they
    are used as-is at t.
  * Numeric (non-binary) features are normalized with a *shifted* EWM z-score
    (mean/std exclude the current observation).
  * Binary rule features are strictly 0/1 and are NOT normalized.
Crossing rules compare t vs t-1 — both observable at t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.indicators import Indicators
from strategy.residual_features import ewm_z


def _cross_up(a: pd.Series, b) -> pd.Series:
    b_prev = b.shift(1) if isinstance(b, pd.Series) else b
    return ((a > b) & (a.shift(1) <= b_prev)).astype(int)


def _cross_dn(a: pd.Series, b) -> pd.Series:
    b_prev = b.shift(1) if isinstance(b, pd.Series) else b
    return ((a < b) & (a.shift(1) >= b_prev)).astype(int)


class TechnicalRuleFeatureBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, ticker: str, prices: pd.DataFrame, highs: pd.DataFrame,
              lows: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
        # single-column frames so we can reuse the vectorized Indicators class
        p = prices[[ticker]]
        ind = Indicators(p, highs[[ticker]], lows[[ticker]], volumes[[ticker]])
        px = p[ticker]

        rsi = ind.rsi()[ticker]
        macd_l, macd_s, macd_h = (df[ticker] for df in ind.macd())
        atr = ind.atr()[ticker]
        obv = ind.obv()[ticker]
        pfe = ind.pfe()[ticker]
        sma20 = px.rolling(20).mean()
        sma50 = px.rolling(50).mean()
        sma200 = px.rolling(200).mean()
        ema20 = px.ewm(span=20, adjust=False).mean()
        ema50 = px.ewm(span=50, adjust=False).mean()
        ema200 = px.ewm(span=200, adjust=False).mean()
        vol = volumes[ticker]
        roll_vol = px.pct_change(fill_method=None).rolling(20).std() * np.sqrt(252)
        bw, w52 = self.cfg.breakout_win, self.cfg.week52_win
        roll_hi = px.rolling(bw).max()
        roll_lo = px.rolling(bw).min()
        hi52 = px.rolling(w52).max()
        lo52 = px.rolling(w52).min()
        vol_ma = vol.rolling(self.cfg.vol_spike_win).mean()
        atr_pct = atr / px
        atr_med = atr_pct.rolling(100).median()
        obv_slope = obv.diff(10)

        # ---- numeric (continuous) features -> EWM-z normalized ---------- #
        numeric = {
            "rsi": rsi, "macd": macd_l, "macd_signal": macd_s, "macd_hist": macd_h,
            "atr": atr, "atr_pct": atr_pct, "obv": obv, "pfe": pfe,
            "px_sma20_ratio": px / sma20 - 1, "px_sma50_ratio": px / sma50 - 1,
            "px_sma200_ratio": px / sma200 - 1,
            "px_ema20_ratio": px / ema20 - 1, "px_ema50_ratio": px / ema50 - 1,
            "px_ema200_ratio": px / ema200 - 1,
            "roll_vol": roll_vol, "vol_ratio": vol / vol_ma - 1, "obv_slope": obv_slope,
        }
        out = {}
        for name, s in numeric.items():
            _, _, z = ewm_z(s, self.cfg.ewm_span)
            out[f"{name}_z"] = z

        # ---- binary rule features (0/1, not normalized) ---------------- #
        binary = {
            "rule_rsi_oversold":     (rsi < 30).astype(int),
            "rule_rsi_overbought":   (rsi > 70).astype(int),
            "rule_rsi_cross_up_30":  _cross_up(rsi, 30),
            "rule_rsi_cross_dn_70":  _cross_dn(rsi, 70),
            "rule_macd_above_sig":   (macd_l > macd_s).astype(int),
            "rule_macd_cross_up":    _cross_up(macd_l, macd_s),
            "rule_macd_cross_dn":    _cross_dn(macd_l, macd_s),
            "rule_sma20_gt_50":      (sma20 > sma50).astype(int),
            "rule_sma50_gt_200":     (sma50 > sma200).astype(int),
            "rule_px_gt_sma20":      (px > sma20).astype(int),
            "rule_px_gt_sma50":      (px > sma50).astype(int),
            "rule_px_gt_sma200":     (px > sma200).astype(int),
            "rule_px_cross_up_sma50": _cross_up(px, sma50),
            "rule_px_cross_dn_sma50": _cross_dn(px, sma50),
            "rule_breakout_20":      (px >= roll_hi).astype(int),
            "rule_breakdown_20":     (px <= roll_lo).astype(int),
            "rule_volume_spike":     (vol > 2 * vol_ma).astype(int),
            "rule_atr_high_regime":  (atr_pct > 1.5 * atr_med).astype(int),
            "rule_atr_low_regime":   (atr_pct < 0.75 * atr_med).astype(int),
            "rule_obv_trend_pos":    (obv_slope > 0).astype(int),
            "rule_obv_trend_neg":    (obv_slope < 0).astype(int),
            "rule_near_52w_high":    (px >= 0.98 * hi52).astype(int),
            "rule_near_52w_low":     (px <= 1.02 * lo52).astype(int),
        }
        out.update(binary)
        return pd.DataFrame(out, index=px.index)
