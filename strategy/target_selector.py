"""SectorTargetSelector — dynamically choose the target stock per sector.

For each sector and each retrain date the selector scores every candidate
member on a handful of suitability criteria and picks the best. **Only data
strictly before the retrain date is used** (the caller slices history; this
class never peeks forward).

Candidate pool = all members of the sector (the original target + predictors).
Whichever member wins becomes the target; the rest form the predictor pool.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class TargetChoice:
    etf:         str
    sector:      str
    target:      str
    candidates:  list[str]
    scores:      pd.DataFrame   # per-candidate component + total scores


def _ols_residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Residual of y regressed on [1, X] via least squares."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def _mean_reversion_score(resid: np.ndarray) -> float:
    """Speed of mean reversion of a residual series (statsmodels-free).

    Estimate the AR(1) decay by regressing Δresid_t on resid_{t-1}; a more
    negative slope => faster reversion. Returned as a positive score in ~[0,1].
    """
    r = resid[~np.isnan(resid)]
    if len(r) < 60 or np.std(r) == 0:
        return 0.0
    lag = r[:-1]
    d = np.diff(r)
    # slope of d on lag
    lag_c = lag - lag.mean()
    denom = np.dot(lag_c, lag_c)
    if denom == 0:
        return 0.0
    slope = np.dot(lag_c, d - d.mean()) / denom
    # slope in (-2, 0) for stationary mean reversion; map -> [0,1]
    return float(np.clip(-slope, 0.0, 1.0))


class SectorTargetSelector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.w = cfg.target_score_weights

    def select(self, etf: str, sector_name: str, members: list[str],
               prices: pd.DataFrame, returns: pd.DataFrame,
               volumes: pd.DataFrame, etf_returns: pd.Series | None) -> TargetChoice:
        """`prices/returns/volumes` are already sliced to history < retrain date."""
        members = [m for m in members if m in prices.columns]
        rows = {}
        for cand in members:
            p = prices[cand].dropna()
            r = returns[cand].dropna()
            if len(p) < self.cfg.target_min_history:
                continue
            peers = [m for m in members if m != cand and m in prices.columns]

            # liquidity: log median dollar volume
            dollar_vol = (prices[cand] * volumes[cand]).replace(0, np.nan).dropna()
            liq = np.log1p(dollar_vol.median()) if len(dollar_vol) else 0.0

            # corr with sector ETF
            etf_corr = 0.0
            if etf_returns is not None:
                j = pd.concat([r, etf_returns], axis=1).dropna()
                if len(j) > 30:
                    etf_corr = abs(j.iloc[:, 0].corr(j.iloc[:, 1]))

            # avg |corr| with peers
            peer_corr = 0.0
            if peers:
                cors = returns[peers].corrwith(r).abs()
                peer_corr = float(cors.mean()) if len(cors) else 0.0

            # mean reversion of residual vs equal-weight basket of peers
            mr = 0.0
            if peers:
                aligned = prices[[cand] + peers].dropna()
                if len(aligned) > 60:
                    basket = aligned[peers].mean(axis=1).values
                    resid = _ols_residual(aligned[cand].values, basket)
                    mr = _mean_reversion_score(resid)

            # volatility fit: 1 inside preferred band, decaying outside
            ann_vol = r.std() * np.sqrt(252)
            lo, hi = self.cfg.target_vol_band
            if lo <= ann_vol <= hi:
                vol_fit = 1.0
            else:
                edge = lo if ann_vol < lo else hi
                vol_fit = float(np.exp(-abs(ann_vol - edge) / 0.20))

            # history coverage
            hist = len(p) / len(prices)

            rows[cand] = dict(liquidity=liq, etf_corr=etf_corr, peer_corr=peer_corr,
                              mean_reversion=mr, vol_fit=vol_fit, history=hist)

        scores = pd.DataFrame(rows).T
        if scores.empty:
            # fall back to the original configured target
            return TargetChoice(etf, sector_name, members[0], members, scores)

        # cross-sectional min-max normalization within the candidate set
        norm = scores.copy()
        for c in norm.columns:
            col = norm[c]
            rng = col.max() - col.min()
            norm[c] = 0.5 if rng == 0 else (col - col.min()) / rng
        norm["total"] = sum(norm[c] * self.w[c] for c in self.w)
        scores = scores.join(norm["total"])
        best = scores["total"].idxmax()
        return TargetChoice(etf, sector_name, best, members,
                            scores.sort_values("total", ascending=False))
