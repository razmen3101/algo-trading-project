"""PredictorSelector — LASSO / ElasticNet *feature selection* only.

Given the target's price series and the candidate predictors' price series
(all within the training window), fit a sparse linear model on either returns
or log-indexed prices and keep the top-N predictors by absolute coefficient.
This model is NEVER used to predict live — it only decides which predictor
stocks feed the XGBoost models.

No future / validation / test data may influence the selection: the caller
passes a history-only slice, and the internal CV is a chronological
TimeSeriesSplit (no shuffling).
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class PredictorChoice:
    target:       str
    selected:     list[str]
    coefficients: pd.Series   # full ranking (abs-sorted), incl. zeros
    method:       str


class PredictorSelector:
    def __init__(self, cfg):
        self.cfg = cfg

    def _prepare_data(self, target: str, candidates: list[str],
                      returns: pd.DataFrame, prices: pd.DataFrame | None):
        if self.cfg.predictor_selection_input == "log_indexed_prices":
            if prices is None:
                return None
            cols = [target] + [c for c in candidates if c in prices.columns and c != target]
            data = prices[cols].dropna()
            if len(data) < 60 or len(cols) <= 1:
                return None
            return np.log(data / data.iloc[0])

        cands = [c for c in candidates if c in returns.columns and c != target]
        data = returns[[target] + cands].dropna()
        if len(data) < 60 or not cands:
            return None
        return data

    def select(self, target: str, candidates: list[str],
               returns: pd.DataFrame, prices: pd.DataFrame | None = None) -> PredictorChoice:
        """History-only selection on returns or log-indexed prices."""
        data = self._prepare_data(target, candidates, returns, prices)
        if data is None:
            cands = [c for c in candidates if c in returns.columns and c != target]
            return PredictorChoice(target, cands[: self.cfg.top_n_predictors],
                                   pd.Series(dtype=float), self.cfg.feature_selection_method)

        cands = [c for c in data.columns if c != target]

        y = data[target].values
        X = data[cands].values
        # standardize X with TRAIN-window stats only (this slice is train history)
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd

        cv = TimeSeriesSplit(n_splits=5)
        if self.cfg.feature_selection_method == "lasso":
            model = LassoCV(n_alphas=self.cfg.selection_alphas, cv=cv,
                            random_state=self.cfg.random_state, max_iter=5000)
        else:
            model = ElasticNetCV(l1_ratio=self.cfg.elasticnet_l1_ratio,
                                 n_alphas=self.cfg.selection_alphas, cv=cv,
                                 random_state=self.cfg.random_state, max_iter=5000)
        with warnings.catch_warnings():
            # n_alphas deprecation (sklearn>=1.7) and convergence chatter
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            model.fit(Xs, y)

        coefs = pd.Series(np.abs(model.coef_), index=cands).sort_values(ascending=False)
        nonzero = coefs[coefs > 0]
        ranked = (nonzero if len(nonzero) else coefs)
        selected = ranked.head(self.cfg.top_n_predictors).index.tolist()
        if not selected:   # degenerate: everything shrank to zero
            selected = cands[: self.cfg.top_n_predictors]
        return PredictorChoice(target, selected, coefs, self.cfg.feature_selection_method)
