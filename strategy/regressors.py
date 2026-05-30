"""Dynamic XGBoost regressors.

DynamicShadowPriceModel
    Learns the target's *fair value* from the selected predictors' prices
    (a contemporaneous cross-sectional relationship). The gap between the
    actual price and this shadow price is the core mispricing signal.

DynamicReturnModel
    Learns the target's *expected forward return* over ``return_horizon`` days
    from predictor prices/returns. The future return is used ONLY as a training
    label — never as a live feature.

Both are fit per walk-forward fold on the expanding training window and predict
strictly out-of-sample on the next block of days.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


def _feature_frame(prices: pd.DataFrame, predictors: list[str],
                   with_returns: bool) -> pd.DataFrame:
    """Build a leakage-free feature frame from predictor PRICES (+ optional
    same-day returns). All columns are observable at time t."""
    feats = prices[predictors].copy()
    feats.columns = [f"px_{c}" for c in predictors]
    if with_returns:
        rets = prices[predictors].pct_change(fill_method=None)
        rets.columns = [f"ret_{c}" for c in predictors]
        feats = feats.join(rets)
    return feats


@dataclass
class ShadowResult:
    shadow_price: pd.Series   # predicted fair value, indexed by date
    val_r2:       float


class DynamicShadowPriceModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = XGBRegressor(**cfg.reg_params)

    def fit(self, prices: pd.DataFrame, target: str, predictors: list[str],
            train_idx, val_idx=None) -> tuple[pd.DataFrame, pd.Series, float, pd.Index]:
        train_dates = pd.Index(train_idx)
        safe_train_idx = train_dates[:-self.cfg.return_horizon] if len(train_dates) > self.cfg.return_horizon else train_dates[:0]

        if self.cfg.price_transform == "log_indexed":
            feats, y, base_target_price = self._indexed_log_frame(prices, predictors, target, safe_train_idx)
        else:
            feats = _feature_frame(prices, predictors, with_returns=False)
            y = prices[target].rename("_y")
            base_target_price = float(prices[target].reindex(safe_train_idx).dropna().iloc[0]) if len(safe_train_idx) and not prices[target].reindex(safe_train_idx).dropna().empty else float("nan")

        tr = feats.loc[safe_train_idx].join(y).dropna()
        if len(tr) < 30:
            return pd.DataFrame(index=prices.index), pd.Series(dtype=float), float("nan"), pd.Index([])

        self.model.fit(tr.drop(columns="_y"), tr["_y"])
        return feats, y, base_target_price, safe_train_idx

    def predict(self, feats: pd.DataFrame, predict_idx, base_target_price: float = float("nan")) -> pd.Series:
        idx = pd.Index(predict_idx)
        pred = pd.Series(np.nan, index=idx, name="shadow_price")
        pred_feats = feats.loc[idx].dropna()
        if len(pred_feats):
            pred_y = self.model.predict(pred_feats)
            if self.cfg.price_transform == "log_indexed" and np.isfinite(base_target_price):
                pred.loc[pred_feats.index] = base_target_price * np.exp(pred_y)
            else:
                pred.loc[pred_feats.index] = pred_y
        return pred

    def _indexed_log_frame(self, prices: pd.DataFrame, predictors: list[str], target: str,
                           base_idx: pd.Index | pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series, float]:
        idx = pd.Index(base_idx)
        if len(idx) == 0:
            return pd.DataFrame(index=prices.index), pd.Series(dtype=float), float("nan")

        target_base = prices[target].reindex(idx).dropna()
        if target_base.empty:
            return pd.DataFrame(index=prices.index), pd.Series(dtype=float), float("nan")

        base_target_price = float(target_base.iloc[0])
        y = np.log(prices[target] / base_target_price).rename("_y")

        feats = {}
        for pred in predictors:
            pred_base = prices[pred].reindex(idx).dropna()
            if pred_base.empty:
                continue
            base_pred_price = float(pred_base.iloc[0])
            feats[f"px_{pred}"] = np.log(prices[pred] / base_pred_price)
        return pd.DataFrame(feats, index=prices.index), y, base_target_price

    def fit_predict(self, prices: pd.DataFrame, target: str, predictors: list[str],
                    train_idx, predict_idx, val_idx=None) -> ShadowResult:
        feats, y, base_target_price, safe_train_idx = self.fit(prices, target, predictors, train_idx, val_idx)
        safe_val_idx = pd.Index(val_idx)[:-self.cfg.return_horizon] if val_idx is not None and len(val_idx) > self.cfg.return_horizon else pd.Index([])

        if len(safe_train_idx) == 0:
            idx = prices.loc[predict_idx].index
            return ShadowResult(pd.Series(np.nan, index=idx), float("nan"))

        val_r2 = float("nan")
        if val_idx is not None and len(safe_val_idx):
            v = feats.loc[safe_val_idx].join(y).dropna()
            if len(v) > 5:
                val_r2 = r2_score(v["_y"], self.model.predict(v.drop(columns="_y")))

        pred_feats = feats.loc[predict_idx].dropna()
        shadow = pd.Series(np.nan, index=prices.loc[predict_idx].index, name="shadow_price")
        if len(pred_feats):
            pred_y = self.model.predict(pred_feats)
            if self.cfg.price_transform == "log_indexed" and np.isfinite(base_target_price):
                shadow.loc[pred_feats.index] = base_target_price * np.exp(pred_y)
            else:
                shadow.loc[pred_feats.index] = pred_y
        return ShadowResult(shadow, val_r2)


@dataclass
class ReturnResult:
    predicted_return: pd.Series
    val_r2:           float


class DynamicReturnModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = XGBRegressor(**cfg.reg_params)

    def fit(self, prices: pd.DataFrame, target: str, predictors: list[str],
            train_idx, val_idx=None) -> tuple[pd.DataFrame, pd.Series, pd.Index]:
        h = self.cfg.return_horizon
        feats = _feature_frame(prices, predictors, with_returns=True)
        fwd_ret = prices[target].shift(-h) / prices[target] - 1.0

        train_dates = pd.Index(train_idx)
        safe_train_idx = train_dates[:-h] if len(train_dates) > h else train_dates[:0]
        tr = feats.loc[safe_train_idx].join(fwd_ret.rename("_y")).dropna()
        if len(tr) < 30:
            return pd.DataFrame(index=prices.index), pd.Series(dtype=float), pd.Index([])

        self.model.fit(tr.drop(columns="_y"), tr["_y"])
        return feats, fwd_ret, safe_train_idx

    def predict(self, feats: pd.DataFrame, predict_idx) -> pd.Series:
        idx = pd.Index(predict_idx)
        pred = pd.Series(np.nan, index=idx, name="predicted_return")
        pred_feats = feats.loc[idx].dropna()
        if len(pred_feats):
            pred.loc[pred_feats.index] = self.model.predict(pred_feats)
        return pred

    def fit_predict(self, prices: pd.DataFrame, target: str, predictors: list[str],
                    train_idx, predict_idx, val_idx=None) -> ReturnResult:
        feats, fwd_ret, safe_train_idx = self.fit(prices, target, predictors, train_idx, val_idx)
        h = self.cfg.return_horizon
        safe_val_idx = pd.Index(val_idx)[:-h] if val_idx is not None and len(val_idx) > h else pd.Index([])

        if len(safe_train_idx) == 0:
            idx = prices.loc[predict_idx].index
            return ReturnResult(pd.Series(np.nan, index=idx), float("nan"))

        val_r2 = float("nan")
        if val_idx is not None and len(safe_val_idx):
            v = feats.loc[safe_val_idx].join(fwd_ret.rename("_y")).dropna()
            if len(v) > 5:
                val_r2 = r2_score(v["_y"], self.model.predict(v.drop(columns="_y")))

        pred_feats = feats.loc[predict_idx].dropna()
        pred = pd.Series(np.nan, index=prices.loc[predict_idx].index, name="predicted_return")
        if len(pred_feats):
            pred.loc[pred_feats.index] = self.model.predict(pred_feats)
        return ReturnResult(pred, val_r2)
