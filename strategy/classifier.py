"""GlobalSignalClassifier — one shared XGBoost classifier across all sectors.

Labels are threshold-based on the realized forward return over
``label_horizon`` (used as a LABEL only, never as a feature):

    fwd > +thr  ->  +1 (long)
    fwd < -thr  ->  -1 (short)
    otherwise   ->   0 (flat)

Tuning (random search) scores candidates on the VALIDATION set only; the test
set is never touched here. The final model is refit on train+val with the
chosen params and used to predict the locked test set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score

# external label <-> xgboost class index
_TO_IDX = {-1: 0, 0: 1, 1: 2}
_FROM_IDX = {0: -1, 1: 0, 2: 1}


def make_labels(price: pd.Series, cfg) -> pd.Series:
    """Threshold (optionally volatility-adjusted) forward-return labels."""
    h = cfg.label_horizon
    fwd = price.shift(-h) / price - 1.0
    pos, neg = cfg.positive_threshold, cfg.negative_threshold
    if cfg.vol_adjusted_labels:
        # scale thresholds by trailing realized vol over the horizon (shifted)
        daily = price.pct_change(fill_method=None)
        scale = daily.rolling(20).std().shift(1) * np.sqrt(h)
        scale = scale / scale.median()
        pos, neg = pos * scale, neg * scale
    lab = pd.Series(0, index=price.index, dtype=float)
    lab[fwd > pos] = 1
    lab[fwd < neg] = -1
    lab[fwd.isna()] = np.nan
    return lab.rename("label")


@dataclass
class ClassifierResult:
    params:       dict
    val_metrics:  dict
    feature_importance: pd.Series
    classes_:     np.ndarray = field(default=None)


class GlobalSignalClassifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_: XGBClassifier | None = None
        self.features_: list[str] = []
        self.result_: ClassifierResult | None = None

    # ---- helpers ------------------------------------------------------- #
    def _new_model(self, params: dict) -> XGBClassifier:
        p = dict(params)
        p.setdefault("num_class", 3)
        return XGBClassifier(**p)

    def _param_grid(self):
        rng = np.random.RandomState(self.cfg.random_state)
        base = self.cfg.clf_params
        for _ in range(self.cfg.random_search_iter):
            yield {**base,
                   "n_estimators": int(rng.choice([200, 300, 400, 600])),
                   "learning_rate": float(rng.choice([0.01, 0.02, 0.03, 0.05])),
                   "max_depth": int(rng.choice([2, 3, 4, 5])),
                   "subsample": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
                   "colsample_bytree": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
                   "reg_lambda": float(rng.choice([0.5, 1.0, 2.0, 5.0])),
                   "reg_alpha": float(rng.choice([0.0, 0.1, 0.5, 1.0]))}

    @staticmethod
    def _score(model, X, y_idx) -> dict:
        pred = model.predict(X)
        return {"accuracy": accuracy_score(y_idx, pred),
                "f1_macro": f1_score(y_idx, pred, average="macro")}

    # ---- fit (tune on val) + refit on train+val ------------------------ #
    def fit(self, X_tr, y_tr, X_val, y_val) -> ClassifierResult:
        self.features_ = list(X_tr.columns)
        ytr_i = y_tr.map(_TO_IDX).astype(int)
        yval_i = y_val.map(_TO_IDX).astype(int)

        if self.cfg.use_random_search:
            best, best_params, best_metrics = -np.inf, None, None
            for params in self._param_grid():
                m = self._new_model(params)
                m.fit(X_tr, ytr_i)
                metrics = self._score(m, X_val, yval_i)
                if metrics["f1_macro"] > best:
                    best, best_params, best_metrics = metrics["f1_macro"], params, metrics
            params, val_metrics = best_params, best_metrics
        else:
            params = dict(self.cfg.clf_params)
            m = self._new_model(params)
            m.fit(X_tr, ytr_i)
            val_metrics = self._score(m, X_val, yval_i)

        # final refit on train+val with chosen params (test stays untouched)
        X_all = pd.concat([X_tr, X_val])
        y_all = pd.concat([ytr_i, yval_i])
        self.model_ = self._new_model(params)
        self.model_.fit(X_all, y_all)

        imp = pd.Series(self.model_.feature_importances_, index=self.features_)
        self.result_ = ClassifierResult(params, val_metrics,
                                         imp.sort_values(ascending=False),
                                         classes_=np.array([-1, 0, 1]))
        return self.result_

    # ---- inference ----------------------------------------------------- #
    def predict(self, X) -> pd.Series:
        idx = self.model_.predict(X[self.features_])
        return pd.Series([_FROM_IDX[i] for i in idx], index=X.index, name="signal")

    def predict_proba(self, X) -> pd.DataFrame:
        proba = self.model_.predict_proba(X[self.features_])
        return pd.DataFrame(proba, index=X.index,
                            columns=["P_short", "P_flat", "P_long"])
