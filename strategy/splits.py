"""Chronological splitting and the expanding-window walk-forward scheduler.

Hard rules enforced here:
  * splits are purely chronological (never shuffled);
  * the test set is carved off the END of the timeline and the walk-forward
    scheduler only ever produces folds whose prediction block lies inside the
    val+test region while training on everything strictly before it.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class ChronoSplit:
    train_end: pd.Timestamp   # exclusive upper bound of train
    val_end:   pd.Timestamp   # exclusive upper bound of validation
    index:     pd.DatetimeIndex

    @property
    def train_idx(self) -> pd.DatetimeIndex:
        return self.index[self.index < self.train_end]

    @property
    def val_idx(self) -> pd.DatetimeIndex:
        return self.index[(self.index >= self.train_end) & (self.index < self.val_end)]

    @property
    def test_idx(self) -> pd.DatetimeIndex:
        return self.index[self.index >= self.val_end]

    def describe(self) -> str:
        def span(ix):
            return f"{ix[0].date()}->{ix[-1].date()} ({len(ix)})" if len(ix) else "(empty)"
        return (f"train {span(self.train_idx)} | "
                f"val {span(self.val_idx)} | test {span(self.test_idx)}")


def chrono_split(index: pd.DatetimeIndex, cfg) -> ChronoSplit:
    """Split a DatetimeIndex into train/val/test by *trading-day counts*.

    Using row counts (not calendar dates) keeps the 3:1:1 ratio exact and
    robust to market holidays."""
    index = pd.DatetimeIndex(index).sort_values()
    n = len(index)
    per_year = n / (cfg.train_years + cfg.val_years + cfg.test_years)
    n_train = int(round(per_year * cfg.train_years))
    n_val = int(round(per_year * cfg.val_years))
    train_end = index[min(n_train, n - 1)]
    val_end = index[min(n_train + n_val, n - 1)]
    return ChronoSplit(train_end=train_end, val_end=val_end, index=index)


@dataclass
class WalkForwardFold:
    fold:        int
    train_idx:   pd.DatetimeIndex   # expanding window: everything < predict_start
    predict_idx: pd.DatetimeIndex   # the next `retrain_every` days to predict

    @property
    def retrain_date(self) -> pd.Timestamp:
        return self.predict_idx[0]


def walk_forward_folds(index: pd.DatetimeIndex, cfg) -> list[WalkForwardFold]:
    """Expanding-window scheduler.

    Start with ``min_train_days`` of history, predict the next
    ``retrain_every`` days, then fold those days into the training history and
    repeat. Every prediction is strictly out-of-sample w.r.t. its train window.
    """
    index = pd.DatetimeIndex(index).sort_values()
    n = len(index)
    folds: list[WalkForwardFold] = []
    start = cfg.min_train_days
    f = 0
    while start < n:
        stop = min(start + cfg.retrain_every, n)
        folds.append(WalkForwardFold(
            fold=f,
            train_idx=index[:start],
            predict_idx=index[start:stop],
        ))
        start = stop
        f += 1
    return folds
