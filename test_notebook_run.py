#!/usr/bin/env python
"""Test script to run DBTS_Train_Only_Diagnostic.ipynb code."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from xgboost import XGBClassifier

from config import SECTORS
from strategy.backtester import Backtester
from strategy.pipeline import StrategyPipeline
from strategy.position_manager import PositionManager, summarize_completed_trades
from strategy.splits import chrono_split
from strategy.strategy_config import StrategyConfig

print("[TEST] Setup and imports successful")

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.figsize": (14, 7),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
})


@dataclass
class TrainOnlyFold:
    retrain_date: pd.Timestamp
    train_idx: pd.DatetimeIndex
    predict_idx: pd.DatetimeIndex


def load_train_only_folds(md, cfg: StrategyConfig, split) -> list[TrainOnlyFold]:
    diag_path = Path("outputs/train_only_deep_diagnostic.json")
    if not diag_path.exists():
        raise FileNotFoundError("outputs/train_only_deep_diagnostic.json is required for the train-only retrain timeline")

    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    retrain_dates = sorted({pd.Timestamp(row["retrain_date"]) for row in diag.get("shadow_selected", [])})
    if not retrain_dates:
        raise ValueError("No retrain dates found in outputs/train_only_deep_diagnostic.json")

    index = pd.DatetimeIndex(md.prices.index).sort_values()
    folds: list[TrainOnlyFold] = []
    for i, retrain_date in enumerate(retrain_dates):
        next_retrain = retrain_dates[i + 1] if i + 1 < len(retrain_dates) else split.train_end
        train_idx = index[index < retrain_date]
        predict_idx = index[(index >= retrain_date) & (index < next_retrain) & (index < split.train_end)]
        if len(train_idx) and len(predict_idx):
            folds.append(TrainOnlyFold(retrain_date=retrain_date, train_idx=train_idx, predict_idx=predict_idx))
    return folds


def train_only_feature_columns(panel: pd.DataFrame) -> list[str]:
    excluded = {
        "date", "etf", "sector", "target", "predictors", "target_price",
        "shadow_price", "next_ret", "label", "spread_signal",
        "ann_vol", "residual_z", "price_residual", "residual_ewm_mean",
        "residual_ewm_std", "residual_roll_mean", "residual_roll_std",
        "fold_role", "retrain_date", "role", "true_label",
    }
    feature_cols = [c for c in panel.columns if c not in excluded]
    return [c for c in feature_cols if pd.api.types.is_numeric_dtype(panel[c])]


def build_position_manager(cfg: StrategyConfig) -> PositionManager:
    return PositionManager(
        long_entry_confidence=float(cfg.pm_entry_confidence),
        short_entry_confidence=float(cfg.pm_entry_confidence),
        flat_probability_block=float(cfg.flat_probability_block),
        entry_residual_threshold=float(cfg.pm_entry_residual_z),
        mean_reversion_exit=float(cfg.pm_exit_residual_z),
        opposite_signal_confidence=float(cfg.pm_opposite_confidence),
        stop_loss=float(cfg.pm_stop_loss),
        take_profit=float(cfg.pm_take_profit),
        max_holding_days=int(cfg.pm_max_holding_days),
        allow_flip=bool(cfg.pm_allow_flip),
    )


def fit_train_only_classifier(train_panel: pd.DataFrame, cfg: StrategyConfig):
    data = train_panel.copy().dropna(subset=["label", "next_ret"])
    feature_cols = train_only_feature_columns(data)
    if not feature_cols:
        raise ValueError("No numeric feature columns found for the train-only classifier")

    X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.groupby(data["target"]).ffill().fillna(0.0)
    y = data["label"].astype(int)
    y_idx = y.map({-1: 0, 0: 1, 1: 2}).astype(int)
    if set(y_idx.unique()) != {0, 1, 2}:
        raise ValueError("TRAIN-only classifier needs all three labels (-1, 0, 1) present in the training panel")

    params = dict(cfg.clf_params)
    params.update({
        "num_class": 3,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": int(cfg.random_state),
        "verbosity": 0,
    })
    model = XGBClassifier(**params)
    model.fit(X, y_idx)

    proba_raw = pd.DataFrame(model.predict_proba(X), index=X.index, columns=[0, 1, 2])
    proba = pd.DataFrame(index=X.index)
    proba["P_short"] = proba_raw[0]
    proba["P_flat"] = proba_raw[1]
    proba["P_long"] = proba_raw[2]

    signal = pd.Series(model.predict(X), index=X.index).map({0: -1, 1: 0, 2: 1}).astype(int)

    scored = data.copy()
    scored["true_label"] = scored["label"].astype(int)
    scored["signal"] = signal
    scored = pd.concat([scored, proba], axis=1)
    return scored, model, feature_cols


def sortino_ratio(returns: pd.Series) -> float:
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return float("nan")
    downside = r[r < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) else float("nan")
    if not np.isfinite(downside_std) or downside_std == 0:
        return float("nan")
    return float((r.mean() / downside_std) * np.sqrt(252))


print("[TEST] Loading market data...")
cfg = StrategyConfig(force_recompute=True, make_plots=False)
pipeline = StrategyPipeline(cfg)
market_data = pipeline.load_data()
split = chrono_split(market_data.prices.index, cfg)
folds = load_train_only_folds(market_data, cfg, split)

print(f"[TEST] TRAIN window: {split.train_idx[0].date()} -> {split.train_end.date()}")
print(f"[TEST] Train-only retrain folds: {len(folds)}")

print("[TEST] Building feature panel...")
panel = pipeline.build_panel(market_data, folds, split)
panel = panel.loc[panel["date"] < split.train_end].copy()
panel = panel.sort_values(["sector", "target", "date"]).reset_index(drop=True)
print(f"[TEST] Train-only panel rows: {len(panel):,}")
print(f"[TEST] Train-only active sectors: {panel['sector'].nunique()}")
print(f"[TEST] Train-only selected targets: {panel['target'].nunique()}")

print("[TEST] Fitting train-only classifier...")
audit_panel, model, feature_cols = fit_train_only_classifier(panel, cfg)
print("[TEST] Classifier fit successful")

print("[TEST] Building position manager and running backtest...")
position_manager = build_position_manager(cfg)
backtester = Backtester(cfg)
result = backtester.run(audit_panel, position_manager=position_manager)
completed_trades = summarize_completed_trades(result.trades)
if len(completed_trades) > 0:
    completed_trades = completed_trades.sort_values(["entry_date", "sector", "target"]).reset_index(drop=True)
else:
    completed_trades = completed_trades.reset_index(drop=True)

portfolio = result.portfolio.copy()
portfolio["drawdown"] = portfolio["equity"] / portfolio["equity"].cummax() - 1.0
portfolio["cum_pnl"] = portfolio["ret"].fillna(0.0).cumsum()

train_returns = portfolio["ret"].dropna()
sortino = sortino_ratio(train_returns)
metrics = dict(result.metrics)
metrics["sortino"] = sortino
metrics["n_completed_trades"] = int(len(completed_trades))
metrics["win_rate_completed_trades"] = float((completed_trades["pnl"] > 0).mean()) if len(completed_trades) else float("nan")
metrics["trade_log_rows"] = int(len(result.trades))
metrics["target_switches"] = int(audit_panel.sort_values(["sector", "date"]).groupby("sector")["target"].apply(lambda s: int(s.ne(s.shift()).sum() - 1 if len(s) else 0)).sum())

position_sign = np.sign(result.trades["position"].fillna(0.0)).astype(int)
position_counts = position_sign.value_counts().reindex([-1, 0, 1], fill_value=0)
selected_target_distribution = pd.crosstab(audit_panel["sector"], audit_panel["target"])
selected_target_distribution = selected_target_distribution.loc[:, selected_target_distribution.sum(axis=0).sort_values(ascending=False).index]
sector_target_switches = (
    audit_panel.sort_values(["sector", "date"])
    .groupby("sector")["target"]
    .apply(lambda s: int(s.ne(s.shift()).sum() - 1 if len(s) else 0))
    .rename("target_switches")
    .reset_index()
)

sector_summary = result.sector_perf.reset_index().rename(columns={"index": "sector"})
sector_summary = sector_summary.merge(
    audit_panel.groupby("sector").agg(
        selected_rows=("target", "size"),
        unique_targets=("target", "nunique"),
        target_switches=("target", lambda s: int(s.ne(s.shift()).sum() - 1 if len(s) else 0)),
    ).reset_index(),
    on="sector",
    how="left",
)

if len(result.target_perf):
    target_summary = result.target_perf.reset_index().rename(columns={"index": "target"})
else:
    target_summary = pd.DataFrame(columns=["target", "net_pnl", "trades", "win_rate", "avg_ret"])

sector_pnl = result.trades.groupby(["date", "sector"], sort=True)["net_pnl"].sum().unstack(fill_value=0).sort_index()
target_pnl = result.trades.groupby(["date", "target"], sort=True)["net_pnl"].sum().unstack(fill_value=0).sort_index()
sector_cum_pnl = sector_pnl.cumsum()
target_cum_pnl = target_pnl.cumsum()

output_dir = Path("outputs/train_only_dbts_report")
output_dir.mkdir(parents=True, exist_ok=True)
portfolio.to_csv(output_dir / "equity_curve.csv")
result.trades.to_csv(output_dir / "trade_log.csv", index=False)
completed_trades.to_csv(output_dir / "completed_trades.csv", index=False)
sector_summary.to_csv(output_dir / "sector_summary.csv", index=False)
target_summary.to_csv(output_dir / "target_summary.csv", index=False)
selected_target_distribution.to_csv(output_dir / "selected_target_distribution.csv")
sector_target_switches.to_csv(output_dir / "sector_target_switches.csv", index=False)
result.confusion.to_csv(output_dir / "confusion_matrix.csv")
Path(output_dir / "classification_report.txt").write_text(str(result.report), encoding="utf-8")

print("\n[TEST] === TRAIN-ONLY PERFORMANCE ===")
for key in ["cumulative_return", "annualized_return", "annualized_vol", "sharpe", "sortino", "max_drawdown", "win_rate", "n_trades"]:
    print(f"[TEST] {key}: {metrics.get(key)}")
print(f"[TEST] completed_trades: {metrics['n_completed_trades']}")
print(f"[TEST] win_rate_completed_trades: {metrics['win_rate_completed_trades']}")
print(f"[TEST] long/short/flat counts: long={position_counts.get(1, 0)}, short={position_counts.get(-1, 0)}, flat={position_counts.get(0, 0)}")
print(f"[TEST] target_switches: {metrics['target_switches']}")

print("\n[TEST] === CONFUSION MATRIX ===")
print(result.confusion)
print("\n[TEST] === CLASSIFICATION REPORT ===")
print(result.report)

print("\n[TEST] === SECTOR SUMMARY ===")
print(sector_summary.sort_values("net_pnl", ascending=False))

print("\n[TEST] SUCCESS: All notebook code executed without errors")
