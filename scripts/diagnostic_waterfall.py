"""Diagnostic waterfall: track signal flow through train-only pipeline.

For each target, show exactly where signals disappear at each filtering stage.
Compare NVDA and FCX (previously worked) to identify the issue.

TRAIN ONLY. Pure debugging.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SECTORS
from strategy.backtester import Backtester
from strategy.classifier import make_labels
from strategy.position_manager import PositionManager, summarize_completed_trades
from strategy.predictor_selector import PredictorSelector
from strategy.regressors import DynamicReturnModel, DynamicShadowPriceModel
from strategy.residual_features import ResidualFeatureBuilder
from strategy.splits import chrono_split
from strategy.strategy_config import StrategyConfig
from strategy.pipeline import StrategyPipeline


@dataclass
class TrainOnlyFold:
    retrain_date: pd.Timestamp
    train_idx: pd.DatetimeIndex
    predict_idx: pd.DatetimeIndex


PAIR_SPECS = [
    {"sector": "Technology", "etf": "XLK", "current": "NVDA", "candidate": "AVGO"},
    {"sector": "Materials", "etf": "XLB", "current": "FCX", "candidate": "NEM"},
]


def _load_folds(md, cfg: StrategyConfig) -> tuple:
    split = chrono_split(md.prices.index, cfg)
    diag_path = Path("outputs/train_only_deep_diagnostic.json")
    if not diag_path.exists():
        raise FileNotFoundError("outputs/train_only_deep_diagnostic.json required")
    
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    retrain_dates = sorted({pd.Timestamp(row["retrain_date"]) for row in diag.get("shadow_selected", [])})
    if not retrain_dates:
        raise ValueError("No retrain dates in diagnostic")

    index = pd.DatetimeIndex(md.prices.index).sort_values()
    folds: list[TrainOnlyFold] = []
    for i, retrain_date in enumerate(retrain_dates):
        next_retrain = retrain_dates[i + 1] if i + 1 < len(retrain_dates) else split.train_end
        train_idx = index[index < retrain_date]
        predict_idx = index[(index >= retrain_date) & (index < next_retrain) & (index < split.train_end)]
        if len(train_idx) and len(predict_idx):
            folds.append(TrainOnlyFold(retrain_date=retrain_date, train_idx=train_idx, predict_idx=predict_idx))
    return split, folds


def _classifier_columns(panel: pd.DataFrame) -> list[str]:
    excluded = {
        "date", "etf", "sector", "target", "predictors", "target_price",
        "shadow_price", "next_ret", "label", "spread_signal",
        "ann_vol", "residual_z", "price_residual", "residual_ewm_mean",
        "residual_ewm_std", "residual_roll_mean", "residual_roll_std",
        "fold_role", "retrain_date", "role",
    }
    feature_cols = [c for c in panel.columns if c not in excluded]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(panel[c])]
    return feature_cols


def _safe_r2(actual: pd.Series, pred: pd.Series) -> float:
    frame = pd.DataFrame({"actual": actual, "pred": pred}).dropna()
    if len(frame) <= 5:
        return float("nan")
    try:
        return float(r2_score(frame["actual"], frame["pred"]))
    except Exception:
        return float("nan")


def _fold_panel(
    cfg: StrategyConfig,
    md,
    tech_cache: dict[str, pd.DataFrame],
    pipe: StrategyPipeline,
    etf: str,
    sector_name: str,
    target: str,
    fold: TrainOnlyFold,
    split,
) -> tuple[pd.DataFrame, dict]:
    """Build feature panel for one target on one fold."""
    sector_cfg = SECTORS[etf]
    members = [sector_cfg["target"]] + sector_cfg["predictors"]
    candidates = [member for member in members if member != target]

    prices = md.prices
    returns = md.returns
    volumes = md.volumes

    predictor_choice = PredictorSelector(cfg).select(target, candidates, returns.reindex(fold.train_idx), prices.loc[fold.train_idx])
    preds = predictor_choice.selected or candidates[: cfg.top_n_predictors]

    shadow_model = DynamicShadowPriceModel(cfg)
    shadow_feats, _, base_target_price, safe_train_idx = shadow_model.fit(prices, target, preds, fold.train_idx)
    return_model = DynamicReturnModel(cfg)
    return_feats, _, safe_return_idx = return_model.fit(prices, target, preds, fold.train_idx)

    combined_idx = pd.Index(fold.train_idx).append(pd.Index(fold.predict_idx)).unique().sort_values()
    combined_idx = pd.DatetimeIndex(combined_idx)

    if len(safe_train_idx) == 0 or len(safe_return_idx) == 0:
        return pd.DataFrame(), {}

    shadow_pred = shadow_model.predict(shadow_feats, combined_idx, base_target_price)
    return_pred = return_model.predict(return_feats, combined_idx)

    price = prices[target].reindex(combined_idx)
    fwd_return = price.shift(-cfg.label_horizon) / price - 1.0
    resid = ResidualFeatureBuilder(cfg).build(price, shadow_pred.reindex(combined_idx), return_pred.reindex(combined_idx), fwd_return)
    resid["target"] = target
    resid = resid.set_index("target", append=True).reorder_levels([0, 1]).sort_index()

    tech_target = tech_cache[target].reindex(combined_idx)
    tech_rows = pd.DataFrame(
        [tech_target.loc[d] if d in tech_target.index else pd.Series(dtype=float) for d in combined_idx],
        index=pd.MultiIndex.from_arrays([combined_idx, [target] * len(combined_idx)], names=["date", "target"]),
    )

    oos = pd.DataFrame({
        "date": combined_idx,
        "target": target,
        "predictors": [tuple(preds)] * len(combined_idx),
        "target_price": price.values,
        "shadow_price": shadow_pred.reindex(combined_idx).values,
        "predicted_return": return_pred.reindex(combined_idx).values,
    })
    sect = pipe._sector_features(md, oos, etf)

    panel = oos.set_index(["date", "target"]).copy()
    panel = panel.drop(columns=["predicted_return"])
    resid_cols = ResidualFeatureBuilder.feature_columns(resid)
    resid_cols = [c for c in resid_cols if c != "shadow_price"]
    panel = panel.join(resid[resid_cols]).join(tech_rows).join(sect)
    panel["next_ret"] = price.shift(-1) / price - 1.0
    panel["ann_vol"] = md.returns[target].rolling(20).std().shift(1) * np.sqrt(252)
    panel["label"] = make_labels(price, cfg)
    panel["residual_z"] = panel["residual_ewm_z"]
    panel["spread_signal"] = -np.sign(panel["residual_z"].fillna(0.0)).astype(int)
    panel["date"] = panel.index.get_level_values(0)
    panel["target"] = panel.index.get_level_values(1)
    panel["sector"] = sector_name
    panel["etf"] = etf
    panel["predictors"] = [tuple(preds)] * len(panel)
    panel["fold_role"] = np.where(panel["date"].isin(fold.train_idx), "train", "predict")
    panel["retrain_date"] = fold.retrain_date

    cutoff_pos = int(pd.Index(md.prices.index).searchsorted(split.train_end, side="left"))
    date_positions = pd.Index(md.prices.index).get_indexer(panel["date"])
    safe_mask = (date_positions >= 0) & ((date_positions + cfg.label_horizon) < cutoff_pos)
    panel = panel.loc[safe_mask].copy()

    return panel.reset_index(drop=True), {}


def _diagnostic_waterfall(cfg: StrategyConfig, panel: pd.DataFrame, target: str, sector: str) -> dict:
    """Track signal flow through classifier and filters."""
    if panel.empty:
        return {
            "target": target,
            "sector": sector,
            "rows_in_classifier": 0,
            "rows_with_label": 0,
            "rows_with_nextret": 0,
            "signal_-1_count": 0,
            "signal_0_count": 0,
            "signal_+1_count": 0,
            "avg_proba_short": float("nan"),
            "avg_proba_flat": float("nan"),
            "avg_proba_long": float("nan"),
            "raw_signals": 0,
            "after_confidence": 0,
            "after_residual_z": 0,
            "after_agreement": 0,
            "after_volatility": 0,
            "pm_entries": 0,
            "pm_exits": 0,
            "backtest_trades": 0,
            "completed_trades": 0,
        }

    data = panel.dropna(subset=["label", "next_ret"]).copy()
    rows_in_classifier = len(panel)
    rows_with_label = len(data)
    
    if data.empty:
        return {
            "target": target,
            "sector": sector,
            "rows_in_classifier": rows_in_classifier,
            "rows_with_label": rows_with_label,
            "rows_with_nextret": 0,
            "signal_-1_count": 0,
            "signal_0_count": 0,
            "signal_+1_count": 0,
            "avg_proba_short": float("nan"),
            "avg_proba_flat": float("nan"),
            "avg_proba_long": float("nan"),
            "raw_signals": 0,
            "after_confidence": 0,
            "after_residual_z": 0,
            "after_agreement": 0,
            "after_volatility": 0,
            "pm_entries": 0,
            "pm_exits": 0,
            "backtest_trades": 0,
            "completed_trades": 0,
        }

    # Train/predict split
    feature_cols = _classifier_columns(data)
    X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.groupby(data["target"]).ffill().fillna(0.0)
    data = data.assign(**{c: X[c] for c in feature_cols})

    train = data[data["fold_role"] == "train"].copy()
    predict = data[data["fold_role"] == "predict"].copy()
    rows_with_nextret = len(predict)

    if train.empty or predict.empty:
        return {
            "target": target,
            "sector": sector,
            "rows_in_classifier": rows_in_classifier,
            "rows_with_label": rows_with_label,
            "rows_with_nextret": rows_with_nextret,
            "signal_-1_count": 0,
            "signal_0_count": 0,
            "signal_+1_count": 0,
            "avg_proba_short": float("nan"),
            "avg_proba_flat": float("nan"),
            "avg_proba_long": float("nan"),
            "raw_signals": 0,
            "after_confidence": 0,
            "after_residual_z": 0,
            "after_agreement": 0,
            "after_volatility": 0,
            "pm_entries": 0,
            "pm_exits": 0,
            "backtest_trades": 0,
            "completed_trades": 0,
        }

    train_y = train["label"].astype(float)
    predict_X = predict[feature_cols]
    train_X = train[feature_cols]

    # Classifier
    if train_y.nunique(dropna=True) < 2:
        signal = pd.Series(0, index=predict.index, name="signal")
        proba_df = pd.DataFrame({"P_short": 0.0, "P_flat": 1.0, "P_long": 0.0}, index=predict.index)
    else:
        y_map = train_y.map({-1: 0, 0: 1, 1: 2}).astype(int)
        params = dict(cfg.clf_params)
        params.setdefault("objective", "multi:softprob")
        params["use_label_encoder"] = False
        model = XGBClassifier(**params)
        try:
            model.fit(train_X, y_map)
            pred_idx = model.predict(predict_X)
            inv = {0: -1, 1: 0, 2: 1}
            signal = pd.Series([inv[int(i)] for i in pred_idx], index=predict.index, name="signal")
            proba_df = pd.DataFrame(model.predict_proba(predict_X), index=predict.index, columns=["P_short", "P_flat", "P_long"])
        except Exception as e:
            print(f"  Classifier failed: {e}")
            signal = pd.Series(0, index=predict.index, name="signal")
            proba_df = pd.DataFrame({"P_short": 0.0, "P_flat": 1.0, "P_long": 0.0}, index=predict.index)

    predict_panel = predict.copy()
    predict_panel["signal"] = signal
    predict_panel = pd.concat([predict_panel, proba_df], axis=1)
    predict_panel = predict_panel.reset_index(drop=True)

    # Signal counts
    signal_counts = signal.value_counts()
    signal_minus1 = signal_counts.get(-1, 0)
    signal_0 = signal_counts.get(0, 0)
    signal_plus1 = signal_counts.get(1, 0)
    avg_proba_short = proba_df["P_short"].mean()
    avg_proba_flat = proba_df["P_flat"].mean()
    avg_proba_long = proba_df["P_long"].mean()

    # Raw signals (non-zero)
    raw_signals = len(predict_panel[predict_panel["signal"] != 0])

    # After confidence filter
    after_confidence = len(predict_panel[
        ((predict_panel["signal"] == 1) & (predict_panel["P_long"] >= cfg.entry_confidence)) |
        ((predict_panel["signal"] == -1) & (predict_panel["P_short"] >= cfg.entry_confidence))
    ])

    # After residual z filter
    after_residual_z = len(predict_panel[
        ((predict_panel["signal"] == 1) & (predict_panel["P_long"] >= cfg.entry_confidence) & (predict_panel["residual_z"] >= cfg.entry_z_threshold)) |
        ((predict_panel["signal"] == -1) & (predict_panel["P_short"] >= cfg.entry_confidence) & (predict_panel["residual_z"] <= -cfg.entry_z_threshold))
    ])

    # After agreement filter (signal + spread_signal agree)
    after_agreement = len(predict_panel[
        ((predict_panel["signal"] == 1) & (predict_panel["P_long"] >= cfg.entry_confidence) & (predict_panel["residual_z"] >= cfg.entry_z_threshold) & (predict_panel["spread_signal"] == 1)) |
        ((predict_panel["signal"] == -1) & (predict_panel["P_short"] >= cfg.entry_confidence) & (predict_panel["residual_z"] <= -cfg.entry_z_threshold) & (predict_panel["spread_signal"] == -1))
    ])

    # After volatility filter
    after_volatility = len(predict_panel[
        ((predict_panel["signal"] == 1) & (predict_panel["P_long"] >= cfg.entry_confidence) & (predict_panel["residual_z"] >= cfg.entry_z_threshold) & (predict_panel["spread_signal"] == 1) & (predict_panel["ann_vol"] < cfg.entry_volatility_cap)) |
        ((predict_panel["signal"] == -1) & (predict_panel["P_short"] >= cfg.entry_confidence) & (predict_panel["residual_z"] <= -cfg.entry_z_threshold) & (predict_panel["spread_signal"] == -1) & (predict_panel["ann_vol"] < cfg.entry_volatility_cap))
    ])

    # Run backtest
    bt = Backtester(cfg)
    result = bt.run(predict_panel)
    backtest_trades = len(result.trades) if not result.trades.empty else 0
    completed = summarize_completed_trades(result.trades)
    completed_trades = completed.get("completed_trades", 0)

    # PositionManager
    pm = PositionManager(
        long_entry_confidence=cfg.pm_entry_confidence,
        short_entry_confidence=cfg.pm_entry_confidence,
        flat_probability_block=cfg.flat_probability_block,
        entry_residual_threshold=cfg.pm_entry_residual_z,
        mean_reversion_exit=cfg.pm_exit_residual_z,
        opposite_signal_confidence=cfg.pm_opposite_confidence,
        stop_loss=cfg.pm_stop_loss,
        take_profit=cfg.pm_take_profit,
        max_holding_days=cfg.pm_max_holding_days,
        allow_flip=cfg.pm_allow_flip,
    )
    result_pm = bt.run(predict_panel, position_manager=pm)
    pm_entries = len(result_pm.trades) if not result_pm.trades.empty else 0
    pm_completed = summarize_completed_trades(result_pm.trades)
    pm_exits = pm_completed.get("completed_trades", 0)

    return {
        "target": target,
        "sector": sector,
        "rows_in_classifier": rows_in_classifier,
        "rows_with_label": rows_with_label,
        "rows_with_nextret": rows_with_nextret,
        "signal_-1_count": int(signal_minus1),
        "signal_0_count": int(signal_0),
        "signal_+1_count": int(signal_plus1),
        "avg_proba_short": float(avg_proba_short),
        "avg_proba_flat": float(avg_proba_flat),
        "avg_proba_long": float(avg_proba_long),
        "raw_signals": raw_signals,
        "after_confidence": after_confidence,
        "after_residual_z": after_residual_z,
        "after_agreement": after_agreement,
        "after_volatility": after_volatility,
        "pm_entries": pm_entries,
        "pm_exits": pm_exits,
        "backtest_trades": backtest_trades,
        "completed_trades": completed_trades,
    }


def main() -> None:
    print("[DIAGNOSTIC WATERFALL] Train-only signal flow analysis")
    print()
    
    cfg = StrategyConfig(force_recompute=False, make_plots=False)
    md = StrategyPipeline(cfg).load_data()
    split, folds = _load_folds(md, cfg)
    pipe = StrategyPipeline(cfg)
    tech_cache = pipe._technical_all(md)

    print(f"[folds] {len(folds)} retrains detected")
    print()

    results = []

    for spec in PAIR_SPECS:
        sector_name = spec["sector"]
        etf = spec["etf"]
        current = spec["current"]
        candidate = spec["candidate"]

        for target in [current, candidate]:
            print(f"[{sector_name}] {target}")
            
            fold_results = []
            combined_panels = []

            for i, fold in enumerate(folds):
                try:
                    panel, _ = _fold_panel(cfg, md, tech_cache, pipe, etf, sector_name, target, fold, split)
                    if not panel.empty:
                        combined_panels.append(panel)
                except Exception as e:
                    print(f"  [fold {i}] error: {e}")
                    continue

            if not combined_panels:
                print(f"  No panels built for {target}")
                continue

            combined = pd.concat(combined_panels, ignore_index=True)
            combined = combined.sort_values(["target", "date"]).reset_index(drop=True)

            diagnostic = _diagnostic_waterfall(cfg, combined, target, sector_name)
            results.append(diagnostic)
            
            print(f"  Rows in: {diagnostic['rows_in_classifier']:4d} -> "
                  f"Label: {diagnostic['rows_with_label']:4d} -> "
                  f"NextRet: {diagnostic['rows_with_nextret']:4d}")
            print(f"  Signals: {diagnostic['signal_-1_count']:3d} short, {diagnostic['signal_0_count']:4d} flat, {diagnostic['signal_+1_count']:3d} long "
                  f"(raw={diagnostic['raw_signals']:3d})")
            print(f"  Waterfall: Confidence={diagnostic['after_confidence']:3d} -> "
                  f"ResidualZ={diagnostic['after_residual_z']:3d} -> "
                  f"Agreement={diagnostic['after_agreement']:3d} -> "
                  f"Volatility={diagnostic['after_volatility']:3d}")
            print(f"  Backtest: {diagnostic['backtest_trades']:3d} trade events -> "
                  f"{diagnostic['completed_trades']:3d} completed | "
                  f"PM: {diagnostic['pm_entries']:3d} entries -> {diagnostic['pm_exits']:3d} exits")
            print()

    # Create waterfall table
    df = pd.DataFrame(results)
    df = df[["sector", "target", "rows_in_classifier", "rows_with_label", "rows_with_nextret",
             "signal_-1_count", "signal_0_count", "signal_+1_count",
             "raw_signals", "after_confidence", "after_residual_z", "after_agreement", "after_volatility",
             "backtest_trades", "completed_trades", "pm_entries", "pm_exits"]]

    print()
    print("=== WATERFALL TABLE ===")
    print()
    print(df.to_string(index=False))

    # Save for reference
    df.to_csv(Path("outputs/diagnostic_waterfall.csv"), index=False)
    print()
    print(f"Full diagnostic saved to outputs/diagnostic_waterfall.csv")


if __name__ == "__main__":
    main()
