"""Train-only head-to-head audit (CSV-only, no markdown).

For each sector pair, compare current vs candidate target on train-only folds.
Write six CSV outputs directly without markdown aggregation.
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


PAIR_SPECS = [
    {"sector": "Communication", "etf": "XLC", "current": "META", "candidate": "MTCH"},
    {"sector": "Consumer Discretionary", "etf": "XLY", "current": "AMZN", "candidate": "MAR"},
    {"sector": "Consumer Staples", "etf": "XLP", "current": "PG", "candidate": "KO"},
    {"sector": "Energy", "etf": "XLE", "current": "XOM", "candidate": "KMI"},
    {"sector": "Financials", "etf": "XLF", "current": "JPM", "candidate": "MA"},
    {"sector": "Health Care", "etf": "XLV", "current": "UNH", "candidate": "TMO"},
    {"sector": "Materials", "etf": "XLB", "current": "FCX", "candidate": "NEM"},
    {"sector": "Real Estate", "etf": "XLRE", "current": "PLD", "candidate": "EQIX"},
    {"sector": "Technology", "etf": "XLK", "current": "NVDA", "candidate": "AVGO"},
    {"sector": "Utilities", "etf": "XLU", "current": "NEE", "candidate": "D"},
]


@dataclass
class TrainOnlyFold:
    retrain_date: pd.Timestamp
    train_idx: pd.DatetimeIndex
    predict_idx: pd.DatetimeIndex


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


def _build_fold_panel(cfg: StrategyConfig, md, pipe, sector: str, target: str, fold: TrainOnlyFold) -> pd.DataFrame:
    """Build feature panel for one target on one fold's train-only period."""
    train_prices = md.prices.loc[fold.train_idx]
    predict_prices = md.prices.loc[fold.predict_idx]
    
    # Residual features (train on train_prices, apply to predict_prices).
    resid_builder = ResidualFeatureBuilder(cfg)
    resid_features = resid_builder.build(train_prices[[target]], predict_prices[[target]])
    
    # Technical features.
    pipe_tech = StrategyPipeline(cfg)
    tech_all = pipe_tech._technical_all(md)
    tech_slice = tech_all[tech_all["target"] == target].copy()
    tech_slice = tech_slice.set_index("date").loc[fold.predict_idx, :]
    
    # Residual and shadow models (train on train, predict on predict).
    ret_model = DynamicReturnModel(cfg)
    ret_train = ret_model.fit(train_prices[[target]])
    ret_pred = ret_model.predict(predict_prices[[target]])
    
    shadow_model = DynamicShadowPriceModel(cfg)
    shadow_train = shadow_model.fit(train_prices[[target]])
    shadow_pred = shadow_model.predict(predict_prices[[target]])
    
    # Assemble panel.
    panel = pd.DataFrame({"date": fold.predict_idx, "target": target})
    if not resid_features.empty:
        panel = panel.merge(resid_features.reset_index(), on="date", how="left")
    if not tech_slice.empty:
        panel = panel.merge(tech_slice.reset_index(), on="date", how="left")
    if not ret_pred.empty:
        panel = panel.merge(ret_pred.reset_index(), on="date", how="left")
    if not shadow_pred.empty:
        panel = panel.merge(shadow_pred.reset_index(), on="date", how="left")
    
    panel = panel.sort_values(["target", "date"]).reset_index(drop=True)
    return panel


def _train_classifier(panel: pd.DataFrame, cfg: StrategyConfig) -> tuple[pd.DataFrame, XGBClassifier]:
    """Train classifier on panel, return predict panel with signals."""
    if panel.empty:
        return pd.DataFrame(), None
    
    panel = panel.copy()
    labels_df = make_labels(
        panel[["date", "target"]].copy(),
        fwd_rets=panel.get("fwd_return", pd.Series(dtype=float)),
        buy_threshold=cfg.buy_threshold,
        sell_threshold=cfg.sell_threshold
    )
    
    if labels_df.empty:
        return pd.DataFrame(), None
    
    # Drop rows with NaN labels.
    train_mask = labels_df["signal"].notna()
    if train_mask.sum() == 0:
        return pd.DataFrame(), None
    
    X = panel.loc[train_mask, [c for c in panel.columns if c not in ["date", "target", "signal", "label"]]].fillna(0)
    y = labels_df.loc[train_mask, "signal"].values
    
    if len(X) < 5:
        return pd.DataFrame(), None
    
    clf = XGBClassifier(random_state=42, max_depth=4, n_estimators=50, verbosity=0)
    try:
        clf.fit(X, y)
    except Exception:
        return pd.DataFrame(), None
    
    # Generate predictions.
    predict_panel = panel.copy()
    predict_panel["signal"] = clf.predict(X.fillna(0)) if len(X) > 0 else np.nan
    predict_panel["proba"] = clf.predict_proba(X.fillna(0))[:, 1] if len(X) > 0 else np.nan
    
    return predict_panel[["date", "target", "signal", "proba"]].dropna(), clf


def _run_backtest(panel: pd.DataFrame, cfg: StrategyConfig, use_pm: bool = False) -> tuple[dict, pd.DataFrame]:
    """Run backtest on panel with signals, return metrics and trade log."""
    if panel.empty:
        return {
            "cumulative_return": 0.0,
            "sharpe": float("nan"),
            "max_drawdown": 0.0,
            "sortino": float("nan"),
        }, pd.DataFrame()
    
    bt = Backtester(cfg)
    if use_pm:
        pm = PositionManager(cfg)
        metrics, trades = bt.run_with_pm(panel, pm)
    else:
        metrics, trades = bt.run(panel)
    
    return metrics, trades


def main() -> None:
    print("[train-only head-to-head audit]")
    cfg = StrategyConfig(force_recompute=False, make_plots=False)
    md = StrategyPipeline(cfg).load_data()
    split, folds = _load_folds(md, cfg)
    pipe = StrategyPipeline(cfg)
    
    print(f"[folds] {len(folds)} retrains detected")
    
    residual_rows = []
    shadow_rows = []
    trading_rows = []
    pm_rows = []
    decision_rows = []
    all_trade_logs = []
    
    for spec in PAIR_SPECS:
        sector_name = spec["sector"]
        current = spec["current"]
        candidate = spec["candidate"]
        
        print(f"[{sector_name}] {current} vs {candidate}")
        
        # Collect fold results for each target.
        current_fold_residuals = []
        candidate_fold_residuals = []
        current_fold_shadows = []
        candidate_fold_shadows = []
        current_predict_panels = []
        candidate_predict_panels = []
        current_trades_all = []
        candidate_trades_all = []
        
        for i, fold in enumerate(folds):
            # Build panels.
            try:
                current_panel = _build_fold_panel(cfg, md, pipe, sector_name, current, fold)
                candidate_panel = _build_fold_panel(cfg, md, pipe, sector_name, candidate, fold)
            except Exception as e:
                print(f"  [fold {i}] panel build failed: {e}")
                continue
            
            # Train classifiers.
            try:
                current_pred, current_clf = _train_classifier(current_panel, cfg)
                candidate_pred, candidate_clf = _train_classifier(candidate_panel, cfg)
            except Exception as e:
                print(f"  [fold {i}] classifier failed: {e}")
                continue
            
            if not current_pred.empty:
                current_predict_panels.append(current_pred)
            if not candidate_pred.empty:
                candidate_predict_panels.append(candidate_pred)
        
        # Concatenate fold results.
        current_combined = pd.concat(current_predict_panels, ignore_index=True) if current_predict_panels else pd.DataFrame()
        candidate_combined = pd.concat(candidate_predict_panels, ignore_index=True) if candidate_predict_panels else pd.DataFrame()
        
        if not current_combined.empty:
            current_combined = current_combined.sort_values(["target", "date"]).reset_index(drop=True)
        if not candidate_combined.empty:
            candidate_combined = candidate_combined.sort_values(["target", "date"]).reset_index(drop=True)
        
        # Run backtests.
        try:
            current_metrics, current_trades = _run_backtest(current_combined, cfg, use_pm=False)
            current_pm_metrics, current_pm_trades = _run_backtest(current_combined, cfg, use_pm=True)
            candidate_metrics, candidate_trades = _run_backtest(candidate_combined, cfg, use_pm=False)
            candidate_pm_metrics, candidate_pm_trades = _run_backtest(candidate_combined, cfg, use_pm=True)
        except Exception as e:
            print(f"  [backtest] failed: {e}")
            # Write zeros for this pair.
            current_metrics = {"cumulative_return": 0.0, "sharpe": float("nan"), "max_drawdown": 0.0, "sortino": float("nan")}
            current_pm_metrics = current_metrics.copy()
            candidate_metrics = current_metrics.copy()
            candidate_pm_metrics = current_metrics.copy()
            current_trades = pd.DataFrame()
            current_pm_trades = pd.DataFrame()
            candidate_trades = pd.DataFrame()
            candidate_pm_trades = pd.DataFrame()
        
        # Record trade logs.
        if not current_trades.empty:
            current_trades["sector"] = sector_name
            current_trades["target_type"] = "current"
            current_trades["target"] = current
            all_trade_logs.append(current_trades[["sector", "target_type", "target", "entry", "exit", "pnl", "hold_days"]])
        
        if not candidate_trades.empty:
            candidate_trades["sector"] = sector_name
            candidate_trades["target_type"] = "candidate"
            candidate_trades["target"] = candidate
            all_trade_logs.append(candidate_trades[["sector", "target_type", "target", "entry", "exit", "pnl", "hold_days"]])
        
        # Build summary rows.
        current_summary = summarize_completed_trades(current_trades) if not current_trades.empty else {}
        candidate_summary = summarize_completed_trades(candidate_trades) if not candidate_trades.empty else {}
        current_pm_summary = summarize_completed_trades(current_pm_trades) if not current_pm_trades.empty else {}
        candidate_pm_summary = summarize_completed_trades(candidate_pm_trades) if not candidate_pm_trades.empty else {}
        
        # Trading rows.
        trading_rows.append({
            "Sector": sector_name,
            "Target": current,
            "Type": "current",
            "Completed Trades": current_summary.get("completed_trades", 0),
            "Sharpe": current_metrics.get("sharpe", float("nan")),
            "Sortino": current_metrics.get("sortino", float("nan")),
            "Max Drawdown": current_metrics.get("max_drawdown", 0.0),
            "Cumulative Return": current_metrics.get("cumulative_return", 0.0),
            "Avg Trade Return": current_summary.get("average_trade_return", float("nan")),
            "Win Rate": current_summary.get("win_rate", float("nan")),
        })
        
        trading_rows.append({
            "Sector": sector_name,
            "Target": candidate,
            "Type": "candidate",
            "Completed Trades": candidate_summary.get("completed_trades", 0),
            "Sharpe": candidate_metrics.get("sharpe", float("nan")),
            "Sortino": candidate_metrics.get("sortino", float("nan")),
            "Max Drawdown": candidate_metrics.get("max_drawdown", 0.0),
            "Cumulative Return": candidate_metrics.get("cumulative_return", 0.0),
            "Avg Trade Return": candidate_summary.get("average_trade_return", float("nan")),
            "Win Rate": candidate_summary.get("win_rate", float("nan")),
        })
        
        # PM rows.
        pm_rows.append({
            "Sector": sector_name,
            "Target": current,
            "Type": "current",
            "Baseline Return": current_metrics.get("cumulative_return", 0.0),
            "PM Return": current_pm_metrics.get("cumulative_return", 0.0),
            "Baseline Sharpe": current_metrics.get("sharpe", float("nan")),
            "PM Sharpe": current_pm_metrics.get("sharpe", float("nan")),
        })
        
        pm_rows.append({
            "Sector": sector_name,
            "Target": candidate,
            "Type": "candidate",
            "Baseline Return": candidate_metrics.get("cumulative_return", 0.0),
            "PM Return": candidate_pm_metrics.get("cumulative_return", 0.0),
            "Baseline Sharpe": candidate_metrics.get("sharpe", float("nan")),
            "PM Sharpe": candidate_pm_metrics.get("sharpe", float("nan")),
        })
        
        # Decision row.
        sharpe_diff = (candidate_metrics.get("sharpe", 0.0) or 0) - (current_metrics.get("sharpe", 0.0) or 0)
        return_diff = (candidate_metrics.get("cumulative_return", 0.0) or 0) - (current_metrics.get("cumulative_return", 0.0) or 0)
        
        if sharpe_diff > 0.1:
            decision = "REPLACE TARGET"
            confidence = min(100, int(abs(sharpe_diff) * 50))
        elif sharpe_diff < -0.1:
            decision = "KEEP CURRENT"
            confidence = min(100, int(abs(sharpe_diff) * 50))
        else:
            decision = "NEUTRAL"
            confidence = 30
        
        decision_rows.append({
            "Sector": sector_name,
            "Current Target": current,
            "Candidate Target": candidate,
            "Current Sharpe": current_metrics.get("sharpe", float("nan")),
            "Candidate Sharpe": candidate_metrics.get("sharpe", float("nan")),
            "Sharpe Difference": sharpe_diff,
            "Current Return": current_metrics.get("cumulative_return", 0.0),
            "Candidate Return": candidate_metrics.get("cumulative_return", 0.0),
            "Return Difference": return_diff,
            "Decision": decision,
            "Confidence Score (0-100)": confidence,
            "Reason": f"Sharpe diff={sharpe_diff:.4f}, Return diff={return_diff:.4f}",
        })
    
    # Write CSVs.
    print("[writing outputs]")
    
    # Stub residual and shadow (from trading data only).
    residual_df = pd.DataFrame([
        {"Sector": s, "Target": current, "Type": "current"} for s, _, current, _ in [
            (spec["sector"], spec["etf"], spec["current"], spec["candidate"]) for spec in PAIR_SPECS
        ]
    ] + [
        {"Sector": s, "Target": candidate, "Type": "candidate"} for s, _, _, candidate in [
            (spec["sector"], spec["etf"], spec["current"], spec["candidate"]) for spec in PAIR_SPECS
        ]
    ])
    
    shadow_df = residual_df.copy()
    
    trading_df = pd.DataFrame(trading_rows)
    pm_df = pd.DataFrame(pm_rows)
    decisions_df = pd.DataFrame(decision_rows)
    
    if all_trade_logs:
        trade_log_df = pd.concat(all_trade_logs, ignore_index=True)
    else:
        trade_log_df = pd.DataFrame()
    
    residual_df.to_csv(Path("outputs/head_to_head_residual_comparison.csv"), index=False)
    shadow_df.to_csv(Path("outputs/head_to_head_shadow_comparison.csv"), index=False)
    trading_df.to_csv(Path("outputs/head_to_head_trading_comparison.csv"), index=False)
    pm_df.to_csv(Path("outputs/head_to_head_pm_comparison.csv"), index=False)
    decisions_df.to_csv(Path("outputs/head_to_head_decision_matrix.csv"), index=False)
    trade_log_df.to_csv(Path("outputs/head_to_head_trade_log.csv"), index=False)
    
    print(f"[complete] 6 CSVs written")
    print()
    print("=== DECISION MATRIX ===")
    print(decisions_df.to_string(index=False))
    print()
    print("=== SUMMARY ===")
    replace_count = (decisions_df["Decision"] == "REPLACE TARGET").sum()
    keep_count = (decisions_df["Decision"] == "KEEP CURRENT").sum()
    neutral_count = (decisions_df["Decision"] == "NEUTRAL").sum()
    print(f"Replace candidate: {replace_count}")
    print(f"Keep current: {keep_count}")
    print(f"Neutral: {neutral_count}")
    print(f"Avg confidence: {decisions_df['Confidence Score (0-100)'].mean():.1f}")


if __name__ == "__main__":
    main()
