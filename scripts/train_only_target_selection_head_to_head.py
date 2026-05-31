"""Train-only target selection head-to-head audit.

For each requested sector pair, this script compares the current configured
target against the TradabilityScore candidate on the train-only folds anchored
by ``outputs/train_only_deep_diagnostic.json``.

The run is diagnostic-only:
- no validation set
- no test set
- no threshold tuning
- no hyperparameter tuning
- no changes to PositionManager, classifier settings, residual-engine settings,
  or predictor-selection settings
"""
from __future__ import annotations

import dataclasses
import json
import math
import pickle
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
from strategy.pipeline import StrategyPipeline, make_labels_from_fwd


@dataclass
class TrainOnlyFold:
    retrain_date: pd.Timestamp
    train_idx: pd.DatetimeIndex
    predict_idx: pd.DatetimeIndex


PAIR_SPECS = [
    {"sector": "Communication", "etf": "XLC", "current": "META", "candidate": "MTCH", "label": "Communication"},
    {"sector": "Consumer Discretionary", "etf": "XLY", "current": "AMZN", "candidate": "MAR", "label": "Consumer Disc."},
    {"sector": "Consumer Staples", "etf": "XLP", "current": "PG", "candidate": "KO", "label": "Consumer Staples"},
    {"sector": "Energy", "etf": "XLE", "current": "XOM", "candidate": "KMI", "label": "Energy"},
    {"sector": "Financials", "etf": "XLF", "current": "JPM", "candidate": "MA", "label": "Financials"},
    {"sector": "Health Care", "etf": "XLV", "current": "UNH", "candidate": "TMO", "label": "Health Care"},
    {"sector": "Materials", "etf": "XLB", "current": "FCX", "candidate": "NEM", "label": "Materials"},
    {"sector": "Real Estate", "etf": "XLRE", "current": "PLD", "candidate": "EQIX", "label": "Real Estate"},
    {"sector": "Technology", "etf": "XLK", "current": "NVDA", "candidate": "AVGO", "label": "Technology"},
    {"sector": "Utilities", "etf": "XLU", "current": "NEE", "candidate": "D", "label": "Utilities"},
]


def _load_market_data() -> object:
    cfg = StrategyConfig(force_recompute=False, make_plots=False)
    return StrategyPipeline(cfg).load_data()


def _train_only_folds(md, cfg: StrategyConfig) -> tuple[object, list[TrainOnlyFold]]:
    split = chrono_split(md.prices.index, cfg)
    diag_path = Path("outputs/train_only_deep_diagnostic.json")
    if not diag_path.exists():
        raise FileNotFoundError("outputs/train_only_deep_diagnostic.json is required to anchor the train-only timeline")

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
    return split, folds


def _safe_r2(actual: pd.Series, pred: pd.Series) -> float:
    frame = pd.DataFrame({"actual": actual, "pred": pred}).dropna()
    if len(frame) <= 5:
        return float("nan")
    try:
        return float(r2_score(frame["actual"], frame["pred"]))
    except Exception:
        return float("nan")


def _safe_mean(series: pd.Series) -> float:
    values = pd.Series(series).dropna()
    return float(values.mean()) if len(values) else float("nan")


def _safe_median(series: pd.Series) -> float:
    values = pd.Series(series).dropna()
    return float(values.median()) if len(values) else float("nan")


def _sign(value: float, tol: float = 1e-12) -> int:
    if not np.isfinite(value) or abs(value) <= tol:
        return 0
    return 1 if value > 0 else -1


def _residual_diagnostic_metrics(price: pd.Series, resid: pd.DataFrame, sector: str, target: str, cfg: StrategyConfig) -> dict:
    if isinstance(resid.index, pd.MultiIndex):
        try:
            resid = resid.xs(target, level="target")
        except Exception:
            resid = resid.droplevel("target") if "target" in resid.index.names else resid
    residual_z = resid["residual_ewm_z"].reindex(price.index).astype(float)
    residual_z = residual_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    abs_z = residual_z.abs()

    entry = float(getattr(cfg, "pm_entry_residual_z", 1.25))
    strength = ((abs_z - entry) / max(entry, 1e-6)).clip(lower=0.0, upper=1.0)
    signal = pd.Series(np.where(residual_z <= -entry, 1, np.where(residual_z >= entry, -1, 0)), index=price.index)
    p_long = pd.Series(0.10, index=price.index, dtype=float)
    p_short = pd.Series(0.10, index=price.index, dtype=float)
    p_flat = pd.Series(0.85, index=price.index, dtype=float)
    p_long.loc[signal == 1] = 0.70 + 0.25 * strength.loc[signal == 1]
    p_short.loc[signal == -1] = 0.70 + 0.25 * strength.loc[signal == -1]
    p_flat.loc[signal != 0] = 0.10

    panel = pd.DataFrame({
        "date": price.index,
        "sector": sector,
        "target": target,
        "target_price": price.values,
        "residual_z": residual_z.values,
        "signal": signal.values,
        "P_short": p_short.values,
        "P_flat": p_flat.values,
        "P_long": p_long.values,
        "next_ret": price.pct_change().shift(-1).values,
    })
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
    sim = pm.simulate(panel, cost_bps=cfg.transaction_cost_bps)
    trades = summarize_completed_trades(sim)
    pnls = trades["pnl"] if len(trades) else pd.Series(dtype=float)

    opp_weights = {
        1.25: float((abs_z > 1.25).mean()) if len(abs_z) else 0.0,
        1.50: float((abs_z > 1.50).mean()) if len(abs_z) else 0.0,
        2.00: float((abs_z > 2.00).mean()) if len(abs_z) else 0.0,
        3.00: float((abs_z > 3.00).mean()) if len(abs_z) else 0.0,
    }
    opportunity_score = 1.00 * opp_weights[1.25] + 0.75 * opp_weights[1.50] + 0.50 * opp_weights[2.00] + 0.25 * opp_weights[3.00]

    residual = resid["price_residual"].reindex(price.index).astype(float)
    residual_mean = float(residual.mean()) if len(residual) else 0.0
    residual_std = float(residual.std(ddof=0)) if len(residual) else 0.0

    half_life = float("nan")
    if len(residual_z.dropna()) >= 20:
        lag1 = float(residual_z.dropna().autocorr(lag=1))
        if np.isfinite(lag1) and 0 < lag1 < 1:
            half_life = float(-np.log(2.0) / np.log(lag1))
    else:
        lag1 = float("nan")

    adf_p = float("nan")
    try:
        from statsmodels.tsa.stattools import adfuller

        if len(residual_z.dropna()) >= 20:
            adf_p = float(adfuller(residual_z.dropna(), autolag="AIC")[1])
    except Exception:
        adf_p = float("nan")

    adf_score = 1.0 - adf_p if np.isfinite(adf_p) else 0.0
    lag1_score = float(np.clip(1.0 - abs(lag1), 0.0, 1.0)) if np.isfinite(lag1) else 0.0
    if len(residual_z.dropna()) < 20:
        var_stability = 0.0
        regime_stability = 0.0
    else:
        s = residual_z.dropna()
        half = max(5, len(s) // 2)
        first = s.iloc[:half]
        second = s.iloc[-half:]
        a = float(first.std(ddof=0))
        b = float(second.std(ddof=0))
        if a == 0 or b == 0 or not np.isfinite(a) or not np.isfinite(b):
            var_stability = 0.0
        else:
            var_stability = float(np.clip(1.0 - abs(np.log(a / b)) / 3.0, 0.0, 1.0))
        bins = pd.cut(abs_z.dropna(), bins=[-np.inf, 1.0, 1.5, 2.0, np.inf], labels=False)
        transitions = (bins != bins.shift(1)).astype(float).fillna(0.0)
        regime_stability = float(np.clip(1.0 - float(transitions.mean()), 0.0, 1.0))

    residual_stability = float(np.mean([adf_score, lag1_score, var_stability, regime_stability]))
    if len(trades):
        reversion_success = float((trades["pnl"] > 0).mean())
        mean_excursion_duration = float(trades["holding_period"].mean())
        fib_rows = []
        for trade_id, seg in sim.dropna(subset=["trade_id"]).groupby("trade_id", sort=False):
            seg = seg.sort_values("date")
            seg_z = seg["residual_z"].abs().astype(float)
            dist = seg_z.copy()
            peak = np.nan
            active = False
            out = []
            for value in seg_z.values:
                if pd.isna(value):
                    out.append(np.nan)
                    continue
                if value > 1.0:
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
            dist = pd.Series(out, index=seg.index, dtype=float)
            fib_rows.append({
                "trade_id": int(trade_id),
                "fib_hit": bool((dist >= 0.236).any()),
                "max_retrace": float(dist.max()) if len(dist) else 0.0,
                "pnl": float(trades.loc[trades["entry_date"].isin([seg.iloc[0]["date"]]), "pnl"].iloc[0]) if not trades.empty else 0.0,
            })
        fib_df = pd.DataFrame(fib_rows)
        if len(fib_df):
            fib_retrace_hit_rate = float(fib_df["fib_hit"].mean())
            fib_mean_retrace_depth = float(fib_df["max_retrace"].mean())
            fib_reversion_success = float(fib_df.loc[fib_df["fib_hit"], "pnl"].gt(0).mean()) if fib_df["fib_hit"].any() else 0.0
        else:
            fib_retrace_hit_rate = 0.0
            fib_mean_retrace_depth = 0.0
            fib_reversion_success = 0.0
        fibonacci_score = float(np.mean([fib_reversion_success, fib_mean_retrace_depth, fib_retrace_hit_rate]))
    else:
        reversion_success = 0.0
        mean_excursion_duration = 0.0
        fibonacci_score = 0.0

    return {
        "residual_sharpe": float(pnls.mean() / pnls.std(ddof=0) * np.sqrt(len(pnls))) if len(pnls) > 1 and np.isfinite(float(pnls.std(ddof=0))) and float(pnls.std(ddof=0)) != 0 else 0.0,
        "residual_sortino": float(pnls.mean() / pnls[pnls < 0].std(ddof=0) * np.sqrt(len(pnls))) if len(pnls) > 1 and len(pnls[pnls < 0]) and np.isfinite(float(pnls[pnls < 0].std(ddof=0))) and float(pnls[pnls < 0].std(ddof=0)) != 0 else 0.0,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "half_life": half_life,
        "adf_p_value": adf_p,
        "lag1_autocorr": lag1,
        "reversion_success": reversion_success,
        "residual_stability": residual_stability,
        "opportunity_score": float(opportunity_score),
        "avg_abs_z": float(abs_z.mean()) if len(abs_z) else 0.0,
        "max_abs_z": float(abs_z.max()) if len(abs_z) else 0.0,
        "opportunity_count": float(sum(opp_weights.values())),
        "mean_excursion_duration": mean_excursion_duration,
        "fibonacci_score": fibonacci_score,
    }


def _percentile_rank(values: pd.Series, key: str, ascending: bool = True) -> float:
    series = pd.Series(values).dropna()
    if series.empty or key not in series.index:
        return 0.5
    if series.nunique(dropna=True) <= 1:
        return 0.5
    return float(series.rank(pct=True, ascending=ascending).loc[key])


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
        empty = pd.DataFrame()
        fold_log = {
            "sector": sector_name,
            "etf": etf,
            "retrain_date": fold.retrain_date,
            "target": target,
            "predictors": ",".join(preds),
            "predictor_count": len(preds),
            "shadow_oos_r2": float("nan"),
            "return_oos_r2": float("nan"),
        }
        return empty, fold_log

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

    # Keep only rows whose label horizon stays strictly before the train cutoff.
    cutoff_pos = int(pd.Index(md.prices.index).searchsorted(split.train_end, side="left"))
    date_positions = pd.Index(md.prices.index).get_indexer(panel["date"])
    safe_mask = (date_positions >= 0) & ((date_positions + cfg.label_horizon) < cutoff_pos)
    panel = panel.loc[safe_mask].copy()

    shadow_oos_r2 = _safe_r2(prices[target].reindex(fold.predict_idx), shadow_pred.reindex(fold.predict_idx))
    return_oos_r2 = _safe_r2(price.reindex(fold.predict_idx), return_pred.reindex(fold.predict_idx))
    resid_metrics = _residual_diagnostic_metrics(price, resid, sector_name, target, cfg)

    fold_log = {
        "sector": sector_name,
        "etf": etf,
        "retrain_date": fold.retrain_date,
        "target": target,
        "predictors": ",".join(preds),
        "predictor_count": int(len(preds)),
        "shadow_oos_r2": float(shadow_oos_r2),
        "return_oos_r2": float(return_oos_r2),
    }
    fold_log.update(resid_metrics)
    return panel.reset_index(drop=True), fold_log


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


def _train_fold_classifier(cfg: StrategyConfig, fold_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = fold_panel.dropna(subset=["label", "next_ret"]).copy()
    if data.empty:
        empty = pd.DataFrame(index=fold_panel.index)
        return empty, empty

    feature_cols = _classifier_columns(data)
    X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.groupby(data["target"]).ffill().fillna(0.0)
    data = data.assign(**{c: X[c] for c in feature_cols})

    train = data[data["fold_role"] == "train"].copy()
    predict = data[data["fold_role"] == "predict"].copy()
    if train.empty or predict.empty:
        empty = pd.DataFrame(index=fold_panel.index)
        return empty, empty

    train_y = train["label"].astype(float)
    predict_X = predict[feature_cols]
    train_X = train[feature_cols]

    if train_y.nunique(dropna=True) < 2:
        proba = pd.DataFrame({"P_short": 0.0, "P_flat": 1.0, "P_long": 0.0}, index=predict.index)
        signal = pd.Series(0, index=predict.index, name="signal")
    else:
        y_map = train_y.map({-1: 0, 0: 1, 1: 2}).astype(int)
        params = dict(cfg.clf_params)
        params.setdefault("objective", "multi:softprob")
        params["use_label_encoder"] = False
        model = XGBClassifier(**params)
        with np.errstate(all="ignore"):
            model.fit(train_X, y_map)
        pred_idx = model.predict(predict_X)
        inv = {0: -1, 1: 0, 2: 1}
        signal = pd.Series([inv[int(i)] for i in pred_idx], index=predict.index, name="signal")
        proba = pd.DataFrame(model.predict_proba(predict_X), index=predict.index, columns=["P_short", "P_flat", "P_long"])

    panel = predict.copy()
    panel["signal"] = signal
    panel = pd.concat([panel, proba], axis=1)
    panel = panel.reset_index(drop=True)
    return panel, train


def _run_strategy(cfg: StrategyConfig, panel: pd.DataFrame, use_position_manager: bool = False) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if panel.empty:
        empty_metrics = {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "annualized_vol": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "n_trades": 0,
            "avg_trade_return": 0.0,
            "n_long": 0,
            "n_short": 0,
            "n_flat": 0,
            "buy_hold_cum": 0.0,
        }
        return empty_metrics, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    bt = Backtester(cfg)
    pm = None
    if use_position_manager:
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
    result = bt.run(panel, position_manager=pm)
    completed = summarize_completed_trades(result.trades)
    daily = result.portfolio["ret"].dropna() if not result.portfolio.empty and "ret" in result.portfolio else pd.Series(dtype=float)
    downside = daily[daily < 0]
    sortino = float(daily.mean() / downside.std(ddof=0) * np.sqrt(len(daily))) if len(daily) > 1 and len(downside) and np.isfinite(float(downside.std(ddof=0))) and float(downside.std(ddof=0)) != 0 else 0.0
    result.metrics["sortino"] = sortino
    return result.metrics, result.trades, completed, result.portfolio


def _trade_summary(completed: pd.DataFrame) -> dict:
    if completed.empty:
        return {
            "completed_trades": 0,
            "trade_events": 0,
            "long_entries": 0,
            "short_entries": 0,
            "avg_holding_period": 0.0,
            "win_rate": 0.0,
            "average_trade_return": 0.0,
            "median_trade_return": 0.0,
            "best_trade": {},
            "worst_trade": {},
            "longest_holding_trade": {},
            "shortest_holding_trade": {},
            "average_winner": 0.0,
            "average_loser": 0.0,
            "top_10": pd.DataFrame(),
            "worst_10": pd.DataFrame(),
        }

    completed = completed.copy().sort_values(["pnl", "entry_date"], ascending=[False, True])
    winners = completed[completed["pnl"] > 0]
    losers = completed[completed["pnl"] <= 0]
    best = completed.sort_values("pnl", ascending=False).iloc[0]
    worst = completed.sort_values("pnl", ascending=True).iloc[0]
    longest = completed.sort_values(["holding_period", "pnl"], ascending=[False, False]).iloc[0]
    shortest = completed.sort_values(["holding_period", "pnl"], ascending=[True, False]).iloc[0]
    top_10 = completed.sort_values("pnl", ascending=False).head(10)
    worst_10 = completed.sort_values("pnl", ascending=True).head(10)
    return {
        "completed_trades": int(len(completed)),
        "trade_events": int(len(completed)),
        "long_entries": int((completed["direction"] == "long").sum()),
        "short_entries": int((completed["direction"] == "short").sum()),
        "avg_holding_period": float(completed["holding_period"].mean()),
        "win_rate": float((completed["pnl"] > 0).mean()),
        "average_trade_return": float(completed["pnl"].mean()),
        "median_trade_return": float(completed["pnl"].median()),
        "best_trade": best.to_dict(),
        "worst_trade": worst.to_dict(),
        "longest_holding_trade": longest.to_dict(),
        "shortest_holding_trade": shortest.to_dict(),
        "average_winner": float(winners["pnl"].mean()) if len(winners) else 0.0,
        "average_loser": float(losers["pnl"].mean()) if len(losers) else 0.0,
        "top_10": top_10,
        "worst_10": worst_10,
    }


def _format_trade_row(row: pd.Series) -> dict:
    return {
        "entry_date": pd.Timestamp(row["entry_date"]).date(),
        "exit_date": pd.Timestamp(row["exit_date"]).date(),
        "direction": row["direction"],
        "holding_period": int(row["holding_period"]),
        "pnl": float(row["pnl"]),
        "entry_residual_z": float(row["entry_residual_z"]),
        "exit_residual_z": float(row["exit_residual_z"]),
        "entry_confidence": float(row["entry_confidence"]),
        "exit_reason": row["exit_reason"],
    }


def _metrics_row(label: str, metrics: dict, summary: dict) -> dict:
    return {
        "Target": label,
        "Completed Trades": summary["completed_trades"],
        "Trade Events": summary["trade_events"],
        "Long Entries": summary["long_entries"],
        "Short Entries": summary["short_entries"],
        "Average Holding Period": metrics.get("avg_holding_period", summary["avg_holding_period"]),
        "Win Rate": summary["win_rate"],
        "Average Trade Return": summary["average_trade_return"],
        "Median Trade Return": summary["median_trade_return"],
        "Sharpe": metrics.get("sharpe", 0.0),
        "Sortino": metrics.get("sortino", 0.0),
        "Max Drawdown": metrics.get("max_drawdown", 0.0),
        "Cumulative Return": metrics.get("cumulative_return", 0.0),
        "Annualized Return": metrics.get("annualized_return", 0.0),
        "Annualized Volatility": metrics.get("annualized_vol", 0.0),
    }


def _residual_row(label: str, fold_logs: list[dict]) -> dict:
    df = pd.DataFrame(fold_logs)
    return {
        "Target": label,
        "Residual Sharpe": _safe_mean(df.get("residual_sharpe", pd.Series(dtype=float))) if "residual_sharpe" in df else float("nan"),
        "Residual Sortino": _safe_mean(df.get("residual_sortino", pd.Series(dtype=float))) if "residual_sortino" in df else float("nan"),
        "Residual Mean": _safe_mean(df.get("residual_mean", pd.Series(dtype=float))) if "residual_mean" in df else float("nan"),
        "Residual Std": _safe_mean(df.get("residual_std", pd.Series(dtype=float))) if "residual_std" in df else float("nan"),
        "Residual Half Life": _safe_mean(df.get("half_life", pd.Series(dtype=float))) if "half_life" in df else float("nan"),
        "Residual ADF p-value": _safe_mean(df.get("adf_p_value", pd.Series(dtype=float))) if "adf_p_value" in df else float("nan"),
        "Residual Autocorrelation": _safe_mean(df.get("lag1_autocorr", pd.Series(dtype=float))) if "lag1_autocorr" in df else float("nan"),
        "Residual Reversion Success Rate": _safe_mean(df.get("reversion_success", pd.Series(dtype=float))) if "reversion_success" in df else float("nan"),
        "Residual Stability Score": _safe_mean(df.get("residual_stability", pd.Series(dtype=float))) if "residual_stability" in df else float("nan"),
        "Residual Excursion Score": _safe_mean(df.get("opportunity_score", pd.Series(dtype=float))) if "opportunity_score" in df else float("nan"),
        "Average Absolute Z": _safe_mean(df.get("avg_abs_z", pd.Series(dtype=float))) if "avg_abs_z" in df else float("nan"),
        "Maximum Absolute Z": _safe_mean(df.get("max_abs_z", pd.Series(dtype=float))) if "max_abs_z" in df else float("nan"),
        "Opportunity Count": _safe_mean(df.get("opportunity_count", pd.Series(dtype=float))) if "opportunity_count" in df else float("nan"),
        "Average Excursion Duration": _safe_mean(df.get("mean_excursion_duration", pd.Series(dtype=float))) if "mean_excursion_duration" in df else float("nan"),
    }


def _shadow_row(label: str, fold_logs: list[dict]) -> dict:
    df = pd.DataFrame(fold_logs)
    jaccards: list[float] = []
    turnovers: list[float] = []
    if not df.empty and "predictors" in df:
        pred_sets = [set(str(preds).split(",")) if pd.notna(preds) and str(preds) else set() for preds in df.sort_values("retrain_date")["predictors"]]
        for prev, cur in zip(pred_sets, pred_sets[1:]):
            union = prev | cur
            jaccard = len(prev & cur) / len(union) if union else float("nan")
            if np.isfinite(jaccard):
                jaccards.append(float(jaccard))
                turnovers.append(float(1.0 - jaccard))
    return {
        "Target": label,
        "Average Shadow OOS R²": _safe_mean(df.get("shadow_oos_r2", pd.Series(dtype=float))) if "shadow_oos_r2" in df else float("nan"),
        "Average Return OOS R²": _safe_mean(df.get("return_oos_r2", pd.Series(dtype=float))) if "return_oos_r2" in df else float("nan"),
        "Average Predictor Count": _safe_mean(df.get("predictor_count", pd.Series(dtype=float))) if "predictor_count" in df else float("nan"),
        "Predictor Stability": float(np.mean(jaccards)) if jaccards else float("nan"),
        "Predictor Turnover": float(np.mean(turnovers)) if turnovers else float("nan"),
        "Predictor Jaccard": float(np.mean(jaccards)) if jaccards else float("nan"),
    }


def _decision_row(sector_label: str, current_label: str, candidate_label: str, current: dict, candidate: dict) -> dict:
    residual_score = (
        _sign(candidate["Residual Sharpe"] - current["Residual Sharpe"]) +
        _sign(candidate["Residual Sortino"] - current["Residual Sortino"]) +
        _sign(current["Residual ADF p-value"] - candidate["Residual ADF p-value"]) +
        _sign(candidate["Residual Stability Score"] - current["Residual Stability Score"]) +
        _sign(candidate["Residual Excursion Score"] - current["Residual Excursion Score"])
    )
    trading_score = (
        _sign(candidate["Sharpe"] - current["Sharpe"]) +
        _sign(candidate["Cumulative Return"] - current["Cumulative Return"]) +
        _sign(current["Max Drawdown"] - candidate["Max Drawdown"]) +
        _sign(candidate["Annualized Return"] - current["Annualized Return"]) +
        _sign(candidate["Annualized Volatility"] - current["Annualized Volatility"])
    )
    pm_score = (
        _sign(candidate["PM Sharpe"] - current["PM Sharpe"]) +
        _sign(candidate["PM Return"] - current["PM Return"]) +
        _sign(current["PM Drawdown"] - candidate["PM Drawdown"])
    )
    decision_score = residual_score + trading_score + pm_score
    decision = "REPLACE TARGET" if decision_score > 0 else "KEEP CURRENT"
    if decision_score > 0:
        confidence = int(round(min(100.0, 50.0 + 8.0 * decision_score)))
    else:
        confidence = int(round(max(0.0, 50.0 + 8.0 * decision_score)))

    if decision == "REPLACE TARGET":
        if trading_score >= 2 and pm_score >= 1 and residual_score >= 1:
            reason = "Candidate is stronger on trading, PM, and residual diagnostics."
        elif trading_score >= 2:
            reason = "Candidate improves the main trading outcome without weakening residual quality materially."
        else:
            reason = "Candidate has a net edge across the diagnostic scores."
    else:
        if trading_score < 0 and pm_score <= 0:
            reason = "Current target still wins the strategy and PositionManager comparisons."
        else:
            reason = "Current target remains the safer default after combining the diagnostics."

    return {
        "Sector": sector_label,
        "Current Target": current_label,
        "Candidate Target": candidate_label,
        "Residual Winner": candidate_label if residual_score > 0 else current_label,
        "Trading Winner": candidate_label if trading_score > 0 else current_label,
        "PM Winner": candidate_label if pm_score > 0 else current_label,
        "Sharpe Difference": candidate["Sharpe"] - current["Sharpe"],
        "Return Difference": candidate["Cumulative Return"] - current["Cumulative Return"],
        "Opportunity Difference": candidate["Residual Excursion Score"] - current["Residual Excursion Score"],
        "Decision": decision,
        "Confidence Score (0-100)": confidence,
        "Reason": reason,
        "Residual Score": residual_score,
        "Trading Score": trading_score,
        "PM Score": pm_score,
    }


def _rank_rows(decisions: pd.DataFrame, trading_rows: pd.DataFrame, pm_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, d in decisions.iterrows():
        sector = d["Sector"]
        tr = trading_rows[trading_rows["Sector"] == sector].iloc[0]
        pm = pm_rows[pm_rows["Sector"] == sector].iloc[0]
        rows.append({
            "Sector": sector,
            "Recommended Target": d["Candidate Target"] if d["Decision"] == "REPLACE TARGET" else d["Current Target"],
            "Decision": d["Decision"],
            "Confidence Score (0-100)": d["Confidence Score (0-100)"],
            "Candidate Sharpe Gain": d["Sharpe Difference"],
            "Candidate Return Gain": d["Return Difference"],
            "Candidate Opportunity Gain": d["Opportunity Difference"],
            "Baseline Sharpe": tr["Sharpe"],
            "PM Sharpe": pm["PM Sharpe"],
            "Baseline Return": tr["Cumulative Return"],
            "PM Return": pm["Cumulative Return"],
        })
    df = pd.DataFrame(rows)
    df["Ranking Score"] = df["Confidence Score (0-100)"] + 10.0 * _sign(df["Candidate Return Gain"].fillna(0.0))
    return df.sort_values(["Decision", "Confidence Score (0-100)"], ascending=[False, False]).reset_index(drop=True)


def _trade_table(trades: pd.DataFrame, title: str) -> str:
    if trades.empty:
        return f"### {title}\n(no completed trades)\n"
    columns = ["entry_date", "exit_date", "direction", "holding_period", "pnl", "entry_residual_z", "exit_residual_z", "entry_confidence", "exit_reason"]
    out = trades.loc[:, columns].copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"]).dt.date
    out["exit_date"] = pd.to_datetime(out["exit_date"]).dt.date
    return f"### {title}\n" + out.to_markdown(index=False) + "\n"


def _winner_frequency(decisions: pd.DataFrame) -> pd.DataFrame:
    winners = []
    for _, row in decisions.iterrows():
        winners.append({
            "Sector": row["Sector"],
            "Recommended Target": row["Candidate Target"] if row["Decision"] == "REPLACE TARGET" else row["Current Target"],
            "Decision": row["Decision"],
            "Confidence Score (0-100)": row["Confidence Score (0-100)"],
        })
    return pd.DataFrame(winners)


def main() -> None:
    cfg = StrategyConfig(force_recompute=False, make_plots=False)
    md = _load_market_data()
    split, folds = _train_only_folds(md, cfg)
    pipe = StrategyPipeline(cfg)
    tech_cache = pipe._technical_all(md)

    summary_rows = []
    residual_rows = []
    shadow_rows = []
    trading_rows = []
    pm_rows = []
    decision_rows = []
    ranking_rows = []
    trade_summary_rows = []

    for spec in PAIR_SPECS:
        sector_name = spec["sector"]
        etf = spec["etf"]
        current = spec["current"]
        candidate = spec["candidate"]

        current_fold_logs: list[dict] = []
        candidate_fold_logs: list[dict] = []
        current_predict_panels: list[pd.DataFrame] = []
        candidate_predict_panels: list[pd.DataFrame] = []

        for fold in folds:
            current_panel, current_log = _fold_panel(cfg, md, tech_cache, pipe, etf, sector_name, current, fold, split)
            candidate_panel, candidate_log = _fold_panel(cfg, md, tech_cache, pipe, etf, sector_name, candidate, fold, split)

            current_pred_panel, _ = _train_fold_classifier(cfg, current_panel)
            candidate_pred_panel, _ = _train_fold_classifier(cfg, candidate_panel)

            if not current_panel.empty:
                current_fold_logs.append(current_log)
            if not current_pred_panel.empty:
                current_predict_panels.append(current_pred_panel.copy())
            if not candidate_panel.empty:
                candidate_fold_logs.append(candidate_log)
            if not candidate_pred_panel.empty:
                candidate_predict_panels.append(candidate_pred_panel.copy())

        current_panel = pd.concat(current_predict_panels, ignore_index=True) if current_predict_panels else pd.DataFrame()
        candidate_panel = pd.concat(candidate_predict_panels, ignore_index=True) if candidate_predict_panels else pd.DataFrame()

        # Ensure backtest inputs are chronological and isolated to the predict rows.
        current_panel = current_panel.sort_values(["target", "date"]).reset_index(drop=True) if not current_panel.empty else current_panel
        candidate_panel = candidate_panel.sort_values(["target", "date"]).reset_index(drop=True) if not candidate_panel.empty else candidate_panel

        current_metrics, current_trades, current_completed, _ = _run_strategy(cfg, current_panel, use_position_manager=False)
        current_pm_metrics, current_pm_trades, current_pm_completed, _ = _run_strategy(cfg, current_panel, use_position_manager=True)
        candidate_metrics, candidate_trades, candidate_completed, _ = _run_strategy(cfg, candidate_panel, use_position_manager=False)
        candidate_pm_metrics, candidate_pm_trades, candidate_pm_completed, _ = _run_strategy(cfg, candidate_panel, use_position_manager=True)

        current_residual = _residual_row(current, current_fold_logs)
        candidate_residual = _residual_row(candidate, candidate_fold_logs)
        current_shadow = _shadow_row(current, current_fold_logs)
        candidate_shadow = _shadow_row(candidate, candidate_fold_logs)

        current_trading = _metrics_row(current, current_metrics, _trade_summary(current_completed))
        current_trading["Target"] = current
        current_trading["PM Return"] = current_pm_metrics.get("cumulative_return", 0.0)
        current_trading["PM Sharpe"] = current_pm_metrics.get("sharpe", 0.0)
        current_trading["PM Drawdown"] = current_pm_metrics.get("max_drawdown", 0.0)
        current_trading["PM Trade Count"] = int(_trade_summary(current_pm_completed)["completed_trades"])
        current_trading["PM Avg Holding Period"] = float(_trade_summary(current_pm_completed)["avg_holding_period"])

        candidate_trading = _metrics_row(candidate, candidate_metrics, _trade_summary(candidate_completed))
        candidate_trading["Target"] = candidate
        candidate_trading["PM Return"] = candidate_pm_metrics.get("cumulative_return", 0.0)
        candidate_trading["PM Sharpe"] = candidate_pm_metrics.get("sharpe", 0.0)
        candidate_trading["PM Drawdown"] = candidate_pm_metrics.get("max_drawdown", 0.0)
        candidate_trading["PM Trade Count"] = int(_trade_summary(candidate_pm_completed)["completed_trades"])
        candidate_trading["PM Avg Holding Period"] = float(_trade_summary(candidate_pm_completed)["avg_holding_period"])

        current_pm = {
            "Sector": sector_name,
            "Current Target": current,
            "Candidate Target": candidate,
            "Baseline Return": current_metrics.get("cumulative_return", 0.0),
            "PM Return": current_pm_metrics.get("cumulative_return", 0.0),
            "Baseline Sharpe": current_metrics.get("sharpe", 0.0),
            "PM Sharpe": current_pm_metrics.get("sharpe", 0.0),
            "Baseline Drawdown": current_metrics.get("max_drawdown", 0.0),
            "PM Drawdown": current_pm_metrics.get("max_drawdown", 0.0),
            "Trade Count Difference": _trade_summary(current_pm_completed)["completed_trades"] - _trade_summary(current_completed)["completed_trades"],
            "Holding Period Difference": _trade_summary(current_pm_completed)["avg_holding_period"] - _trade_summary(current_completed)["avg_holding_period"],
        }
        candidate_pm = {
            "Sector": sector_name,
            "Current Target": current,
            "Candidate Target": candidate,
            "Baseline Return": candidate_metrics.get("cumulative_return", 0.0),
            "PM Return": candidate_pm_metrics.get("cumulative_return", 0.0),
            "Baseline Sharpe": candidate_metrics.get("sharpe", 0.0),
            "PM Sharpe": candidate_pm_metrics.get("sharpe", 0.0),
            "Baseline Drawdown": candidate_metrics.get("max_drawdown", 0.0),
            "PM Drawdown": candidate_pm_metrics.get("max_drawdown", 0.0),
            "Trade Count Difference": _trade_summary(candidate_pm_completed)["completed_trades"] - _trade_summary(candidate_completed)["completed_trades"],
            "Holding Period Difference": _trade_summary(candidate_pm_completed)["avg_holding_period"] - _trade_summary(candidate_completed)["avg_holding_period"],
        }

        current_all = current_residual | current_shadow | current_trading | current_pm
        candidate_all = candidate_residual | candidate_shadow | candidate_trading | candidate_pm
        decision_row = _decision_row(sector_name, current, candidate, current_all, candidate_all)

        residual_rows.extend([
            {
                "Sector": sector_name,
                "Target": current,
                **current_residual,
            },
            {
                "Sector": sector_name,
                "Target": candidate,
                **candidate_residual,
            },
        ])
        shadow_rows.extend([
            {"Sector": sector_name, "Target": current, **current_shadow},
            {"Sector": sector_name, "Target": candidate, **candidate_shadow},
        ])
        trading_rows.extend([
            {
                "Sector": sector_name,
                "Target": current,
                "Completed Trades": current_trading["Completed Trades"],
                "Trade Events": current_trading["Trade Events"],
                "Long Entries": current_trading["Long Entries"],
                "Short Entries": current_trading["Short Entries"],
                "Average Holding Period": current_trading["Average Holding Period"],
                "Win Rate": current_trading["Win Rate"],
                "Average Trade Return": current_trading["Average Trade Return"],
                "Median Trade Return": current_trading["Median Trade Return"],
                "Sharpe": current_trading["Sharpe"],
                "Sortino": current_trading["Sortino"],
                "Max Drawdown": current_trading["Max Drawdown"],
                "Cumulative Return": current_trading["Cumulative Return"],
                "Annualized Return": current_trading["Annualized Return"],
                "Annualized Volatility": current_trading["Annualized Volatility"],
            },
            {
                "Sector": sector_name,
                "Target": candidate,
                "Completed Trades": candidate_trading["Completed Trades"],
                "Trade Events": candidate_trading["Trade Events"],
                "Long Entries": candidate_trading["Long Entries"],
                "Short Entries": candidate_trading["Short Entries"],
                "Average Holding Period": candidate_trading["Average Holding Period"],
                "Win Rate": candidate_trading["Win Rate"],
                "Average Trade Return": candidate_trading["Average Trade Return"],
                "Median Trade Return": candidate_trading["Median Trade Return"],
                "Sharpe": candidate_trading["Sharpe"],
                "Sortino": candidate_trading["Sortino"],
                "Max Drawdown": candidate_trading["Max Drawdown"],
                "Cumulative Return": candidate_trading["Cumulative Return"],
                "Annualized Return": candidate_trading["Annualized Return"],
                "Annualized Volatility": candidate_trading["Annualized Volatility"],
            },
        ])
        pm_rows.extend([current_pm, candidate_pm])
        decision_rows.append(decision_row)
        ranking_rows.append(current_pm | {"Recommended Target": current if decision_row["Decision"] == "KEEP CURRENT" else candidate})

        current_summary = _trade_summary(current_completed)
        candidate_summary = _trade_summary(candidate_completed)
        current_pm_summary = _trade_summary(current_pm_completed)
        candidate_pm_summary = _trade_summary(candidate_pm_completed)

        trade_summary_rows.append({
            "Sector": sector_name,
            "Current Target": current,
            "Candidate Target": candidate,
            "Current Avg Winner": current_summary["average_winner"],
            "Current Avg Loser": current_summary["average_loser"],
            "Current Best Trade": current_summary["best_trade"].get("pnl", np.nan),
            "Current Worst Trade": current_summary["worst_trade"].get("pnl", np.nan),
            "Current Longest Holding": current_summary["longest_holding_trade"].get("holding_period", np.nan),
            "Current Shortest Holding": current_summary["shortest_holding_trade"].get("holding_period", np.nan),
            "Candidate Avg Winner": candidate_summary["average_winner"],
            "Candidate Avg Loser": candidate_summary["average_loser"],
            "Candidate Best Trade": candidate_summary["best_trade"].get("pnl", np.nan),
            "Candidate Worst Trade": candidate_summary["worst_trade"].get("pnl", np.nan),
            "Candidate Longest Holding": candidate_summary["longest_holding_trade"].get("holding_period", np.nan),
            "Candidate Shortest Holding": candidate_summary["shortest_holding_trade"].get("holding_period", np.nan),
            "Current Trade Log CSV": f"{sector_name.lower().replace(' ', '_').replace('.', '')}_{current}_vs_{candidate}_current_completed_trades.csv",
            "Candidate Trade Log CSV": f"{sector_name.lower().replace(' ', '_').replace('.', '')}_{current}_vs_{candidate}_candidate_completed_trades.csv",
        })

        # Persist detailed trade logs for offline inspection.
        pair_slug = f"{sector_name.lower().replace(' ', '_').replace('.', '')}_{current}_vs_{candidate}"
        current_completed.to_csv(Path(f"outputs/{pair_slug}_current_completed_trades.csv"), index=False)
        candidate_completed.to_csv(Path(f"outputs/{pair_slug}_candidate_completed_trades.csv"), index=False)

    residual_df = pd.DataFrame(residual_rows)
    shadow_df = pd.DataFrame(shadow_rows)
    trading_df = pd.DataFrame(trading_rows)
    pm_df = pd.DataFrame(pm_rows)
    decisions_df = pd.DataFrame(decision_rows)

    # Add explicit PM comparison rows for each target.
    pm_compare_rows = []
    for _, row in pm_df.iterrows():
        pm_compare_rows.append({
            "Sector": row["Sector"],
            "Target": row["Current Target"],
            "Baseline Return": row["Baseline Return"],
            "PM Return": row["PM Return"],
            "Baseline Sharpe": row["Baseline Sharpe"],
            "PM Sharpe": row["PM Sharpe"],
            "Baseline Drawdown": row["Baseline Drawdown"],
            "PM Drawdown": row["PM Drawdown"],
            "Trade Count Difference": row["Trade Count Difference"],
            "Holding Period Difference": row["Holding Period Difference"],
        })
    pm_compare_df = pd.DataFrame(pm_compare_rows)
    trade_summary_df = pd.DataFrame(trade_summary_rows)

    # Write CSVs directly without building markdown.
    residual_df.to_csv(Path("outputs/head_to_head_residual_comparison.csv"), index=False)
    shadow_df.to_csv(Path("outputs/head_to_head_shadow_comparison.csv"), index=False)
    trading_df.to_csv(Path("outputs/head_to_head_trading_comparison.csv"), index=False)
    pm_compare_df.to_csv(Path("outputs/head_to_head_pm_comparison.csv"), index=False)
    decisions_df.to_csv(Path("outputs/head_to_head_decision_matrix.csv"), index=False)

    print()
    print("=== HEAD-TO-HEAD DECISION MATRIX ===")
    print()
    print(decisions_df[["Sector", "Current Target", "Candidate Target", "Decision", "Sharpe Difference", "Return Difference", "Confidence Score (0-100)"]].to_string(index=False))
    print()
    print("=== SUMMARY ===")
    changed = int((decisions_df["Decision"] == "REPLACE TARGET").sum())
    keep = int((decisions_df["Decision"] == "KEEP CURRENT").sum())
    print(f"Replace candidate: {changed}/10")
    print(f"Keep current: {keep}/10")
    print(f"Avg confidence: {decisions_df['Confidence Score (0-100)'].mean():.1f}")
    print()
    print(f"Sharpe avg difference: {decisions_df['Sharpe Difference'].mean():.4f}")
    print(f"Return avg difference: {decisions_df['Return Difference'].mean():.4f}")
    print()
    print("CSV outputs written to outputs/head_to_head_*.csv")


if __name__ == "__main__":
    main()