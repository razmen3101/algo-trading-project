from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import math
import warnings

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from strategy.target_selector import SectorTargetSelector
from strategy.predictor_selector import PredictorSelector
from strategy.regressors import DynamicShadowPriceModel, DynamicReturnModel
from strategy.residual_features import ResidualFeatureBuilder, distance_from_peak
from strategy.position_manager import PositionManager, summarize_completed_trades

try:
    from statsmodels.tsa.stattools import adfuller
except Exception:  # pragma: no cover - optional dependency in some environments
    adfuller = None


SelectionMode = Literal["legacy", "tradability_score", "meta_target"]


@dataclass
class TargetChoice:
    etf: str
    sector: str
    target: str
    candidates: list[str]
    scores: pd.DataFrame
    mode: str
    selected_score: float | None = None
    meta_prediction: float | None = None
    selected_predictors: list[str] = field(default_factory=list)
    predictor_coefficients: pd.Series | None = None


@dataclass
class _SectorState:
    prev_target: str | None = None
    prev_predictors: set[str] = field(default_factory=set)
    selected_counts: dict[str, int] = field(default_factory=dict)
    meta_history: list[dict] = field(default_factory=list)


def _pct_rank(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    if series.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    return series.rank(pct=True, method="average").fillna(0.5).astype(float)


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _sharpe(pnls: pd.Series) -> float:
    s = pnls.dropna()
    if len(s) < 2:
        return 0.0
    mu = float(s.mean())
    sd = float(s.std(ddof=0))
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float(mu / sd * np.sqrt(len(s)))


def _sortino(pnls: pd.Series) -> float:
    s = pnls.dropna()
    if len(s) < 2:
        return 0.0
    downside = s[s < 0]
    denom = float(downside.std(ddof=0)) if len(downside) else 0.0
    if denom == 0 or not np.isfinite(denom):
        return 0.0
    return float(s.mean() / denom * np.sqrt(len(s)))


def _safe_adf_score(series: pd.Series) -> tuple[float, float]:
    s = series.dropna()
    if len(s) < 20:
        return float("nan"), 0.0
    pval = float("nan")
    if adfuller is not None:
        try:
            pval = float(adfuller(s, autolag="AIC")[1])
        except Exception:
            pval = float("nan")
    score = 1.0 - pval if np.isfinite(pval) else 0.0
    return pval, _clip01(score)


def _variance_stability(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 20:
        return 0.0
    half = max(5, len(s) // 2)
    first = s.iloc[:half]
    second = s.iloc[-half:]
    a = float(first.std(ddof=0))
    b = float(second.std(ddof=0))
    if a == 0 or b == 0 or not np.isfinite(a) or not np.isfinite(b):
        return 0.0
    ratio = a / b
    if ratio <= 0 or not np.isfinite(ratio):
        return 0.0
    return _clip01(1.0 - abs(np.log(ratio)) / 3.0)


def _regime_stability(abs_z: pd.Series) -> float:
    s = abs_z.dropna()
    if len(s) < 20:
        return 0.0
    bins = pd.cut(s, bins=[-np.inf, 1.0, 1.5, 2.0, np.inf], labels=False)
    transitions = (bins != bins.shift(1)).astype(float).fillna(0.0)
    return _clip01(1.0 - float(transitions.mean()))


def _half_life(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 20:
        return float("nan")
    lag1 = float(s.autocorr(lag=1))
    if not np.isfinite(lag1) or lag1 <= 0 or lag1 >= 1:
        return float("nan")
    return float(-np.log(2.0) / np.log(lag1))


def _half_life_score(half_life: float, target_half_life: float = 3.0) -> float:
    if not np.isfinite(half_life):
        return 0.0
    return float(np.exp(-abs(half_life - target_half_life)))


def _synthetic_trade_panel(price: pd.Series, residual_z: pd.Series, sector: str, target: str, cfg) -> pd.DataFrame:
    z = residual_z.reindex(price.index).astype(float)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    abs_z = z.abs()
    entry = float(getattr(cfg, "pm_entry_residual_z", 1.25))
    strength = ((abs_z - entry) / max(entry, 1e-6)).clip(lower=0.0, upper=1.0)

    signal = pd.Series(np.where(z <= -entry, 1, np.where(z >= entry, -1, 0)), index=price.index)
    p_long = pd.Series(0.10, index=price.index, dtype=float)
    p_short = pd.Series(0.10, index=price.index, dtype=float)
    p_flat = pd.Series(0.85, index=price.index, dtype=float)
    p_long.loc[signal == 1] = 0.70 + 0.25 * strength.loc[signal == 1]
    p_short.loc[signal == -1] = 0.70 + 0.25 * strength.loc[signal == -1]
    p_flat.loc[signal != 0] = 0.10

    return pd.DataFrame({
        "date": price.index,
        "sector": sector,
        "target": target,
        "target_price": price.values,
        "residual_z": z.values,
        "signal": signal.values,
        "P_short": p_short.values,
        "P_flat": p_flat.values,
        "P_long": p_long.values,
        "next_ret": price.pct_change().shift(-1).values,
    })


def _trade_metrics(price: pd.Series, residual_z: pd.Series, sector: str, target: str, cfg) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    panel = _synthetic_trade_panel(price, residual_z, sector, target, cfg)
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

    abs_z = residual_z.abs().dropna()
    opp_weights = {
        1.25: float((abs_z > 1.25).mean()) if len(abs_z) else 0.0,
        1.50: float((abs_z > 1.50).mean()) if len(abs_z) else 0.0,
        2.00: float((abs_z > 2.00).mean()) if len(abs_z) else 0.0,
        3.00: float((abs_z > 3.00).mean()) if len(abs_z) else 0.0,
    }
    opportunity_score = 1.00 * opp_weights[1.25] + 0.75 * opp_weights[1.50] + 0.50 * opp_weights[2.00] + 0.25 * opp_weights[3.00]

    half_life = _half_life(residual_z)
    adf_p, adf_score = _safe_adf_score(residual_z)
    lag1 = float(residual_z.dropna().autocorr(lag=1)) if residual_z.dropna().shape[0] > 2 else float("nan")
    lag1_score = _clip01(1.0 - abs(lag1)) if np.isfinite(lag1) else 0.0
    var_stability = _variance_stability(residual_z)
    regime_stability = _regime_stability(abs_z)
    residual_stability = float(np.mean([adf_score, lag1_score, var_stability, regime_stability]))

    if len(trades):
        reversion_success = float((trades["pnl"] > 0).mean())
        mean_excursion_duration = float(trades["holding_period"].mean())
        avg_abs_z = float(abs_z.mean()) if len(abs_z) else 0.0
        max_abs_z = float(abs_z.max()) if len(abs_z) else 0.0
    else:
        reversion_success = 0.0
        mean_excursion_duration = 0.0
        avg_abs_z = float(abs_z.mean()) if len(abs_z) else 0.0
        max_abs_z = float(abs_z.max()) if len(abs_z) else 0.0

    fib_rows = []
    if len(trades):
        for trade_id, seg in sim.dropna(subset=["trade_id"]).groupby("trade_id", sort=False):
            seg = seg.sort_values("date")
            seg_z = seg["residual_z"].abs().astype(float)
            dist = distance_from_peak(seg_z, reset_threshold=1.0).fillna(0.0)
            fib_hit = bool((dist >= 0.236).any())
            fib_rows.append({
                "trade_id": int(trade_id),
                "fib_hit": fib_hit,
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

    metrics = {
        "residual_sharpe": _sharpe(pnls),
        "residual_sortino": _sortino(pnls),
        "opportunity_score": float(opportunity_score),
        "opportunity_count": float(sum(opp_weights.values())),
        "reversion_success": float(reversion_success),
        "half_life": float(half_life),
        "half_life_score": _half_life_score(half_life),
        "adf_p_value": float(adf_p),
        "lag1_autocorr": float(lag1),
        "variance_stability": float(var_stability),
        "regime_stability": float(regime_stability),
        "residual_stability": float(residual_stability),
        "avg_abs_z": float(avg_abs_z),
        "max_abs_z": float(max_abs_z),
        "mean_excursion_duration": float(mean_excursion_duration),
        "fib_reversion_success": float(fib_reversion_success),
        "fib_mean_retrace_depth": float(fib_mean_retrace_depth),
        "fib_retrace_hit_rate": float(fib_retrace_hit_rate),
        "fibonacci_score": float(fibonacci_score),
        "future_residual_sharpe": float(_sharpe(pnls)),
        "future_strategy_return": float(pnls.sum()) if len(pnls) else 0.0,
        "future_tradability_score": float(0.6 * _sharpe(pnls) + 0.4 * (float(pnls.sum()) if len(pnls) else 0.0)),
        "trade_count": float(len(trades)),
    }
    return metrics, sim, trades


class _SelectionStateStore:
    def __init__(self):
        self.by_sector: dict[str, _SectorState] = {}

    def get(self, sector: str) -> _SectorState:
        if sector not in self.by_sector:
            self.by_sector[sector] = _SectorState()
        return self.by_sector[sector]


class TargetSelectionEngine:
    def __init__(self, cfg, mode: SelectionMode | None = None):
        self.cfg = cfg
        self.mode = mode or getattr(cfg, "target_selection_mode", "tradability_score")
        self._legacy = SectorTargetSelector(cfg)
        self._states = _SelectionStateStore()

    def _candidate_frame(self, etf: str, sector_name: str, members: list[str], prices: pd.DataFrame,
                         returns: pd.DataFrame, volumes: pd.DataFrame, etf_returns: pd.Series | None,
                         train_idx, predict_idx=None, split=None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        print(f"[target-select] stage=candidate_frame sector={sector_name} retrain_date={pd.Index(predict_idx)[0].date() if predict_idx is not None and len(pd.Index(predict_idx)) else 'n/a'} candidates={len(members)}")
        train_prices = prices.loc[pd.Index(train_idx)]
        train_returns = returns.reindex(train_idx)
        train_volumes = volumes.loc[pd.Index(train_idx)]
        future_idx = pd.Index(predict_idx) if predict_idx is not None else pd.Index([])
        state = self._states.get(sector_name)

        rows: list[dict] = []
        predictor_map: dict[str, list[str]] = {}
        completed_candidates = 0
        for cand in [m for m in members if m in train_prices.columns]:
            print(f"[target-select]   sector={sector_name} candidate={cand} status=score_start completed_candidates={completed_candidates}")
            p = train_prices[cand].dropna()
            r = train_returns[cand].dropna() if cand in train_returns.columns else pd.Series(dtype=float)
            if len(p) < self.cfg.target_min_history:
                print(f"[target-select]   sector={sector_name} candidate={cand} status=skip_short_history")
                continue

            peers = [m for m in members if m != cand and m in train_prices.columns]
            predictor_choice = PredictorSelector(self.cfg).select(cand, peers, train_returns, train_prices)
            preds = predictor_choice.selected
            predictor_map[cand] = preds

            shadow_model = DynamicShadowPriceModel(self.cfg)
            shadow_feats, _, base_target_price, safe_train_idx = shadow_model.fit(prices, cand, preds, train_idx)
            return_model = DynamicReturnModel(self.cfg)
            return_feats, _, safe_return_idx = return_model.fit(prices, cand, preds, train_idx)
            if len(safe_train_idx) == 0 or len(safe_return_idx) == 0:
                print(f"[target-select]   sector={sector_name} candidate={cand} status=skip_no_train_data")
                continue

            train_shadow = shadow_model.predict(shadow_feats, train_idx, base_target_price)
            train_predret = return_model.predict(return_feats, train_idx)
            train_resid = ResidualFeatureBuilder(self.cfg).build(
                train_prices[cand].reindex(train_idx),
                train_shadow.reindex(train_idx),
                train_predret.reindex(train_idx),
                None,
            )

            train_metrics, _, _ = _trade_metrics(train_prices[cand].reindex(train_idx), train_resid["residual_ewm_z"], sector_name, cand, self.cfg)

            future_metrics = {}
            if len(future_idx):
                future_shadow = shadow_model.predict(shadow_feats, future_idx, base_target_price)
                future_predret = return_model.predict(return_feats, future_idx)
                future_resid = ResidualFeatureBuilder(self.cfg).build(
                    prices[cand].reindex(future_idx),
                    future_shadow.reindex(future_idx),
                    future_predret.reindex(future_idx),
                    None,
                )
                future_metrics, _, _ = _trade_metrics(prices[cand].reindex(future_idx), future_resid["residual_ewm_z"], sector_name, cand, self.cfg)

            target_frequency = state.selected_counts.get(cand, 0)
            target_total = sum(state.selected_counts.values())
            target_stability = float(target_frequency / target_total) if target_total else 0.0
            if state.prev_predictors:
                union = state.prev_predictors | set(preds)
                jaccard = len(state.prev_predictors & set(preds)) / len(union) if union else 0.5
            else:
                jaccard = 0.5
            predictor_turnover = 1.0 - float(jaccard)

            row = {
                "etf": etf,
                "sector": sector_name,
                "candidate": cand,
                "selected_predictors": ",".join(preds),
                "predictor_count": len(preds),
                "predictor_jaccard": float(jaccard),
                "predictor_turnover": float(predictor_turnover),
                "target_stability": float(target_stability),
                "residual_quality_score": float(np.mean([
                    _pct_rank(pd.Series([train_metrics["avg_abs_z"]])).iloc[0],
                    _pct_rank(pd.Series([train_metrics["max_abs_z"]])).iloc[0],
                    _pct_rank(pd.Series([train_metrics["mean_excursion_duration"]])).iloc[0],
                ])),
            }
            row.update(train_metrics)
            row.update({f"future_{k}": v for k, v in future_metrics.items()})
            rows.append(row)
            completed_candidates += 1
            print(f"[target-select]   sector={sector_name} candidate={cand} status=score_done completed_candidates={completed_candidates}")

        scores = pd.DataFrame(rows)
        if scores.empty:
            return scores, predictor_map

        for col in [
            "residual_sharpe", "residual_sortino", "opportunity_score", "reversion_success",
            "half_life_score", "residual_stability", "predictor_stability", "target_stability",
            "fibonacci_score", "residual_quality_score",
        ]:
            if col not in scores:
                scores[col] = 0.0

        scores["predictor_stability"] = scores["predictor_jaccard"].fillna(0.0)

        # robust, sector-local normalization
        for col in [
            "residual_sharpe", "residual_sortino", "opportunity_score", "reversion_success",
            "half_life_score", "residual_stability", "predictor_stability", "target_stability",
            "fibonacci_score",
        ]:
            scores[f"rank_{col}"] = _pct_rank(scores[col].astype(float))

        scores["tradability_score"] = (
            0.25 * scores["rank_residual_sharpe"]
            + 0.15 * scores["rank_residual_sortino"]
            + 0.15 * scores["rank_opportunity_score"]
            + 0.15 * scores["rank_reversion_success"]
            + 0.10 * scores["rank_half_life_score"]
            + 0.05 * scores["rank_residual_stability"]
            + 0.05 * scores["rank_predictor_stability"]
            + 0.05 * scores["rank_target_stability"]
            + 0.05 * scores["rank_fibonacci_score"]
        )

        meta_prediction = pd.Series(np.nan, index=scores.index, dtype=float)
        if self.mode == "meta_target":
            hist = pd.DataFrame(state.meta_history)
            if len(hist) >= int(getattr(self.cfg, "target_selection_meta_min_rows", 100)):
                train_df = hist.dropna(subset=["future_tradability_score"]).copy()
                feature_cols = [c for c in scores.columns if c not in {
                    "etf", "sector", "candidate", "selected_predictors", "tradability_score",
                    "future_residual_sharpe", "future_strategy_return", "future_tradability_score",
                } and pd.api.types.is_numeric_dtype(scores[c])]
                feature_cols = [c for c in feature_cols if c in train_df.columns]
                if feature_cols and len(train_df) >= int(getattr(self.cfg, "target_selection_meta_min_rows", 100)):
                    model = XGBRegressor(**self.cfg.reg_params)
                    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                    y_train = train_df["future_tradability_score"].astype(float)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=FutureWarning)
                        warnings.simplefilter("ignore", category=UserWarning)
                        model.fit(X_train, y_train)
                    X_current = scores[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                    meta_prediction = pd.Series(model.predict(X_current), index=scores.index, dtype=float)
                    scores["meta_prediction"] = meta_prediction
                else:
                    scores["meta_prediction"] = scores["tradability_score"]
            else:
                scores["meta_prediction"] = scores["tradability_score"]
        else:
            scores["meta_prediction"] = np.nan

        if self.mode == "meta_target" and scores["meta_prediction"].notna().any():
            selected_row = scores["meta_prediction"].astype(float).idxmax()
            selected_score = float(scores.loc[selected_row, "meta_prediction"])
        else:
            selected_row = scores["tradability_score"].astype(float).idxmax()
            selected_score = float(scores.loc[selected_row, "tradability_score"])

        selected_idx = str(scores.loc[selected_row, "candidate"])

        choice = TargetChoice(
            etf=etf,
            sector=sector_name,
            target=selected_idx,
            candidates=[m for m in members if m in train_prices.columns],
            scores=scores.sort_values("tradability_score", ascending=False),
            mode=self.mode,
            selected_score=selected_score,
            meta_prediction=float(scores.loc[selected_row, "meta_prediction"]) if "meta_prediction" in scores else None,
            selected_predictors=predictor_map.get(selected_idx, []),
            predictor_coefficients=predictor_choice.coefficients if "predictor_choice" in locals() else None,
        )

        # Update train-only history for meta learning and stability tracking.
        state.prev_target = choice.target
        state.prev_predictors = set(choice.selected_predictors)
        state.selected_counts[choice.target] = state.selected_counts.get(choice.target, 0) + 1

        if len(future_idx) and split is not None and len(pd.Index(future_idx)) and pd.Index(future_idx)[-1] < split.train_end:
            for _, row in scores.iterrows():
                rec = row.to_dict()
                rec.update({
                    "selected_target": choice.target,
                    "selector_mode": self.mode,
                    "retrain_date": pd.Index(train_idx)[-1] if len(train_idx) else None,
                })
                state.meta_history.append(rec)

        return scores, predictor_map

    def select(self, etf: str, sector_name: str, members: list[str],
               prices: pd.DataFrame, returns: pd.DataFrame,
               volumes: pd.DataFrame, etf_returns: pd.Series | None,
               train_idx, predict_idx=None, split=None) -> TargetChoice:
        if self.mode == "legacy":
            print(f"[target-select] stage=legacy sector={sector_name}")
            legacy_choice = self._legacy.select(etf, sector_name, members, prices.loc[pd.Index(train_idx)], returns.reindex(train_idx), volumes.loc[pd.Index(train_idx)], None if etf_returns is None else etf_returns.reindex(train_idx))
            return TargetChoice(
                etf=legacy_choice.etf,
                sector=legacy_choice.sector,
                target=legacy_choice.target,
                candidates=legacy_choice.candidates,
                scores=legacy_choice.scores,
                mode=self.mode,
                selected_score=float(legacy_choice.scores["total"].max()) if "total" in legacy_choice.scores else None,
                meta_prediction=None,
                selected_predictors=[],
                predictor_coefficients=None,
            )

        scores, predictor_map = self._candidate_frame(etf, sector_name, members, prices, returns, volumes, etf_returns, train_idx, predict_idx, split)
        if scores.empty:
            print(f"[target-select] sector={sector_name} fallback=legacy_due_to_empty_scores")
            fallback = self._legacy.select(etf, sector_name, members, prices.loc[pd.Index(train_idx)], returns.reindex(train_idx), volumes.loc[pd.Index(train_idx)], None if etf_returns is None else etf_returns.reindex(train_idx))
            return TargetChoice(
                etf=fallback.etf,
                sector=fallback.sector,
                target=fallback.target,
                candidates=fallback.candidates,
                scores=fallback.scores,
                mode=self.mode,
                selected_score=float(fallback.scores["total"].max()) if "total" in fallback.scores else None,
                meta_prediction=None,
                selected_predictors=[],
                predictor_coefficients=None,
            )

        select_col = "meta_prediction" if self.mode == "meta_target" and scores["meta_prediction"].notna().any() else "tradability_score"
        selected_row = scores[select_col].astype(float).idxmax()
        selected_idx = str(scores.loc[selected_row, "candidate"])
        print(f"[target-select] sector={sector_name} selected={selected_idx} select_col={select_col}")
        state = self._states.get(sector_name)
        choice = TargetChoice(
            etf=etf,
            sector=sector_name,
            target=selected_idx,
            candidates=[m for m in members if m in prices.columns],
            scores=scores.sort_values(select_col, ascending=False),
            mode=self.mode,
            selected_score=float(scores.loc[selected_row, select_col]),
            meta_prediction=float(scores.loc[selected_row, "meta_prediction"]) if "meta_prediction" in scores else None,
            selected_predictors=predictor_map.get(selected_idx, []),
            predictor_coefficients=None,
        )
        state.prev_target = choice.target
        state.prev_predictors = set(choice.selected_predictors)
        state.selected_counts[choice.target] = state.selected_counts.get(choice.target, 0) + 1
        return choice


def build_target_selector(cfg, mode: SelectionMode | None = None) -> TargetSelectionEngine:
    return TargetSelectionEngine(cfg, mode=mode)
