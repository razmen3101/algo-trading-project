"""Train-only target selection diagnostic report.

Compares legacy, tradability_score, and meta_target selector modes using only
walk-forward folds whose prediction windows stay strictly before split.train_end.
The report is diagnostic-only: it does not change production behavior.
"""
from __future__ import annotations

import dataclasses
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SECTORS
from strategy.strategy_config import StrategyConfig
from strategy.splits import chrono_split, walk_forward_folds
from strategy.target_selection import TargetSelectionEngine


@dataclass
class TrainOnlyFold:
    retrain_date: pd.Timestamp
    train_idx: pd.DatetimeIndex
    predict_idx: pd.DatetimeIndex


def _load_market_data() -> object:
    cache_files = sorted(Path(".cache").glob("market_data__*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cache_files:
        raise FileNotFoundError("No market_data__*.pkl cache found in .cache/")
    with cache_files[0].open("rb") as f:
        return pickle.load(f)


def _train_only_folds(md, cfg):
    split = chrono_split(md.prices.index, cfg)
    diag_path = Path("outputs/train_only_deep_diagnostic.json")
    if not diag_path.exists():
        raise FileNotFoundError("outputs/train_only_deep_diagnostic.json is required to anchor the train-only retrain timeline")

    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    retrain_dates = sorted({pd.Timestamp(r["retrain_date"]) for r in diag.get("shadow_selected", [])})
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


def _collect_mode_rows(cfg: StrategyConfig, md, split, folds, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = TargetSelectionEngine(cfg, mode=mode)
    rows: list[dict] = []
    decisions: list[dict] = []

    for fold in folds:
        for etf, sector_cfg in SECTORS.items():
            members = [sector_cfg["target"]] + sector_cfg["predictors"]
            etf_ret = md.returns[etf] if etf in md.returns else None
            choice = engine.select(
                etf,
                sector_cfg["name"],
                members,
                md.prices,
                md.returns,
                md.volumes,
                etf_ret,
                train_idx=fold.train_idx,
                predict_idx=fold.predict_idx,
                split=split,
            )

            score_col = "meta_prediction" if mode == "meta_target" and choice.scores.get("meta_prediction", pd.Series(dtype=float)).notna().any() else "tradability_score"
            scored = choice.scores.copy()
            if "candidate" not in scored.columns:
                scored = scored.reset_index().rename(columns={scored.index.name or "index": "candidate"})
            scored["sector"] = sector_cfg["name"]
            scored["etf"] = etf
            scored["mode"] = mode
            scored["retrain_date"] = fold.retrain_date
            scored["current_target"] = sector_cfg["target"]
            scored["selected_target"] = choice.target
            scored["is_selected"] = scored["candidate"].eq(choice.target)
            rows.extend(scored.to_dict("records"))

            selected_matches = scored.loc[scored["is_selected"]]
            if selected_matches.empty:
                selected_row = pd.Series({"candidate": choice.target})
            else:
                selected_row = selected_matches.iloc[0]
            decisions.append({
                "mode": mode,
                "sector": sector_cfg["name"],
                "etf": etf,
                "retrain_date": fold.retrain_date,
                "current_target": sector_cfg["target"],
                "selected_target": choice.target,
                "selector_score": float(selected_row.get(score_col, np.nan)),
                "future_residual_sharpe": float(selected_row.get("future_residual_sharpe", np.nan)),
                "future_strategy_return": float(selected_row.get("future_strategy_return", np.nan)),
                "future_tradability_score": float(selected_row.get("future_tradability_score", np.nan)),
                "tradability_score": float(selected_row.get("tradability_score", np.nan)),
                "meta_prediction": float(selected_row.get("meta_prediction", np.nan)),
            })

    return pd.DataFrame(rows), pd.DataFrame(decisions)


def _winner_frequency(decisions: pd.DataFrame) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(columns=["sector", "mode", "current_target", "most_frequent_selected", "selection_count", "retrain_count", "turnover_rate"])
    out = []
    for (mode, sector), g in decisions.groupby(["mode", "sector"], sort=True):
        g = g.sort_values("retrain_date")
        counts = g["selected_target"].value_counts()
        selected = counts.index[0]
        selection_count = int(counts.iloc[0])
        retrain_count = int(len(g))
        turnover_rate = float((g["selected_target"] != g["selected_target"].shift(1)).iloc[1:].mean()) if len(g) > 1 else 0.0
        out.append({
            "mode": mode,
            "sector": sector,
            "current_target": g["current_target"].iloc[0],
            "most_frequent_selected": selected,
            "selection_count": selection_count,
            "retrain_count": retrain_count,
            "turnover_rate": turnover_rate,
        })
    return pd.DataFrame(out)


def _aggregate_candidates(rows: pd.DataFrame, mode: str) -> pd.DataFrame:
    if rows.empty:
        return rows
    numeric_cols = [
        "tradability_score", "meta_prediction", "residual_sharpe", "residual_sortino",
        "opportunity_score", "reversion_success", "half_life_score", "residual_stability",
        "predictor_stability", "target_stability", "fibonacci_score", "future_residual_sharpe",
        "future_strategy_return", "future_tradability_score",
    ]
    agg = rows.groupby(["sector", "candidate"], sort=True)[numeric_cols].mean(numeric_only=True)
    agg["retrain_count"] = rows.groupby(["sector", "candidate"], sort=True).size()
    agg = agg.reset_index()
    score_col = "meta_prediction" if mode == "meta_target" else "tradability_score"
    agg = agg.sort_values(["sector", score_col], ascending=[True, False])
    return agg


def _top_contributors(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    cols = [
        "residual_sharpe", "residual_sortino", "opportunity_score", "reversion_success",
        "half_life_score", "residual_stability", "predictor_stability", "target_stability",
        "fibonacci_score",
    ]
    scored = rows.copy()
    for c in cols:
        scored[f"rank_{c}"] = scored.groupby(["sector", "retrain_date"])[c].transform(lambda s: s.rank(pct=True, method="average"))
    selected = scored[scored["is_selected"]]
    out = []
    for c in cols:
        winner_mean = float(selected[f"rank_{c}"].mean())
        all_mean = float(scored[f"rank_{c}"].mean())
        out.append({
            "metric": c,
            "winner_mean_rank": winner_mean,
            "all_mean_rank": all_mean,
            "lift_vs_universe": winner_mean - all_mean,
        })
    return pd.DataFrame(out).sort_values("lift_vs_universe", ascending=False)


def _fib_contribution(rows: pd.DataFrame, mode: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    trad = rows.copy()
    trad["score_no_fib"] = trad["tradability_score"] - 0.05 * trad["fibonacci_score"].fillna(0.0)
    out = []
    for (sector, retrain_date), g in trad.groupby(["sector", "retrain_date"], sort=True):
        winner_with = g.loc[g["tradability_score"].astype(float).idxmax()]
        winner_without = g.loc[g["score_no_fib"].astype(float).idxmax()]
        out.append({
            "sector": sector,
            "retrain_date": retrain_date,
            "winner_with_fib": winner_with["candidate"],
            "winner_without_fib": winner_without["candidate"],
            "winner_changed": bool(winner_with["candidate"] != winner_without["candidate"]),
            "future_tradability_with_fib": float(winner_with["future_tradability_score"]),
            "future_tradability_without_fib": float(winner_without["future_tradability_score"]),
        })
    return pd.DataFrame(out)


def _build_report(cfg: StrategyConfig, split, folds, mode_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> str:
    legacy_rows, legacy_decisions = mode_outputs["legacy"]
    trad_rows, trad_decisions = mode_outputs["tradability_score"]
    meta_rows, meta_decisions = mode_outputs["meta_target"]
    future_lookup = trad_rows.reindex(columns=["sector", "retrain_date", "candidate", "future_residual_sharpe", "future_strategy_return", "future_tradability_score"]).drop_duplicates()

    current = pd.DataFrame([
        {"sector": cfg_sector["name"], "current_target": cfg_sector["target"]}
        for cfg_sector in SECTORS.values()
    ]).sort_values("sector")

    trad_summary = _winner_frequency(trad_decisions)
    meta_summary = _winner_frequency(meta_decisions)

    table1 = current.merge(trad_summary[["sector", "most_frequent_selected", "turnover_rate"]], on="sector", how="left")
    table1 = table1.rename(columns={"most_frequent_selected": "tradability_target", "turnover_rate": "tradability_turnover_rate"})
    table2 = current.merge(meta_summary[["sector", "most_frequent_selected", "turnover_rate"]], on="sector", how="left")
    table2 = table2.rename(columns={"most_frequent_selected": "meta_target", "turnover_rate": "meta_turnover_rate"})

    trad_rankings = _aggregate_candidates(trad_rows, "tradability_score")
    meta_rankings = _aggregate_candidates(meta_rows, "meta_target")

    # future performance table
    perf_rows = []
    for mode, decisions in [("legacy", legacy_decisions), ("tradability_score", trad_decisions), ("meta_target", meta_decisions)]:
        perf_rows.append({
            "mode": mode,
            "avg_future_residual_sharpe": float(decisions["future_residual_sharpe"].mean()),
            "avg_future_strategy_return": float(decisions["future_strategy_return"].mean()),
            "avg_future_tradability_score": float(decisions["future_tradability_score"].mean()),
            "avg_turnover_rate": float(_winner_frequency(decisions)["turnover_rate"].mean()),
        })
    table8 = pd.DataFrame(perf_rows)

    # top contributors and fibonacci contribution use tradability mode.
    table6 = _top_contributors(trad_rows)
    fib = _fib_contribution(trad_rows, "tradability_score")
    table7 = pd.DataFrame([
        {
            "winner_changed_rate": float(fib["winner_changed"].mean()) if len(fib) else 0.0,
            "avg_future_tradability_with_fib": float(fib["future_tradability_with_fib"].mean()) if len(fib) else 0.0,
            "avg_future_tradability_without_fib": float(fib["future_tradability_without_fib"].mean()) if len(fib) else 0.0,
        }
    ])

    lines: list[str] = []
    lines.append("# TRAIN-ONLY TARGET SELECTION DIAGNOSTIC")
    lines.append("")
    lines.append(f"Train-only window: dates strictly before {split.train_end.date()}")
    lines.append(f"Train-only retrains evaluated: {len(folds)}")
    lines.append("")

    lines.append("## Table 1 - Current selector vs TradabilityScore selector")
    lines.append(table1.to_markdown(index=False))
    lines.append("")

    lines.append("## Table 2 - Current selector vs Meta selector")
    lines.append(table2.to_markdown(index=False))
    lines.append("")

    lines.append("## Table 3 - Per-sector ranking sorted by TradabilityScore")
    for sector in sorted(trad_rankings["sector"].unique()):
        lines.append(f"### {sector}")
        lines.append(trad_rankings.loc[trad_rankings["sector"] == sector].reindex(columns=["candidate", "tradability_score", "future_tradability_score", "future_residual_sharpe", "future_strategy_return"]).to_markdown(index=False))
        lines.append("")

    lines.append("## Table 4 - Per-sector ranking sorted by MetaPrediction")
    for sector in sorted(meta_rankings["sector"].unique()):
        lines.append(f"### {sector}")
        show_cols = ["candidate", "meta_prediction", "tradability_score", "future_tradability_score", "future_residual_sharpe", "future_strategy_return"]
        lines.append(meta_rankings.loc[meta_rankings["sector"] == sector].reindex(columns=show_cols).to_markdown(index=False))
        lines.append("")

    lines.append("## Table 5 - Target turnover")
    legacy_summary = _winner_frequency(legacy_decisions)
    lines.append(pd.concat([legacy_summary.assign(mode="legacy"), trad_summary.assign(mode="tradability_score"), meta_summary.assign(mode="meta_target")], ignore_index=True).to_markdown(index=False))
    lines.append("")

    lines.append("## Table 6 - Top contributors")
    lines.append(table6.to_markdown(index=False))
    lines.append("")

    lines.append("## Table 7 - Fibonacci contribution")
    lines.append(table7.to_markdown(index=False))
    lines.append("")

    lines.append("## Table 8 - Future performance")
    lines.append(table8.to_markdown(index=False))
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    cfg = StrategyConfig(force_recompute=True)
    diag_cfg = dataclasses.replace(cfg, target_min_history=60)
    md = _load_market_data()
    split, folds = _train_only_folds(md, diag_cfg)

    mode_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for mode in ["legacy", "tradability_score", "meta_target"]:
        mode_outputs[mode] = _collect_mode_rows(diag_cfg, md, split, folds, mode)

    report = _build_report(diag_cfg, split, folds, mode_outputs)
    out = Path("outputs/train_only_target_selection_diagnostic.md")
    out.write_text(report, encoding="utf-8")

    # Save supporting CSVs for later inspection.
    for mode, (rows, decisions) in mode_outputs.items():
        rows.to_csv(Path(f"outputs/train_only_target_selection_{mode}_rows.csv"), index=False)
        decisions.to_csv(Path(f"outputs/train_only_target_selection_{mode}_decisions.csv"), index=False)

    print(report)


if __name__ == "__main__":
    main()
