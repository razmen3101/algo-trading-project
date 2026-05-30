"""Run train-only residual-family ablation experiments.

This script runs the seven experiments described in the task on the TRAIN
period ending before 2024-04-01. It does not modify any production defaults
and writes a concise JSON report to stdout.
"""
from __future__ import annotations

import json
from collections import Counter
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from strategy.strategy_config import StrategyConfig
from strategy.pipeline import StrategyPipeline
from strategy.position_manager import PositionManager, summarize_completed_trades
from strategy.backtester import Backtester


RAW_PREFIX = "raw_"
PERCENT_PREFIX = "percent_"
LOG_PREFIX = "log_"
RETURN_EXACT_FEATURES = {
    "predicted_return",
    "predicted_return_z",
    "predicted_return_rank",
    "predicted_return_direction",
    "predicted_return_percentile",
    "predicted_return_ewm_z",
    "predicted_return_velocity",
    "predicted_return_acceleration",
    "predicted_return_distance_from_extreme",
    "predicted_return_days_positive",
    "predicted_return_days_negative",
}
LEGACY_RESIDUAL_EXACT = {
    "price_residual",
    "price_residual_z",
    "residual_ewm_z",
    "residual_rank",
    "residual_percentile",
    "residual_abs_z",
    "residual_sign",
    "residual_distance_from_zero",
    "residual_distance_from_peak",
    "shadow_price_gap_pct",
    "residual_excursion_bucket",
    "residual_half_life_proxy",
    "residual_ewm_mean",
    "residual_ewm_std",
    "residual_roll_mean",
    "residual_roll_std",
    "residual_ewm_slope",
    "shadow_price",
}


def is_return_feature(col):
    return col in RETURN_EXACT_FEATURES or col.startswith("predicted_return_regime_")


def is_legacy_residual_feature(col):
    if col.startswith((RAW_PREFIX, PERCENT_PREFIX, LOG_PREFIX)):
        return False
    return col in LEGACY_RESIDUAL_EXACT or col.startswith("residual_") or col.startswith("price_residual")


def select_features(feature_cols, families, include_return):
    out = []
    for c in feature_cols:
        if c == "predicted_return_regime":
            continue
        if c.startswith((RAW_PREFIX, PERCENT_PREFIX, LOG_PREFIX)):
            keep = any(
                (fam == "raw" and c.startswith(RAW_PREFIX))
                or (fam == "percent" and c.startswith(PERCENT_PREFIX))
                or (fam == "log" and c.startswith(LOG_PREFIX))
                for fam in families
            )
            if keep:
                out.append(c)
            continue
        if is_return_feature(c):
            if include_return:
                out.append(c)
            continue
        if is_legacy_residual_feature(c):
            continue
        # otherwise keep (technical, sector, non-residual controls)
        out.append(c)
    return out


def feature_sanity(selected_cols, include_return):
    raw_features = [c for c in selected_cols if c.startswith(RAW_PREFIX)]
    percent_features = [c for c in selected_cols if c.startswith(PERCENT_PREFIX)]
    log_features = [c for c in selected_cols if c.startswith(LOG_PREFIX)]
    return_features = [c for c in selected_cols if is_return_feature(c)]
    legacy_residual_features = [c for c in selected_cols if is_legacy_residual_feature(c)]
    return {
        "n_raw_features": len(raw_features),
        "n_percent_features": len(percent_features),
        "n_log_features": len(log_features),
        "n_legacy_residual_features_used": len(legacy_residual_features),
        "n_return_features": len(return_features),
        "raw_features_used": raw_features,
        "percent_features_used": percent_features,
        "log_features_used": log_features,
        "legacy_residual_features_used": legacy_residual_features,
        "return_features_used": return_features,
        "with_return_expected": bool(include_return),
    }


def run_experiment(cfg, panel, feature_cols, families, include_return, train_end):
    data = panel.dropna(subset=["label"]).copy()
    # numeric features and leakage-free fill
    X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.groupby(data["target"]).ffill().fillna(0.0)
    data = data.assign(**{c: X[c] for c in feature_cols})
    # filter to training period (strictly before the chronological validation boundary)
    data = data[data["date"] < train_end].copy()

    feats = select_features(feature_cols, families, include_return)
    sanity = feature_sanity(feats, include_return)
    X_tr = data[feats]
    y_tr = data["label"]

    # train-only XGBoost (no val/test, no tuning)
    params = dict(cfg.clf_params)
    params.setdefault("objective", "multi:softprob")
    model = XGBClassifier(**{**params, "use_label_encoder": False},)
    model.fit(X_tr, y_tr.map({-1: 0, 0: 1, 1: 2}))

    # predictions on train
    y_idx = y_tr.map({-1: 0, 0: 1, 1: 2}).astype(int)
    pred_idx = model.predict(X_tr)
    inv = {0: -1, 1: 0, 2: 1}
    pred_series = pd.Series([inv[i] for i in pred_idx], index=X_tr.index, name="signal")
    proba = pd.DataFrame(model.predict_proba(X_tr), index=X_tr.index, columns=["P_short", "P_flat", "P_long"])

    # classification diagnostics
    acc = accuracy_score(y_idx, pred_idx)
    f1_macro = f1_score(y_idx, pred_idx, average="macro")
    f1_weighted = f1_score(y_idx, pred_idx, average="weighted")
    cm = confusion_matrix(y_idx, pred_idx, labels=[0,1,2])
    class_report = classification_report(y_idx, pred_idx, labels=[0,1,2], zero_division=0, output_dict=True)

    # build panel for backtest (train period only)
    test = data.copy()
    test = test.assign(signal=pred_series.reindex(test.index).values,
                       P_short=proba["P_short"].reindex(test.index).values,
                       P_flat=proba["P_flat"].reindex(test.index).values,
                       P_long=proba["P_long"].reindex(test.index).values)

    # enable PositionManager with cfg parameters
    pm = PositionManager(long_entry_confidence=cfg.pm_entry_confidence,
                         short_entry_confidence=cfg.pm_entry_confidence,
                         flat_probability_block=cfg.flat_probability_block,
                         entry_residual_threshold=cfg.pm_entry_residual_z,
                         mean_reversion_exit=cfg.pm_exit_residual_z,
                         opposite_signal_confidence=cfg.pm_opposite_confidence,
                         stop_loss=cfg.pm_stop_loss,
                         take_profit=cfg.pm_take_profit,
                         max_holding_days=cfg.pm_max_holding_days,
                         allow_flip=cfg.pm_allow_flip)

    sim = pm.simulate(test, cost_bps=cfg.transaction_cost_bps)

    # Backtester metrics (use run_with_positions)
    bt = Backtester(cfg)
    bt_res = bt.run_with_positions(sim)

    # feature importance grouping
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    top30 = imp.head(30).to_dict()
    family_imp = {"raw": 0.0, "percent": 0.0, "log": 0.0, "Return": 0.0, "Other": 0.0}
    for k, v in imp.items():
        if k.startswith("raw_"):
            family_imp["raw"] += v
        elif k.startswith("percent_"):
            family_imp["percent"] += v
        elif k.startswith("log_"):
            family_imp["log"] += v
        elif k.startswith("predicted_return"):
            family_imp["Return"] += v
        else:
            family_imp["Other"] += v

    # fibonacci diagnostics (counts + conditional avg trade return)
    fib_levels = ["fib_23_hit", "fib_38_hit", "fib_50_hit", "fib_61_hit", "fib_78_hit"]
    fib_stats = {}
    for lvl in fib_levels:
        col = None
        for c in test.columns:
            if c.endswith(lvl) or c.endswith("_"+lvl):
                col = c
                break
        if col is None:
            fib_stats[lvl] = {"hit_count": 0, "trade_count": 0, "avg_trade_return": None, "win_rate": None}
            continue
        hits = sim[sim[col] == 1]
        trades = summarize_completed_trades(sim)
        trade_fib = trades.merge(
            test[["date", "target", col]],
            left_on=["entry_date", "target"],
            right_on=["date", "target"],
            how="left",
        )
        trades_hit = trade_fib[trade_fib[col] == 1]
        fib_stats[lvl] = {
            "hit_count": int(hits.shape[0]),
            "trade_count": int(trades_hit.shape[0]),
            "avg_trade_return": float(trades_hit["pnl"].mean()) if len(trades_hit) else None,
            "win_rate": float((trades_hit["pnl"] > 0).mean()) if len(trades_hit) else None,
        }

    # residual diagnostics
    resid_stats = {}
    for fam in families:
        prefix = fam + "_"
        zcol = None
        for c in data.columns:
            if c.startswith(prefix) and c.endswith("_residual_z"):
                zcol = c
                break
        if zcol is None:
            resid_stats[fam] = None
            continue
        s = data[zcol].dropna()
        resid_stats[fam] = {
            "mean_z": float(s.mean()),
            "std_z": float(s.std()),
            "pct_abs_gt_1_25": float((s.abs() > 1.25).mean()),
            "pct_abs_gt_1_5": float((s.abs() > 1.5).mean()),
            "pct_abs_gt_2": float((s.abs() > 2.0).mean()),
            "pct_abs_gt_3": float((s.abs() > 3.0).mean()),
        }

    out = {
        "families": list(families),
        "include_return": include_return,
        "feature_count": len(feats),
        "feature_breakdown": {
            "raw": sanity["n_raw_features"],
            "percent": sanity["n_percent_features"],
            "log": sanity["n_log_features"],
            "return": sanity["n_return_features"],
        },
        "sanity": sanity,
        "classification": {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm.tolist(),
            "class_report": class_report,
        },
        "backtest_metrics": bt_res.metrics,
        "sector_perf": bt_res.sector_perf.fillna(0).to_dict(),
        "target_perf": bt_res.target_perf.fillna(0).to_dict(),
        "top30_features": top30,
        "family_importance": family_imp,
        "fibonacci": fib_stats,
        "residual_diagnostics": resid_stats,
    }
    return out


def main():
    import dataclasses
    base_cfg = StrategyConfig()
    cfg_no_return = dataclasses.replace(
        base_cfg,
        end="2024-03-31",
        enable_multi_residual_engine=True,
        enable_return_feature_expansion=False,
    )
    cfg_with_return = dataclasses.replace(
        base_cfg,
        end="2024-03-31",
        enable_multi_residual_engine=True,
        enable_return_feature_expansion=True,
    )

    from strategy.splits import chrono_split, walk_forward_folds

    sp_no_return = StrategyPipeline(cfg_no_return)
    md = sp_no_return.load_data()
    split = chrono_split(md.prices.index, cfg_no_return)
    folds = walk_forward_folds(md.prices.index, cfg_no_return)
    panel_no_return = sp_no_return.build_panel(md, folds, split)

    sp_with_return = StrategyPipeline(cfg_with_return)
    panel_with_return = sp_with_return.build_panel(md, folds, split)

    # base feature columns from pipeline logic
    excluded = {"date", "etf", "sector", "target", "predictors", "target_price",
                "shadow_price", "next_ret", "label", "spread_signal",
                "ann_vol", "residual_z"}
    feature_cols_no_return = [c for c in panel_no_return.columns if c not in excluded]
    feature_cols_with_return = [c for c in panel_with_return.columns if c not in excluded]

    experiments = [
        ("A_RAW", {"raw"}),
        ("B_PERCENT", {"percent"}),
        ("C_LOG", {"log"}),
        ("D_RAW_PERCENT", {"raw", "percent"}),
        ("E_RAW_LOG", {"raw", "log"}),
        ("F_PERCENT_LOG", {"percent", "log"}),
        ("G_ALL", {"raw", "percent", "log"}),
    ]

    results = {}
    for name, fams in experiments:
        res = run_experiment(cfg_with_return, panel_with_return, feature_cols_with_return, fams, include_return=True, train_end=split.train_end)
        results[f"{name}_with_return"] = res
        print(name + "_with_return", res["sanity"])
        res2 = run_experiment(cfg_no_return, panel_no_return, feature_cols_no_return, fams, include_return=False, train_end=split.train_end)
        results[f"{name}_no_return"] = res2
        print(name + "_no_return", res2["sanity"])

    # save raw JSON results
    import os
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/residual_ablation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=lambda x: str(x))

    # generate a Markdown report
    def md_row(k, v):
        return f"- **{k}**: {v}"

    lines = []
    lines.append("# Residual-family Ablation Report (Train Only)\n")
    lines.append("Generated experiments and diagnostics for residual-family ablation. Train period: before 2024-04-01.\n")

    summary_table = []
    # per-experiment sections
    for ex, res in results.items():
        lines.append(f"## Experiment: {ex}\n")
        lines.append("### Feature counts")
        fb = res['feature_breakdown']
        lines.append(md_row('Total features', res['feature_count']))
        lines.append(md_row('Raw residual features', fb.get('raw', 0)))
        lines.append(md_row('Percent residual features', fb.get('percent', 0)))
        lines.append(md_row('Log residual features', fb.get('log', 0)))
        lines.append(md_row('Return features', fb.get('return', 0)))
        sanity = res.get('sanity', {})
        if sanity:
            lines.append("\n### Feature sanity")
            lines.append(md_row('n_raw_features', sanity.get('n_raw_features', 0)))
            lines.append(md_row('n_percent_features', sanity.get('n_percent_features', 0)))
            lines.append(md_row('n_log_features', sanity.get('n_log_features', 0)))
            lines.append(md_row('n_legacy_residual_features_used', sanity.get('n_legacy_residual_features_used', 0)))
            lines.append(md_row('n_return_features', sanity.get('n_return_features', 0)))
            lines.append(md_row('raw_features_used', sanity.get('raw_features_used', [])))
            lines.append(md_row('percent_features_used', sanity.get('percent_features_used', [])))
            lines.append(md_row('log_features_used', sanity.get('log_features_used', [])))
            lines.append(md_row('legacy_residual_features_used', sanity.get('legacy_residual_features_used', [])))
            lines.append(md_row('return_features_used', sanity.get('return_features_used', [])))
        lines.append("\n### Classification diagnostics")
        c = res['classification']
        lines.append(md_row('Accuracy', round(c['accuracy'], 4)))
        lines.append(md_row('Macro F1', round(c['f1_macro'], 4)))
        lines.append(md_row('Weighted F1', round(c['f1_weighted'], 4)))
        lines.append('\nConfusion matrix:')
        lines.append('```')
        for row in c['confusion_matrix']:
            lines.append(' | '.join(str(x) for x in row))
        lines.append('```')

        lines.append('\n### Backtest metrics (PositionManager enabled)')
        bm = res['backtest_metrics']
        for k in ['cumulative_return','annualized_return','annualized_vol','sharpe','max_drawdown','win_rate','n_trades','avg_trade_return','n_long','n_short']:
            lines.append(md_row(k, round(bm.get(k, 0), 6) if isinstance(bm.get(k,0), float) else bm.get(k)))

        lines.append('\n### Top 30 features')
        for k,v in res['top30_features'].items():
            lines.append(f"- {k}: {v:.6f}")

        lines.append('\n### Family importance')
        for fam, val in res['family_importance'].items():
            lines.append(md_row(fam, round(val, 6)))

        lines.append('\n### Fibonacci diagnostics')
        for lvl, info in res['fibonacci'].items():
            lines.append(md_row(lvl, f"hits={info['hit_count']}, trades={info['trade_count']}, avg_ret={info['avg_trade_return']}, win_rate={info['win_rate']}"))

        lines.append('\n### Residual diagnostics')
        for fam, info in (res.get('residual_diagnostics') or {}).items():
            if info is None:
                lines.append(md_row(fam, 'N/A'))
            else:
                lines.append(md_row(fam, f"mean_z={info['mean_z']:.4f}, std_z={info['std_z']:.4f}, pct>|1.25|={info['pct_abs_gt_1_25']:.3f}"))

        # collect summary row for tables
        summary_table.append({
            'experiment': ex,
            'accuracy': res['classification']['accuracy'],
            'macro_f1': res['classification']['f1_macro'],
            'sharpe': res['backtest_metrics'].get('sharpe', 0),
            'cumulative_return': res['backtest_metrics'].get('cumulative_return', 0),
            'max_dd': res['backtest_metrics'].get('max_drawdown', 0),
            'completed_trades': res['backtest_metrics'].get('n_trades', 0),
            'long_entries': res['backtest_metrics'].get('n_long', 0),
            'short_entries': res['backtest_metrics'].get('n_short', 0),
        })

        lines.append('\n---\n')

    # summary tables
    lines.append('## Summary Tables\n')
    lines.append('### Table 1: Residual Family Performance')
    lines.append('| Experiment | Accuracy | Macro F1 | Sharpe | Cumulative Return | Max DD | Completed Trades | Long Entries | Short Entries |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in summary_table:
        lines.append(f"| {r['experiment']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['sharpe']:.4f} | {r['cumulative_return']:.4f} | {r['max_dd']:.4f} | {r['completed_trades']} | {r['long_entries']} | {r['short_entries']} |")

    # write markdown
    with open('outputs/residual_ablation_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('WROTE outputs/residual_ablation_results.json and outputs/residual_ablation_report.md')


if __name__ == "__main__":
    main()
