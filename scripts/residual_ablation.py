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


DATE_CUTOFF = pd.Timestamp("2024-04-01")


def select_features(feature_cols, families, include_return):
    # keep base features (those not in family prefixes) and only include
    # family-prefixed columns according to `families` set
    family_prefixes = {
        "raw": "raw_",
        "percent": "percent_",
        "log": "log_",
    }
    out = []
    for c in feature_cols:
        if c.startswith(tuple(family_prefixes.values())):
            # family-specific column
            keep = any(c.startswith(family_prefixes[f]) for f in families)
            if keep:
                out.append(c)
            continue
        if c.startswith("predicted_return") or c.startswith("predicted_return_"):
            if include_return:
                out.append(c)
            continue
        # otherwise keep (technical, sector, shared residuals)
        out.append(c)
    return out


def run_experiment(cfg, panel, feature_cols, families, include_return):
    data = panel.dropna(subset=["label"]).copy()
    # numeric features and leakage-free fill
    X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.groupby(data["target"]).ffill().fillna(0.0)
    data = data.assign(**{c: X[c] for c in feature_cols})
    # filter to training period (train-only per request)
    data = data[data["date"] < DATE_CUTOFF].copy()

    feats = select_features(feature_cols, families, include_return)
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
    test = test[test["date"] < DATE_CUTOFF]
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
        trades_hit = trades[trades["target"].isin(hits["target"])]
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
            "raw": sum(1 for c in feats if c.startswith("raw_")),
            "percent": sum(1 for c in feats if c.startswith("percent_")),
            "log": sum(1 for c in feats if c.startswith("log_")),
            "return": sum(1 for c in feats if c.startswith("predicted_return")),
        },
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
    import dataclasses
    cfg = dataclasses.replace(StrategyConfig(), end="2024-03-31", enable_multi_residual_engine=True)

    sp = StrategyPipeline(cfg)
    md = sp.load_data()
    from strategy.splits import chrono_split, walk_forward_folds
    split = chrono_split(md.prices.index, cfg)
    folds = walk_forward_folds(md.prices.index, cfg)
    panel = sp.build_panel(md, folds, split)

    # base feature columns from pipeline logic
    excluded = {"date", "etf", "sector", "target", "predictors", "target_price",
                "shadow_price", "next_ret", "label", "spread_signal",
                "ann_vol", "residual_z", "price_residual", "residual_ewm_mean",
                "residual_ewm_std", "residual_roll_mean", "residual_roll_std"}
    feature_cols = [c for c in panel.columns if c not in excluded]

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
        # prioritize residual + return features variant
        res = run_experiment(cfg, panel, feature_cols, fams, include_return=True)
        results[f"{name}_with_return"] = res
        # try without return features if quick (best-effort)
        try:
            res2 = run_experiment(cfg, panel, feature_cols, fams, include_return=False)
            results[f"{name}_no_return"] = res2
        except Exception:
            results[f"{name}_no_return"] = None

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
