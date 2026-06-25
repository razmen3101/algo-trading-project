"""Take the best VALIDATION config (from the sweep CSVs) and evaluate it ONCE
on the TEST set — no tuning on test. Prints VAL vs TEST metrics side by side.

    python eval_on_test.py
"""
from __future__ import annotations

import pandas as pd

import sweep_q_pm_val as S
from strategy.strategy_config import StrategyConfig
from strategy.splits import chrono_split


def main():
    md, val_idx, pools, REG, QUAL = S.load_inputs()
    cfg = StrategyConfig(force_recompute=False, make_plots=False)
    sp = chrono_split(md.prices.index, cfg)
    test_idx = pd.DatetimeIndex(sp.test_idx).sort_values()

    # ---- best config straight from the validation leaderboards -------------
    lb_q = pd.read_csv(S.CACHE_DIR / "leaderboard_qweights_val.csv")
    lb_pm = pd.read_csv(S.CACHE_DIR / "leaderboard_pm_val.csv")
    bq, bpm = lb_q.iloc[0], lb_pm.iloc[0]
    w_mr, w_ic = float(bq["w_mr"]), float(bq["w_ic"])
    w_stab = round(1.0 - w_mr - w_ic, 2)
    stop = -float(bpm["stop_pct"]) / 100.0
    take = float(bpm["take_pct"]) / 100.0

    print("\nBest VALIDATION config (greedy: Q first, then PM):")
    print(f"  score={S.SCORE_FN}  universe={S.UNIVERSE}  stickiness={S.STICK}")
    print(f"  Q weights: w_mr_hit={w_mr}  w_pred_ic={w_ic}  w_stab={w_stab}")
    print(f"  stop_loss={stop:.1%}   take_profit={take:.1%}")

    # ---- evaluate the SAME config on VAL and on TEST -----------------------
    eng_val = S.build_engine(md, val_idx, pools, REG, QUAL)
    eng_test = S.build_engine(md, test_idx, pools, REG, QUAL)
    m_val = eng_val["run_combo"](w_mr, w_ic, stop=stop, take=take)
    m_test = eng_test["run_combo"](w_mr, w_ic, stop=stop, take=take)

    cols = ["days", "entries", "long", "short", "cum_ret", "ann_ret",
            "sharpe", "sortino", "max_dd", "win_rate_days"]
    comp = pd.DataFrame({"VAL": {k: m_val.get(k) for k in cols},
                         "TEST": {k: m_test.get(k) for k in cols}})
    # human-friendly percent formatting for the ratio rows
    for k in ("cum_ret", "ann_ret", "max_dd", "win_rate_days"):
        comp.loc[k] = comp.loc[k].apply(lambda v: f"{v:.2%}" if pd.notna(v) else v)

    print("\n=== Same config: VALIDATION vs TEST ===")
    print(comp.to_string())

    out = S.CACHE_DIR / "best_val_config_on_test.csv"
    pd.DataFrame([
        {"split": "VAL", **m_val},
        {"split": "TEST", **m_test},
    ]).to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
