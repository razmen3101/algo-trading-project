"""Proportional buy-and-hold benchmark on TEST.

Idea: instead of the model's active timing, just HOLD each stock in proportion
to how much the model traded it on TEST ("bought a lot of X -> hold a lot of X").
Weights come from the model's signed position-days per stock; we then hold that
fixed-weight basket across the whole test window and measure the same metrics.

Two weightings are reported:
  - signed   : long/short book (long days +, short days -), matches the model
  - long-only: only the long exposure (the literal "stocks I bought"), held long

Buy-and-hold is modelled as a constant-weight basket (daily-rebalanced to keep
the proportions), GROSS of costs — there is no active turnover to charge.

    python bh_benchmark_test.py
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

import sweep_q_pm_val as S
from strategy.strategy_config import StrategyConfig
from strategy.splits import chrono_split


def metrics(daily, ppy=252):
    daily = np.asarray(daily, dtype=float)
    daily = daily[~np.isnan(daily)]
    n = len(daily)
    if n == 0:
        return dict(days=0, cum_ret=np.nan, ann_ret=np.nan, sharpe=np.nan,
                    sortino=np.nan, max_dd=np.nan, win_rate_days=np.nan)
    eq = np.cumprod(1 + daily)
    cum = float(eq[-1] - 1.0)
    ann_ret = float((1 + cum) ** (ppy / max(n, 1)) - 1.0)
    ann_vol = float(np.std(daily, ddof=1) * math.sqrt(ppy)) if n > 1 else np.nan
    sh = ann_ret / ann_vol if ann_vol and np.isfinite(ann_vol) and ann_vol != 0 else np.nan
    dn = daily[daily < 0]
    dnv = float(np.std(dn, ddof=1) * math.sqrt(ppy)) if len(dn) > 1 else np.nan
    so = ann_ret / dnv if dnv and np.isfinite(dnv) and dnv != 0 else np.nan
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    return dict(days=n, cum_ret=round(cum, 4), ann_ret=round(ann_ret, 4),
                sharpe=round(sh, 4) if np.isfinite(sh) else np.nan,
                sortino=round(so, 4) if np.isfinite(so) else np.nan,
                max_dd=round(mdd, 4),
                win_rate_days=round(float((daily > 0).mean()), 4))


def main():
    md, val_idx, pools, REG, QUAL = S.load_inputs()
    cfg = StrategyConfig(force_recompute=False, make_plots=False)
    sp = chrono_split(md.prices.index, cfg)
    test_idx = pd.DatetimeIndex(sp.test_idx).sort_values()

    lb_q = pd.read_csv(S.CACHE_DIR / "leaderboard_qweights_val.csv")
    lb_pm = pd.read_csv(S.CACHE_DIR / "leaderboard_pm_val.csv")
    bq, bpm = lb_q.iloc[0], lb_pm.iloc[0]
    w_mr, w_ic = float(bq["w_mr"]), float(bq["w_ic"])
    stop = -float(bpm["stop_pct"]) / 100.0
    take = float(bpm["take_pct"]) / 100.0
    print(f"\nBest VAL config -> Q(w_mr={w_mr}, w_ic={w_ic}, w_stab={round(1-w_mr-w_ic,2)}) "
          f"stop={stop:.1%} take={take:.1%}")

    # ---- model on TEST, with per-(date,sector) panel -----------------------
    eng = S.build_engine(md, test_idx, pools, REG, QUAL)
    m_model, panel = eng["run_combo"](w_mr, w_ic, stop=stop, take=take, return_panel=True)

    # ---- per-stock signed position-days (exposure) on TEST -----------------
    panel["key"] = list(zip(panel["sector"], panel["stock"]))
    expo = panel.groupby("key")["position"].sum()          # signed position-days
    expo = expo[expo != 0.0]
    if expo.empty:
        print("No exposure on test — nothing to hold.")
        return

    # ---- daily returns of every held stock across the test window ----------
    ret = pd.DataFrame(
        {k: REG[k]["next_ret"].reindex(test_idx) for k in expo.index}
    ).fillna(0.0)

    # ---- weightings --------------------------------------------------------
    w_signed = expo / expo.abs().sum()                     # long/short book, |w| sums to 1
    long_expo = expo.clip(lower=0.0)
    w_long = long_expo / long_expo.sum() if long_expo.sum() > 0 else long_expo

    bh_signed = ret.mul(w_signed, axis=1).sum(axis=1).values
    bh_long = ret.mul(w_long, axis=1).sum(axis=1).values

    m_signed = metrics(bh_signed)
    m_long = metrics(bh_long)

    # ---- report ------------------------------------------------------------
    cols = ["days", "cum_ret", "ann_ret", "sharpe", "sortino", "max_dd", "win_rate_days"]
    comp = pd.DataFrame({
        "MODEL (test)": {k: m_model.get(k) for k in cols},
        "B&H signed": {k: m_signed.get(k) for k in cols},
        "B&H long-only": {k: m_long.get(k) for k in cols},
    })
    for k in ("cum_ret", "ann_ret", "max_dd", "win_rate_days"):
        comp.loc[k] = comp.loc[k].apply(lambda v: f"{v:.2%}" if pd.notna(v) else v)
    print("\n=== TEST: active model vs proportional buy-and-hold ===")
    print(comp.to_string())

    print(f"\nHeld names: {len(expo)} | long sleeves: {(expo>0).sum()} | "
          f"short sleeves: {(expo<0).sum()}")
    top = w_signed.reindex(expo.abs().sort_values(ascending=False).index).head(10)
    print("\nTop-10 holdings by |weight| (signed):")
    for (sec, stk), w in top.items():
        print(f"  {stk:6s} ({sec:16s}) {w:+.3%}   pos-days={expo[(sec, stk)]:+.0f}")

    out = S.CACHE_DIR / "bh_benchmark_test.csv"
    pd.DataFrame([
        {"strategy": "model_test", **m_model},
        {"strategy": "bh_signed", **m_signed},
        {"strategy": "bh_long_only", **m_long},
    ]).to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
