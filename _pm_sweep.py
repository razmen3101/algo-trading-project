"""
Fast PM gate sweep using cached selected_panel.

For each (mean_reversion_exit, entry_residual_threshold) combo, re-simulate
PM stepping over the existing selected_panel (with the one-position-per-sector
+ target stickiness logic as in patched Cell 8), and report metrics.

Selected target sequence is held fixed from the previous full run — this is a
fast screen, not a fully end-to-end simulation (bandit feedback is frozen).
"""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from strategy.position_manager import PositionManager, PositionState

CACHE = Path("outputs/train_only_dbts_cache")
PM_COST = 5.0 / 1e4
ENTRY_ACTIONS = {"ENTER_LONG", "ENTER_SHORT", "FLIP_LONG_TO_SHORT", "FLIP_SHORT_TO_LONG"}
EXIT_ACTIONS  = {"EXIT",       "FLIP_LONG_TO_SHORT", "FLIP_SHORT_TO_LONG"}

# Stickiness from patched Cell 8
MIN_TARGET_HOLD_DAYS = 5
TARGET_SWITCH_MARGIN = 0.05  # not used here (selected target is fixed)

print("Loading cached selected_panel...")
sp = pd.read_pickle(CACHE / "selected_panel.pkl")
print(f"  rows={len(sp):,}  sectors={sp['sector'].nunique()}  "
      f"dates={sp['date'].nunique()}")

# Ensure sorted by date then sector for deterministic stepping
sp = sp.sort_values(["date", "sector"]).reset_index(drop=True)
dates = sorted(sp["date"].unique())


def simulate(mre, rz_thr, conf=0.70, flat_block=0.25, max_hold=10):
    pm = PositionManager(
        long_entry_confidence=conf,
        short_entry_confidence=conf,
        flat_probability_block=flat_block,
        entry_residual_threshold=rz_thr,
        mean_reversion_exit=mre,
        opposite_signal_confidence=0.70,
        stop_loss=-0.02,
        take_profit=0.03,
        max_holding_days=max_hold,
        allow_flip=False,
    )

    # per-sector state (one position per sector)
    state_by_sec = {}
    prev_pos_by_sec = {}
    trade_seq_by_sec = {}
    cur_target_by_sec = {}
    hold_days_by_sec = {}

    rows = []  # one row per (date, sector) PM step

    for date, day_df in sp.groupby("date", sort=True):
        for _, r in day_df.iterrows():
            sec = r["sector"]
            selected = r["target"]
            prev_target = cur_target_by_sec.get(sec)
            prev_pos = prev_pos_by_sec.get(sec, 0)

            # Forced exit on target switch with open position
            if prev_target is not None and prev_target != selected and prev_pos != 0:
                # Need prev_target's next_ret on this date. We only have the SELECTED
                # target's next_ret in selected_panel. As a screen approximation, use
                # 0.0 for the forced-exit next_ret (cost-only). This matches the spirit
                # of "we couldn't carry the position" without overstating PnL.
                gross = 0.0
                turn = abs(0 - prev_pos)
                net = gross - turn * PM_COST
                st = state_by_sec.get(sec, PositionState())
                realised = float(st.trade_pnl) + float(net)
                rows.append({
                    "date": date, "sector": sec, "action": "EXIT",
                    "position": 0.0, "prev_pos": float(prev_pos),
                    "net_pnl": net, "gross_pnl": gross,
                    "is_entry": False, "is_exit": True,
                    "trade_pnl": realised, "closed": True,
                })
                state_by_sec[sec] = PositionState()
                prev_pos_by_sec[sec] = 0
                prev_pos = 0

            cur_target_by_sec[sec] = selected
            state = state_by_sec.get(sec) or PositionState()
            next_trade_id = trade_seq_by_sec.get(sec, 0)

            nr = float(r["next_ret"]) if pd.notna(r["next_ret"]) else 0.0
            pm_row = pd.Series({
                "date": date, "target": selected, "sector": sec,
                "signal": int(r["signal"]),
                "P_short": float(r["P_short"]), "P_flat": float(r["P_flat"]),
                "P_long": float(r["P_long"]),
                "residual_z": float(r["residual_z"]) if pd.notna(r["residual_z"]) else 0.0,
                "next_ret": nr,
                "target_price": r["target_price"],
            })
            d = pm.decide(pm_row, state)
            pos = d.position
            turn = abs(pos - prev_pos)
            gross = pos * nr
            net = gross - turn * PM_COST
            is_entry = d.action in ENTRY_ACTIONS
            is_exit  = d.action in EXIT_ACTIONS

            realised = None
            if is_exit and state.trade_id is not None:
                if is_entry:
                    realised = float(state.trade_pnl)
                else:
                    realised = float(state.trade_pnl) + float(net)

            rows.append({
                "date": date, "sector": sec, "action": d.action,
                "position": float(pos), "prev_pos": float(prev_pos),
                "net_pnl": float(net), "gross_pnl": float(gross),
                "is_entry": bool(is_entry), "is_exit": bool(is_exit),
                "trade_pnl": realised if realised is not None else 0.0,
                "closed": realised is not None,
            })

            if is_entry:
                next_trade_id += 1
                state_by_sec[sec] = PositionState(
                    current_position=int(pos),
                    days_in_position=1,
                    entry_residual_z=pm_row["residual_z"],
                    entry_confidence=r["P_long"] if pos == 1 else r["P_short"],
                    trade_pnl=net,
                    trade_id=next_trade_id,
                )
            elif pos == 0:
                state_by_sec[sec] = PositionState()
            elif pos == prev_pos and prev_pos != 0:
                state.current_position = pos
                state.days_in_position += 1
                state.trade_pnl += net
                state_by_sec[sec] = state

            prev_pos_by_sec[sec] = pos
            trade_seq_by_sec[sec] = next_trade_id

    trades = pd.DataFrame(rows)
    if trades.empty:
        return None

    # Per-day aggregated PnL across sectors
    daily = trades.groupby("date")["net_pnl"].sum().sort_index()
    entries = int(trades["is_entry"].sum())
    if entries == 0:
        return dict(entries=0, sharpe=np.nan, sortino=np.nan,
                    cumret=0.0, ann_ret=0.0, ann_vol=0.0, max_dd=0.0,
                    win_rate_days=0.0, completed=0, target_switches=0)

    cumret = float(daily.sum())
    n_days = len(daily)
    ann_ret = cumret / n_days * 252.0
    ann_vol = float(daily.std(ddof=0)) * math.sqrt(252.0)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    downside = daily[daily < 0]
    dn_vol = float(downside.std(ddof=0)) * math.sqrt(252.0) if len(downside) > 1 else 0.0
    sortino = ann_ret / dn_vol if dn_vol > 0 else 0.0

    equity = daily.cumsum()
    peak = equity.cummax()
    dd = (equity - peak)
    max_dd = float(dd.min())

    win_rate_days = float((daily > 0).sum() / (daily != 0).sum()) if (daily != 0).any() else 0.0
    completed = int(trades["closed"].sum())

    # Target switches: per sector, count distinct target changes
    sw = (sp.sort_values(["sector", "date"])
            .groupby("sector")["target"]
            .apply(lambda s: (s.shift(1).fillna(s.iloc[0]) != s).sum()).sum())

    return dict(
        entries=entries,
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        cumret=round(cumret, 4),
        ann_ret=round(ann_ret, 4),
        ann_vol=round(ann_vol, 4),
        max_dd=round(max_dd, 4),
        win_rate_days=round(win_rate_days, 3),
        completed=completed,
        target_switches=int(sw),
    )


MRE_GRID = [0.50, 0.40, 0.35, 0.30, 0.25, 0.20]
RZ_GRID  = [1.5, 1.4, 1.3, 1.2, 1.1, 1.0]

results = []
combos = list(itertools.product(MRE_GRID, RZ_GRID))
print(f"\nSweeping {len(combos)} combos (conf=0.70, flat_block=0.25, max_hold=10, no flip)...")
for i, (mre, rz) in enumerate(combos, 1):
    m = simulate(mre, rz)
    row = {"mean_reversion_exit": mre, "entry_residual_threshold": rz, **m}
    results.append(row)
    print(f"  [{i:2}/{len(combos)}] mre={mre:.2f} rz={rz:.1f} | "
          f"entries={m['entries']:4d}  sharpe={m['sharpe']:+.2f}  "
          f"sortino={m['sortino']:+.2f}  cumret={m['cumret']:+.3f}  "
          f"maxdd={m['max_dd']:+.3f}")

df = pd.DataFrame(results)
df.to_csv("_pm_sweep_results.csv", index=False)
print("\nSaved _pm_sweep_results.csv")

# Print pivot tables
print("\n=== SHARPE pivot (rows=mre, cols=rz) ===")
piv = df.pivot(index="mean_reversion_exit", columns="entry_residual_threshold", values="sharpe")
print(piv.to_string())

print("\n=== ENTRIES pivot ===")
print(df.pivot(index="mean_reversion_exit", columns="entry_residual_threshold", values="entries").to_string())

print("\n=== CUMRET pivot ===")
print(df.pivot(index="mean_reversion_exit", columns="entry_residual_threshold", values="cumret").to_string())

# Top 5 by sharpe
print("\n=== TOP 5 by Sharpe ===")
top = df.sort_values("sharpe", ascending=False).head(5)
print(top.to_string(index=False))
