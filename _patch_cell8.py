"""Apply user-requested fixes to Cell 8 of DBTS_Train_Only_Diagnostic_FIXED.ipynb:
  1. Tighter PM gates (conf=0.70, flat_block=0.25, rz_thr=2.0, mr_exit=0.30,
     max_hold=10, allow_flip=False).
  2. Target stickiness: min_target_hold_days=5, target_switch_margin=0.05.
  3. ONE position per sector: state keyed by sector_name only. When the
     selected target switches while a position is open, force EXIT on the old
     target first (using its same-day next_ret), update bandit with realised
     pnl, then continue with the new target.
"""
import json, copy
from pathlib import Path

P = Path("DBTS_Train_Only_Diagnostic_FIXED.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))
cell = nb["cells"][16]
src = "".join(cell["source"])

# ── (1) PM ctor + header ────────────────────────────────────────────────────
old_pm_block = '''    print("\\nPM gate parameters: conf=0.45, residual_threshold=1.0, mr_exit=0.50, max_hold=15d")
    pm = PositionManager(
        long_entry_confidence=0.45,
        short_entry_confidence=0.45,
        flat_probability_block=0.50,
        entry_residual_threshold=1.0,
        mean_reversion_exit=0.50,
        opposite_signal_confidence=float(getattr(cfg, "pm_opposite_confidence", 0.70)),
        stop_loss=float(getattr(cfg, "pm_stop_loss", -0.02)),
        take_profit=float(getattr(cfg, "pm_take_profit", 0.03)),
        max_holding_days=15,
        allow_flip=bool(getattr(cfg, "pm_allow_flip", True)),
    )'''
new_pm_block = '''    print("\\nPM gate parameters: conf=0.70, flat_block=0.25, rz_thr=2.0, mr_exit=0.30, max_hold=10d, allow_flip=False")
    print("Target-selection guards: min_target_hold_days=5, target_switch_margin=0.05, one-position-per-sector")
    pm = PositionManager(
        long_entry_confidence=0.70,
        short_entry_confidence=0.70,
        flat_probability_block=0.25,
        entry_residual_threshold=2.0,
        mean_reversion_exit=0.30,
        opposite_signal_confidence=float(getattr(cfg, "pm_opposite_confidence", 0.70)),
        stop_loss=float(getattr(cfg, "pm_stop_loss", -0.02)),
        take_profit=float(getattr(cfg, "pm_take_profit", 0.03)),
        max_holding_days=10,
        allow_flip=False,
    )
    # Target-selection guards
    MIN_TARGET_HOLD_DAYS = 5
    TARGET_SWITCH_MARGIN = 0.05
    target_hold_days  = {}   # sector -> int days held current target
    pm_current_target = {}   # sector -> str (target with open position, or None)'''
assert old_pm_block in src, "PM ctor block not found"
src = src.replace(old_pm_block, new_pm_block)

# ── (2) Selection + stickiness ──────────────────────────────────────────────
old_select = '''            finite   = {k: v for k, v in scores.items() if np.isfinite(v)}
            selected = max(finite, key=finite.get) if finite else members[0]
            target_switched = selected != last_selected_by_sector.get(sector_name, selected)
            last_selected_by_sector[sector_name] = selected'''
new_select = '''            finite = {k: v for k, v in scores.items() if np.isfinite(v)}
            top    = max(finite, key=finite.get) if finite else members[0]
            current_sel = last_selected_by_sector.get(sector_name)
            held_days   = target_hold_days.get(sector_name, 0)
            if current_sel is None or current_sel == top:
                selected = top
            elif held_days < MIN_TARGET_HOLD_DAYS:
                selected = current_sel
            elif finite.get(top, -np.inf) < finite.get(current_sel, -np.inf) + TARGET_SWITCH_MARGIN:
                selected = current_sel
            else:
                selected = top
            target_switched = (current_sel is not None) and (selected != current_sel)
            target_hold_days[sector_name] = 1 if target_switched else (held_days + 1)
            last_selected_by_sector[sector_name] = selected'''
assert old_select in src, "selection block not found"
src = src.replace(old_select, new_select)

# ── (3) Per-sector PM state with forced exit on target switch ───────────────
old_pm_step_head = '''            # ── ONLINE PM STEP for (sector_name, selected) ───────────────
            sk = (sector_name, selected)
            state = pm_state.get(sk) or PositionState()
            prev_position = pm_prev_pos.get(sk, 0)
            next_trade_id = pm_trade_seq.get(sk, 0)'''
new_pm_step_head = '''            # ── ONLINE PM STEP — one position per SECTOR ────────────────
            sk = sector_name
            prev_target_held = pm_current_target.get(sk)
            prev_position    = pm_prev_pos.get(sk, 0)
            next_trade_id    = pm_trade_seq.get(sk, 0)

            # FORCED EXIT if the selected target changed while a position is open
            if (prev_target_held is not None and prev_target_held != selected
                    and prev_position != 0):
                old_state = pm_state.get(sk) or PositionState()
                closed_id_force = old_state.trade_id
                old_key = (date, prev_target_held)
                old_nr = 0.0
                if old_key in panel_indexed.index:
                    old_row = panel_indexed.loc[old_key]
                    nr_val = (old_row.get("next_ret", np.nan)
                              if hasattr(old_row, "get")
                              else (old_row["next_ret"] if "next_ret" in old_row.index else np.nan))
                    old_nr = float(nr_val) if pd.notna(nr_val) else 0.0
                gross_fx = float(prev_position) * old_nr
                turn_fx  = float(abs(prev_position))
                net_fx   = gross_fx - turn_fx * PM_COST
                realised_force = float(old_state.trade_pnl) + net_fx
                pm_rows.append({
                    "date": date, "target": prev_target_held, "sector": sector_name,
                    "signal": 0, "P_short": np.nan, "P_flat": np.nan, "P_long": np.nan,
                    "residual_z": np.nan, "next_ret": old_nr,
                    "target_price": safe_price(md.prices, prev_target_held, date),
                    "action": "EXIT", "action_reason": "target_switch_forced_exit",
                    "position": 0.0, "turnover": turn_fx,
                    "gross_pnl": gross_fx, "net_pnl": net_fx,
                    "prev_pos": float(prev_position),
                    "trade_id": int(closed_id_force) if closed_id_force is not None else None,
                    "entry_id": int(closed_id_force) if closed_id_force is not None else None,
                    "closed_trade_id": int(closed_id_force) if closed_id_force is not None else None,
                    "is_entry": False, "is_exit": True,
                    "days_in_position": 0,
                    "trade_pnl": realised_force,
                    "entry_residual_z": (float(old_state.entry_residual_z)
                                         if old_state.entry_residual_z is not None
                                         and np.isfinite(old_state.entry_residual_z) else np.nan),
                    "entry_confidence": (float(old_state.entry_confidence)
                                         if old_state.entry_confidence is not None
                                         and np.isfinite(old_state.entry_confidence) else np.nan),
                })
                if closed_id_force is not None and np.isfinite(realised_force):
                    bandit.update(sector_name, prev_target_held, realised_force)
                    bandit_updates_applied += 1
                pm_state[sk] = PositionState()
                pm_prev_pos[sk] = 0
                prev_position = 0

            state = pm_state.get(sk) or PositionState()'''
assert old_pm_step_head in src, "PM step head not found"
src = src.replace(old_pm_step_head, new_pm_step_head)

# ── Track per-sector current target after PM step ───────────────────────────
old_tail = '''            pm_prev_pos[sk]  = position
            pm_trade_seq[sk] = next_trade_id'''
new_tail = '''            pm_prev_pos[sk]  = position
            pm_trade_seq[sk] = next_trade_id
            pm_current_target[sk] = selected if position != 0 else None'''
assert old_tail in src, "tail not found"
src = src.replace(old_tail, new_tail)

cell["source"] = src.splitlines(keepends=True)
# Clear stale outputs (so re-execution regenerates them cleanly)
cell["outputs"] = []
cell["execution_count"] = None

P.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Cell 8 patched: tighter PM, target stickiness, one-position-per-sector with forced exit.")
