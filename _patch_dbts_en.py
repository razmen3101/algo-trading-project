"""Patch DBTS_NoBandit_EN_Regime.ipynb:
 1. Add MIN_TARGET_HOLD + TARGET_SWITCH_MARGIN, use stickiness in winner selection.
 2. Force-recompute decision panel + add a low-conf gate sweep.
"""
import json, nbformat
from pathlib import Path

P = Path("DBTS_NoBandit_EN_Regime.ipynb")
nb = nbformat.read(P, as_version=4)

# locate cell containing DBTS_W
for i, c in enumerate(nb.cells):
    if c.cell_type != "code": continue
    if "DBTS_W = {" in c.source and "winner_ix = np.argmax" in c.source:
        old = c.source
        # add stickiness constants
        new = old.replace(
            "DBTS_W = {'residual': 0.50, 'pred_ret': 0.25, 'adf': 0.25}",
            "DBTS_W = {'residual': 0.50, 'pred_ret': 0.25, 'adf': 0.25}\n"
            "MIN_TARGET_HOLD = 5\n"
            "TARGET_SWITCH_MARGIN = 0.10"
        )
        # replace winner selection with sticky version
        old_winner = """        scores = np.where(valid, scores, -np.inf)
        winner_ix = np.argmax(scores, axis=1)
        winner    = np.array(cand_list)[winner_ix]
        winner_score = scores[np.arange(len(idx)), winner_ix]"""
        new_winner = """        scores = np.where(valid, scores, -np.inf)
        # Stickiness: keep current target unless held >= MIN_TARGET_HOLD AND
        # challenger exceeds current by TARGET_SWITCH_MARGIN.
        n_days, n_cand = scores.shape
        cand_arr = np.array(cand_list)
        winner = np.empty(n_days, dtype=object)
        winner_score = np.full(n_days, -np.inf)
        cur_ix = -1; cur_held = 0
        for ii in range(n_days):
            ds = scores[ii]
            if not np.any(np.isfinite(ds)):
                cur_ix = -1; cur_held = 0; continue
            if cur_ix == -1 or not np.isfinite(ds[cur_ix]):
                cur_ix = int(np.argmax(ds)); cur_held = 1
            else:
                best = int(np.argmax(ds))
                if (best != cur_ix and cur_held >= MIN_TARGET_HOLD
                    and ds[best] >= ds[cur_ix] + TARGET_SWITCH_MARGIN):
                    cur_ix = best; cur_held = 1
                else:
                    cur_held += 1
            winner[ii] = cand_arr[cur_ix]
            winner_score[ii] = ds[cur_ix]"""
        assert old_winner in new, "could not find winner block"
        new = new.replace(old_winner, new_winner)
        c.source = new
        print(f"patched cell {i}: stickiness + constants")
        break
else:
    raise SystemExit("DBTS_W cell not found")

# Find cache HIT branch in same cell to force recompute regardless
for i, c in enumerate(nb.cells):
    if c.cell_type == "code" and "PANEL_CACHE.exists()" in c.source and "decision_panel" in c.source:
        c.source = c.source.replace(
            "if PANEL_CACHE.exists() and not FORCE_RECOMPUTE:",
            "if PANEL_CACHE.exists() and not FORCE_RECOMPUTE and False:  # force rebuild after stickiness patch",
        )
        print(f"patched cell {i}: force panel rebuild")
        break

# Cell 11 — replace gates with full sweep + best selection
for i, c in enumerate(nb.cells):
    if c.cell_type == "code" and "PRIMARY_GATES" in c.source and "splits = {" in c.source:
        c.source = (
            "# Cell 11 — Gate sweep on TRAIN and VAL\n"
            "GRID = list(__import__('itertools').product(\n"
            "    [1.2, 1.5, 1.8, 2.0],     # rz_thr\n"
            "    [0.35, 0.40, 0.45, 0.55], # conf_thr\n"
            "    [0.30, 0.50],             # mr_exit\n"
            "))\n"
            "splits = {\n"
            "    'TRAIN': decision_panel[decision_panel['date'].isin(train_idx)].copy(),\n"
            "    'VAL':   decision_panel[decision_panel['date'].isin(val_idx)].copy(),\n"
            "}\n"
            "all_results = []\n"
            "for split_name, pan in splits.items():\n"
            "    print(f'--- {split_name} ---')\n"
            "    for rz, c_, mre in GRID:\n"
            "        td = regime_pm_simulate(pan, rz_thr=rz, conf_thr=c_, mr_exit=mre)\n"
            "        m = portfolio_metrics(td)\n"
            "        all_results.append({\n"
            "            'split': split_name, 'rz_thr': rz, 'conf_thr': c_, 'mr_exit': mre,\n"
            "            **m.to_dict()\n"
            "        })\n"
            "summary = pd.DataFrame(all_results)\n"
            "summary_view = summary[['split','rz_thr','conf_thr','mr_exit','total_entries',\n"
            "                        'long_entries','short_entries','sharpe','sortino',\n"
            "                        'cumulative_return','max_drawdown']]\n"
            "print('\\n=== TRAIN top 5 by Sharpe ===')\n"
            "print(summary_view[summary_view['split']=='TRAIN'].sort_values('sharpe', ascending=False).head(5).to_string(index=False))\n"
            "print('\\n=== VAL top 5 by Sharpe ===')\n"
            "print(summary_view[summary_view['split']=='VAL'].sort_values('sharpe', ascending=False).head(5).to_string(index=False))\n"
            "print('\\n=== VAL baseline-config (rz=1.5,conf=0.45,mre=0.5) ===')\n"
            "row = summary_view[(summary_view['split']=='VAL') & (summary_view['rz_thr']==1.5) &\n"
            "                   (summary_view['conf_thr']==0.45) & (summary_view['mr_exit']==0.5)]\n"
            "print(row.to_string(index=False) if not row.empty else 'no match')\n"
            "summary.to_csv(CACHE_DIR / 'gate_sweep.csv', index=False)\n"
        )
        print(f"patched cell {i}: gate sweep")
        break

# Cell 12 — log best TRAIN + best VAL only
for i, c in enumerate(nb.cells):
    if c.cell_type == "code" and "run_history.csv" in c.source and "log_rows" in c.source:
        c.source = (
            "# Cell 12 — Log best (TRAIN and VAL) to run_history\n"
            "from datetime import datetime as _dt\n"
            "RUN_HISTORY = Path('outputs/tuning_cache/run_history.csv')\n"
            "RUN_HISTORY.parent.mkdir(parents=True, exist_ok=True)\n"
            "HIST_COLS = ['timestamp','source','tag','clf_trial','f1_val','rz_thr','conf','mr_exit',\n"
            "             'total_entries','sharpe','sortino','cumulative_return',\n"
            "             'annualized_return','annualized_volatility','max_drawdown',\n"
            "             'win_rate_days','active_days']\n"
            "ts = _dt.now().strftime('%Y-%m-%d %H:%M:%S')\n"
            "rows = []\n"
            "for split_name in ['TRAIN','VAL']:\n"
            "    sub = summary[summary['split']==split_name].dropna(subset=['sharpe'])\n"
            "    if sub.empty: continue\n"
            "    best = sub.sort_values('sharpe', ascending=False).iloc[0]\n"
            "    rows.append({\n"
            "        'timestamp': ts, 'source': f'DBTS_NoBandit_EN_{split_name}',\n"
            "        'tag': f'sweep_best_rz{best.rz_thr}_conf{best.conf_thr}_mre{best.mr_exit}',\n"
            "        'clf_trial': None, 'f1_val': None,\n"
            "        'rz_thr': float(best.rz_thr), 'conf': float(best.conf_thr), 'mr_exit': float(best.mr_exit),\n"
            "        **{k: best.get(k) for k in HIST_COLS if k not in\n"
            "           ('timestamp','source','tag','clf_trial','f1_val','rz_thr','conf','mr_exit')},\n"
            "    })\n"
            "new_df = pd.DataFrame([{k: r.get(k) for k in HIST_COLS} for r in rows])\n"
            "if RUN_HISTORY.exists():\n"
            "    prev = pd.read_csv(RUN_HISTORY)\n"
            "    combined = pd.concat([prev, new_df], ignore_index=True)\n"
            "else:\n"
            "    combined = new_df\n"
            "combined.to_csv(RUN_HISTORY, index=False)\n"
            "print(f'Logged {len(new_df)} rows. Total: {len(combined)}')\n"
            "view = combined[combined['source'].str.contains('NoBandit') | combined['source'].str.startswith('Regime')][\n"
            "    ['source','tag','rz_thr','conf','mr_exit','total_entries','sharpe','sortino',\n"
            "     'cumulative_return','max_drawdown']\n"
            "].copy()\n"
            "print('\\nAll DBTS_NoBandit_EN + Regime_* runs:')\n"
            "print(view.to_string(index=False))\n"
        )
        print(f"patched cell {i}: log best")
        break

nbformat.write(nb, P)
print(f"\nSaved {P.name}")
