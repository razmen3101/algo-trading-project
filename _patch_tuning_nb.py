"""Patch Validation_Tuning.ipynb:
  1. Add run_history.csv logging to PM sweep cell
  2. Add new cell at end that displays history table
"""
import json
from pathlib import Path

P = Path("Validation_Tuning.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

# --- Patch the sweep cell (find by content) ---
SWEEP_MARKER = "sweep_df.to_csv(CACHE_DIR / 'pm_sweep_val.csv', index=False)"
LOG_INSERT = """from datetime import datetime
RUN_HISTORY = CACHE_DIR / 'run_history.csv'
_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
_hist_cols = ['timestamp','source','tag','clf_trial','f1_val','rz_thr','conf','mr_exit',
              'total_entries','sharpe','sortino','cumulative_return',
              'annualized_return','annualized_volatility','max_drawdown',
              'win_rate_days','active_days']
_hist_rows = []
for _, r in sweep_df.iterrows():
    _hist_rows.append({
        'timestamp': _ts,
        'source': 'Validation_Tuning',
        'tag': f\"t{int(r['clf_trial'])}_rz{r['rz_thr']}_c{r['conf']}_mre{r['mr_exit']}\",
        **{k: r.get(k) for k in _hist_cols if k not in ('timestamp','source','tag')},
    })
_new_hist = pd.DataFrame(_hist_rows)[_hist_cols]
if RUN_HISTORY.exists():
    _prev = pd.read_csv(RUN_HISTORY)
    _combined = pd.concat([_prev, _new_hist], ignore_index=True)
else:
    _combined = _new_hist
_combined.to_csv(RUN_HISTORY, index=False)
print(f'Appended {len(_new_hist)} rows to {RUN_HISTORY.name} (total: {len(_combined)})')
"""

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if SWEEP_MARKER in src:
        # Insert the logging block right after the to_csv line
        old = "sweep_df.to_csv(CACHE_DIR / 'pm_sweep_val.csv', index=False)\nprint(f'\\nSaved sweep -> {CACHE_DIR/\"pm_sweep_val.csv\"}')"
        new = (old + "\n\n# --- Append to persistent run history ---\n"
               + LOG_INSERT)
        assert old in src, "to_csv block not found"
        src = src.replace(old, new)
        cell["source"] = src.splitlines(keepends=True)
        print("Patched sweep cell with run_history logging")
        break
else:
    raise SystemExit("sweep cell not found")

# --- Append new cell: display run history ---
history_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Cell 11 — Show full run history (all sweeps + external runs from _log_run.py)\n",
        "RUN_HISTORY = CACHE_DIR / 'run_history.csv'\n",
        "if RUN_HISTORY.exists():\n",
        "    hist = pd.read_csv(RUN_HISTORY)\n",
        "    print(f'Total runs logged: {len(hist)}')\n",
        "    print(f'Sources: {hist[\"source\"].value_counts().to_dict()}')\n",
        "    print('\\nTop 15 by Sharpe across ALL runs:')\n",
        "    display(hist.sort_values(\"sharpe\", ascending=False).head(15))\n",
        "    print('\\nMost recent 10 runs:')\n",
        "    display(hist.tail(10))\n",
        "else:\n",
        "    print('No history yet — run sweep cell first or use _log_run.py')\n",
    ],
}
nb["cells"].append(history_cell)
print(f"Appended history display cell (total cells: {len(nb['cells'])})")

P.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Saved Validation_Tuning.ipynb")
