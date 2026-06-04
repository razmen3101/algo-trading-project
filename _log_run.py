"""Log a single run's metrics to outputs/tuning_cache/run_history.csv.

Use this to record ad-hoc DBTS notebook runs alongside the validation sweep,
so all attempts live in one CSV for comparison.

USAGE:
  Edit the RUN dict below, then:
      python _log_run.py
"""
from datetime import datetime
from pathlib import Path
import pandas as pd

CACHE = Path("outputs/tuning_cache")
CACHE.mkdir(parents=True, exist_ok=True)
RUN_HISTORY = CACHE / "run_history.csv"

# ============================================================
# Edit this dict with the run you want to log:
# ============================================================
RUN = {
    "source": "DBTS_Train_Only",   # or "DBTS_Test", "main_strategy", etc.
    "tag":    "baseline_rz2.0_c0.70",
    "clf_trial":  None,
    "f1_val":     None,
    "rz_thr":     2.0,
    "conf":       0.70,
    "mr_exit":    0.30,
    "total_entries":         306,
    "sharpe":                1.0561,
    "sortino":               1.6173,
    "cumulative_return":     0.122,
    "annualized_return":     0.0528,
    "annualized_volatility": 0.05,
    "max_drawdown":         -0.0563,
    "win_rate_days":         0.3989,
    "active_days":           790,
}
# ============================================================

COLS = ["timestamp","source","tag","clf_trial","f1_val","rz_thr","conf","mr_exit",
        "total_entries","sharpe","sortino","cumulative_return",
        "annualized_return","annualized_volatility","max_drawdown",
        "win_rate_days","active_days"]

row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **RUN}
new_df = pd.DataFrame([{c: row.get(c) for c in COLS}])

if RUN_HISTORY.exists():
    prev = pd.read_csv(RUN_HISTORY)
    combined = pd.concat([prev, new_df], ignore_index=True)
else:
    combined = new_df

combined.to_csv(RUN_HISTORY, index=False)
print(f"Logged: {RUN['tag']}")
print(f"Total runs in history: {len(combined)}")
print(f"File: {RUN_HISTORY}")
