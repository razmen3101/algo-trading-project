"""Seed run_history.csv with the two DBTS runs we already executed,
so they appear alongside the upcoming validation sweep."""
from datetime import datetime
from pathlib import Path
import pandas as pd

CACHE = Path("outputs/tuning_cache")
CACHE.mkdir(parents=True, exist_ok=True)
RUN_HISTORY = CACHE / "run_history.csv"

COLS = ["timestamp","source","tag","clf_trial","f1_val","rz_thr","conf","mr_exit",
        "total_entries","sharpe","sortino","cumulative_return",
        "annualized_return","annualized_volatility","max_drawdown",
        "win_rate_days","active_days"]

ts_baseline = "2026-06-04 12:00:00"
ts_test2    = "2026-06-04 13:00:00"

rows = [
    {"timestamp": ts_baseline, "source": "DBTS_Train_Only",
     "tag": "baseline_rz2.0_c0.70_mre0.30",
     "clf_trial": None, "f1_val": None,
     "rz_thr": 2.0, "conf": 0.70, "mr_exit": 0.30,
     "total_entries": 306, "sharpe": 1.0561, "sortino": 1.6173,
     "cumulative_return": 0.122, "annualized_return": 0.0528,
     "annualized_volatility": 0.05, "max_drawdown": -0.0563,
     "win_rate_days": 0.3989, "active_days": 790},
    {"timestamp": ts_test2, "source": "DBTS_Train_Only",
     "tag": "rz1.2_c0.89_mre0.30",
     "clf_trial": None, "f1_val": None,
     "rz_thr": 1.2, "conf": 0.89, "mr_exit": 0.30,
     "total_entries": 710, "sharpe": 0.0597, "sortino": 0.0843,
     "cumulative_return": 0.0093, "annualized_return": 0.0041,
     "annualized_volatility": 0.0694, "max_drawdown": -0.090,
     "win_rate_days": 0.4929, "active_days": 2016},
]
df = pd.DataFrame(rows)[COLS]

if RUN_HISTORY.exists():
    prev = pd.read_csv(RUN_HISTORY)
    # avoid duplicate seed if rerun
    mask = prev["tag"].isin(df["tag"]) & (prev["source"] == "DBTS_Train_Only")
    prev = prev[~mask]
    combined = pd.concat([prev, df], ignore_index=True)
else:
    combined = df

combined.to_csv(RUN_HISTORY, index=False)
print(f"Seeded {len(df)} DBTS runs. Total in history: {len(combined)}")
print(combined.to_string(index=False))
