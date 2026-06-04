"""Run full notebook fresh kernel, save executed copy, print final metrics."""
import sys, time
from pathlib import Path
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import nbformat
from nbclient import NotebookClient

P = Path("DBTS_Train_Only_Diagnostic_FIXED.ipynb")
nb = nbformat.read(P, as_version=4)

# Append a verification cell at the end
verify_src = r"""
import pandas as pd, numpy as np
print("=" * 60)
print("FINAL METRICS")
print("=" * 60)
try:
    print(metrics.to_string())
except Exception:
    pass
print()
print(f"selected_panel rows : {len(selected_panel):,}")
print(f"trades rows         : {len(trades):,}")
print(f"completed_trades    : {len(completed_trades):,}")
print(f"bandit updates      : {bandit_updates_applied}")
# Switch / hold stats
sp = selected_panel.copy()
sp["date"] = pd.to_datetime(sp["date"])
sw = int(sp.groupby("sector")["target"].apply(lambda s: (s.shift(1).fillna(s.iloc[0]) != s).sum()).sum())
print(f"target switches     : {sw}")
print(f"avg entries/sector  : {len(completed_trades)/sp['sector'].nunique():.1f}")
print("PASS_BLOCK")
"""
nb.cells.append(nbformat.v4.new_code_cell(verify_src))

t0 = time.time()
client = NotebookClient(nb, timeout=3600, kernel_name="python3", allow_errors=False)
print(f"Executing {len(nb.cells)} cells (fresh kernel)...")
try:
    client.execute()
except Exception as e:
    print(f"EXEC_FAILED: {type(e).__name__}: {e}")
    # Print first error cell
    for i, c in enumerate(nb.cells):
        for out in c.get("outputs", []):
            if out.get("output_type") == "error":
                src_head = "".join(c.get("source", []))[:120].replace("\n"," | ")
                print(f"--- Cell [{i}] FAILED: {src_head}")
                import re
                tb = "\n".join(re.sub(r"\x1b\[[0-9;]*m", "", l) for l in out.get("traceback", []))
                print(tb[-3000:])
                sys.exit(1)
    sys.exit(1)
print(f"\n=========== EXEC OK ({time.time()-t0:.1f}s) ===========")
nbformat.write(nb, "DBTS_Train_Only_Diagnostic_FIXED.executed.ipynb")
# Print last cell output
for out in nb.cells[-1].get("outputs", []):
    if "text" in out: print(out["text"])
    elif "data" in out and "text/plain" in out["data"]: print(out["data"]["text/plain"])
