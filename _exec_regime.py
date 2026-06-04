"""Execute MeanRev_vs_Momentum.ipynb fresh kernel, force-recompute all caches.
Print final metrics on stdout.
"""
import sys, re
from pathlib import Path
import nbformat
from nbclient import NotebookClient

P = Path("MeanRev_vs_Momentum.ipynb")
nb = nbformat.read(P, as_version=4)

# Inject FORCE_RECOMPUTE = True override at start of cell 2
for cell in nb.cells:
    if cell.cell_type == "code" and "FORCE_RECOMPUTE = False" in cell.source:
        cell.source = cell.source.replace("FORCE_RECOMPUTE = False",
                                          "FORCE_RECOMPUTE = True")
        break

# Append verification cell
verify = nbformat.v4.new_code_cell(source="""
import numpy as np
print("=" * 60)
print("FINAL METRICS")
print("=" * 60)
print(metrics.to_string())
print(f"\\n  long_entries / short_entries : "
      f"{int(metrics.get('long_entries',0))} / {int(metrics.get('short_entries',0))}")
print(f"  total entries                : {int(metrics.get('total_entries',0))}")
print(f"  decision_panel rows          : {len(decision_panel):,}")
print(f"  rz stats                     : "
      f"mean={decision_panel['residual_z'].mean():+.3f} "
      f"std={decision_panel['residual_z'].std():+.3f} "
      f"|rz|>1.5 frac={(decision_panel['residual_z'].abs()>1.5).mean():.3f}")
print("PASS_BLOCK")
""")
nb.cells.append(verify)

import time
t0 = time.time()
print(f"Executing {len(nb.cells)} cells (fresh kernel)...")
client = NotebookClient(nb, timeout=3600, kernel_name="python3", allow_errors=False)
try:
    client.execute()
    elapsed = time.time() - t0
    print(f"\n=========== EXEC OK ({elapsed:.1f}s) ===========")
except Exception as e:
    print(f"\n!!! EXEC FAILED after {time.time()-t0:.1f}s !!!")
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            for out in c.get("outputs", []):
                if out.get("output_type") == "error":
                    tb = "\n".join(out.get("traceback", []))
                    tb = re.sub(r"\x1b\[[0-9;]*m", "", tb)
                    print(f"\n--- error in cell {i} ---\n{tb}")
                    break
    sys.exit(1)

# Print verification block output
for c in nb.cells[-3:]:
    if c.cell_type == "code":
        for out in c.get("outputs", []):
            if out.get("output_type") == "stream":
                print(out.get("text", ""))

nbformat.write(nb, "MeanRev_vs_Momentum.executed.ipynb")
print("Wrote MeanRev_vs_Momentum.executed.ipynb")
