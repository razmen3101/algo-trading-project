"""Execute Regime_TEST.ipynb and print last cells' outputs."""
import sys, re, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

P = Path("Regime_TEST.ipynb")
nb = nbformat.read(P, as_version=4)

t0 = time.time()
print(f"Executing {len(nb.cells)} cells...")
client = NotebookClient(nb, timeout=1800, kernel_name="python3", allow_errors=False)
try:
    client.execute()
    print(f"OK ({time.time()-t0:.1f}s)")
except Exception as e:
    print(f"FAIL ({time.time()-t0:.1f}s): {e}")
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            for out in c.get("outputs", []):
                if out.get("output_type") == "error":
                    tb = "\n".join(out.get("traceback", []))
                    tb = re.sub(r"\x1b\[[0-9;]*m", "", tb)
                    print(f"\n--- error in cell {i} ---\n{tb}")
    nbformat.write(nb, "Regime_TEST.executed.ipynb")
    sys.exit(1)

# print stream outputs of all code cells
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    streams = [o.get("text","") for o in c.get("outputs",[]) if o.get("output_type")=="stream"]
    if streams:
        print(f"\n=== cell {i} ===")
        print("".join(streams))

nbformat.write(nb, "Regime_TEST.executed.ipynb")
print("\nWrote Regime_TEST.executed.ipynb")
