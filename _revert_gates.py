"""Revert Cell 8 PM gates to baseline (conf=0.70, rz_thr=2.0)."""
import json
from pathlib import Path

P = Path("DBTS_Train_Only_Diagnostic_FIXED.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))
cell = nb["cells"][16]
src = "".join(cell["source"])

old = '''    print("\\nPM gate parameters: conf=0.89, flat_block=0.25, rz_thr=1.2, mr_exit=0.30, max_hold=10d, allow_flip=False")
    print("Target-selection guards: min_target_hold_days=5, target_switch_margin=0.05, one-position-per-sector")
    pm = PositionManager(
        long_entry_confidence=0.89,
        short_entry_confidence=0.89,
        flat_probability_block=0.25,
        entry_residual_threshold=1.2,
        mean_reversion_exit=0.30,'''
new = '''    print("\\nPM gate parameters: conf=0.70, flat_block=0.25, rz_thr=2.0, mr_exit=0.30, max_hold=10d, allow_flip=False")
    print("Target-selection guards: min_target_hold_days=5, target_switch_margin=0.05, one-position-per-sector")
    pm = PositionManager(
        long_entry_confidence=0.70,
        short_entry_confidence=0.70,
        flat_probability_block=0.25,
        entry_residual_threshold=2.0,
        mean_reversion_exit=0.30,'''
assert old in src, "not found"
src = src.replace(old, new)
cell["source"] = src.splitlines(keepends=True)
cell["outputs"] = []
cell["execution_count"] = None
P.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Reverted to baseline: conf=0.70, rz_thr=2.0")
