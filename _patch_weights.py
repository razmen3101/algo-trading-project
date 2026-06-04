"""Patch Cell 8: revert to USE_DBTS=True with bandit weight=0 (pre-fix behavior).
Selection driven by opportunity components (residual_z + pred_ret + adf).
"""
import json
from pathlib import Path

P = Path("DBTS_Train_Only_Diagnostic_FIXED.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))
cell = nb["cells"][16]
src = "".join(cell["source"])

# 1. Turn DBTS back on
old1 = "USE_DBTS = False"
new1 = "USE_DBTS = True"
assert old1 in src, "USE_DBTS=False not found"
src = src.replace(old1, new1)

# 2. Reweight DBTS: drop bandit to 0, redistribute to opportunity components
old2 = 'DBTS_WEIGHTS = {"bandit": 0.40, "residual": 0.25, "pred_ret": 0.20, "adf": 0.15}'
new2 = 'DBTS_WEIGHTS = {"bandit": 0.00, "residual": 0.45, "pred_ret": 0.35, "adf": 0.20}'
assert old2 in src, "DBTS_WEIGHTS not found"
src = src.replace(old2, new2)

cell["source"] = src.splitlines(keepends=True)
cell["outputs"] = []
cell["execution_count"] = None
P.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Patched: USE_DBTS=True, DBTS_WEIGHTS = bandit:0 / residual:0.45 / pred_ret:0.35 / adf:0.20")
