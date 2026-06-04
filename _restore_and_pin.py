"""Restore original DBTS_WEIGHTS and pin XGBoost determinism."""
import json
from pathlib import Path

P = Path("DBTS_Train_Only_Diagnostic_FIXED.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

# 1. Cell 8 (index 16): restore weights
cell8 = nb["cells"][16]
src8 = "".join(cell8["source"])
old = 'DBTS_WEIGHTS = {"bandit": 0.00, "residual": 0.45, "pred_ret": 0.35, "adf": 0.20}'
new = 'DBTS_WEIGHTS = {"bandit": 0.40, "residual": 0.25, "pred_ret": 0.20, "adf": 0.15}'
assert old in src8, "weights block not found"
src8 = src8.replace(old, new)
cell8["source"] = src8.splitlines(keepends=True)
cell8["outputs"] = []
cell8["execution_count"] = None

# 2. Cell 6 (index 6): pin XGBoost to deterministic
cell6 = nb["cells"][6]
src6 = "".join(cell6["source"])
old_xgb = '''_p = dict(cfg.clf_params)
_p.update({"num_class": 3, "objective": "multi:softprob", "verbosity": 0,
           "random_state": int(cfg.random_state)})'''
new_xgb = '''_p = dict(cfg.clf_params)
_p.update({"num_class": 3, "objective": "multi:softprob", "verbosity": 0,
           "random_state": int(cfg.random_state),
           "n_jobs": 1, "tree_method": "exact", "seed": int(cfg.random_state)})'''
assert old_xgb in src6, "XGB ctor not found"
src6 = src6.replace(old_xgb, new_xgb)
cell6["source"] = src6.splitlines(keepends=True)
cell6["outputs"] = []
cell6["execution_count"] = None

P.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Restored DBTS_WEIGHTS (bandit=0.40) and pinned XGBoost determinism")
