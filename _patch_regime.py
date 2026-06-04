"""Patch MeanRev_vs_Momentum.ipynb with 4 fixes:
  1. Contemporaneous price regressor (y = price[t], not shift(-1))
  2. DECAY_ALPHA = 0.995
  3. REGIME_CONF_THR = 0.55
  4. MOMENTUM logic: direction follows residual_z (rz>0 → LONG)
"""
import json
from pathlib import Path

P = Path("MeanRev_vs_Momentum.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))

def patch_cell(idx, old, new, label):
    cell = nb["cells"][idx]
    src = "".join(cell["source"])
    assert old in src, f"[cell {idx}] '{label}' not found"
    src = src.replace(old, new)
    cell["source"] = src.splitlines(keepends=True)
    cell["outputs"] = []
    cell["execution_count"] = None
    print(f"  patched cell {idx}: {label}")

# Find cells by content (indices may shift). Build name->index map.
def find_cell(marker):
    for i, c in enumerate(nb["cells"]):
        if marker in "".join(c.get("source", [])):
            return i
    raise KeyError(marker)

# --- (1) DECAY_ALPHA ---
i = find_cell("DECAY_ALPHA   = 0.99")
patch_cell(i,
    "DECAY_ALPHA   = 0.99   # half-life ~69 days",
    "DECAY_ALPHA   = 0.995  # half-life ~138 days",
    "DECAY_ALPHA 0.99 -> 0.995")

# --- (2) Contemporaneous price regressor + simpler return regressor alignment ---
i = find_cell("# === PRICE regressor: peers' price today")
old2 = """        # === PRICE regressor: peers' price today \u2192 target's price next day ===
        X_price = md.prices[peers].copy()
        y_price = md.prices[tgt].shift(-1)   # predict next day
        pred_price = walk_forward_elasticnet(X_price, y_price)
        residual    = md.prices[tgt] - pred_price.shift(1)   # align: pred_price is for t+1, residual at t+1
        # Simpler/consistent: residual_today = today's actual \u2212 model's prediction for today (made from yesterday's peers)
        # walk_forward_elasticnet's preds[t] was fit on data <= t with y = price.shift(-1) at t, so it predicts t+1.
        # We shift back so residual at date d = actual[d] - predicted_for_d (from d-1 peers).
        residual_z = (residual - residual.rolling(RESID_Z_WIN, min_periods=20).mean()) / \\
                      residual.rolling(RESID_Z_WIN, min_periods=20).std()"""
new2 = """        # === PRICE regressor: CONTEMPORANEOUS — peers' price today \u2192 target price today ===
        # Cross-sectional spread: residual_z captures how 'over/under-valued' the
        # target is vs its peer-implied price RIGHT NOW. This is the canonical
        # pairs-trading signal.
        X_price = md.prices[peers].copy()
        y_price = md.prices[tgt].copy()        # contemporaneous, not shifted
        pred_price = walk_forward_elasticnet(X_price, y_price)
        residual   = md.prices[tgt] - pred_price
        residual_z = (residual - residual.rolling(RESID_Z_WIN, min_periods=20).mean()) / \\
                      residual.rolling(RESID_Z_WIN, min_periods=20).std()"""
patch_cell(i, old2, new2, "contemporaneous price regressor")

# --- (3) REGIME_CONF_THR ---
i = find_cell("REGIME_CONF_THR    = 0.45")
patch_cell(i,
    "REGIME_CONF_THR    = 0.45    # min regime probability",
    "REGIME_CONF_THR    = 0.55    # min regime probability",
    "REGIME_CONF_THR 0.45 -> 0.55")

# --- (4) MOMENTUM logic flip ---
i = find_cell("# MOMENTUM (+1) \u2014 use predicted_return for direction")
old4 = """            else:           # MOMENTUM (+1) \u2014 use predicted_return for direction
                if prr > pred_ret_thr:
                    desired = 1;  action = 'ENTER_LONG'
                elif prr < -pred_ret_thr:
                    desired = -1; action = 'ENTER_SHORT'
                # else stay flat (insufficient pred_ret confirmation)"""
new4 = """            else:           # MOMENTUM (+1) \u2014 trend continuation: follow residual_z direction
                # rz > 0 \u2192 target rich vs peers; in momentum it keeps drifting up \u2192 LONG
                # rz < 0 \u2192 target cheap vs peers; in momentum it keeps drifting down \u2192 SHORT
                # Require predicted_return agreement as sanity check.
                if rz > 0 and prr > -pred_ret_thr:
                    desired = 1;  action = 'ENTER_LONG'
                elif rz < 0 and prr < pred_ret_thr:
                    desired = -1; action = 'ENTER_SHORT'
                # else stay flat (rz and pred_ret disagree)"""
patch_cell(i, old4, new4, "MOMENTUM direction = sign(rz)")

P.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("\nSaved MeanRev_vs_Momentum.ipynb")
