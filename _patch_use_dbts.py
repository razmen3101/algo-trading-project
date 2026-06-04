"""Patch Cell 8 to:
  1. Add USE_DBTS flag (default False) — when False, always use cfg target
     instead of dynamic DBTS selection. Removes bandit randomness entirely.
  2. Pin RNG seeds (random + numpy) for full determinism.
  3. When USE_DBTS=False, skip the heavy scoring loop and just pick members[0]
     (which IS the config target since members = [cfg_sector['target'], *predictors]).
"""
import json
from pathlib import Path

P = Path("DBTS_Train_Only_Diagnostic_FIXED.ipynb")
nb = json.loads(P.read_text(encoding="utf-8"))
cell = nb["cells"][16]
src = "".join(cell["source"])

# --- (1) Add USE_DBTS flag + RNG seed pin at top of else branch ---
old1 = '''CACHE_DIR = Path("outputs/train_only_dbts_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECOMPUTE_DBTS = True'''
new1 = '''CACHE_DIR = Path("outputs/train_only_dbts_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RECOMPUTE_DBTS = True

# ============================================================
# USE_DBTS flag:
#   True  -> dynamic per-day target selection (bandit + residual + adf + pred_ret)
#   False -> always trade the static config target (deterministic, no bandit)
# Set to False to remove bandit randomness and DBTS scoring noise.
# ============================================================
USE_DBTS = False

# Pin RNG seeds for determinism
import random as _random
_random.seed(int(getattr(cfg, "random_state", 42)))
np.random.seed(int(getattr(cfg, "random_state", 42)))'''
assert old1 in src, "block 1 not found"
src = src.replace(old1, new1)

# --- (2) Replace the selection block to branch on USE_DBTS ---
old2 = '''            finite = {k: v for k, v in scores.items() if np.isfinite(v)}
            top    = max(finite, key=finite.get) if finite else members[0]
            current_sel = last_selected_by_sector.get(sector_name)
            held_days   = target_hold_days.get(sector_name, 0)
            if current_sel is None or current_sel == top:
                selected = top
            elif held_days < MIN_TARGET_HOLD_DAYS:
                selected = current_sel
            elif finite.get(top, -np.inf) < finite.get(current_sel, -np.inf) + TARGET_SWITCH_MARGIN:
                selected = current_sel
            else:
                selected = top
            target_switched = (current_sel is not None) and (selected != current_sel)
            target_hold_days[sector_name] = 1 if target_switched else (held_days + 1)
            last_selected_by_sector[sector_name] = selected'''
new2 = '''            if USE_DBTS:
                finite = {k: v for k, v in scores.items() if np.isfinite(v)}
                top    = max(finite, key=finite.get) if finite else members[0]
                current_sel = last_selected_by_sector.get(sector_name)
                held_days   = target_hold_days.get(sector_name, 0)
                if current_sel is None or current_sel == top:
                    selected = top
                elif held_days < MIN_TARGET_HOLD_DAYS:
                    selected = current_sel
                elif finite.get(top, -np.inf) < finite.get(current_sel, -np.inf) + TARGET_SWITCH_MARGIN:
                    selected = current_sel
                else:
                    selected = top
                target_switched = (current_sel is not None) and (selected != current_sel)
                target_hold_days[sector_name] = 1 if target_switched else (held_days + 1)
                last_selected_by_sector[sector_name] = selected
            else:
                # STATIC MODE: always trade the config target (members[0] = cfg_sector["target"])
                selected = members[0]
                target_switched = False
                last_selected_by_sector[sector_name] = selected'''
assert old2 in src, "block 2 not found"
src = src.replace(old2, new2)

# --- (3) Add print of mode at top of else branch ---
old3 = '    print("Running DBTS scoring loop on TRAIN only (online bandit updates)...")'
new3 = '    print(f"Running scoring loop on TRAIN only | USE_DBTS={USE_DBTS}  "\n          f"(static config-target mode)" if not USE_DBTS else \n          "Running DBTS scoring loop on TRAIN only (online bandit updates)...")'
# Use a cleaner approach
old3_actual = 'print("Running DBTS scoring loop on TRAIN only (online bandit updates)...")'
new3_actual = '''print(f"Running scoring loop on TRAIN only | USE_DBTS={USE_DBTS}")
    if not USE_DBTS:
        print("  STATIC MODE: trading config target per sector, no bandit, no DBTS scoring")'''
assert old3_actual in src, "block 3 not found"
src = src.replace(old3_actual, new3_actual)

cell["source"] = src.splitlines(keepends=True)
cell["outputs"] = []
cell["execution_count"] = None
P.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Patched Cell 8: USE_DBTS flag + seed pinning")
