"""Fresh-start architectural diagnostic.

Mirrors notebook DBTS_Train_Only_Diagnostic_FIXED.ipynb cells 0..6 (load,
fresh-start purge, build panel) and runs DIAG-1..DIAG-5 from
ARCHITECTURE_REVIEW_AND_FIX_PLAN.txt without executing the full DBTS+PM loop.

Purpose: empirically prove / disprove the suspected panel-coverage bug
("panel_indexed contains only one target row per sector/date") before any
strategy code is changed.

NO CACHE IS READ. Every cache file under .cache/ and
outputs/train_only_dbts_cache/ is deleted up-front. force_recompute=True is
set on StrategyConfig so any new artifact written during the run is fine,
but no previous artifact can influence results.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# ------------------------------------------------------------------------
# imports from the project (after sys.path tweak)
# ------------------------------------------------------------------------
from config import SECTORS  # noqa: E402
from strategy.strategy_config import StrategyConfig  # noqa: E402
from strategy.pipeline import StrategyPipeline  # noqa: E402
from strategy.splits import chrono_split, walk_forward_folds  # noqa: E402


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


# ========================================================================
# DIAG-6 + Section 5 — Fresh start / cache purge
# ========================================================================
def fresh_start_purge() -> None:
    hr("FRESH-START — cache audit and purge (DIAG-6)")

    cache_dirs = [
        PROJECT_ROOT / ".cache",
        PROJECT_ROOT / "outputs" / "train_only_dbts_cache",
    ]

    print("\n[5.2] Enumerating caches BEFORE deletion:")
    any_found = False
    for d in cache_dirs:
        if not d.exists():
            print(f"  (missing) {d}")
            continue
        files = sorted(d.glob("*"))
        if not files:
            print(f"  (empty)   {d}")
            continue
        for f in files:
            if f.is_file():
                stat = f.stat()
                mt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
                print(f"  exists    {f.relative_to(PROJECT_ROOT)}  "
                      f"size={stat.st_size:>10d}  mtime={mt}")
                any_found = True
    if not any_found:
        print("  (no cache files found — nothing to delete)")

    print("\n[5.3] Deleting EVERY cache file under the directories above:")
    deleted = 0
    for d in cache_dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("*")):
            if f.is_file():
                f.unlink()
                print(f"  [delete] {f.relative_to(PROJECT_ROOT)}")
                deleted += 1
    print(f"  Total files deleted: {deleted}")

    print("\n[5.4] Recreating cache directories empty:")
    for d in cache_dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ready     {d.relative_to(PROJECT_ROOT)}")

    print("\n[5.5] Policy:")
    print("  - cfg.force_recompute = True (CacheManager will NOT READ any cache)")
    print("  - any cache file written during this run is acceptable")
    print("  - notebook-local RECOMPUTE_DBTS is not used here (no DBTS loop)")

    print("\n[5.6] Will be REBUILT FROM SCRATCH this run:")
    for name in ("market_data", "technical_all", "feature_panel"):
        print(f"  - {name}")
    print("  (classifier / model_store / bandit / selected_panel / pm_trades / "
          "daily_scores / bandit_states are exercised in the notebook, not here)")

    print("\n[5.7] Cache files IGNORED on disk after purge: 0 "
          "(all relevant files were deleted).")


# ========================================================================
# Cells 5–6 condensed — load data + build panel
# ========================================================================
def build_fresh_panel():
    hr("LOAD MARKET DATA (Cell 5 equivalent)")
    cfg = StrategyConfig(force_recompute=True, make_plots=False)
    assert cfg.force_recompute is True
    pipeline = StrategyPipeline(cfg)
    md = pipeline.load_data()
    split = chrono_split(md.prices.index, cfg)

    train_idx = pd.DatetimeIndex(split.train_idx).sort_values()
    horizon_buffer = int(max(getattr(cfg, "label_horizon", 1),
                             getattr(cfg, "return_horizon", 1)))
    train_fit_idx = (train_idx[:-horizon_buffer]
                     if len(train_idx) > horizon_buffer else train_idx[:0])

    print(f"TRAIN range:     {train_idx[0].date()} -> {train_idx[-1].date()} "
          f"| n={len(train_idx)}")
    print(f"train_fit_idx:   {train_fit_idx[0].date()} -> {train_fit_idx[-1].date()} "
          f"| n={len(train_fit_idx)}")

    hr("BUILD PANEL (Cell 6 equivalent, fresh from raw market data)")
    diag_cfg = dataclasses.replace(cfg, min_train_days=120)
    train_folds = walk_forward_folds(train_fit_idx, diag_cfg)
    print(f"folds: {len(train_folds)} (diagnostic min_train_days=120)")
    for i, f in enumerate(train_folds, 1):
        print(f"  fold {i:02d}: train={len(f.train_idx):4d}  "
              f"predict={len(f.predict_idx):3d}  retrain={f.retrain_date.date()}")

    t0 = time.time()
    panel = pipeline.build_panel(md, train_folds, split)
    print(f"\nbuild_panel completed in {time.time()-t0:0.1f}s")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"].isin(train_fit_idx)].copy()
    print(f"panel rows (TRAIN-only filter): {len(panel):,}")
    print(f"panel date range:               "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"panel unique dates:             {panel['date'].nunique()}")
    print(f"panel unique 'target' values:   {panel['target'].nunique()}")
    print(f"panel unique sectors:           {panel['sector'].nunique()}")

    return cfg, md, split, train_fit_idx, train_folds, panel


# ========================================================================
# DIAG-1  — Panel coverage matrix (per date x sector)
# ========================================================================
def diag1_panel_coverage(panel: pd.DataFrame, outputs_dir: Path) -> pd.DataFrame:
    hr("DIAG-1 — Panel coverage by (date, sector)")
    expected_by_sector = {
        cfg_sector["name"]: len([cfg_sector["target"]] + cfg_sector["predictors"])
        for _, cfg_sector in SECTORS.items()
    }
    actual = (panel.groupby(["date", "sector"]).size()
              .rename("actual_rows").reset_index())
    actual["expected_candidates"] = actual["sector"].map(expected_by_sector)
    actual["coverage_pct"] = (actual["actual_rows"]
                              / actual["expected_candidates"] * 100.0).round(2)

    # summary per sector
    summ = (actual.groupby("sector")
            .agg(expected_candidates=("expected_candidates", "first"),
                 min_actual=("actual_rows", "min"),
                 median_actual=("actual_rows", "median"),
                 max_actual=("actual_rows", "max"),
                 mean_coverage_pct=("coverage_pct", "mean"),
                 min_coverage_pct=("coverage_pct", "min"),
                 n_dates=("actual_rows", "size"))
            .reset_index()
            .sort_values("sector"))

    actual.to_csv(outputs_dir / "diag_panel_coverage_by_date_sector.csv", index=False)
    summ.to_csv(outputs_dir / "diag_panel_coverage_summary.csv", index=False)

    print("\nPer-sector coverage summary:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summ.to_string(index=False))

    full_cov = (actual["coverage_pct"] == 100.0).all()
    print(f"\nALL (date, sector) rows at 100% coverage? -> {full_cov}")
    if not full_cov:
        bad = actual[actual["coverage_pct"] < 100.0]
        print(f"  rows below 100%: {len(bad):,}")
        print(f"  worst 10:")
        print(bad.nsmallest(10, "coverage_pct").to_string(index=False))

    return actual


# ========================================================================
# DIAG-2  — Per-candidate feature availability
# ========================================================================
def diag2_candidate_features(panel: pd.DataFrame, outputs_dir: Path) -> pd.DataFrame:
    hr("DIAG-2 — Per-candidate feature availability")

    def pct_nonnull(s):
        return float(s.notna().mean() * 100.0) if len(s) else 0.0

    rows = []
    for (sector, cand), sub in panel.groupby(["sector", "target"]):
        rows.append({
            "sector": sector,
            "candidate": cand,
            "n_rows": len(sub),
            "residual_z_pct":        round(pct_nonnull(sub.get("residual_z", pd.Series(dtype=float))), 2),
            "predicted_return_pct":  round(pct_nonnull(sub.get("predicted_return", pd.Series(dtype=float))), 2),
            "shadow_price_pct":      round(pct_nonnull(sub.get("shadow_price", pd.Series(dtype=float))), 2),
            "dbts_score_pct":        round(pct_nonnull(sub.get("dbts_score", pd.Series(dtype=float))), 2),
            "was_selected_by_dbts_sum": int(sub.get("was_selected_by_dbts", pd.Series(dtype=int)).sum())
                                       if "was_selected_by_dbts" in sub.columns else 0,
        })
    out = pd.DataFrame(rows).sort_values(["sector", "candidate"])
    out.to_csv(outputs_dir / "diag_candidate_feature_availability.csv", index=False)

    print(f"\nCandidates seen in panel: {len(out)}")
    print("\nFirst 25 rows:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out.head(25).to_string(index=False))

    # aggregate: any candidate with <80% predicted_return availability?
    weak_pr = out[out["predicted_return_pct"] < 80.0]
    weak_rz = out[out["residual_z_pct"] < 80.0]
    print(f"\nCandidates with predicted_return availability < 80%: {len(weak_pr)}")
    if len(weak_pr):
        print(weak_pr.head(20).to_string(index=False))
    print(f"\nCandidates with residual_z availability < 80%: {len(weak_rz)}")
    if len(weak_rz):
        print(weak_rz.head(20).to_string(index=False))

    return out


# ========================================================================
# DIAG-3  — panel_indexed lookup-success rate (simulates Cell 8)
# ========================================================================
def diag3_panel_indexed_lookup(panel: pd.DataFrame, train_fit_idx,
                               cfg, outputs_dir: Path) -> pd.DataFrame:
    hr("DIAG-3 — panel_indexed lookup-success simulating Cell 8")
    panel_indexed = panel.set_index(["date", "target"])
    panel_dates = set(panel_indexed.index.get_level_values("date"))

    FEATURE_WARMUP = max(getattr(cfg, "week52_win", 252), 200)
    h = int(getattr(cfg, "label_horizon", 5))
    eval_dates = list(train_fit_idx[FEATURE_WARMUP:-h]) if h > 0 else list(train_fit_idx[FEATURE_WARMUP:])
    eval_dates = [d for d in eval_dates if d in panel_dates]
    print(f"eval_dates: {len(eval_dates)} "
          f"({eval_dates[0].date() if eval_dates else 'n/a'} -> "
          f"{eval_dates[-1].date() if eval_dates else 'n/a'})")

    rows = []
    pr_col_exists = "predicted_return" in panel_indexed.columns
    rz_col_exists = "residual_z" in panel_indexed.columns
    print(f"predicted_return column in panel_indexed: {pr_col_exists}")
    print(f"residual_z       column in panel_indexed: {rz_col_exists}")

    for etf, cfg_sector in SECTORS.items():
        sector_name = cfg_sector["name"]
        members = [cfg_sector["target"]] + cfg_sector["predictors"]
        n_dates = len(eval_dates)
        total_lookups = n_dates * len(members)
        hits = 0
        pr_ok = 0
        rz_ok = 0
        for d in eval_dates:
            for cand in members:
                if (d, cand) in panel_indexed.index:
                    hits += 1
                    if pr_col_exists:
                        v = panel_indexed.at[(d, cand), "predicted_return"]
                        if pd.notna(v):
                            pr_ok += 1
                    if rz_col_exists:
                        v = panel_indexed.at[(d, cand), "residual_z"]
                        if pd.notna(v):
                            rz_ok += 1
        rows.append({
            "sector": sector_name,
            "n_members": len(members),
            "n_eval_dates": n_dates,
            "expected_lookups": total_lookups,
            "key_hits": hits,
            "key_hit_pct": round(hits / total_lookups * 100.0, 2) if total_lookups else 0.0,
            "predicted_return_ok": pr_ok,
            "predicted_return_ok_pct": round(pr_ok / total_lookups * 100.0, 2) if total_lookups else 0.0,
            "residual_z_ok": rz_ok,
            "residual_z_ok_pct": round(rz_ok / total_lookups * 100.0, 2) if total_lookups else 0.0,
        })
    out = pd.DataFrame(rows).sort_values("sector")
    out.to_csv(outputs_dir / "diag_panel_indexed_lookup.csv", index=False)

    print("\nLookup audit (what DBTS in Cell 8 would actually see):")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(out.to_string(index=False))

    overall = {
        "sector": "TOTAL",
        "n_members": out["n_members"].sum(),
        "n_eval_dates": out["n_eval_dates"].iloc[0] if len(out) else 0,
        "expected_lookups": out["expected_lookups"].sum(),
        "key_hits": out["key_hits"].sum(),
        "key_hit_pct": round(out["key_hits"].sum() / max(out["expected_lookups"].sum(), 1) * 100, 2),
        "predicted_return_ok": out["predicted_return_ok"].sum(),
        "predicted_return_ok_pct": round(out["predicted_return_ok"].sum() / max(out["expected_lookups"].sum(), 1) * 100, 2),
        "residual_z_ok": out["residual_z_ok"].sum(),
        "residual_z_ok_pct": round(out["residual_z_ok"].sum() / max(out["expected_lookups"].sum(), 1) * 100, 2),
    }
    print("\nOVERALL:")
    for k, v in overall.items():
        print(f"  {k:>26s}: {v}")
    return out


# ========================================================================
# DIAG-4  — DBTS component contribution (reuses build_panel scoring)
# ========================================================================
def diag4_dbts_components(panel: pd.DataFrame, outputs_dir: Path) -> pd.DataFrame:
    hr("DIAG-4 — DBTS component decomposition (from build_panel scoring)")
    # build_panel already computed dbts_score using:
    #   final = 0.40*bandit + 0.35*|residual_z|/3 capped + 0.25*|pred_ret|/0.05 capped
    # We don't have per-component logs persisted from build_panel, but we can
    # recompute the deterministic residual and pred_ret components from the
    # panel itself. The bandit component is stochastic; we report the residual
    # contribution from the formula above.
    df = panel.copy()
    rz = df["residual_z"].astype(float) if "residual_z" in df.columns else pd.Series(np.nan, index=df.index)
    pr = df.get("predicted_return", pd.Series(np.nan, index=df.index)).astype(float)
    df["residual_component"] = (rz.abs() / 3.0).clip(upper=1.0).where(rz.notna(), other=np.nan)
    df["pred_ret_component"] = (pr / 0.05).clip(-1.0, 1.0).abs().where(pr.notna(), other=np.nan)

    summary = (df.groupby("sector")
                 .agg(n_rows=("dbts_score", "size"),
                      dbts_score_mean=("dbts_score", "mean"),
                      dbts_score_std=("dbts_score", "std"),
                      residual_component_mean=("residual_component", "mean"),
                      residual_component_std=("residual_component", "std"),
                      pred_ret_component_mean=("pred_ret_component", "mean"),
                      pred_ret_component_std=("pred_ret_component", "std"))
                 .round(4)
                 .reset_index()
                 .sort_values("sector"))
    summary.to_csv(outputs_dir / "diag_dbts_component_decomposition.csv", index=False)
    print("\nPer-sector mean/std of deterministic DBTS components and final dbts_score:")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary.to_string(index=False))

    print("\nNote: residual_component (weight 0.35) and pred_ret_component "
          "(weight 0.25) shown above are DETERMINISTIC. The bandit component "
          "(weight 0.40) is stochastic and not persisted in panel; high "
          "variance in dbts_score vs near-zero variance of deterministic "
          "components would imply bandit dominates ranking.")
    return summary


# ========================================================================
# DIAG-5  — Selection stability
# ========================================================================
def diag5_selection_distribution(panel: pd.DataFrame, outputs_dir: Path) -> pd.DataFrame:
    hr("DIAG-5 — Selection stability per sector")
    if "was_selected_by_dbts" not in panel.columns:
        print("was_selected_by_dbts column missing; skipping.")
        return pd.DataFrame()

    sel = panel[panel["was_selected_by_dbts"] == 1].copy()
    sel = sel.sort_values(["sector", "date"])

    rows = []
    for sector, sub in sel.groupby("sector"):
        winners = sub["target"].value_counts()
        # day-to-day switch rate
        switches = (sub["target"].shift(1) != sub["target"]).iloc[1:].sum()
        switch_rate = float(switches) / max(len(sub) - 1, 1)
        rows.append({
            "sector": sector,
            "n_days": len(sub),
            "unique_winners": int(sub["target"].nunique()),
            "top_winner": winners.index[0] if len(winners) else "",
            "top_winner_pct": round(float(winners.iloc[0] / len(sub) * 100), 2) if len(winners) else 0.0,
            "second_winner": winners.index[1] if len(winners) > 1 else "",
            "second_winner_pct": round(float(winners.iloc[1] / len(sub) * 100), 2) if len(winners) > 1 else 0.0,
            "switch_rate_day_to_day_pct": round(switch_rate * 100, 2),
        })
    out = pd.DataFrame(rows).sort_values("sector")
    out.to_csv(outputs_dir / "diag_selection_distribution.csv", index=False)

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(out.to_string(index=False))
    return out


# ========================================================================
# main
# ========================================================================
def main() -> int:
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    fresh_start_purge()
    cfg, md, split, train_fit_idx, train_folds, panel = build_fresh_panel()

    diag1_panel_coverage(panel, outputs_dir)
    diag2_candidate_features(panel, outputs_dir)
    diag3_panel_indexed_lookup(panel, train_fit_idx, cfg, outputs_dir)
    diag4_dbts_components(panel, outputs_dir)
    diag5_selection_distribution(panel, outputs_dir)

    hr("ALL DIAGNOSTICS WRITTEN")
    for name in ("diag_panel_coverage_by_date_sector.csv",
                 "diag_panel_coverage_summary.csv",
                 "diag_candidate_feature_availability.csv",
                 "diag_panel_indexed_lookup.csv",
                 "diag_dbts_component_decomposition.csv",
                 "diag_selection_distribution.csv"):
        p = outputs_dir / name
        if p.exists():
            print(f"  ok  {p.relative_to(PROJECT_ROOT)}  ({p.stat().st_size} bytes)")
        else:
            print(f"  MISSING  {p.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
