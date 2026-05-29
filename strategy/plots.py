"""Best-effort plotting. Uses a non-interactive backend and writes PNGs to
``cfg.plots_dir``. Never raises into the pipeline (caller guards)."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make_all_plots(cfg, md, panel, bt, clf_res, split) -> list[str]:
    out = Path(cfg.plots_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    # pick a representative sector with the most rows for price/residual plots
    sect = panel["sector"].value_counts().idxmax()
    sp = panel[panel["sector"] == sect].sort_values("date")

    # 1. actual price vs shadow price
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(sp["date"], sp["target_price"], label="actual price")
    ax.plot(sp["date"], sp["shadow_price"], label="shadow price", alpha=0.8)
    ax.set_title(f"Actual vs Shadow price — {sect}"); ax.legend()
    paths.append(_save(fig, out / "01_price_vs_shadow.png"))

    # 2. residual over time
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(sp["date"], sp["price_residual"]); ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"Price residual — {sect}")
    paths.append(_save(fig, out / "02_residual.png"))

    # 3. residual z-score
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(sp["date"], sp["residual_z"]); ax.axhline(0, color="k", lw=0.6)
    for lvl in (-2, 2): ax.axhline(lvl, color="r", ls="--", lw=0.6)
    ax.set_title(f"Residual z-score — {sect}")
    paths.append(_save(fig, out / "03_residual_z.png"))

    # 4. signals over price (test region)
    ts = sp[sp["date"] >= split.val_end]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ts["date"], ts["target_price"], color="gray", label="price")
    for sig, c, mk in [(1, "g", "^"), (-1, "r", "v")]:
        m = ts[ts.get("label") == sig] if "label" in ts else ts.iloc[0:0]
        ax.scatter(m["date"], m["target_price"], color=c, marker=mk, s=20)
    ax.set_title(f"Labels over price (test) — {sect}"); ax.legend()
    paths.append(_save(fig, out / "04_signals.png"))

    # 5. cumulative strategy vs buy & hold
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(bt.portfolio.index, bt.portfolio["equity"], label="strategy")
    ax.plot(bt.portfolio.index, bt.portfolio["bh"], label="buy & hold basket", alpha=0.8)
    ax.set_title("Cumulative return — strategy vs buy&hold"); ax.legend()
    paths.append(_save(fig, out / "05_equity_curve.png"))

    # 6. feature importance
    imp = clf_res["importance"].head(20)[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(imp.index, imp.values); ax.set_title("Top-20 feature importance")
    paths.append(_save(fig, out / "06_feature_importance.png"))

    # 7. sector performance
    fig, ax = plt.subplots(figsize=(9, 4))
    bt.sector_perf["net_pnl"].sort_values().plot.barh(ax=ax)
    ax.set_title("Sector net PnL")
    paths.append(_save(fig, out / "07_sector_perf.png"))

    return paths


def _save(fig, path: Path) -> str:
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return str(path)
