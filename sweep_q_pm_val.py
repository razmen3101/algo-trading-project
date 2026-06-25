"""Greedy VAL-only tuning of (1) Q-component weights and (2) PM stop/take.

Standalone script — run it yourself with:  python sweep_q_pm_val.py

It does NOT recompute the heavy NS_spline regressors: it loads the cached
artifacts produced by DBTS_Variants.ipynb (outputs/dbts_variants/*.pkl).
Everything is scored on the VALIDATION split only.

Outputs (saved to outputs/dbts_variants/):
  - sweep_qweights_val.csv          full Q-weight grid results
  - sweep_pm_stop_take_val.csv      full PM stop/take grid results
  - fig_sweeps_val.png              SINGLE figure: 2 rows x 7 metric heatmaps
                                    (row 1 = Q weights, row 2 = PM stop/take)
  - leaderboard_qweights_val.csv    top-10 Q-weight combos by VAL Sharpe
  - leaderboard_pm_val.csv          top-10 PM combos by VAL Sharpe

Methodology note (no look-ahead): the Q components (mr_hit, pred_ic, adf_p) are
TRAIN-only and cached; we only re-normalize + re-weight them here. Candidate
selection is warmed on the full timeline (identical to the 18-variant bake-off)
and P&L is measured on the VAL slice only.
"""
from __future__ import annotations

import os
import sys
import math
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: save figures, never open a window
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# 0. Project setup (mirror DBTS_Variants.ipynb cells 2-3)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "strategy").exists():
    cand = Path(r"C:\algo-trading-project")
    if cand.exists():
        PROJECT_ROOT = cand
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

SEED = 42
np.random.seed(SEED)

CACHE_DIR = Path("outputs/dbts_variants")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REG_CACHE = CACHE_DIR / "ns_regressors.pkl"
QUALITY_CACHE = CACHE_DIR / "quality.pkl"

from config import SECTORS
from strategy.strategy_config import StrategyConfig
from strategy.pipeline import StrategyPipeline
from strategy.splits import chrono_split

# used only when the regressor/quality caches must be (re)built (notebook cells 4-5)
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler, SplineTransformer
try:
    from statsmodels.tsa.stattools import adfuller
except Exception:
    adfuller = None

# ---- regressor build constants (verbatim from notebook Cell 4) ------------- #
RETRAIN_EVERY = 50
MIN_TRAIN_DAYS = 100
ROLL_WIN = 200
DECAY_ALPHA = 0.995
TOP_K_FEATS = 5
SPLINE_K = 4
SPLINE_DEG = 3
RIDGE_ALPHA = 1.0
EN_ALPHA = 0.01
EN_L1_RATIO = 0.5

# ---- quality build constants (verbatim from notebook Cell 5) --------------- #
RZ_ENTRY_FOR_HIT = 1.5
RZ_EXIT_FOR_HIT = 0.5
HIT_WINDOW = 10
IC_HORIZON = 5

# ---- constants copied verbatim from the notebook engine cells -------------- #
RESID_Z_WIN = 60            # (already baked into the cached residual_z)
RZ_ENTRY = 1.5              # PM entry threshold on |residual_z|
RZ_EXIT = 0.30             # PM mean-reversion exit
MAX_HOLD = 10
PM_COST_BPS = 5.0 / 1e4
STOP_LOSS_DEFAULT = -0.02   # used while tuning Q weights
TAKE_PROFIT_DEFAULT = 0.03

# structural axes locked at the bake-off champion
SCORE_FN, UNIVERSE, STICK = "rz_x_q", "all", "soft"
SOFT_MARGIN = 0.10
MIN_HOLD = {"none": 0, "soft": 5, "hard": 20}[STICK]

# 7 metrics shown in every heatmap row (matches the attached leaderboard table)
METRICS = [
    ("sharpe", "Sharpe"),
    ("sortino", "Sortino"),
    ("cum_ret", "Cum. Return"),
    ("max_dd", "Max Drawdown"),
    ("entries", "Entries"),
    ("win_rate_days", "Win-rate (days)"),
    ("ann_ret", "Ann. Return"),
]


# --------------------------------------------------------------------------- #
# 1a. (Re)build the cached regressors + quality — port of notebook cells 4-5
#     Only runs when the .pkl caches are missing; results are cached afterwards.
# --------------------------------------------------------------------------- #
def _decay_weights(n, alpha=DECAY_ALPHA):
    w = alpha ** np.arange(n - 1, -1, -1, dtype=float)
    return w / w.sum() * n


def _feature_select(X, y, w, top_k=TOP_K_FEATS):
    sc = StandardScaler().fit(X)
    en = ElasticNet(alpha=EN_ALPHA, l1_ratio=EN_L1_RATIO, max_iter=5000, random_state=SEED)
    en.fit(sc.transform(X), y, sample_weight=w)
    coef = np.abs(en.coef_)
    if not np.any(coef > 0):
        return np.arange(min(top_k, X.shape[1]))
    return np.argsort(-coef)[:top_k]


def _fit_ns(X, y, w):
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    sp_ = SplineTransformer(n_knots=SPLINE_K, degree=SPLINE_DEG, knots="quantile",
                            extrapolation="linear").fit(Xs)
    Z = sp_.transform(Xs)
    m = Ridge(alpha=RIDGE_ALPHA, random_state=SEED).fit(Z, y, sample_weight=w)
    return lambda Xq: m.predict(sp_.transform(sc.transform(Xq)))


def _wf_ns(X_df, y_ser):
    idx = X_df.index
    n = len(idx)
    pred1 = np.full(n, np.nan)
    last_refit = -10 ** 9
    predict_fn = None
    sel_idx = None
    Xv = X_df.values
    yv = y_ser.values
    for t in range(n):
        if t >= MIN_TRAIN_DAYS and (t - last_refit) >= RETRAIN_EVERY:
            lo = max(0, t - ROLL_WIN)
            Xt = Xv[lo:t]
            yt = yv[lo:t]
            mask = (~np.isnan(Xt).any(axis=1)) & (~np.isnan(yt))
            if mask.sum() < MIN_TRAIN_DAYS:
                continue
            Xt = Xt[mask]
            yt = yt[mask]
            w = _decay_weights(len(yt))
            try:
                sel_idx = _feature_select(Xt, yt, w)
                predict_fn = _fit_ns(Xt[:, sel_idx], yt, w)
                last_refit = t
            except Exception:
                predict_fn = None
        if predict_fn is not None and sel_idx is not None and not np.isnan(Xv[t]).any():
            try:
                pred1[t] = float(predict_fn(Xv[t:t + 1, sel_idx])[0])
            except Exception:
                pass
    return pd.Series(pred1, index=idx)


def build_regressors(md, sector_pools):
    print("[build] NS_spline rolling regressors (cache miss) ...")
    REG = {}
    t0 = time.time()
    for sector, pool in sector_pools.items():
        for cand in pool:
            peers = [p for p in pool if p != cand]
            if not peers:
                continue
            Xp = md.prices[peers].copy()
            yp = md.prices[cand].copy()
            mask = (~Xp.isna().any(axis=1)) & yp.notna()
            X = Xp[mask]
            y = yp[mask]
            if len(y) < MIN_TRAIN_DAYS + 50:
                continue
            pred = _wf_ns(X, y).reindex(md.prices.index)
            price = md.prices[cand].reindex(md.prices.index)
            residual = price - pred
            resz = (residual - residual.rolling(RESID_Z_WIN).mean()) / residual.rolling(RESID_Z_WIN).std()
            pr = (pred.shift(-1) / price - 1.0)
            nxt = price.pct_change().shift(-1)
            REG[(sector, cand)] = pd.DataFrame({
                "price": price, "pred": pred, "residual": residual,
                "residual_z": resz, "predicted_return": pr, "next_ret": nxt,
            })
        print(f"  {sector:18s} done ({sum(1 for k in REG if k[0] == sector)} cands)")
    with open(REG_CACHE, "wb") as f:
        pickle.dump(REG, f)
    print(f"[build] saved {REG_CACHE.name} ({time.time() - t0:.1f}s, {len(REG)} pairs)")
    return REG


def _safe_adf(s, fallback=1.0):
    s = pd.Series(s).dropna()
    if len(s) < 30 or adfuller is None:
        return fallback
    try:
        return float(adfuller(s, autolag="AIC")[1])
    except Exception:
        return fallback


def _mr_hit_rate(rz, win=HIT_WINDOW, enter=RZ_ENTRY_FOR_HIT, exitt=RZ_EXIT_FOR_HIT):
    rz = rz.dropna().values
    if len(rz) < win + 10:
        return np.nan
    cnt = 0
    hits = 0
    i = 0
    while i < len(rz) - win:
        if abs(rz[i]) >= enter:
            cnt += 1
            window = rz[i + 1:i + 1 + win]
            if np.any(np.abs(window) <= exitt):
                hits += 1
            i += win
        else:
            i += 1
    return hits / cnt if cnt > 0 else np.nan


def _pred_return_ic(pr, next_ret_h):
    df = pd.DataFrame({"p": pr, "n": next_ret_h}).dropna()
    if len(df) < 30:
        return np.nan
    rho, _ = spearmanr(df["p"].values, df["n"].values)
    return float(rho) if np.isfinite(rho) else np.nan


def build_quality(REG, train_idx):
    print("[build] quality scores on TRAIN (cache miss) ...")
    raws = {}
    for key, df in REG.items():
        sub_train = df.loc[df.index.isin(train_idx)]
        price = sub_train["price"]
        next_ret_h = price.pct_change(IC_HORIZON).shift(-IC_HORIZON)
        raws[key] = {
            "mr_hit": _mr_hit_rate(sub_train["residual_z"]),
            "pred_ic": _pred_return_ic(sub_train["predicted_return"], next_ret_h),
            "adf_p": _safe_adf(sub_train["residual"]),
        }
    raw_df = pd.DataFrame(raws).T

    def norm01(x):
        x = x.copy().astype(float)
        if x.notna().sum() == 0:
            return x.fillna(0.5)
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        if hi == lo:
            return pd.Series(0.5, index=x.index)
        return (x - lo) / (hi - lo)

    n_hit = norm01(raw_df["mr_hit"])
    n_ic = norm01(raw_df["pred_ic"])
    n_stab = norm01(1.0 - raw_df["adf_p"].clip(0, 1))
    quality = (0.5 * n_hit + 0.3 * n_ic + 0.2 * n_stab).fillna(0.0)
    QUAL = {k: {"quality": float(quality[k]),
                "mr_hit": float(raw_df.loc[k, "mr_hit"]) if pd.notna(raw_df.loc[k, "mr_hit"]) else np.nan,
                "pred_ic": float(raw_df.loc[k, "pred_ic"]) if pd.notna(raw_df.loc[k, "pred_ic"]) else np.nan,
                "adf_p": float(raw_df.loc[k, "adf_p"])}
            for k in raw_df.index}
    with open(QUALITY_CACHE, "wb") as f:
        pickle.dump(QUAL, f)
    print(f"[build] saved {QUALITY_CACHE.name} ({len(QUAL)} pairs)")
    return QUAL


# --------------------------------------------------------------------------- #
# 1b. Load (or build) data, regressors, quality
# --------------------------------------------------------------------------- #
def load_inputs():
    cfg = StrategyConfig(force_recompute=False, make_plots=False)
    pipeline = StrategyPipeline(cfg)
    md = pipeline.load_data()
    sp = chrono_split(md.prices.index, cfg)
    train_idx = pd.DatetimeIndex(sp.train_idx).sort_values()
    val_idx = pd.DatetimeIndex(sp.val_idx).sort_values()

    sector_pools = {}
    for _etf, s in SECTORS.items():
        pool = [s["target"]] + list(s["predictors"])
        pool = [t for t in pool if t in md.prices.columns]
        sector_pools[s["name"]] = pool

    if REG_CACHE.exists():
        with open(REG_CACHE, "rb") as f:
            REG = pickle.load(f)
        print(f"[load] REG cache hit: {len(REG)} pairs")
    else:
        REG = build_regressors(md, sector_pools)

    if QUALITY_CACHE.exists():
        with open(QUALITY_CACHE, "rb") as f:
            QUAL = pickle.load(f)
        print(f"[load] QUAL cache hit: {len(QUAL)} pairs")
    else:
        QUAL = build_quality(REG, train_idx)

    print(f"[load] REG pairs={len(REG)} | QUAL pairs={len(QUAL)} | VAL days={len(val_idx)}")
    return md, val_idx, sector_pools, REG, QUAL


# --------------------------------------------------------------------------- #
# 2. Build the fast sweep engine (faithful to the notebook selector + PM)
# --------------------------------------------------------------------------- #
def _norm01_arr(x):
    x = np.asarray(x, dtype=float).copy()
    if np.all(np.isnan(x)):
        return np.full_like(x, 0.5)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi == lo:
        return np.full_like(x, 0.5)
    return (x - lo) / (hi - lo)


def build_engine(md, val_idx, sector_pools, REG, QUAL):
    """Return a dict of precomputed arrays + closures used by run_combo."""
    # per-pair stacks (full timeline)
    full_idx = md.prices.index
    stacks = {}
    for sector, pool in sector_pools.items():
        cands = [c for c in pool if (sector, c) in REG]
        if not cands:
            continue
        resz = pd.DataFrame({c: REG[(sector, c)]["residual_z"] for c in cands}).reindex(full_idx)
        nxt = pd.DataFrame({c: REG[(sector, c)]["next_ret"] for c in cands}).reindex(full_idx)
        quality_vec = np.array([QUAL[(sector, c)]["quality"] for c in cands])
        top3_ix = np.argsort(-quality_vec)[:3]
        stacks[sector] = dict(cands=cands, resz=resz, nxt=nxt, top3_ix=top3_ix)

    # global raw quality components (TRAIN-only values from cache), aligned to KEYS
    keys = list(QUAL.keys())
    mr_raw = np.array([QUAL[k]["mr_hit"] for k in keys], dtype=float)
    ic_raw = np.array([QUAL[k]["pred_ic"] for k in keys], dtype=float)
    adf_raw = np.array([QUAL[k]["adf_p"] for k in keys], dtype=float)
    NH = _norm01_arr(mr_raw)
    NI = _norm01_arr(ic_raw)
    NS = _norm01_arr(1.0 - np.clip(adf_raw, 0.0, 1.0))
    key_ix = {k: i for i, k in enumerate(keys)}

    val_pos = np.where(full_idx.isin(val_idx))[0]
    sectors_list = list(stacks.keys())
    sec_arr = {}
    for s in sectors_list:
        st = stacks[s]
        cands = st["cands"]
        keep = np.arange(len(cands)) if UNIVERSE == "all" else st["top3_ix"]
        cand_keep = [cands[i] for i in keep]
        resz = st["resz"].values[:, keep].astype(float)
        nxt = st["nxt"].values[:, keep].astype(float)
        sec_arr[s] = dict(
            resz=resz,
            nxt=nxt,
            abs_rz=np.abs(np.nan_to_num(resz, nan=0.0)),
            valid=~np.isnan(resz),
            qix=np.array([key_ix[(s, c)] for c in cand_keep]),
            cands=cand_keep,
        )
    eval_dates = full_idx[val_pos]

    def quality_from_weights(w_mr, w_ic, w_stab):
        q = w_mr * NH + w_ic * NI + w_stab * NS  # nan if any component nan
        return np.nan_to_num(q, nan=0.0)          # == original .fillna(0.0)

    def select_winners(scores):
        n = scores.shape[0]
        if STICK == "none":
            fin = np.isfinite(scores).any(axis=1)
            return np.where(fin, np.argmax(scores, axis=1), -1)
        win = np.full(n, -1, dtype=int)
        cur = -1
        held = 0
        for i in range(n):
            ds = scores[i]
            if not np.isfinite(ds).any():
                cur = -1
                held = 0
                continue
            if cur == -1 or not np.isfinite(ds[cur]):
                cur = int(np.argmax(ds))
                held = 1
            else:
                best = int(np.argmax(ds))
                if best != cur and held >= MIN_HOLD:
                    switch = (ds[best] >= ds[cur] + SOFT_MARGIN) if STICK == "soft" \
                        else (ds[cur] < 0.90 * ds[best])
                    if switch:
                        cur = best
                        held = 1
                    else:
                        held += 1
                else:
                    held += 1
            win[i] = cur
        return win

    def pm_sector(winners, resz, nxt, rz_thr, rz_exit, max_hold, stop, take, cost):
        n = len(winners)
        pnl = np.full(n, np.nan)
        is_entry = np.zeros(n, dtype=bool)
        pos_rec = np.zeros(n)
        pos = 0
        days_in = 0
        trade_pnl = 0.0
        pos_cand = -1
        for i in range(n):
            c = winners[i]
            if c < 0:                       # no valid candidate -> no row (state persists)
                continue
            rz = resz[i, c]
            if not np.isfinite(rz):
                continue
            nr = nxt[i, c]
            nr = 0.0 if not np.isfinite(nr) else nr
            prev = pos
            desired = pos
            entered = False
            if pos != 0 and pos_cand != c:
                desired = 0
            elif pos != 0:
                days_in += 1
                if (days_in >= max_hold or abs(rz) <= rz_exit
                        or trade_pnl <= stop or trade_pnl >= take):
                    desired = 0
            if desired == 0 and abs(rz) >= rz_thr:
                desired = 1 if rz < 0 else -1
                entered = True
            net = desired * nr - abs(desired - prev) * cost
            if entered:
                pos, days_in, trade_pnl, pos_cand = desired, 1, net, c
                is_entry[i] = True
            elif desired == 0:
                pos, days_in, trade_pnl, pos_cand = 0, 0, 0.0, -1
            else:
                trade_pnl += net
            pnl[i] = net
            pos_rec[i] = desired
        return pnl, is_entry, pos_rec

    def metrics(daily, entries, longs, shorts, ppy=252):
        n = len(daily)
        if n == 0:
            return dict(days=0, entries=entries, long=longs, short=shorts, cum_ret=np.nan,
                        ann_ret=np.nan, sharpe=np.nan, sortino=np.nan, max_dd=np.nan,
                        win_rate_days=np.nan)
        eq = np.cumprod(1 + daily)
        cum = float(eq[-1] - 1.0)
        ann_ret = float((1 + cum) ** (ppy / max(n, 1)) - 1.0)
        ann_vol = float(np.std(daily, ddof=1) * math.sqrt(ppy)) if n > 1 else np.nan
        sh = ann_ret / ann_vol if ann_vol and np.isfinite(ann_vol) and ann_vol != 0 else np.nan
        dn = daily[daily < 0]
        dnv = float(np.std(dn, ddof=1) * math.sqrt(ppy)) if len(dn) > 1 else np.nan
        so = ann_ret / dnv if dnv and np.isfinite(dnv) and dnv != 0 else np.nan
        peak = np.maximum.accumulate(eq)
        mdd = float((eq / peak - 1.0).min())
        return dict(days=n, entries=int(entries), long=int(longs), short=int(shorts),
                    cum_ret=round(cum, 4), ann_ret=round(ann_ret, 4),
                    sharpe=round(sh, 4) if np.isfinite(sh) else np.nan,
                    sortino=round(so, 4) if np.isfinite(so) else np.nan,
                    max_dd=round(mdd, 4),
                    win_rate_days=round(float((daily > 0).mean()), 4))

    def run_combo(w_mr, w_ic, stop, take, rz_thr=RZ_ENTRY, rz_exit=RZ_EXIT,
                  max_hold=MAX_HOLD, cost=PM_COST_BPS, return_panel=False):
        """stop is NEGATIVE (e.g. -0.02). Returns a metrics dict for the eval
        split. If return_panel=True, also returns a per-(date,sector) DataFrame
        with the held stock, position and forward return (for benchmarking)."""
        q = quality_from_weights(w_mr, w_ic, 1.0 - w_mr - w_ic)
        nval = len(val_pos)
        pnl_mat = np.full((nval, len(sectors_list)), np.nan)
        entries = longs = shorts = 0
        panel_rows = []
        for j, s in enumerate(sectors_list):
            a = sec_arr[s]
            qv = q[a["qix"]]
            if SCORE_FN == "rz_x_q":
                scores = a["abs_rz"] * qv[None, :]
            elif SCORE_FN == "q":
                scores = np.broadcast_to(qv[None, :], a["abs_rz"].shape).copy()
            else:
                scores = a["abs_rz"].copy()
            scores = np.where(a["valid"], scores, -np.inf)
            winners_full = select_winners(scores)
            wv = winners_full[val_pos]
            rzv = a["resz"][val_pos]
            nxv = a["nxt"][val_pos]
            pnl, is_e, pos_r = pm_sector(wv, rzv, nxv,
                                         rz_thr, rz_exit, max_hold, stop, take, cost)
            pnl_mat[:, j] = pnl
            entries += int(is_e.sum())
            longs += int((pos_r[is_e] == 1).sum())
            shorts += int((pos_r[is_e] == -1).sum())
            if return_panel:
                for i in range(nval):
                    c = wv[i]
                    if c < 0:
                        continue
                    panel_rows.append({
                        "date": eval_dates[i], "sector": s,
                        "stock": a["cands"][c], "position": float(pos_r[i]),
                        "next_ret": float(nxv[i, c]) if np.isfinite(nxv[i, c]) else 0.0,
                    })
        daily = np.nanmean(pnl_mat, axis=1)
        daily = daily[~np.isnan(daily)]
        m = metrics(daily, entries, longs, shorts)
        if return_panel:
            return m, pd.DataFrame(panel_rows)
        return m

    return dict(run_combo=run_combo, n_sectors=len(sectors_list), n_val=len(val_pos),
                eval_dates=eval_dates)


# --------------------------------------------------------------------------- #
# 3. Sweeps
# --------------------------------------------------------------------------- #
def sweep_q_weights(run_combo, step=0.01):
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 2)
    rows = []
    total = sum(1 for wm in grid for wi in grid if wm + wi <= 1.0 + 1e-9)
    t0 = time.time()
    done = 0
    for wm in grid:
        for wi in grid:
            if wm + wi > 1.0 + 1e-9:
                continue
            m = run_combo(float(wm), float(wi), stop=STOP_LOSS_DEFAULT, take=TAKE_PROFIT_DEFAULT)
            rows.append({"w_mr": round(float(wm), 2), "w_ic": round(float(wi), 2),
                         "w_stab": round(1.0 - float(wm) - float(wi), 2), **m})
            done += 1
            if done % 500 == 0:
                print(f"  Q sweep {done}/{total}  ({time.time() - t0:.0f}s)")
    print(f"  Q sweep done: {done} combos in {time.time() - t0:.0f}s")
    return pd.DataFrame(rows), grid


def sweep_pm(run_combo, w_mr_best, w_ic_best):
    take_grid = np.round(np.arange(0.2, 5.0 + 1e-9, 0.2), 1)   # percent
    stop_grid = np.round(np.arange(0.2, 2.5 + 1e-9, 0.2), 1)   # percent (negated)
    rows = []
    t0 = time.time()
    for tp in take_grid:
        for sl in stop_grid:
            m = run_combo(w_mr_best, w_ic_best, stop=-float(sl) / 100.0, take=float(tp) / 100.0)
            rows.append({"stop_pct": float(sl), "take_pct": float(tp), **m})
    print(f"  PM sweep done: {len(rows)} combos in {time.time() - t0:.0f}s")
    return pd.DataFrame(rows), stop_grid, take_grid


# --------------------------------------------------------------------------- #
# 4. Plotting + leaderboard
# --------------------------------------------------------------------------- #
def plot_row(fig, axes_row, df, xcol, ycol, xvals, yvals, topk=10):
    """Fill one row of 7 metric heatmaps (one Axes per metric)."""
    xi = {round(float(v), 2): i for i, v in enumerate(xvals)}
    yi = {round(float(v), 2): i for i, v in enumerate(yvals)}
    ranked = df.sort_values("sharpe", ascending=False)
    top = ranked.head(topk)
    best = ranked.iloc[0]
    for ax, (col, label) in zip(axes_row, METRICS):
        mat = np.full((len(yvals), len(xvals)), np.nan)
        for _, r in df.iterrows():
            mat[yi[round(float(r[ycol]), 2)], xi[round(float(r[xcol]), 2)]] = r[col]
        cmap = "viridis_r" if col == "max_dd" else "viridis"
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap,
                       extent=[min(xvals), max(xvals), min(yvals), max(yvals)])
        ax.scatter(top[xcol], top[ycol], s=24, facecolors="none",
                   edgecolors="red", linewidths=1.1)
        ax.scatter([best[xcol]], [best[ycol]], marker="*", s=140,
                   color="red", edgecolors="white", linewidths=0.6)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(xcol, fontsize=8)
        ax.set_ylabel(ycol, fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def leaderboard(df, param_cols, out_csv):
    d = df.sort_values("sharpe", ascending=False).head(10).copy().reset_index(drop=True)
    d.index = d.index + 1
    out = pd.DataFrame(index=d.index)
    for c in param_cols:
        out[c] = d[c]
    out["entries"] = d["entries"].astype(int)
    out["cum_ret_%"] = (d["cum_ret"] * 100).round(2)
    out["sortino"] = d["sortino"].round(2)
    out["sharpe"] = d["sharpe"].round(2)
    out["max_dd_%"] = (d["max_dd"] * 100).round(2)
    out["win_days_%"] = (d["win_rate_days"] * 100).round(1)
    out.to_csv(out_csv)
    return out


# --------------------------------------------------------------------------- #
# 5. Main
# --------------------------------------------------------------------------- #
def main():
    md, val_idx, sector_pools, REG, QUAL = load_inputs()
    eng = build_engine(md, val_idx, sector_pools, REG, QUAL)
    run_combo = eng["run_combo"]

    # ---- SWEEP 1: Q-component weights (step 0.01, simplex) -----------------
    print("\n=== SWEEP 1: Q-component weights (VAL only) ===")
    q_df, q_grid = sweep_q_weights(run_combo, step=0.01)
    q_df.to_csv(CACHE_DIR / "sweep_qweights_val.csv", index=False)

    best_q = q_df.sort_values("sharpe", ascending=False).iloc[0]
    w_mr_best, w_ic_best = float(best_q["w_mr"]), float(best_q["w_ic"])
    w_stab_best = round(1.0 - w_mr_best - w_ic_best, 2)
    print(f"  best Q weights: w_mr={w_mr_best}  w_ic={w_ic_best}  w_stab={w_stab_best}  "
          f"(Sharpe={best_q['sharpe']}, cum={best_q['cum_ret']:.2%}, maxDD={best_q['max_dd']:.2%})")

    lb_q = leaderboard(q_df, ["w_mr", "w_ic", "w_stab"], CACHE_DIR / "leaderboard_qweights_val.csv")
    print("\n  TOP-10 Q-weight combos by VAL Sharpe "
          f"(fixed score={SCORE_FN}|universe={UNIVERSE}|stick={STICK}):")
    print(lb_q.to_string())

    # ---- SWEEP 2: PM stop/take at the best Q weights -----------------------
    print("\n=== SWEEP 2: PM stop_loss x take_profit (VAL only) ===")
    pm_df, stop_grid, take_grid = sweep_pm(run_combo, w_mr_best, w_ic_best)
    pm_df.to_csv(CACHE_DIR / "sweep_pm_stop_take_val.csv", index=False)

    best_pm = pm_df.sort_values("sharpe", ascending=False).iloc[0]
    print(f"  best PM: stop=-{best_pm['stop_pct']:.1f}%  take=+{best_pm['take_pct']:.1f}%  "
          f"(Sharpe={best_pm['sharpe']}, cum={best_pm['cum_ret']:.2%}, maxDD={best_pm['max_dd']:.2%})")

    lb_pm = leaderboard(pm_df, ["stop_pct", "take_pct"], CACHE_DIR / "leaderboard_pm_val.csv")
    print("\n  TOP-10 PM combos by VAL Sharpe "
          f"(fixed Q: w_mr={w_mr_best}|w_ic={w_ic_best}|w_stab={w_stab_best}):")
    print(lb_pm.to_string())

    # ---- SINGLE combined figure: 2 rows x 7 metric heatmaps ----------------
    fig, axes = plt.subplots(2, len(METRICS), figsize=(4.0 * len(METRICS), 9.0))
    plot_row(fig, axes[0], q_df, "w_mr", "w_ic", q_grid, q_grid)
    plot_row(fig, axes[1], pm_df, "stop_pct", "take_pct", stop_grid, take_grid)
    fig.subplots_adjust(left=0.035, right=0.99, top=0.88, bottom=0.07,
                        hspace=0.62, wspace=0.5)
    fig.text(0.5, 0.96,
             f"VAL sweeps | score={SCORE_FN}  universe={UNIVERSE}  stickiness={STICK} | "
             f"rz_entry={RZ_ENTRY}  rz_exit={RZ_EXIT}  max_hold={MAX_HOLD} | "
             "red O = top-10 Sharpe, * = best",
             ha="center", fontsize=11, weight="bold")
    fig.text(0.5, 0.915,
             f"ROW 1 - Q weights  (x=w_mr_hit, y=w_ic, w_stab=1-x-y)  |  "
             f"stop={STOP_LOSS_DEFAULT:.1%}  take={TAKE_PROFIT_DEFAULT:.1%}",
             ha="center", fontsize=9.5)
    fig.text(0.5, 0.455,
             f"ROW 2 - PM stop/take  (x=stop_loss %, y=take_profit %)  |  "
             f"Q weights: w_mr={w_mr_best}  w_ic={w_ic_best}  w_stab={w_stab_best}",
             ha="center", fontsize=9.5)
    fig_path = CACHE_DIR / "fig_sweeps_val.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"\n  saved combined figure -> {fig_path}")

    # ---- Final tuned configuration ----------------------------------------
    final = run_combo(w_mr_best, w_ic_best,
                      stop=-float(best_pm["stop_pct"]) / 100.0,
                      take=float(best_pm["take_pct"]) / 100.0)
    print("\n" + "=" * 70)
    print("FINAL GREEDY-TUNED CONFIG (VAL only):")
    print(f"  score_fn={SCORE_FN}  universe={UNIVERSE}  stickiness={STICK}")
    print(f"  Q weights : w_mr_hit={w_mr_best}  w_pred_ic={w_ic_best}  w_stab={w_stab_best}")
    print(f"  stop_loss=-{best_pm['stop_pct']:.1f}%   take_profit=+{best_pm['take_pct']:.1f}%")
    print(f"  VAL: Sharpe={final['sharpe']}  Sortino={final['sortino']}  "
          f"cum={final['cum_ret']:.2%}  maxDD={final['max_dd']:.2%}  entries={final['entries']}")
    print("=" * 70)
    print(f"\nAll artifacts written under: {(PROJECT_ROOT / CACHE_DIR).resolve()}")


if __name__ == "__main__":
    main()
