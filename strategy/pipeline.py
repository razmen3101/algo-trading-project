"""StrategyPipeline — end-to-end orchestration.

Flow:
  1. download + clean (existing Loader), universe = sector members + ETFs
  2. chronological 3/1/1 split; expanding-window walk-forward folds
  3. per sector x fold (history-only): target select -> predictor select ->
     shadow-price regressor -> return regressor  (all out-of-sample)
  4. residual/anomaly + technical + sector features over the OOS region
  5. one global XGBoost classifier (train+tune on pre-test, predict locked test)
  6. backtest on the locked test set, metrics, logs, plots

Look-ahead guards live in each component; this module only ever passes
history-up-to-t slices into selectors/regressors and keeps the test region
untouched until the final backtest.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from config import SECTORS
from data.loader import Loader
from strategy.strategy_config import StrategyConfig
from strategy.cache import CacheManager
from strategy.splits import chrono_split, walk_forward_folds
from strategy.target_selection import TargetSelectionEngine
from strategy.predictor_selector import PredictorSelector
from strategy.regressors import DynamicShadowPriceModel, DynamicReturnModel
from strategy.residual_features import ResidualFeatureBuilder
from strategy.technical_features import TechnicalRuleFeatureBuilder
from strategy.classifier import GlobalSignalClassifier
from strategy.backtester import Backtester
from sklearn.metrics import r2_score


def hr(title: str) -> None:
    print(f"\n{'='*78}\n  {title}\n{'='*78}")


class StrategyPipeline:
    def __init__(self, cfg: StrategyConfig | None = None):
        self.cfg = cfg or StrategyConfig()
        self.cache = CacheManager(self.cfg)
        self.log_rows: list[dict] = []
        self.selection_rows: list[dict] = []

    # ================================================================== #
    # 1. data
    # ================================================================== #
    def load_data(self):
        def _build():
            members = [t for c in SECTORS.values() for t in [c["target"]] + c["predictors"]]
            universe = sorted(set(members) | set(SECTORS.keys()))  # + sector ETFs
            session = self._maybe_insecure_session()
            ld = Loader(tickers=universe, start=self.cfg.start,
                        end=self.cfg.end, interval=self.cfg.interval, session=session)
            return ld.load()
        return self.cache.cached("market_data", _build, "data", verbose=True)

    def _maybe_insecure_session(self):
        """Build a TLS-verification-skipping session iff cfg.insecure_ssl.

        Only needed on networks with a TLS-intercepting proxy whose root CA is
        absent from certifi's bundle. Returns None otherwise (normal verified
        download)."""
        if not self.cfg.insecure_ssl:
            return None
        try:
            from curl_cffi import requests as creq
            print("[fetch] WARNING: insecure_ssl=True — skipping TLS verification")
            return creq.Session(verify=False)
        except Exception as e:
            print(f"[fetch] insecure session unavailable ({e}); using default")
            return None

    # ================================================================== #
    # 2-3. walk-forward OOS shadow price / return per sector
    # ================================================================== #
    def _sector_oos(self, etf, cfg_sector, md, folds, split):
        """Return per-date OOS frame for one sector with dynamic target/predictors."""
        members = [cfg_sector["target"]] + cfg_sector["predictors"]
        prices, returns, volumes = md.prices, md.returns, md.volumes
        etf_ret = returns[etf] if etf in returns else None

        tsel = TargetSelectionEngine(self.cfg, mode=self.cfg.target_selection_mode)
        psel = PredictorSelector(self.cfg)
        shadow_m = DynamicShadowPriceModel(self.cfg)
        return_m = DynamicReturnModel(self.cfg)

        records = []
        prev_target = None
        prev_predictors: set[str] | None = None
        for fold in folds:
            tr_idx, pr_idx = fold.train_idx, fold.predict_idx
            # validation slice = the val region inside the train window (for scoring)
            val_idx = tr_idx[(tr_idx >= split.train_end) & (tr_idx < split.val_end)]

            # ---- dynamic target (history only) ----
            # returns drops the first calendar date (pct_change), so reindex
            # the returns-based slices onto tr_idx (missing rows -> NaN, dropped).
            tc = tsel.select(etf, cfg_sector["name"], members,
                             prices, returns, volumes,
                             None if etf_ret is None else etf_ret,
                             train_idx=tr_idx, predict_idx=pr_idx, split=split)
            target = tc.target
            cands = [m for m in members if m != target]

            # ---- dynamic predictors (history only, LASSO/ElasticNet) ----
            preds = tc.selected_predictors or psel.select(target, cands, returns.reindex(tr_idx), prices.loc[tr_idx]).selected
            pc = psel.select(target, cands, returns.reindex(tr_idx), prices.loc[tr_idx])

            # ---- regressors (fit history, predict OOS block) ----
            sh = shadow_m.fit_predict(prices, target, preds, tr_idx, pr_idx, val_idx)
            rr = return_m.fit_predict(prices, target, preds, tr_idx, pr_idx, val_idx)

            pr_dates = pd.Index(pr_idx)
            sh_eval = pd.DataFrame({
                "actual": prices[target].reindex(pr_dates),
                "pred": sh.shadow_price.reindex(pr_dates),
            }).dropna()
            shadow_oos_r2 = float(r2_score(sh_eval["actual"], sh_eval["pred"])) if len(sh_eval) > 5 else float("nan")

            h = self.cfg.return_horizon
            fwd = prices[target].shift(-h) / prices[target] - 1.0
            rr_eval = pd.DataFrame({
                "actual": fwd.reindex(pr_dates),
                "pred": rr.predicted_return.reindex(pr_dates),
            }).dropna()
            return_oos_r2 = float(r2_score(rr_eval["actual"], rr_eval["pred"])) if len(rr_eval) > 5 else float("nan")

            cur_predictors = set(preds)
            predictor_jaccard = float("nan")
            predictor_turnover = float("nan")
            target_turnover = 0
            if prev_predictors is not None:
                union = prev_predictors | cur_predictors
                predictor_jaccard = len(prev_predictors & cur_predictors) / len(union) if union else float("nan")
                predictor_turnover = 1.0 - predictor_jaccard if np.isfinite(predictor_jaccard) else float("nan")
            if prev_target is not None:
                target_turnover = int(target != prev_target)

            self.log_rows.append(dict(
                sector=cfg_sector["name"], etf=etf, retrain_date=fold.retrain_date.date(),
                selector_mode=tc.mode,
                selected_target=target, selected_predictors=",".join(preds),
                selection_score=tc.selected_score,
                meta_prediction=tc.meta_prediction,
                shadow_val_r2=round(sh.val_r2, 4), return_val_r2=round(rr.val_r2, 4),
                shadow_oos_r2=round(shadow_oos_r2, 4) if np.isfinite(shadow_oos_r2) else np.nan,
                return_oos_r2=round(return_oos_r2, 4) if np.isfinite(return_oos_r2) else np.nan,
                target_turnover=target_turnover,
                predictor_jaccard=round(predictor_jaccard, 4) if np.isfinite(predictor_jaccard) else np.nan,
                predictor_turnover=round(predictor_turnover, 4) if np.isfinite(predictor_turnover) else np.nan,
                top_coefficients=",".join(f"{k}:{v:.3f}" for k, v in pc.coefficients.head(3).items()),
            ))

            prev_target = target
            prev_predictors = cur_predictors

            if isinstance(tc.scores, pd.DataFrame) and not tc.scores.empty:
                sel = tc.scores.copy()
                sel["sector"] = cfg_sector["name"]
                sel["etf"] = etf
                sel["retrain_date"] = fold.retrain_date.date()
                sel["selector_mode"] = tc.mode
                sel["selected_target"] = target
                self.selection_rows.extend(sel.to_dict("records"))

            for d in pr_idx:
                records.append(dict(date=d, etf=etf, sector=cfg_sector["name"],
                                    target=target,
                                    target_price=prices.at[d, target],
                                    shadow_price=sh.shadow_price.get(d, np.nan),
                                    predicted_return=rr.predicted_return.get(d, np.nan),
                                    predictors=tuple(preds)))
        return pd.DataFrame(records)

    # ================================================================== #
    # 4. feature assembly (residual + technical + sector)
    # ================================================================== #
    def _technical_all(self, md):
        def _build():
            tb = TechnicalRuleFeatureBuilder(self.cfg)
            return {t: tb.build(t, md.prices, md.highs, md.lows, md.volumes)
                    for t in md.prices.columns}
        return self.cache.cached("technical_all", _build, "tech", verbose=True)

    def _sector_features(self, md, oos: pd.DataFrame, etf: str) -> pd.DataFrame:
        """Continuous sector features (EWM-z normalized) for a sector OOS frame."""
        from strategy.residual_features import ewm_z
        prices, returns = md.prices, md.returns
        etf_ret = returns[etf] if etf in returns else pd.Series(0.0, index=returns.index)
        etf_px = prices[etf] if etf in prices else pd.Series(np.nan, index=prices.index)
        span = self.cfg.ewm_span

        rows = []
        for _, r in oos.iterrows():
            d, t, preds = r["date"], r["target"], r["predictors"]
            tr = returns[t]
            # rolling (shifted) relationships
            corr_etf = tr.rolling(60).corr(etf_ret).shift(1).get(d, np.nan)
            basket = returns[list(preds)].mean(axis=1) if preds else pd.Series(np.nan, index=returns.index)
            corr_pred = tr.rolling(60).corr(basket).shift(1).get(d, np.nan)
            vol_t = tr.rolling(20).std().shift(1)
            vol_e = etf_ret.rolling(20).std().shift(1)
            vol_rel = (vol_t / vol_e).get(d, np.nan)
            rows.append(dict(date=d,
                             target=t,
                             sect_etf_ret=etf_ret.get(d, np.nan),
                             sect_target_minus_etf=tr.get(d, np.nan) - etf_ret.get(d, np.nan),
                             sect_px_rel_etf=(prices[t] / etf_px).get(d, np.nan),
                             sect_vol_rel_etf=vol_rel,
                             sect_corr_etf=corr_etf,
                             sect_corr_pred=corr_pred))
        sf = pd.DataFrame(rows).set_index(["date", "target"])
        # EWM-z normalize the non-bounded continuous ones
        for c in ["sect_px_rel_etf", "sect_vol_rel_etf"]:
            _, _, z = ewm_z(sf[c], span)
            sf[c] = z
        return sf

    def build_panel(self, md, folds, split) -> pd.DataFrame:
        def _build():
            self.log_rows = []
            tech = self._technical_all(md)
            rfb = ResidualFeatureBuilder(self.cfg)
            sector_keys = list(SECTORS.keys())
            panels = []
            for etf, cfg_sector in SECTORS.items():
                hr(f"OOS modelling — {cfg_sector['name']} ({etf})")
                oos = self._sector_oos(etf, cfg_sector, md, folds, split)
                if oos.empty:
                    continue
                oos = oos.sort_values("date").reset_index(drop=True)

                # residual / anomaly features are normalized per target ticker so
                # the active target switch inside a sector never mixes scales.
                resid_parts = []
                label_parts = []
                for target, sub in oos.groupby("target", sort=False):
                    sub = sub.sort_values("date")
                    price = sub.set_index("date")["target_price"]
                    shadow = sub.set_index("date")["shadow_price"]
                    pret = sub.set_index("date")["predicted_return"]
                    fwd_sub = self._forward_return(md, sub, self.cfg.label_horizon)
                    resid_sub = rfb.build(price, shadow, pret.reindex(price.index), fwd_sub)
                    resid_sub["target"] = target
                    resid_parts.append(resid_sub)
                    label_parts.append(make_labels_from_fwd(fwd_sub, self.cfg).rename(target))
                resid = pd.concat(resid_parts).set_index("target", append=True).reorder_levels([0, 1]).sort_index()
                labels = pd.concat(label_parts, axis=1).stack().sort_index()
                resid_cols = ResidualFeatureBuilder.feature_columns(resid)

                # technical features per row (active target on that date)
                tech_rows = pd.DataFrame(
                    [tech[t].loc[d] if d in tech[t].index else pd.Series(dtype=float)
                     for d, t in zip(oos["date"], oos["target"])]
                ).set_index(pd.MultiIndex.from_arrays([oos["date"].values, oos["target"].values], names=["date", "target"]))

                # sector features
                sect = self._sector_features(md, oos, etf)

                df = oos.set_index(["date", "target"]).copy()
                # df already carries shadow_price (kept for plots); take the rest
                # of the residual features from the builder, incl. predicted_return.
                df = df.drop(columns=["predicted_return"])
                resid_join = [c for c in resid_cols if c != "shadow_price"]
                df = df.join(resid[resid_join]).join(tech_rows).join(sect)
                df["next_ret"] = self._next_return(md, oos).values
                df["ann_vol"] = self._trailing_vol(md, oos).values
                df["label"] = labels.reindex(df.index).values
                df["residual_z"] = resid["residual_ewm_z"]
                # spread signal: overpriced (resid_z>0) -> short, underpriced -> long
                df["spread_signal"] = -np.sign(df["residual_z"].fillna(0)).astype(int)
                # sector one-hot
                for k in sector_keys:
                    df[f"oh_{k}"] = int(etf == k)
                df["date"] = df.index.get_level_values(0)
                df["target"] = df.index.get_level_values(1)
                panels.append(df.reset_index(drop=True))
            panel = pd.concat(panels, ignore_index=True)
            return {"panel": panel, "retrain_log": pd.DataFrame(self.log_rows), "selection_log": pd.DataFrame(self.selection_rows)}
        obj = self.cache.cached("feature_panel", _build, "panel", verbose=True)
        if isinstance(obj, dict):
            self.log_rows = obj.get("retrain_log", pd.DataFrame()).to_dict("records")
            self.selection_rows = obj.get("selection_log", pd.DataFrame()).to_dict("records")
            return obj["panel"]
        self.log_rows = []
        self.selection_rows = []
        return obj

    # ---- per-ticker return lookups (active target varies by row) -------- #
    @staticmethod
    def _forward_return(md, oos, h):
        out = []
        for d, t in zip(oos["date"], oos["target"]):
            s = md.prices[t]
            fut = s.shift(-h)
            out.append(fut.get(d, np.nan) / s.get(d, np.nan) - 1.0)
        return pd.Series(out, index=oos["date"].values)

    @staticmethod
    def _next_return(md, oos):
        out = []
        for d, t in zip(oos["date"], oos["target"]):
            s = md.prices[t]
            out.append(s.shift(-1).get(d, np.nan) / s.get(d, np.nan) - 1.0)
        return pd.Series(out, index=oos["date"].values)

    @staticmethod
    def _trailing_vol(md, oos):
        """Annualized 20d realized vol of the active target, shifted(1)
        (observable at t) — feeds the backtest's high-volatility filter."""
        out = []
        for d, t in zip(oos["date"], oos["target"]):
            v = md.returns[t].rolling(20).std().shift(1) * np.sqrt(252)
            out.append(v.get(d, np.nan))
        return pd.Series(out, index=oos["date"].values)

    # ================================================================== #
    # 5. global classifier
    # ================================================================== #
    def train_classifier(self, panel, split):
        # exclude identifiers, raw price levels, labels, and backtest-only helpers
        # (ann_vol/residual_z drive trade filters; residual_ewm_z is the modelled
        # feature, so residual_z would just duplicate it).
        feature_cols = [c for c in panel.columns if c not in (
            "date", "etf", "sector", "target", "predictors", "target_price",
            "shadow_price", "next_ret", "label", "spread_signal",
            "ann_vol", "residual_z", "price_residual", "residual_ewm_mean",
            "residual_ewm_std", "residual_roll_mean", "residual_roll_std")]
        # keep only numeric features, drop rows without label
        data = panel.dropna(subset=["label"]).copy()
        X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
        # leakage-free fill: forward then zero (no future info used)
        X = X.groupby(data["target"]).ffill().fillna(0.0)
        data = data.assign(**{c: X[c] for c in feature_cols})

        is_test = data["date"] >= split.val_end
        dev, test = data[~is_test], data[is_test]      # dev = pre-test (yr4), test locked
        # chronological train/val split of dev for tuning
        cut = dev["date"].quantile(0.8)
        tr, val = dev[dev["date"] <= cut], dev[dev["date"] > cut]

        split_diagnostics = {
            "train_class_dist": tr["label"].value_counts(dropna=False).sort_index().to_dict(),
            "val_class_dist": val["label"].value_counts(dropna=False).sort_index().to_dict(),
            "test_class_dist": test["label"].value_counts(dropna=False).sort_index().to_dict(),
            "train_rows": int(len(tr)),
            "val_rows": int(len(val)),
            "test_rows": int(len(test)),
        }

        clf = GlobalSignalClassifier(self.cfg)

        def _fit():
            clf.fit(tr[feature_cols], tr["label"], val[feature_cols], val["label"])
            feature_group_map = {c: self._feature_group(c) for c in feature_cols}
            feature_group_counts = pd.Series(feature_group_map).value_counts().to_dict()
            return dict(params=clf.result_.params, val_metrics=clf.result_.val_metrics,
                        importance=clf.result_.feature_importance, features=feature_cols,
                        split_diagnostics=split_diagnostics,
                        feature_group_map=feature_group_map,
                        feature_group_counts=feature_group_counts)
        res = self.cache.cached("classifier", _fit, "clf", verbose=True)

        # rebuild a live model from cached params (model object itself also cacheable,
        # but params + a quick refit keeps the cache small and portable)
        if clf.model_ is None:
            clf.features_ = res["features"]
            clf.fit(tr[feature_cols], tr["label"], val[feature_cols], val["label"])

        return clf, test, feature_cols, res

    @staticmethod
    def _feature_group(col: str) -> str:
        if col.startswith("oh_") or col.startswith("sect_"):
            return "Sector"
        if col.startswith("predicted_return"):
            return "Return"
        if col.startswith(("residual_", "raw_", "percent_", "log_")) or col in {
            "price_residual", "price_residual_z", "shadow_price_gap_pct",
        }:
            return "Residual"
        return "Technical"

    # ================================================================== #
    # 6. backtest
    # ================================================================== #
    def backtest(self, clf, test, feature_cols):
        proba = clf.predict_proba(test)
        signal = clf.predict(test)
        panel = pd.DataFrame({
            "date": test["date"].values,
            "sector": test["sector"].values,
            "target": test["target"].values,
            "signal": signal.values,
            "pre_filter_signal": signal.values,
            "residual_z": test["residual_z"].values,
            "spread_signal": test["spread_signal"].values,
            "next_ret": test["next_ret"].values,
            "ann_vol": test["ann_vol"].values,
            "true_label": test["label"].values,
        })
        panel = pd.concat([panel.reset_index(drop=True), proba.reset_index(drop=True)], axis=1)
        panel = panel.dropna(subset=["next_ret"])
        bt = Backtester(self.cfg)
        return bt.run(panel)

    # ================================================================== #
    # orchestrate
    # ================================================================== #
    def run(self):
        hr("LOAD DATA")
        md = self.load_data()
        split = chrono_split(md.prices.index, self.cfg)
        folds = walk_forward_folds(md.prices.index, self.cfg)
        print(f"[split] {split.describe()}")
        print(f"[walk-forward] {len(folds)} folds, retrain every {self.cfg.retrain_every}d")

        panel = self.build_panel(md, folds, split)

        hr("RETRAIN LOG (per sector x fold)")
        log = pd.DataFrame(self.log_rows)
        if not log.empty:
            self.cache.save(log, "retrain_log", "log")
            print(log.to_string(index=False, max_rows=60))

        hr("GLOBAL CLASSIFIER")
        clf, test, feature_cols, res = self.train_classifier(panel, split)
        print(f"[clf] best params: {res['params']}")
        print(f"[clf] validation metrics: {res['val_metrics']}")

        hr("BACKTEST (locked test set)")
        bt = self.backtest(clf, test, feature_cols)
        self._print_results(bt, res, log)

        if self.cfg.make_plots:
            try:
                from strategy.plots import make_all_plots
                paths = make_all_plots(self.cfg, md, panel, bt, res, split)
                print(f"\n[plots] saved {len(paths)} figures to {self.cfg.plots_dir}/")
            except Exception as e:   # plotting is best-effort
                print(f"[plots] skipped: {e}")

        self.cache.save(bt.metrics, "backtest_metrics", "bt")
        return dict(market_data=md, panel=panel, classifier=clf, backtest=bt,
                    clf_result=res, split=split)

    def _print_results(self, bt, res, log):
        m = bt.metrics
        print("  Performance:")
        for k in ["cumulative_return", "annualized_return", "annualized_vol", "sharpe",
                  "max_drawdown", "win_rate", "n_trades", "avg_trade_return",
                  "n_long", "n_short", "n_flat", "buy_hold_cum"]:
            print(f"    {k:>20}: {m[k]:.4f}" if isinstance(m[k], float) else f"    {k:>20}: {m[k]}")
        print("\n  Sector performance:\n", bt.sector_perf.round(4).to_string())
        print("\n  Target performance:\n", bt.target_perf.round(4).to_string())
        print("\n  Confusion matrix:\n", bt.confusion.to_string())
        print("\n  Classification report:\n", bt.report)
        print("\n  Top feature importances:\n", res["importance"].head(15).round(4).to_string())
        if "split_diagnostics" in res:
            print("\n  Split diagnostics:\n", pd.DataFrame(res["split_diagnostics"]).to_string())

        print("\n  Signal distribution (pre-filter):\n", bt.trades["signal"].value_counts().sort_index().to_string())
        print("\n  Position distribution (post-filter):\n", bt.trades["position"].value_counts().sort_index().to_string())
        print("\n  Average confidence by class:\n", bt.trades.groupby("signal")["confidence"].mean().round(4).to_string())

        if log is not None and not log.empty:
            print("\n  Target selection counts:\n", log["selected_target"].value_counts().to_string())
            predictor_counts = pd.Series(
                [p for row in log["selected_predictors"].fillna("") for p in str(row).split(",") if p]
            ).value_counts()
            print("\n  Predictor selection counts:\n", predictor_counts.to_string())


def make_labels_from_fwd(fwd: pd.Series, cfg) -> pd.Series:
    """Threshold labels directly from a precomputed forward-return series."""
    pos, neg = cfg.positive_threshold, cfg.negative_threshold
    lab = pd.Series(0.0, index=fwd.index)
    lab[fwd > pos] = 1
    lab[fwd < neg] = -1
    lab[fwd.isna()] = np.nan
    return lab
