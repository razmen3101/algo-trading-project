from __future__ import annotations

import os
import math
import pandas as pd
import numpy as np

from config import SECTORS
from data.loader import Loader
from strategy.pipeline import StrategyPipeline
from strategy.splits import chrono_split
from strategy.predictor_selector import PredictorSelector
from strategy.regressors import DynamicShadowPriceModel, DynamicReturnModel
from strategy.residual_features import ResidualFeatureBuilder
from strategy.technical_features import TechnicalRuleFeatureBuilder
from strategy.bandit_target_selector import BanditTargetSelector


def _safe_get_price(prices: pd.DataFrame, ticker: str, date):
    try:
        return float(prices.at[date, ticker])
    except Exception:
        return float("nan")


class DBTSRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline = StrategyPipeline(cfg)
        self.bandit = BanditTargetSelector(cfg)
        self.logs = []

    def run(self):
        print("[dbts] stage=load_data")
        md = self.pipeline.load_data()
        print("[dbts] stage=split")
        split = chrono_split(md.prices.index, self.cfg)
        # build panel and train classifier on dev (train+val)
        from strategy.splits import walk_forward_folds
        folds = walk_forward_folds(md.prices.index, self.cfg)
        print(f"[dbts] stage=train_panel folds={len(folds)}")
        panel = self.pipeline.build_panel(md, folds, split)
        print("[dbts] stage=train_classifier")
        clf, test_df, feature_cols, res = self.pipeline.train_classifier(panel, split)

        # prepare dev index (train+val)
        dev_idx = split.train_idx.union(split.val_idx)

        # fit per-sector per-candidate models on dev
        model_store = {}
        completed_candidates = 0
        for etf, cfg_sector in SECTORS.items():
            sector_name = cfg_sector["name"]
            members = [cfg_sector["target"]] + cfg_sector["predictors"]
            model_store[etf] = {}
            print(f"[dbts] stage=fit_models sector={sector_name} candidates={len(members)}")
            for cand in members:
                print(f"[dbts]   candidate={cand} status=fit_start")
                # predictor selection on dev
                peers = [m for m in members if m != cand and m in md.prices.columns]
                if not peers or cand not in md.prices.columns:
                    print(f"[dbts]   candidate={cand} status=skip_no_peers")
                    continue
                psel = PredictorSelector(self.cfg)
                pred_choice = psel.select(cand, peers, md.returns.reindex(dev_idx), md.prices.loc[dev_idx])
                preds = pred_choice.selected
                shadow_m = DynamicShadowPriceModel(self.cfg)
                shadow_feats, _, base_price, safe_idx = shadow_m.fit(md.prices, cand, preds, dev_idx)
                return_m = DynamicReturnModel(self.cfg)
                return_feats, _, _ = return_m.fit(md.prices, cand, preds, dev_idx)
                model_store[etf][cand] = dict(
                    predictors=preds,
                    shadow_model=shadow_m,
                    shadow_feats=shadow_feats,
                    base_price=base_price,
                    return_model=return_m,
                    return_feats=return_feats,
                )
                completed_candidates += 1
                print(f"[dbts]   candidate={cand} status=fit_done completed_candidates={completed_candidates}")
            # init bandit state
            self.bandit.init_sector(sector_name, members)

        # iterate test dates
        test_dates = list(split.test_idx)
        print(f"[dbts] stage=test_loop days={len(test_dates)}")
        tech_builder = TechnicalRuleFeatureBuilder(self.cfg)
        resid_builder = ResidualFeatureBuilder(self.cfg)

        trade_log_rows = []
        daily_scores_rows = []

        for date in test_dates:
            print(f"[dbts] date_block={date.date()} completed_candidates={completed_candidates}")
            for etf, cfg_sector in SECTORS.items():
                sector_name = cfg_sector["name"]
                members = [cfg_sector["target"]] + cfg_sector["predictors"]
                print(f"[dbts]   sector={sector_name} current_date={date.date()} current_candidate_set={len(members)}")
                # compute scores per candidate
                bandit_samples = self.bandit.sample_scores(sector_name)
                scores = {}
                residual_z_map = {}
                adf_p_map = {}
                for cand in members:
                    print(f"[dbts]     candidate={cand} stage=score_start")
                    rec = model_store.get(etf, {}).get(cand)
                    if rec is None:
                        scores[cand] = float("nan")
                        residual_z_map[cand] = float("nan")
                        adf_p_map[cand] = float("nan")
                        print(f"[dbts]     candidate={cand} status=missing_model")
                        continue
                    feats = rec["shadow_feats"]
                    predict_idx = feats.loc[:date].index if not feats.empty else pd.Index([])
                    if len(predict_idx) == 0:
                        scores[cand] = float("nan")
                        residual_z_map[cand] = float("nan")
                        adf_p_map[cand] = float("nan")
                        print(f"[dbts]     candidate={cand} status=no_predict_idx")
                        continue
                    shadow_pred = rec["shadow_model"].predict(feats, predict_idx, rec["base_price"])
                    price_hist = md.prices[cand].reindex(predict_idx)
                    # residual series up to date
                    resid_series = (price_hist - shadow_pred).dropna()
                    if resid_series.empty:
                        residual_z = float("nan")
                    else:
                        rf = resid_builder.build(price_hist, shadow_pred, rec["return_model"].predict(rec["return_feats"], predict_idx))
                        residual_z = float(rf["residual_ewm_z"].get(date, float("nan")))
                    residual_z_map[cand] = residual_z
                    residual_score = min(abs(residual_z) / 3.0, 1.0) if (residual_z == residual_z) else 0.0
                    # rolling ADF p-value
                    try:
                        from statsmodels.tsa.stattools import adfuller
                        if len(resid_series) >= 20:
                            adf_p = float(adfuller(resid_series, autolag="AIC")[1])
                        else:
                            adf_p = float("nan")
                    except Exception:
                        adf_p = float("nan")
                    adf_p_map[cand] = adf_p
                    adf_score = 1.0 - min(adf_p, 1.0) if adf_p == adf_p else 0.0
                    bandit_score = bandit_samples.get(cand, 0.5)
                    final_score = 0.5 * bandit_score + 0.3 * residual_score + 0.2 * adf_score
                    scores[cand] = final_score
                    print(f"[dbts]     candidate={cand} score_done final={final_score:.4f}")

                # record daily scores
                for cand in members:
                    daily_scores_rows.append({
                        "date": date, "sector": sector_name, "candidate": cand,
                        "bandit_score": bandit_samples.get(cand, float("nan")),
                        "residual_z": residual_z_map.get(cand, float("nan")),
                        "residual_score": min(abs(residual_z_map.get(cand, 0.0)) / 3.0, 1.0) if residual_z_map.get(cand, cand) == residual_z_map.get(cand, cand) else float("nan"),
                        "adf_pvalue": adf_p_map.get(cand, float("nan")),
                        "adf_score": 1.0 - min(adf_p_map.get(cand, 1.0), 1.0) if adf_p_map.get(cand, adf_p_map.get(cand)) == adf_p_map.get(cand, adf_p_map.get(cand)) else float("nan"),
                        "final_score": scores.get(cand, float("nan")),
                    })

                # select best
                best = max(scores.items(), key=lambda kv: (kv[1] if kv[1] == kv[1] else -1.0))
                selected = best[0]
                print(f"[dbts]   sector={sector_name} selected={selected}")

                # build a simple feature row for classifier prediction (best-effort)
                feat_row = {}
                # technical features
                try:
                    tech = tech_builder.build(selected, md.prices, md.highs, md.lows, md.volumes)
                    tech_row = tech.loc[date] if date in tech.index else pd.Series(dtype=float)
                    for c, v in tech_row.items():
                        feat_row[c] = float(v) if (v == v) else 0.0
                except Exception:
                    pass
                # residual features at date
                rec = model_store.get(etf, {}).get(selected)
                if rec is not None:
                    feats = rec["shadow_feats"]
                    predict_idx = feats.loc[:date].index if not feats.empty else pd.Index([])
                    shadow_pred = rec["shadow_model"].predict(feats, predict_idx, rec["base_price"]) if len(predict_idx) else pd.Series(dtype=float)
                    pred_ret = rec["return_model"].predict(rec["return_feats"], predict_idx) if len(predict_idx) else pd.Series(dtype=float)
                    rf = resid_builder.build(md.prices[selected].reindex(predict_idx), shadow_pred, pred_ret)
                    if date in rf.index:
                        for c in rf.columns:
                            feat_row[c] = float(rf.at[date, c]) if (rf.at[date, c] == rf.at[date, c]) else 0.0

                # assemble DataFrame for classifier features
                X_row = pd.DataFrame([ {c: feat_row.get(c, 0.0) for c in clf.features_} ], index=[date])
                proba = clf.predict_proba(X_row)
                signal = clf.predict(X_row).iloc[0]
                p_short, p_flat, p_long = proba.iloc[0].tolist()

                # execute simple trade: entry at date close, exit at date+h close
                h = self.cfg.label_horizon
                dates_idx = list(md.prices.index)
                try:
                    i = dates_idx.index(date)
                    exit_i = i + h
                    if exit_i < len(dates_idx):
                        exit_date = dates_idx[exit_i]
                        entry_price = _safe_get_price(md.prices, selected, date)
                        exit_price = _safe_get_price(md.prices, selected, exit_date)
                        if math.isfinite(entry_price) and math.isfinite(exit_price) and entry_price > 0:
                            realized = exit_price / entry_price - 1.0
                        else:
                            realized = float("nan")
                    else:
                        exit_date = None
                        realized = float("nan")
                except Exception:
                    exit_date = None
                    realized = float("nan")

                # update bandit
                alpha_before, beta_before, updated = self.bandit.get_state(sector_name, selected)[0], self.bandit.get_state(sector_name, selected)[1], "none"
                a0, b0, which = self.bandit.update(sector_name, selected, realized)
                alpha_after, beta_after = self.bandit.get_state(sector_name, selected)
                print(f"[dbts]   sector={sector_name} update={which} alpha={alpha_after} beta={beta_after} realized={realized}")

                trade_log_rows.append({
                    "date": date, "sector": sector_name, "selected_target": selected,
                    "bandit_score": bandit_samples.get(selected, float("nan")),
                    "residual_z": residual_z_map.get(selected, float("nan")),
                    "adf_pvalue": adf_p_map.get(selected, float("nan")),
                    "final_target_score": scores.get(selected, float("nan")),
                    "signal": int(signal), "P_short": p_short, "P_flat": p_flat, "P_long": p_long,
                    "entry_date": date, "exit_date": exit_date, "entry_price": entry_price if 'entry_price' in locals() else float("nan"),
                    "exit_price": exit_price if 'exit_price' in locals() else float("nan"), "realized_return": realized,
                    "alpha_before": a0, "beta_before": b0, "alpha_after": alpha_after, "beta_after": beta_after,
                    "bandit_updated": which,
                })

        # save outputs
        out_dir = "outputs"
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(trade_log_rows).to_csv(os.path.join(out_dir, "dbts_test_trade_log.csv"), index=False)
        pd.DataFrame(daily_scores_rows).to_csv(os.path.join(out_dir, "dbts_daily_target_scores.csv"), index=False)
        invalid_price_rows = int((~np.isfinite(md.prices) | (md.prices <= 0)).sum().sum())
        invalid_log_rows = int((~np.isfinite(pd.DataFrame(trade_log_rows)[[c for c in ["entry_price","exit_price","realized_return"] if c in pd.DataFrame(trade_log_rows).columns]])).sum().sum()) if trade_log_rows else 0
        print(f"DBTS run complete. Saved dbts_test_trade_log.csv and dbts_daily_target_scores.csv in outputs/")
        print(f"[dbts] invalid_price_rows={invalid_price_rows} invalid_log_rows={invalid_log_rows}")


def main():
    from strategy.strategy_config import StrategyConfig
    cfg = StrategyConfig()
    r = DBTSRunner(cfg)
    r.run()


if __name__ == '__main__':
    main()
