"""Backtester — evaluate the strategy on the locked test set.

Input is a long panel (one row per date x sector) carrying the classifier
signal/probabilities, the mispricing (residual_z / spread signal), the target's
realized NEXT-day return, and the true label. The signal decided at the close of
day t earns the target return from t to t+1; position changes pay transaction
costs. Sectors are combined equal-weight into a daily portfolio return.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


@dataclass
class BacktestResult:
    portfolio:     pd.DataFrame   # daily portfolio return / equity
    trades:        pd.DataFrame   # per row positions & pnl
    metrics:       dict
    sector_perf:   pd.DataFrame
    target_perf:   pd.DataFrame
    confusion:     pd.DataFrame
    report:        str


class Backtester:
    def __init__(self, cfg):
        self.cfg = cfg

    def run_with_positions(self, panel: pd.DataFrame) -> BacktestResult:
        """Evaluate a panel that already contains the desired position series.

        This is useful for rule-based PositionManager experiments while keeping
        the baseline classifier-only path unchanged.
        """
        df = panel.copy()
        if "position" not in df:
            raise ValueError("run_with_positions() requires a 'position' column")

        if "confidence" not in df and {"P_short", "P_flat", "P_long"}.issubset(df.columns):
            df["confidence"] = np.where(
                df["position"] == 1, df["P_long"],
                np.where(df["position"] == -1, df["P_short"], df["P_flat"])
            )

        df = df.sort_values(["target", "date"]).reset_index(drop=True)
        cost = self.cfg.transaction_cost_bps / 1e4

        df["prev_pos"] = df.groupby("target")["position"].shift(1).fillna(0.0)
        df["turnover"] = (df["position"] - df["prev_pos"]).abs()
        df["gross_pnl"] = df["position"] * df["next_ret"]
        df["net_pnl"] = df["gross_pnl"] - df["turnover"] * cost

        port = df.groupby("date").agg(
            ret=("net_pnl", "mean"),
            gross=("gross_pnl", "mean"),
            n_active=("position", lambda s: int((s != 0).sum())),
        )
        port["equity"] = (1 + port["ret"]).cumprod()
        port["bh"] = (1 + df.groupby("date")["next_ret"].mean()).cumprod()

        metrics = self._metrics(port, df)
        sector_perf = self._group_perf(df, "sector")
        target_perf = self._group_perf(df, "target")
        confusion, report = self._classification(df)
        return BacktestResult(port, df, metrics, sector_perf, target_perf, confusion, report)

    # ---- position construction (applies all trade filters) ------------- #
    def _positions(self, panel: pd.DataFrame) -> pd.DataFrame:
        df = panel.copy()
        df["confidence"] = np.where(
            df["signal"] == 1, df["P_long"],
            np.where(df["signal"] == -1, df["P_short"], df["P_flat"])
        )

        raw = df["signal"].astype(float)

        keep = np.ones(len(df), dtype=bool)
        keep &= raw != 0
        keep &= df["confidence"].fillna(0).values >= self.cfg.confidence_threshold
        if "P_flat" in df:
            keep &= df["P_flat"].fillna(0).values < self.cfg.flat_probability_block
        if "residual_z" in df:
            keep &= df["residual_z"].abs().fillna(0).values >= self.cfg.min_residual_z
        if "ann_vol" in df:
            keep &= df["ann_vol"].fillna(0).values <= self.cfg.max_vol_filter
        if self.cfg.require_agreement and "spread_signal" in df:
            # spread signal must not contradict the classifier (0 spread is neutral)
            agree = (np.sign(raw) == np.sign(df["spread_signal"])) | (df["spread_signal"] == 0)
            keep &= agree.values

        size = df["confidence"].clip(lower=0).values if self.cfg.size_by_confidence else np.ones(len(df))
        df["position"] = np.where(keep, raw * size, 0.0)
        return df

    def run(self, panel: pd.DataFrame, position_manager=None) -> BacktestResult:
        if position_manager is None:
            df = self._positions(panel)
        else:
            df = position_manager.simulate(panel)
        df = df.sort_values(["target", "date"]).reset_index(drop=True)
        cost = self.cfg.transaction_cost_bps / 1e4

        # per-target position change cost (first appearance pays full entry)
        df["prev_pos"] = df.groupby("target")["position"].shift(1).fillna(0.0)
        df["turnover"] = (df["position"] - df["prev_pos"]).abs()
        df["gross_pnl"] = df["position"] * df["next_ret"]
        df["net_pnl"] = df["gross_pnl"] - df["turnover"] * cost

        # portfolio: equal-weight across active sectors each day
        port = df.groupby("date").agg(
            ret=("net_pnl", "mean"),
            gross=("gross_pnl", "mean"),
            n_active=("position", lambda s: int((s != 0).sum())),
        )
        port["equity"] = (1 + port["ret"]).cumprod()
        port["bh"] = (1 + df.groupby("date")["next_ret"].mean()).cumprod()  # buy&hold basket

        metrics = self._metrics(port, df)
        sector_perf = self._group_perf(df, "sector")
        target_perf = self._group_perf(df, "target")
        confusion, report = self._classification(df)
        return BacktestResult(port, df, metrics, sector_perf, target_perf, confusion, report)

    # ---- metrics ------------------------------------------------------- #
    def _metrics(self, port: pd.DataFrame, df: pd.DataFrame) -> dict:
        r = port["ret"].dropna()
        ann = 252
        cum = port["equity"].iloc[-1] - 1 if len(port) else 0.0
        ann_ret = (1 + cum) ** (ann / max(len(r), 1)) - 1 if len(r) else 0.0
        ann_vol = r.std() * np.sqrt(ann)
        sharpe = ann_ret / ann_vol if ann_vol else 0.0
        dd = (port["equity"] / port["equity"].cummax() - 1).min() if len(port) else 0.0
        traded = df[df["position"] != 0]
        wins = (traded["net_pnl"] > 0).mean() if len(traded) else 0.0
        sig = df["signal"]
        return {
            "cumulative_return": float(cum),
            "annualized_return": float(ann_ret),
            "annualized_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(dd),
            "win_rate": float(wins),
            "n_trades": int((df["turnover"] > 0).sum()),
            "avg_trade_return": float(traded["net_pnl"].mean()) if len(traded) else 0.0,
            "n_long": int((sig == 1).sum()),
            "n_short": int((sig == -1).sum()),
            "n_flat": int((sig == 0).sum()),
            "buy_hold_cum": float(port["bh"].iloc[-1] - 1) if len(port) else 0.0,
        }

    def _group_perf(self, df: pd.DataFrame, by: str) -> pd.DataFrame:
        g = df.groupby(by)
        out = pd.DataFrame({
            "net_pnl": g["net_pnl"].sum(),
            "trades": g["turnover"].apply(lambda s: int((s > 0).sum())),
            "win_rate": g.apply(lambda x: (x.loc[x["position"] != 0, "net_pnl"] > 0).mean()
                                if (x["position"] != 0).any() else np.nan, include_groups=False),
            "avg_ret": g.apply(lambda x: x.loc[x["position"] != 0, "net_pnl"].mean()
                               if (x["position"] != 0).any() else np.nan, include_groups=False),
        })
        return out.sort_values("net_pnl", ascending=False)

    def _classification(self, df: pd.DataFrame):
        if "true_label" not in df:
            return pd.DataFrame(), "(no labels available)"
        m = df.dropna(subset=["true_label", "signal"])
        labels = [-1, 0, 1]
        cm = confusion_matrix(m["true_label"], m["signal"], labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                             columns=[f"pred_{l}" for l in labels])
        rep = classification_report(m["true_label"], m["signal"], labels=labels,
                                    zero_division=0)
        return cm_df, rep
