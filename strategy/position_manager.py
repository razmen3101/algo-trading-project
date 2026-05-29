"""Rule-based PositionManager for trade lifecycle control.

Version 1 is intentionally deterministic and non-learned. It turns the
classifier's directional edge into an explicit trade lifecycle:

    ENTER -> HOLD -> EXIT -> FLIP

The manager operates on a chronological per-target panel and produces a
position series plus action diagnostics. It does not change the classifier;
it only decides whether the existing signal should become an open trade or
remain flat/held/closed.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PositionState:
    current_position: int = 0
    days_in_position: int = 0
    entry_residual_z: float | None = None
    entry_confidence: float | None = None
    trade_pnl: float = 0.0


@dataclass
class PositionDecision:
    action: str
    position: int
    reason: str


class PositionManager:
    def __init__(
        self,
        long_entry_confidence: float = 0.65,
        short_entry_confidence: float = 0.65,
        flat_probability_block: float = 0.40,
        entry_residual_threshold: float = 1.25,
        mean_reversion_exit: float = 0.25,
        opposite_signal_confidence: float = 0.70,
        stop_loss: float = -0.02,
        take_profit: float = 0.03,
        max_holding_days: int = 10,
        allow_flip: bool = True,
    ):
        self.long_entry_confidence = long_entry_confidence
        self.short_entry_confidence = short_entry_confidence
        self.flat_probability_block = flat_probability_block
        self.entry_residual_threshold = entry_residual_threshold
        self.mean_reversion_exit = mean_reversion_exit
        self.opposite_signal_confidence = opposite_signal_confidence
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_holding_days = max_holding_days
        self.allow_flip = allow_flip

    def _entry_long(self, row) -> bool:
        return (
            int(row.get("signal", 0)) == 1
            and float(row.get("P_long", 0.0)) >= self.long_entry_confidence
            and float(row.get("residual_z", np.nan)) <= -self.entry_residual_threshold
            and float(row.get("P_flat", 0.0)) < self.flat_probability_block
        )

    def _entry_short(self, row) -> bool:
        return (
            int(row.get("signal", 0)) == -1
            and float(row.get("P_short", 0.0)) >= self.short_entry_confidence
            and float(row.get("residual_z", np.nan)) >= self.entry_residual_threshold
            and float(row.get("P_flat", 0.0)) < self.flat_probability_block
        )

    def decide(self, row: pd.Series, state: PositionState) -> PositionDecision:
        signal = int(row.get("signal", 0) or 0)
        p_short = float(row.get("P_short", np.nan))
        p_flat = float(row.get("P_flat", np.nan))
        p_long = float(row.get("P_long", np.nan))
        residual_z = float(row.get("residual_z", np.nan))

        can_open_new = not np.isnan(p_flat) and p_flat < self.flat_probability_block

        if state.current_position == 0:
            if not can_open_new:
                return PositionDecision("HOLD_FLAT", 0, "flat_probability_block")
            if self._entry_long(row):
                return PositionDecision("ENTER_LONG", 1, "long_entry")
            if self._entry_short(row):
                return PositionDecision("ENTER_SHORT", -1, "short_entry")
            return PositionDecision("HOLD_FLAT", 0, "no_entry")

        if state.current_position == 1:
            mean_reversion_exit = residual_z >= -self.mean_reversion_exit
            opposite_signal_exit = signal == -1 and p_short >= self.opposite_signal_confidence
            if mean_reversion_exit:
                if self.allow_flip and can_open_new and opposite_signal_exit and residual_z >= self.entry_residual_threshold:
                    return PositionDecision("FLIP_LONG_TO_SHORT", -1, "mean_reversion_and_opposite_signal")
                return PositionDecision("EXIT", 0, "mean_reversion_exit")
            if state.trade_pnl <= self.stop_loss:
                return PositionDecision("EXIT", 0, "stop_loss")
            if state.trade_pnl >= self.take_profit:
                return PositionDecision("EXIT", 0, "take_profit")
            if state.days_in_position >= self.max_holding_days:
                return PositionDecision("EXIT", 0, "time_exit")
            if opposite_signal_exit:
                if self.allow_flip and can_open_new and residual_z >= self.entry_residual_threshold and self._entry_short(row):
                    return PositionDecision("FLIP_LONG_TO_SHORT", -1, "opposite_signal_flip")
                return PositionDecision("EXIT", 0, "opposite_signal_exit")
            return PositionDecision("HOLD_LONG", 1, "hold_long")

        if state.current_position == -1:
            mean_reversion_exit = residual_z <= self.mean_reversion_exit
            opposite_signal_exit = signal == 1 and p_long >= self.opposite_signal_confidence
            if mean_reversion_exit:
                if self.allow_flip and can_open_new and opposite_signal_exit and residual_z <= -self.entry_residual_threshold:
                    return PositionDecision("FLIP_SHORT_TO_LONG", 1, "mean_reversion_and_opposite_signal")
                return PositionDecision("EXIT", 0, "mean_reversion_exit")
            if state.trade_pnl <= self.stop_loss:
                return PositionDecision("EXIT", 0, "stop_loss")
            if state.trade_pnl >= self.take_profit:
                return PositionDecision("EXIT", 0, "take_profit")
            if state.days_in_position >= self.max_holding_days:
                return PositionDecision("EXIT", 0, "time_exit")
            if opposite_signal_exit:
                if self.allow_flip and can_open_new and residual_z <= -self.entry_residual_threshold and self._entry_long(row):
                    return PositionDecision("FLIP_SHORT_TO_LONG", 1, "opposite_signal_flip")
                return PositionDecision("EXIT", 0, "opposite_signal_exit")
            return PositionDecision("HOLD_SHORT", -1, "hold_short")

        return PositionDecision("HOLD_FLAT", 0, "invalid_state")

    def simulate(self, panel: pd.DataFrame, cost_bps: float = 5.0) -> pd.DataFrame:
        """Simulate the lifecycle row by row.

        The input panel should be chronological and must contain at least:
        signal, P_short, P_flat, P_long, residual_z, next_ret, target, date.
        """
        df = panel.sort_values(["target", "date"]).reset_index(drop=True).copy()
        cost = cost_bps / 1e4

        rows = []
        for target, g in df.groupby("target", sort=False):
            state = PositionState()
            prev_position = 0
            for _, row in g.iterrows():
                decision = self.decide(row, state)
                position = decision.position
                turnover = abs(position - prev_position)
                gross_pnl = position * float(row.get("next_ret", 0.0))
                net_pnl = gross_pnl - turnover * cost

                if decision.action in {"ENTER_LONG", "ENTER_SHORT", "FLIP_LONG_TO_SHORT", "FLIP_SHORT_TO_LONG"}:
                    entry_residual_z = float(row.get("residual_z", np.nan))
                    entry_confidence = float(row.get("P_long", np.nan)) if position == 1 else float(row.get("P_short", np.nan))
                    days_in_position = 1
                    trade_pnl = net_pnl
                elif position != 0:
                    entry_residual_z = state.entry_residual_z if state.entry_residual_z is not None else np.nan
                    entry_confidence = state.entry_confidence if state.entry_confidence is not None else np.nan
                    days_in_position = state.days_in_position
                    trade_pnl = state.trade_pnl
                else:
                    entry_residual_z = np.nan
                    entry_confidence = np.nan
                    days_in_position = 0
                    trade_pnl = 0.0

                out = row.to_dict()
                out.update({
                    "action": decision.action,
                    "position": float(position),
                    "turnover": float(turnover),
                    "gross_pnl": float(gross_pnl),
                    "net_pnl": float(net_pnl),
                    "prev_pos": float(prev_position),
                    "days_in_position": int(days_in_position),
                    "trade_pnl": float(trade_pnl),
                    "entry_residual_z": float(entry_residual_z),
                    "entry_confidence": float(entry_confidence),
                })
                rows.append(out)

                if position == 0:
                    state = PositionState()
                elif position == prev_position and prev_position != 0:
                    state.current_position = position
                    state.days_in_position += 1
                    state.trade_pnl += net_pnl
                else:
                    state = PositionState(
                        current_position=int(position),
                        days_in_position=1,
                        entry_residual_z=float(row.get("residual_z", np.nan)),
                        entry_confidence=float(row.get("P_long", np.nan)) if position == 1 else float(row.get("P_short", np.nan)),
                        trade_pnl=net_pnl,
                    )

                prev_position = position

        out = pd.DataFrame(rows)
        out["prev_pos"] = out.groupby("target")["position"].shift(1).fillna(0.0)
        return out


def summarize_completed_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a lifecycle panel into completed trade blocks."""
    rows = []
    for target, g in df.sort_values(["target", "date"]).groupby("target", sort=False):
        current_sign = 0
        start = None
        for i, (_, row) in enumerate(g.iterrows()):
            sign = int(np.sign(row["position"]))
            if sign == 0:
                if start is not None:
                    seg = g.iloc[start:i]
                    if len(seg):
                        rows.append(_trade_row(seg))
                start = None
                current_sign = 0
                continue
            if start is None:
                start = i
                current_sign = sign
                continue
            if sign != current_sign:
                seg = g.iloc[start:i]
                if len(seg):
                    rows.append(_trade_row(seg))
                start = i
                current_sign = sign
        if start is not None:
            seg = g.iloc[start:]
            if len(seg):
                rows.append(_trade_row(seg))
    return pd.DataFrame(rows)


def _trade_row(seg: pd.DataFrame) -> dict:
    return {
        "date": seg.iloc[0]["date"],
        "sector": seg.iloc[0]["sector"],
        "target": seg.iloc[0]["target"],
        "signal": int(seg.iloc[0]["signal"]),
        "entry_residual_z": float(seg.iloc[0].get("entry_residual_z", seg.iloc[0].get("residual_z", np.nan))),
        "entry_confidence": float(seg.iloc[0].get("entry_confidence", seg.iloc[0].get("confidence", np.nan))),
        "exit_date": seg.iloc[-1]["date"],
        "holding_period": int(len(seg)),
        "pnl": float(seg["net_pnl"].sum()),
    }
