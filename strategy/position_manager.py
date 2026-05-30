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
    trade_id: int | None = None


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
                    return PositionDecision("FLIP_LONG_TO_SHORT", -1, "mean_reversion_exit")
                return PositionDecision("EXIT", 0, "mean_reversion_exit")
            if state.trade_pnl <= self.stop_loss:
                return PositionDecision("EXIT", 0, "stop_loss_exit")
            if state.trade_pnl >= self.take_profit:
                return PositionDecision("EXIT", 0, "take_profit_exit")
            if state.days_in_position >= self.max_holding_days:
                return PositionDecision("EXIT", 0, "time_exit")
            if opposite_signal_exit:
                if self.allow_flip and can_open_new and residual_z >= self.entry_residual_threshold and self._entry_short(row):
                    return PositionDecision("FLIP_LONG_TO_SHORT", -1, "opposite_signal_exit")
                return PositionDecision("EXIT", 0, "opposite_signal_exit")
            return PositionDecision("HOLD_LONG", 1, "hold_long")

        if state.current_position == -1:
            mean_reversion_exit = residual_z <= self.mean_reversion_exit
            opposite_signal_exit = signal == 1 and p_long >= self.opposite_signal_confidence
            if mean_reversion_exit:
                if self.allow_flip and can_open_new and opposite_signal_exit and residual_z <= -self.entry_residual_threshold:
                    return PositionDecision("FLIP_SHORT_TO_LONG", 1, "mean_reversion_exit")
                return PositionDecision("EXIT", 0, "mean_reversion_exit")
            if state.trade_pnl <= self.stop_loss:
                return PositionDecision("EXIT", 0, "stop_loss_exit")
            if state.trade_pnl >= self.take_profit:
                return PositionDecision("EXIT", 0, "take_profit_exit")
            if state.days_in_position >= self.max_holding_days:
                return PositionDecision("EXIT", 0, "time_exit")
            if opposite_signal_exit:
                if self.allow_flip and can_open_new and residual_z <= -self.entry_residual_threshold and self._entry_long(row):
                    return PositionDecision("FLIP_SHORT_TO_LONG", 1, "opposite_signal_exit")
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
            next_trade_id = 0
            for _, row in g.iterrows():
                decision = self.decide(row, state)
                position = decision.position
                turnover = abs(position - prev_position)
                gross_pnl = position * float(row.get("next_ret", 0.0))
                net_pnl = gross_pnl - turnover * cost
                is_entry = decision.action in {"ENTER_LONG", "ENTER_SHORT", "FLIP_LONG_TO_SHORT", "FLIP_SHORT_TO_LONG"}
                is_exit = decision.action in {"EXIT", "FLIP_LONG_TO_SHORT", "FLIP_SHORT_TO_LONG"}
                closed_trade_id = state.trade_id if is_exit and state.trade_id is not None else np.nan

                if is_entry:
                    next_trade_id += 1
                    trade_id = next_trade_id
                    entry_residual_z = float(row.get("residual_z", np.nan))
                    entry_confidence = float(row.get("P_long", np.nan)) if position == 1 else float(row.get("P_short", np.nan))
                    days_in_position = 1
                    trade_pnl = net_pnl
                elif position != 0:
                    trade_id = state.trade_id if state.trade_id is not None else np.nan
                    entry_residual_z = state.entry_residual_z if state.entry_residual_z is not None else np.nan
                    entry_confidence = state.entry_confidence if state.entry_confidence is not None else np.nan
                    days_in_position = state.days_in_position
                    trade_pnl = state.trade_pnl
                else:
                    trade_id = state.trade_id if is_exit and state.trade_id is not None else np.nan
                    entry_residual_z = np.nan
                    entry_confidence = np.nan
                    days_in_position = 0
                    trade_pnl = state.trade_pnl if is_exit else 0.0

                out = row.to_dict()
                out.update({
                    "action": decision.action,
                    "action_reason": decision.reason,
                    "position": float(position),
                    "turnover": float(turnover),
                    "gross_pnl": float(gross_pnl),
                    "net_pnl": float(net_pnl),
                    "prev_pos": float(prev_position),
                    "trade_id": None if pd.isna(trade_id) else int(trade_id),
                    "entry_id": None if pd.isna(trade_id) else int(trade_id),
                    "closed_trade_id": None if pd.isna(closed_trade_id) else int(closed_trade_id),
                    "is_entry": bool(is_entry),
                    "is_exit": bool(is_exit),
                    "days_in_position": int(days_in_position),
                    "trade_pnl": float(trade_pnl),
                    "entry_residual_z": float(entry_residual_z),
                    "entry_confidence": float(entry_confidence),
                })
                rows.append(out)

                if is_entry:
                    state = PositionState(
                        current_position=int(position),
                        days_in_position=1,
                        entry_residual_z=float(row.get("residual_z", np.nan)),
                        entry_confidence=float(row.get("P_long", np.nan)) if position == 1 else float(row.get("P_short", np.nan)),
                        trade_pnl=net_pnl,
                        trade_id=next_trade_id,
                    )
                elif position == 0:
                    state = PositionState()
                elif position == prev_position and prev_position != 0:
                    state.current_position = position
                    state.days_in_position += 1
                    state.trade_pnl += net_pnl

                prev_position = position

        out = pd.DataFrame(rows)
        out["prev_pos"] = out.groupby("target")["position"].shift(1).fillna(0.0)
        return out


def summarize_completed_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a lifecycle panel into completed trade blocks."""
    rows = []
    for target, g in df.sort_values(["target", "date"]).groupby("target", sort=False):
        g = g.reset_index(drop=True)
        entry_rows = g.index[g.get("is_entry", False)].tolist() if "is_entry" in g.columns else []
        if not entry_rows:
            entry_rows = [i for i, (_, row) in enumerate(g.iterrows()) if int(np.sign(row["position"])) != 0 and (i == 0 or int(np.sign(g.iloc[i - 1]["position"])) == 0)]

        for start in entry_rows:
            entry_row = g.iloc[start]
            trade_id = entry_row.get("trade_id", entry_row.get("entry_id", None))
            if pd.isna(trade_id):
                continue
            exit_matches = g.index[(g.get("closed_trade_id", pd.Series(index=g.index, dtype=float)) == trade_id)].tolist() if "closed_trade_id" in g.columns else []
            exit_idx = exit_matches[0] if exit_matches else None
            seg = g.iloc[start:exit_idx] if exit_idx is not None else g.iloc[start:]
            if seg.empty:
                continue
            exit_row = g.iloc[exit_idx] if exit_idx is not None else seg.iloc[-1]
            exit_pos = int(np.sign(exit_row["position"]))
            exit_cost = float(exit_row["net_pnl"]) if exit_pos == 0 and exit_idx is not None else 0.0
            direction = "long" if int(np.sign(entry_row["position"])) > 0 else "short"
            rows.append({
                "entry_date": entry_row["date"],
                "exit_date": exit_row["date"] if exit_idx is not None else seg.iloc[-1]["date"],
                "sector": entry_row["sector"],
                "target": entry_row["target"],
                "direction": direction,
                "entry_price": float(entry_row.get("target_price", np.nan)),
                "exit_price": float(exit_row.get("target_price", seg.iloc[-1].get("target_price", np.nan))),
                "holding_period": int(len(seg)),
                "pnl": float(seg["net_pnl"].sum() + exit_cost),
                "entry_residual_z": float(entry_row.get("entry_residual_z", entry_row.get("residual_z", np.nan))),
                "exit_residual_z": float(exit_row.get("residual_z", np.nan)),
                "entry_confidence": float(entry_row.get("entry_confidence", np.nan)),
                "exit_reason": _normalize_exit_reason(exit_row.get("action_reason", "period_end") if exit_idx is not None else "period_end"),
                "trade_id": None if pd.isna(trade_id) else int(trade_id),
                "entry_id": None if pd.isna(trade_id) else int(trade_id),
            })
    return pd.DataFrame(rows)


def _normalize_exit_reason(reason: str) -> str:
    mapping = {
        "stop_loss": "stop_loss_exit",
        "take_profit": "take_profit_exit",
        "mean_reversion_and_opposite_signal": "mean_reversion_exit",
        "opposite_signal_flip": "opposite_signal_exit",
        "hold_long": "opposite_signal_exit",
        "hold_short": "opposite_signal_exit",
    }
    return mapping.get(reason, reason)
