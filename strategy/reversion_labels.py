"""Reversion-based labels for the global signal classifier (OPTION C).

Standalone module — does NOT modify production make_labels / make_labels_from_fwd.
Used only by diagnostic notebooks for A/B evaluation of label schemes.

Definition (per candidate time series of residual_z):

    rz_now    = residual_z[t]
    rz_future = residual_z[t+h]
    delta     = rz_future - rz_now

    label = +1   if rz_now < -entry_band  AND  delta > +close_band   (cheap → reverted up)
    label = -1   if rz_now > +entry_band  AND  delta < -close_band   (expensive → reverted down)
    label =  0   otherwise
    label =  NaN if rz_future or rz_now is NaN

Aligned with the PositionManager:
    entry_band = 1.0   matches entry_residual_threshold
    close_band = 0.5   half-way toward mean_reversion_exit=0.30
    h          = 5     typical reversion horizon (vs. label_horizon=1 for raw fwd-return)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_reversion_labels(
    residual_z: pd.Series,
    h: int = 5,
    entry_band: float = 1.0,
    close_band: float = 0.5,
) -> pd.Series:
    """Build reversion labels for a single candidate's residual_z time series.

    Parameters
    ----------
    residual_z : pd.Series
        Date-indexed residual_z values for one candidate. Must be sorted by date.
    h : int
        Forward horizon (trading days) over which reversion is measured.
    entry_band : float
        Only candidates with |rz_now| > entry_band can receive a non-flat label.
    close_band : float
        Reversion magnitude required (in rz units) over h days.

    Returns
    -------
    pd.Series of {-1.0, 0.0, +1.0, NaN}, same index as input.
    """
    rz_now = residual_z.astype(float)
    rz_future = rz_now.shift(-h)
    delta = rz_future - rz_now

    lab = pd.Series(0.0, index=rz_now.index)
    long_mask = (rz_now < -entry_band) & (delta > close_band)
    short_mask = (rz_now > entry_band) & (delta < -close_band)
    lab[long_mask] = 1.0
    lab[short_mask] = -1.0
    lab[rz_now.isna() | rz_future.isna()] = np.nan
    return lab.rename("label_reversion")


def make_reversion_labels_on_panel(
    panel: pd.DataFrame,
    h: int = 5,
    entry_band: float = 1.0,
    close_band: float = 0.5,
    date_col: str = "date",
    cand_col: str = "target",
    rz_col: str = "residual_z",
) -> pd.Series:
    """Apply make_reversion_labels per candidate over a long-format panel.

    Returns a pd.Series aligned with the panel's row order.
    """
    if rz_col not in panel.columns:
        raise KeyError(f"panel missing '{rz_col}'")
    out = pd.Series(np.nan, index=panel.index, dtype=float)
    sorted_panel = panel.sort_values([cand_col, date_col])
    for cand, sub in sorted_panel.groupby(cand_col, sort=False):
        rz = sub.set_index(date_col)[rz_col]
        lab = make_reversion_labels(rz, h=h, entry_band=entry_band, close_band=close_band)
        # map back via the sub-frame's original index
        out.loc[sub.index] = lab.values
    return out.rename("label_reversion")
