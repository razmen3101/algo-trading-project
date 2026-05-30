"""Central configuration for the stat-arb strategy.

A single frozen dataclass holds every knob in the system. ``config_hash()``
produces a short stable hash of the *settings that affect computed results* so
the cache can version artifacts and never silently reuse stale results when a
key parameter changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json


@dataclass(frozen=True)
class StrategyConfig:
    # ---- Data / universe ----
    start:        str = "2021-01-01"   # ~5y window (3 train + 1 val + 1 test)
    end:          str | None = None
    interval:     str = "1d"

    # ---- Chronological split (fractions of the full timeline) ----
    # 3y train / 1y val / 1y test  ->  3:1:1
    train_years:  float = 3.0
    val_years:    float = 1.0
    test_years:   float = 1.0

    # ---- Walk-forward / dynamic retraining ----
    retrain_every:    int = 50    # trading days between retrains
    min_train_days:   int = 252 * 3  # initial expanding-window length

    # ---- Target selection ----
    # weights for the composite target-suitability score (historical only)
    target_min_history:   int = 252 * 2
    target_score_weights: dict = field(default_factory=lambda: {
        "liquidity":      0.20,   # log dollar volume
        "etf_corr":       0.20,   # |corr| of returns with sector ETF
        "peer_corr":      0.15,   # avg |corr| with other sector stocks
        "mean_reversion": 0.25,   # residual stationarity (ADF) vs basket
        "vol_fit":        0.10,   # closeness to a target annualized vol band
        "history":        0.10,   # fraction of valid history
    })
    target_vol_band: tuple = (0.20, 0.45)  # preferred annualized vol range

    # ---- Predictor selection (LASSO / ElasticNet for FEATURE SELECTION only) ----
    predictor_selection_method: str = "elasticnet"  # "elasticnet" | "lasso" | "elasticnet_bic_hybrid"
    feature_selection_method: str = "elasticnet"   # "elasticnet" | "lasso"
    top_n_predictors:         int = 5
    elasticnet_l1_ratio:      float = 0.5
    selection_alphas:         int = 20    # alphas tried in the CV path

    # ---- Regressors (XGBoost) ----
    return_horizon: int = 1   # trading days for the return regressor label
    reg_params: dict = field(default_factory=lambda: dict(
        n_estimators=300, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.1, random_state=42, n_jobs=4,
    ))

    # ---- Residual / anomaly features ----
    residual_type:   str = "raw"   # "raw" | "percent" | "log"
    enable_multi_residual_engine: bool = False
    enable_return_feature_expansion: bool = False
    ewm_span:        int = 50
    resid_roll_win:  int = 50

    # ---- Technical features ----
    breakout_win:    int = 20
    vol_spike_win:   int = 20
    week52_win:      int = 252

    # ---- Labels ----
    label_horizon:        int = 1
    positive_threshold:   float = 0.01
    negative_threshold:   float = -0.01
    vol_adjusted_labels:  bool = False   # if True, thresholds scale by rolling vol

    # ---- Global classifier (XGBoost) ----
    clf_params: dict = field(default_factory=lambda: dict(
        n_estimators=400, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.1, objective="multi:softprob",
        random_state=42, n_jobs=4,
    ))

    # ---- Hyperparameter search ----
    use_random_search:  bool = False
    random_search_iter: int = 25

    # ---- Backtest ----
    confidence_threshold: float = 0.65
    transaction_cost_bps: float = 5.0
    min_residual_z:       float = 1.25    # skip trades with |residual_z| below this
    max_vol_filter:       float = 0.60   # skip trades when annualized vol above this
    require_agreement:    bool = True    # spread signal must agree with classifier
    size_by_confidence:   bool = True
    flat_probability_block: float = 0.40

    # ---- Position Manager (version 1, rule-based) ----
    use_position_manager: bool = False
    pm_entry_confidence: float = 0.65
    pm_entry_residual_z: float = 1.25
    pm_exit_residual_z: float = 0.25
    pm_opposite_confidence: float = 0.70
    pm_stop_loss: float = -0.02
    pm_take_profit: float = 0.03
    pm_max_holding_days: int = 10
    pm_allow_flip: bool = True

    # ---- Infra ----
    cache_dir:        str = ".cache"
    force_recompute:  bool = True
    # Opt-in: skip TLS verification when downloading (needed only on networks
    # with a corporate proxy / MITM whose root CA isn't in certifi's bundle).
    insecure_ssl:     bool = False
    make_plots:       bool = True
    plots_dir:        str = "reports"
    random_state:     int = 42
    price_transform:  str = "log_indexed"
    predictor_selection_input: str = "returns"

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self, *extra: str) -> str:
        """Stable short hash over settings that affect computed results.

        Infra-only knobs (cache_dir, force_recompute, make_plots, plots_dir)
        are excluded so toggling them does not invalidate the cache.
        ``extra`` lets a caller scope the hash to a step (e.g. a date)."""
        d = self.to_dict()
        for k in ("cache_dir", "force_recompute", "make_plots", "plots_dir", "insecure_ssl"):
            d.pop(k, None)
        blob = json.dumps(d, sort_keys=True, default=str) + "|".join(map(str, extra))
        return hashlib.sha1(blob.encode()).hexdigest()[:12]
