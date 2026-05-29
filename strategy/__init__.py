"""
Statistical-arbitrage trading system across 10 sectors.

The "brain" of the strategy, built on top of the existing project helpers
(config.SECTORS, data.loader.Loader, analysis.indicators.Indicators).

Pipeline (see strategy.pipeline.StrategyPipeline):
    download -> clean -> walk-forward {target select -> predictor select ->
    shadow-price regressor -> return regressor -> residual/anomaly features} ->
    technical + sector features -> global classifier -> backtest on locked test set.

Every feature, normalization, fit, selection and label uses only information
available at that point in time (no look-ahead bias). See README in module
docstrings for the specific guards.
"""
from strategy.strategy_config import StrategyConfig

__all__ = ["StrategyConfig"]
