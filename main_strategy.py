"""Entry point for the statistical-arbitrage strategy.

    python main_strategy.py                # run with default config
    python main_strategy.py --recompute    # ignore cache, recompute everything
    python main_strategy.py --no-plots     # skip figures
    python main_strategy.py --random-search# tune classifier on the validation set

Also importable from a notebook:

    from strategy.pipeline import StrategyPipeline
    from strategy.strategy_config import StrategyConfig
    out = StrategyPipeline(StrategyConfig()).run()
"""
from __future__ import annotations

import argparse
import dataclasses
import sys

from strategy.strategy_config import StrategyConfig
from strategy.pipeline import StrategyPipeline


def parse_args(argv=None) -> StrategyConfig:
    p = argparse.ArgumentParser(description="Stat-arb strategy pipeline")
    p.add_argument("--recompute", action="store_true", help="bypass cache")
    p.add_argument("--no-plots", action="store_true", help="skip plotting")
    p.add_argument("--random-search", action="store_true", help="tune classifier on val")
    p.add_argument("--selection", choices=["elasticnet", "lasso"], default=None)
    p.add_argument("--target-selection-mode", choices=["legacy", "tradability_score", "meta_target"], default=None)
    p.add_argument("--top-n", type=int, default=None, help="top N predictors")
    p.add_argument("--start", default=None, help="data start date (YYYY-MM-DD)")
    p.add_argument("--insecure-ssl", action="store_true",
                   help="skip TLS verification on download (corporate proxy/MITM only)")
    args = p.parse_args(argv)

    overrides = {}
    if args.recompute:      overrides["force_recompute"] = True
    if args.no_plots:       overrides["make_plots"] = False
    if args.random_search:  overrides["use_random_search"] = True
    if args.selection:      overrides["feature_selection_method"] = args.selection
    if args.target_selection_mode: overrides["target_selection_mode"] = args.target_selection_mode
    if args.top_n:          overrides["top_n_predictors"] = args.top_n
    if args.start:          overrides["start"] = args.start
    if args.insecure_ssl:   overrides["insecure_ssl"] = True
    return dataclasses.replace(StrategyConfig(), **overrides)


def main(argv=None):
    cfg = parse_args(argv)
    print(f"[config] hash={cfg.config_hash()}  selection={cfg.feature_selection_method}  "
            f"top_n={cfg.top_n_predictors}  target_mode={cfg.target_selection_mode}  random_search={cfg.use_random_search}  "
          f"force_recompute={cfg.force_recompute}")
    StrategyPipeline(cfg).run()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
