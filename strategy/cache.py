"""Config-hash-versioned cache.

Every expensive step calls :meth:`CacheManager.cached` (or the explicit
load/save pair). Artifacts are keyed by ``<name>__<config_hash>.<ext>`` so a
change to any result-affecting setting produces a fresh key and stale results
are never silently reused. ``force_recompute`` bypasses reads.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Any
import pickle

import pandas as pd


class CacheManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg.cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- key construction ---------------------------------------------- #
    def _key(self, name: str, *scope: str) -> str:
        h = self.cfg.config_hash(*scope)
        return f"{name}__{h}"

    def _path(self, name: str, ext: str, *scope: str) -> Path:
        return self.root / f"{self._key(name, *scope)}.{ext}"

    # ---- generic pickle ------------------------------------------------ #
    def exists(self, name: str, *scope: str, ext: str = "pkl") -> bool:
        return self._path(name, ext, *scope).exists()

    def load(self, name: str, *scope: str, ext: str = "pkl") -> Any:
        p = self._path(name, ext, *scope)
        if ext == "parquet":
            return pd.read_parquet(p)
        with open(p, "rb") as fh:
            return pickle.load(fh)

    def save(self, obj: Any, name: str, *scope: str, ext: str = "pkl") -> None:
        p = self._path(name, ext, *scope)
        if ext == "parquet":
            obj.to_parquet(p)
        else:
            with open(p, "wb") as fh:
                pickle.dump(obj, fh)

    # ---- the workhorse ------------------------------------------------- #
    def cached(self, name: str, builder: Callable[[], Any], *scope: str,
               ext: str = "pkl", verbose: bool = True) -> Any:
        """Return cached artifact or compute+store it.

        ``builder`` is only called on a miss or when ``force_recompute``."""
        if not self.cfg.force_recompute and self.exists(name, *scope, ext=ext):
            if verbose:
                print(f"[cache] HIT  {self._key(name, *scope)}.{ext}")
            return self.load(name, *scope, ext=ext)
        if verbose:
            reason = "force" if self.cfg.force_recompute else "miss"
            print(f"[cache] {reason.upper():4} {self._key(name, *scope)}.{ext} -> computing")
        obj = builder()
        self.save(obj, name, *scope, ext=ext)
        return obj

    def path_for(self, name: str, ext: str, *scope: str) -> Path:
        """Public path (used for plots / reports written by other writers)."""
        return self._path(name, ext, *scope)
