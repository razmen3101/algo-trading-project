from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class BanditState:
    alpha: int = 1
    beta: int = 1


class BanditTargetSelector:
    """Simple Thompson-sampling bandit state manager per sector.

    This class only manages alpha/beta and sampling; the heavy scoring logic
    (residuals, ADF, feature construction) is done by the runner which uses
    this object to sample and update.
    """
    def __init__(self, cfg=None, seed: int | None = None):
        self.cfg = cfg
        self._states: dict[str, dict[str, BanditState]] = {}
        self._rnd = random.Random(seed if seed is not None else getattr(cfg, "random_state", 42))

    def init_sector(self, sector: str, candidates: list[str]) -> None:
        if sector not in self._states:
            self._states[sector] = {c: BanditState() for c in candidates}

    def sample_scores(self, sector: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for c, s in self._states.get(sector, {}).items():
            # sample from Beta(alpha, beta)
            out[c] = self._rnd.betavariate(s.alpha, s.beta)
        return out

    def get_state(self, sector: str, candidate: str) -> Tuple[int, int]:
        s = self._states.get(sector, {}).get(candidate)
        if s is None:
            return 1, 1
        return s.alpha, s.beta

    def update(self, sector: str, candidate: str, reward: float) -> Tuple[int, int, str]:
        """Update alpha/beta for a candidate based on realized reward.

        Returns (alpha_before, beta_before, which_updated)
        """
        s = self._states.setdefault(sector, {}).setdefault(candidate, BanditState())
        a0, b0 = s.alpha, s.beta
        updated = "none"
        if reward is None or not (reward == reward):
            return a0, b0, updated
        if reward > 0:
            s.alpha += 1
            updated = "alpha"
        elif reward < 0:
            s.beta += 1
            updated = "beta"
        else:
            # tiny neutral update to avoid complete stagnation
            pass
        return a0, b0, updated

