"""
Classic (fixed) strategies for the Public Goods Game.

These never learn (`training_mode="fixed"`); they are baselines and social "characters" to mix into
a match. They interpret the observation index as the previous round's cooperator count k_prev
(with the game's start index meaning "first round").

Included: AllC, AllD, Random(p), Majority-TFT (cooperate iff at least half of the players
cooperated last round).
"""

from __future__ import annotations

import numpy as np

from ..engine.agent import Agent
from ..games.public_goods import COOPERATE, DEFECT


class _Classic(Agent):
    kind = "classic"
    training_mode = "fixed"
    strategy = "classic"
    rule = ""

    def __init__(self, name: str) -> None:
        self.name = name

    def render_brain(self) -> dict:
        return {"kind": self.kind, "strategy": self.strategy, "rule": self.rule}


class AllC(_Classic):
    strategy = "AllC"
    rule = "Always cooperate."

    def act(self, observation: int) -> int:
        return COOPERATE


class AllD(_Classic):
    strategy = "AllD"
    rule = "Always defect."

    def act(self, observation: int) -> int:
        return DEFECT


class RandomAgent(_Classic):
    strategy = "Random"

    def __init__(self, name: str, p_cooperate: float = 0.5, seed: int = 0) -> None:
        super().__init__(name)
        self.p = p_cooperate
        self.rule = f"Cooperate with probability {p_cooperate:g}."
        self.rng = np.random.default_rng(seed)

    def act(self, observation: int) -> int:
        return COOPERATE if self.rng.random() < self.p else DEFECT

    def render_brain(self) -> dict:
        d = super().render_brain()
        d["p_cooperate"] = self.p
        return d


class MajorityTFT(_Classic):
    """Cooperate iff at least half the players cooperated last round; cooperate on round 1."""

    strategy = "Majority-TFT"

    def __init__(self, name: str, n_agents: int) -> None:
        super().__init__(name)
        self.n_agents = n_agents
        self.start_state = n_agents + 1
        self.threshold = n_agents / 2.0
        self.rule = f"Cooperate iff >= {self.threshold:g} players cooperated last round."

    def act(self, observation: int) -> int:
        if observation == self.start_state:      # first round
            return COOPERATE
        k_prev = observation                     # observation index == cooperator count
        return COOPERATE if k_prev >= self.threshold else DEFECT
