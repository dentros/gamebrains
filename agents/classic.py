"""
Classic (fixed) strategies, defined by meaning rather than by action number.

These never learn (`training_mode="fixed"`); they are baselines and social "characters" to mix
into a match. Included: AllC, AllD, Random(p), Majority-TFT.

**Why these ask the game instead of hardcoding an index.** They used to import `COOPERATE` and
`DEFECT` from the Public Goods Game, which pinned "cooperate" to action 1 forever. That is right
in a social dilemma and backwards in a congestion game, where action 1 is grabbing the contested
resource and the restrained act is holding back. So `AllC` would have played the most aggressive
strategy available while still calling itself cooperative. Each agent now resolves its action
through `game.action_for(...)` at `on_match_start`, so it means the same thing everywhere and
declines games that cannot give it that meaning. See `engine/game.py`'s module docstring.

**A caveat worth carrying into any anti-coordination game.** There, no constant strategy is
collectively good: if everyone concedes the payoff is zero, exactly as it is if everyone claims.
The collectively best behaviour is *taking turns*, which no fixed strategy can express. AllC and
AllD remain useful reference points in such a game, but neither is "the cooperative agent" there.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..engine.agent import Agent

# No import of any game here, deliberately. This module briefly kept re-exporting the Public Goods
# Game's COOPERATE/DEFECT constants for callers that had imported them from here, which left an
# agents -> games edge behind after the strategies themselves had stopped using them. Every caller
# takes them from `games.public_goods` directly, so the re-export was dead weight holding up a
# dependency the architecture says should not exist.


class _Classic(Agent):
    kind = "classic"
    training_mode = "fixed"
    # Every strategy here is defined by meaning ("always give way"), never by number,
    # so each one resolves its role against the game before play. This is the family
    # the original defect lived in.
    semantics = "role-bound"
    strategy = "classic"
    rule = ""

    def __init__(self, name: str) -> None:
        self.name = name

    def _unbound(self) -> str:
        return (f"{self.strategy} has not been matched to a game yet, so it does not know which "
                f"action it should play. The runner calls on_match_start before play begins; a "
                f"caller driving agents directly has to do the same.")

    def render_brain(self) -> dict:
        return {"kind": self.kind, "strategy": self.strategy, "rule": self.rule}


class _ConstantRole(_Classic):
    """Always plays one named role, whichever action happens to fill it in this game."""

    role = "concede"

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._action: int | None = None

    def on_match_start(self, game: Any) -> None:
        self._action = game.action_for(self.role)

    def act(self, observation: int) -> int:
        if self._action is None:
            raise RuntimeError(self._unbound())
        return self._action


class AllC(_ConstantRole):
    strategy = "AllC"
    role = "concede"
    rule = "Always take the conceding action (cooperate; give way in a congestion game)."


class AllD(_ConstantRole):
    strategy = "AllD"
    role = "claim"
    rule = "Always take the claiming action (defect; go for the resource in a congestion game)."


class RandomAgent(_Classic):
    """Uniform-ish mixer, and the reference baseline the coordination score is measured against.

    Its own seed, distinct per agent: a roster of random agents sharing one seed would move in
    lockstep and collide every single round, which is the opposite of a random baseline.
    """

    strategy = "Random"

    def __init__(self, name: str, p_cooperate: float = 0.5, seed: int = 0) -> None:
        super().__init__(name)
        self.p = p_cooperate
        self.rule = f"Take the conceding action with probability {p_cooperate:g}."
        self.rng = np.random.default_rng(seed)
        self._concede: int | None = None
        self._claim: int | None = None

    def on_match_start(self, game: Any) -> None:
        self._concede = game.action_for("concede")
        self._claim = game.action_for("claim")

    def act(self, observation: int) -> int:
        if self._concede is None:
            raise RuntimeError(self._unbound())
        return self._concede if self.rng.random() < self.p else self._claim

    def render_brain(self) -> dict:
        d = super().render_brain()
        d["p_cooperate"] = self.p
        return d


class MajorityTFT(_Classic):
    """Concede iff at least half the players conceded last round; concede on round 1.

    Doubly game-dependent, and both halves are checked at match start: it needs a role to play
    *and* an observation it can actually read, since it counts what others did. A game handing
    out an opaque board index cannot support it, and says so rather than letting it act on a
    number that means something else.
    """

    strategy = "Majority-TFT"

    def __init__(self, name: str, n_agents: int) -> None:
        super().__init__(name)
        self.n_agents = n_agents
        self.start_state = n_agents + 1
        self.threshold = n_agents / 2.0
        self.rule = f"Concede iff >= {self.threshold:g} players conceded last round."
        self._concede: int | None = None
        self._claim: int | None = None

    def on_match_start(self, game: Any) -> None:
        game.require_observation_kind("concede_count")
        self._concede = game.action_for("concede")
        self._claim = game.action_for("claim")

    def act(self, observation: int) -> int:
        if self._concede is None:
            raise RuntimeError(self._unbound())
        if observation == self.start_state:      # first round
            return self._concede
        k_prev = observation                     # observation index == count of conceders
        return self._concede if k_prev >= self.threshold else self._claim
