"""
A compact agent in the Distributed Adaptive Control tradition: reactive, adaptive and contextual
layers, with the layer in control visible in every round.

Distributed Adaptive Control (DAC) is Verschure's layered architecture for behaviour, developed since
Verschure, Kröse and Pfeifer (1992) and stated as a theory of the mind-brain-body nexus in Verschure
(2012). A **reactive** layer of prewired reflexes keeps the agent alive before it knows anything; an
**adaptive** layer learns which actions pay in which states; a **contextual** layer stores sequences
of experience and replays the ones that led to reward. Two descendants of it shape this file most
directly. Freire et al. (2020, PLoS ONE 15:e0234434) built a reactive plus adaptive DAC agent for a
multi-agent turn-taking game, the Battle of the Exes, which is the direct ancestor of this platform's
congestion family. In their two-agent game it was the complete agent that formed turn-taking
conventions and reached human-level efficiency and fairness, while each layer alone was less
efficient and less stable, its rounds ending more often in ties. (An earlier version of this
docstring said their reactive layer alone produced turn-taking. It did not; the claim was checked
against the paper and removed.) The
contextual layer here follows Sequential Episodic Control (Freire, Amil and Verschure, arXiv:2112.14734):
a short-term memory of state-action couplets is consolidated into long-term memory when it ends in
reward, and retrieved by matching the current state, with a bias toward the couplet that follows the
one a sequence last supplied. The arbitration between the contextual layer and the rest follows the
model-trust idea of Verschure (2026, Phil. Trans. R. Soc. B 381:20250156): the agent acts on its
memories only while they have been earning its trust, and trust is lost faster than it is gained.

**What this is not.** It is not a full DAC implementation. There is no perception layer, no
allostatic control over several drives, no cerebellar timing model and no planning. Each layer is
the smallest mechanism that does that layer's job in a two-action repeated game, so that the
platform can host the architecture's shape, show which layer decided each move, and test that the
layers do what they are said to do. Calling it more would be the overclaim the paper argues against.

The layers, as implemented:

* **Reactive.** One homeostatic variable, a *deficit* in [0, 1], falls by the fraction `deficit_relief`
  in a round in which the agent gains, and rises by `deficit_growth` in a round in which it loses or a
  contest ends with it having gained nothing. Other rounds leave it alone, so the steps of an episode
  still in progress are not counted as hunger (a first version counted every round without a gain,
  and in the congestion family, where gains arrive only at the end of an episode, the reflex was in
  urgent control in 97% of rounds and every agent rushed the jar every time). Two reflexes run
  beneath every other layer, as the avoidance reflexes do in Freire et al.'s agents: at or above
  `urgency` the agent claims, and in the round after a contest that ended with it having claimed and
  gained nothing it concedes with probability `withdraw`. Otherwise, when no other layer is ready,
  the reactive layer claims with probability equal to the deficit.
* **Adaptive.** Tabular one-step Q-learning over (observation, deficit band), epsilon-greedy with a
  fixed `explore`, learning from the payoff as Freire et al.'s adaptive layer does. It takes control of a state once it has visited it `min_visits` times; before that
  the reflexes act, which is how the reactive layer bootstraps learning in DAC.
* **Contextual.** Up to `stm_length` recent couplets in short-term memory. When a round ends with a
  gain above the agent's running baseline, the short-term memory is copied into long-term memory
  with that excess as its value (at most `ltm_capacity` sequences, oldest dropped first). To decide,
  every stored couplet whose observation matches the current one votes for its action, weighted by
  its sequence's value, by `distance_decay` per step it lies before the reward, and by
  `1 + succession_bonus` if it directly follows the couplet that sequence supplied last round.
* **Arbitration.** The contextual layer acts when it has a proposal and `trust >= trust_threshold`.
  Trust is tested whenever the move made was the one the contextual layer proposed, whichever layer
  made it: an outcome at least as good as the adaptive layer expected (a non-negative temporal
  difference error) raises it by `trust_up * (1 - trust)`, a worse one lowers it by
  `trust_down * trust`, with `trust_down > trust_up` by default.

The agent binds to the game's declared `concede` and `claim` roles and treats the observation as an
opaque token, so the same unchanged agent plays the Public Goods Game and the congestion family. A
game declaring neither role is refused.

`layers` selects which layers exist. Removing one is a lesion, recorded with the run because it
changes behaviour, and `tests/test_ablation.py` uses it to check that the contextual layer, when it
takes control, actually changes what the agent does.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..engine.agent import Agent

#: Internal ordering of this agent's own vectors, deliberately not the game's action indices.
_I_CLAIM = 0
_I_CONCEDE = 1
_ROLE_NAMES = ("claim", "concede")

LAYERS = ("reactive", "adaptive", "contextual")


@dataclass
class _Sequence:
    """One consolidated episode of experience: couplets in order, and the gain that ended it."""

    uid: int
    couplets: list[tuple[Any, int]]     # (observation, internal action index)
    value: float


class DACAgent(Agent):
    kind = "dac"
    training_mode = "online"
    semantics = "role-bound"
    #: The deficit, the value table and both memories carry the past into every decision.
    information = "observation+memory"

    def __init__(
        self,
        name: str,
        seed: int = 0,
        *,
        deficit_growth: float = 0.15,
        deficit_relief: float = 0.8,
        urgency: float = 0.9,
        withdraw: float = 0.8,
        alpha: float = 0.2,
        gamma: float = 0.9,
        explore: float = 0.1,
        min_visits: int = 3,
        deficit_bands: int = 3,
        stm_length: int = 4,
        ltm_capacity: int = 60,
        distance_decay: float = 0.7,
        succession_bonus: float = 1.0,
        trust_up: float = 0.05,
        trust_down: float = 0.2,
        trust_threshold: float = 0.5,
        initial_trust: float = 0.5,
        baseline_rate: float = 0.05,
        layers: tuple[str, ...] = LAYERS,
    ) -> None:
        unknown = sorted(set(layers) - set(LAYERS))
        if unknown:
            raise ValueError(f"unknown DAC layer(s) {unknown}; the layers are {list(LAYERS)}")
        self.name = name
        self.seed = seed
        self.deficit_growth = deficit_growth
        self.deficit_relief = deficit_relief
        self.urgency = urgency
        self.withdraw = withdraw
        self.alpha = alpha
        self.gamma = gamma
        self.explore = explore
        self.min_visits = min_visits
        self.deficit_bands = deficit_bands
        self.stm_length = stm_length
        self.ltm_capacity = ltm_capacity
        self.distance_decay = distance_decay
        self.succession_bonus = succession_bonus
        self.trust_up = trust_up
        self.trust_down = trust_down
        self.trust_threshold = trust_threshold
        self.initial_trust = initial_trust
        self.baseline_rate = baseline_rate
        # Canonical order, so ("contextual", "reactive") and ("reactive", "contextual") are one
        # design in the record rather than two.
        self.layers = tuple(layer for layer in LAYERS if layer in layers)
        self.rng = np.random.default_rng(seed)

        # Learned across matches, like a Q-table: what pays, and what has paid before.
        self._q: dict[tuple[Any, int], np.ndarray] = {}
        self._visits: dict[tuple[Any, int], int] = {}
        self._ltm: list[_Sequence] = []
        self._next_uid = 0
        self.trust = initial_trust
        self.baseline = 0.0

        self._concede: Optional[int] = None
        self._claim: Optional[int] = None
        self._label = {"concede": "Concede", "claim": "Claim"}
        self.layer_counts = {"reactive-urgent": 0, "reactive-withdraw": 0, "reactive": 0,
                             "adaptive": 0, "contextual": 0, "none": 0}
        self._reset_episode_state()

    # --- per-match state -----------------------------------------------------------------------
    def _reset_episode_state(self) -> None:
        self.deficit = 0.0
        self._withdrawing = False
        self._stm: deque[tuple[Any, int]] = deque(maxlen=self.stm_length)
        self._pointers: dict[int, int] = {}       # sequence uid -> couplet it supplied last round
        self._last_key: Optional[tuple[Any, int]] = None
        self._last_obs: Any = None
        self._last_internal: Optional[int] = None
        self._proposal: Optional[int] = None
        self._vote = np.zeros(2)
        self._matched: list[tuple[int, int, int, float]] = []
        self.last_layer = "none"

    def on_match_start(self, game: Any) -> None:
        self._concede = game.action_for("concede")
        self._claim = game.action_for("claim")
        roles = getattr(game, "action_roles", {}) or {}
        if roles.get("cooperate") == self._concede and roles.get("defect") == self._claim:
            self._label = {"concede": "Cooperate", "claim": "Defect"}
        elif roles.get("yield") == self._concede and roles.get("contest") == self._claim:
            self._label = {"concede": "Yield", "claim": "Contest"}
        else:
            self._label = {"concede": "Concede", "claim": "Claim"}
        self._reset_episode_state()

    # --- the three layers ----------------------------------------------------------------------
    def _band(self) -> int:
        return min(int(self.deficit * self.deficit_bands), self.deficit_bands - 1)

    def _reactive(self) -> int:
        return _I_CLAIM if self.rng.random() < self.deficit else _I_CONCEDE

    def _adaptive(self, key: tuple[Any, int]) -> int:
        if self.rng.random() < self.explore:
            return int(self.rng.integers(2))
        q = self._q.get(key, np.zeros(2))
        # Ties go to conceding, so an untrained state does not default to the aggressive move.
        return _I_CONCEDE if q[_I_CONCEDE] >= q[_I_CLAIM] else _I_CLAIM

    def _contextual(self, obs: Any) -> Optional[int]:
        """Vote over long-term memory; None when nothing stored matches this observation."""
        vote = np.zeros(2)
        matched: list[tuple[int, int, int, float]] = []
        for seq in self._ltm:
            last = len(seq.couplets) - 1
            for j, (stored_obs, action) in enumerate(seq.couplets):
                if stored_obs != obs:
                    continue
                weight = seq.value * self.distance_decay ** (last - j)
                if self._pointers.get(seq.uid) == j - 1:
                    weight *= 1.0 + self.succession_bonus
                vote[action] += weight
                matched.append((seq.uid, j, action, weight))
        self._vote, self._matched = vote, matched
        if vote.sum() <= 0.0 or vote[_I_CLAIM] == vote[_I_CONCEDE]:
            return None
        return int(np.argmax(vote))

    # --- Agent interface -----------------------------------------------------------------------
    def act(self, observation: int) -> int:
        if self._concede is None or self._claim is None:
            raise RuntimeError(f"{type(self).__name__} has not been matched to a game yet, so it "
                               f"does not know which action concedes")
        key = (observation, self._band())
        self._proposal = self._contextual(observation) if "contextual" in self.layers else None

        reflexes = "reactive" in self.layers
        if reflexes and self.deficit >= self.urgency:
            internal, layer = _I_CLAIM, "reactive-urgent"
        elif reflexes and self._withdrawing and self.rng.random() < self.withdraw:
            internal, layer = _I_CONCEDE, "reactive-withdraw"
        elif self._proposal is not None and self.trust >= self.trust_threshold:
            internal, layer = self._proposal, "contextual"
        elif "adaptive" in self.layers and self._visits.get(key, 0) >= self.min_visits:
            internal, layer = self._adaptive(key), "adaptive"
        elif "reactive" in self.layers:
            internal, layer = self._reactive(), "reactive"
        elif "adaptive" in self.layers:
            internal, layer = self._adaptive(key), "adaptive"
        else:
            internal, layer = int(self.rng.integers(2)), "none"

        # Succession: a sequence whose couplet backed the move made points at that couplet, so its
        # next couplet is favoured next round. Every other pointer lapses.
        self._pointers = {uid: j for uid, j, voted, _ in self._matched if voted == internal}
        self.layer_counts[layer] += 1
        self.last_layer = layer
        self._last_key, self._last_obs, self._last_internal = key, observation, internal
        return self._claim if internal == _I_CLAIM else self._concede

    def update(self, observation: int, action: int, reward: float, next_observation: int,
               done: bool) -> None:
        if self._last_key is None:
            return
        internal = _I_CLAIM if action == self._claim else _I_CONCEDE
        key = self._last_key
        reward = float(reward)
        baseline_before = self.baseline

        # Deficit first, because the next state's band depends on it. A step of an episode still in
        # progress, with nothing gained or lost, is not hunger.
        if reward > 0.0:
            self.deficit *= 1.0 - self.deficit_relief
        elif reward < 0.0 or done:
            self.deficit = min(1.0, self.deficit + self.deficit_growth)
        self._withdrawing = bool(done and internal == _I_CLAIM and reward <= 0.0)

        td_error: Optional[float] = None
        if "adaptive" in self.layers:
            q = self._q.setdefault(key, np.zeros(2))
            nxt = self._q.get((next_observation, self._band()), np.zeros(2))
            target = reward + (0.0 if done else self.gamma * float(nxt.max()))
            td_error = target - float(q[internal])
            q[internal] += self.alpha * td_error
            self._visits[key] = self._visits.get(key, 0) + 1

        if self._proposal is not None and self._proposal == internal:
            good = (td_error >= 0.0) if td_error is not None else (reward >= baseline_before)
            if good:
                self.trust += self.trust_up * (1.0 - self.trust)
            else:
                self.trust -= self.trust_down * self.trust

        self.baseline += self.baseline_rate * (reward - baseline_before)

        if "contextual" in self.layers:
            self._stm.append((self._last_obs, internal))
            if reward > baseline_before:
                self._ltm.append(_Sequence(self._next_uid, list(self._stm),
                                           reward - baseline_before))
                self._next_uid += 1
                if len(self._ltm) > self.ltm_capacity:
                    self._ltm.pop(0)
            if done:
                self._stm.clear()
                self._pointers = {}

    # --- transparency --------------------------------------------------------------------------
    def _q_now(self) -> np.ndarray:
        return self._q.get(self._last_key, np.zeros(2)) if self._last_key else np.zeros(2)

    def inspect(self) -> dict:
        return {
            "layer": self.last_layer,
            "layer_counts": dict(self.layer_counts),
            "deficit": self.deficit,
            "trust": self.trust,
            "baseline": self.baseline,
            "ltm": [{"uid": s.uid, "length": len(s.couplets), "value": s.value} for s in self._ltm],
            "q_states": len(self._q),
            "bound_actions": {"concede": self._concede, "claim": self._claim},
        }

    def render_brain(self) -> dict:
        q = self._q_now()
        return {
            "kind": self.kind,
            "layers": list(self.layers),
            "layer": self.last_layer,
            "layer_counts": dict(self.layer_counts),
            "deficit": self.deficit,
            "urgency": self.urgency,
            "trust": self.trust,
            "trust_threshold": self.trust_threshold,
            "ltm_size": len(self._ltm),
            "ltm_capacity": self.ltm_capacity,
            "stm": [{"observation": str(o), "action": _ROLE_NAMES[a]} for o, a in self._stm],
            "contextual_vote": {"claim": float(self._vote[_I_CLAIM]),
                                "concede": float(self._vote[_I_CONCEDE])},
            "matched_couplets": len(self._matched),
            "q_values": {"claim": float(q[_I_CLAIM]), "concede": float(q[_I_CONCEDE])},
            "q_states": len(self._q),
            "role_labels": dict(self._label),
        }
