"""
Free-Energy-Principle / Active-Inference agent (a Theory-of-Mind brain).

This brain does two things, both straight from the FEP:

  1. Perception = variational free-energy minimization. It maintains a categorical posterior
     `qs` over a hidden state — "how many of the OTHER players will cooperate this round" — and
     updates it by exact Bayesian filtering: predict through a slow-drift transition B, then
     multiply by the observation likelihood A and renormalize. That posterior *is* its little
     model of the others' minds, which is why this is our Theory-of-Mind brain.

  2. Action = expected-free-energy minimization. For each action it computes expected value under
     its beliefs (pragmatic value); because its own move does not change the others' disposition,
     the epistemic term is action-independent here, so action selection reduces to a precision-
     weighted softmax over expected value. With `reciprocity=0` the agent is purely self-interested
     (and, correctly, free-rides — defection dominates the Public Goods Game); a positive
     `reciprocity` turns it into a belief-driven conditional cooperator.

We implement the active-inference math directly in numpy (rather than via pymdp's current
JAX API) for robustness and full transparency; a pymdp backend can be added later.

**This agent binds to the game's declared semantics at `on_match_start`, and refuses games it
cannot read.** It used to import `COOPERATE`/`DEFECT` from `games/public_goods.py`, which pinned
"cooperate" to action index 1 exactly as `agents/classic.py` once did. In a congestion game index 1
is the *claiming* action, so this agent played the most aggressive move available while its own
transparency panel labelled it "Cooperate" -- the same defect the declaration mechanism was built
to end, still live in this file long after the classic strategies were migrated. Worse here than
there, because two things were wrong at once: the action indices, and the observation. Its whole
generative model is over "how many OTHERS conceded", so it needs `observation_kind ==
"concede_count"`; handed a `board_index` it filtered a board position through a likelihood over
cooperator counts and reported the result as a belief.

It now requires that observation kind explicitly, which means it correctly *refuses* the congestion
family rather than misreading it. That refusal is the honest answer and not a limitation to work
around: an agent whose hidden state is a count of conceders has nothing to infer from a position.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..engine.agent import Agent

#: Internal ordering of this agent's own value/probability vectors. Deliberately NOT the game's
#: action indices: those are resolved per game at `on_match_start` and may be either way round.
_I_CLAIM = 0
_I_CONCEDE = 1


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


class FEPAgent(Agent):
    kind = "fep"
    training_mode = "online"

    def __init__(
        self,
        name: str,
        n_agents: int,
        mpcr: float,
        cost: float = 1.0,
        obs_noise: float = 0.75,   # σ of the likelihood A over hidden "others cooperating"
        drift: float = 0.1,        # how fast others' disposition is assumed to drift (B)
        precision: float = 4.0,    # γ: how sharply value drives action
        reciprocity: float = 0.0,  # >0 -> prefer cooperating when it believes others do
        seed: int = 0,
        start_state: Optional[int] = None,
    ) -> None:
        self.name = name
        self.n_agents = n_agents
        self.n_others = n_agents - 1
        self.mpcr = mpcr
        self.cost = cost
        self.precision = precision
        self.reciprocity = reciprocity
        # Stored purely for introspection (repository/record.py's config_hash needs the actual
        # hyperparameter values); obs_noise/drift only feed self.A/self.B below otherwise.
        self.obs_noise = obs_noise
        self.drift = drift
        self.rng = np.random.default_rng(seed)
        self.start_state = start_state if start_state is not None else n_agents + 1

        # Hidden states s = 0..n_others  (number of OTHER players cooperating this round)
        self.n_states = self.n_others + 1
        states = np.arange(self.n_states)

        # Likelihood A[o, s] ∝ exp(-(o - s)^2 / 2σ^2), normalized over observations per state.
        diff = states[:, None] - states[None, :]
        A = np.exp(-(diff ** 2) / (2.0 * obs_noise ** 2))
        self.A = A / A.sum(axis=0, keepdims=True)

        # Transition B = (1 - drift) I + drift * uniform  (slow, action-independent drift)
        self.B = (1.0 - drift) * np.eye(self.n_states) + drift * (np.ones((self.n_states, self.n_states)) / self.n_states)

        self.qs = np.ones(self.n_states) / self.n_states   # posterior over others conceding
        self.last_action_probs = np.array([0.5, 0.5])

        # Resolved at on_match_start from the game's own declarations. None until then, so acting
        # before binding fails loudly instead of guessing an index.
        self._concede: Optional[int] = None
        self._claim: Optional[int] = None
        self._label = {"concede": "Concede", "claim": "Claim"}
        #: Whether our own previous move was the conceding one. Tracked as a fact about the move,
        #: not as its index, because the observation arithmetic below counts conceders and the
        #: conceding action is not index 1 in every game.
        self._last_was_concede = False

    # --- binding to a specific game -----------------------------------------------
    def on_match_start(self, game: Any) -> None:
        """Resolve roles and refuse any game whose observation this agent cannot interpret."""
        game.require_observation_kind("concede_count")
        self._concede = game.action_for("concede")
        self._claim = game.action_for("claim")
        # Prefer the game's own vocabulary for display when it declares one, so a Public Goods
        # panel still reads "Cooperate"/"Defect" while a different concede-count game gets neutral
        # words instead of borrowed ones.
        roles = getattr(game, "action_roles", {}) or {}
        if roles.get("cooperate") == self._concede and roles.get("defect") == self._claim:
            self._label = {"concede": "Cooperate", "claim": "Defect"}
        else:
            self._label = {"concede": "Concede", "claim": "Claim"}

    def _unbound(self) -> str:
        return (f"{type(self).__name__} has not been matched to a game yet, so it does not know "
                f"which action concedes. The runner calls on_match_start before play begins; a "
                f"caller driving agents directly has to do the same.")

    # --- perception ---------------------------------------------------------------
    def _perceive(self, others_conceding: int) -> None:
        pred = self.B @ self.qs                       # prior after drift
        like = self.A[int(np.clip(others_conceding, 0, self.n_states - 1)), :]
        post = pred * like
        s = post.sum()
        self.qs = post / s if s > 0 else np.ones(self.n_states) / self.n_states

    # --- expected value under beliefs ---------------------------------------------
    def _expected_values(self) -> np.ndarray:
        s = np.arange(self.n_states)
        ev_defect = float(np.sum(self.qs * (self.mpcr * s * self.cost)))
        ev_coop = float(np.sum(self.qs * (self.mpcr * (s + 1) * self.cost - self.cost)))
        if self.reciprocity:
            ev_coop += self.reciprocity * float(np.sum(self.qs * s)) / max(self.n_others, 1)
        ev = np.empty(2)
        ev[_I_CLAIM] = ev_defect
        ev[_I_CONCEDE] = ev_coop
        return ev

    # --- Agent interface ----------------------------------------------------------
    def act(self, observation: int) -> int:
        if self._concede is None or self._claim is None:
            raise RuntimeError(self._unbound())

        if observation != self.start_state:
            # The observation counts conceders, so subtract our own contribution to it. This is a
            # count, not an index: using the raw action index here is only correct in a game where
            # conceding happens to be action 1, which is exactly the assumption that broke before.
            others = observation - (1 if self._last_was_concede else 0)
            self._perceive(others)

        ev = self._expected_values()
        self.last_action_probs = _softmax(self.precision * ev)
        concede = self.rng.random() < self.last_action_probs[_I_CONCEDE]
        self._last_was_concede = bool(concede)
        return self._concede if concede else self._claim

    def expected_others(self) -> float:
        return float(np.sum(np.arange(self.n_states) * self.qs))

    def inspect(self) -> dict:
        return {
            "belief_others_conceding": self.qs.tolist(),
            "expected_others": self.expected_others(),
            "expected_values": self._expected_values().tolist(),
            "action_probs": self.last_action_probs.tolist(),
            "bound_actions": {"concede": self._concede, "claim": self._claim},
        }

    def render_brain(self) -> dict:
        ev = self._expected_values()
        # Keyed by ROLE, with the display wording carried alongside. Keying the numbers by the
        # display word instead is what let a renderer read `probs["Cooperate"]` and quietly get
        # nothing when the wording changed, so the two are kept separate on purpose.
        return {
            "kind": self.kind,
            "belief": self.qs.tolist(),
            "state_labels": [f"others={s}" for s in range(self.n_states)],
            "expected_others": self.expected_others(),
            "expected_values": {"claim": float(ev[_I_CLAIM]), "concede": float(ev[_I_CONCEDE])},
            "action_probs": {"claim": float(self.last_action_probs[_I_CLAIM]),
                             "concede": float(self.last_action_probs[_I_CONCEDE])},
            "role_labels": dict(self._label),
            "reciprocity": self.reciprocity,
        }
