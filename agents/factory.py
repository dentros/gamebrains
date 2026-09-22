"""One place that turns a kind name and a seat index into an agent.

The interface built rosters one way and every batch tool built them another, which is a difference
that does not show up as a bug. It shows up as two runs with the same `config_hash` and different
numbers, because a seed offset or a default moved in one place and not the other. So the
construction lives here and both callers go through it.

What each kind is given is deliberate and should not be changed casually:

* **Seat seeds are offset by kind**, so seat 0 of a Q-learner and seat 0 of a DQN in the same
  roster do not share a stream. The offsets are 100, 200, 300, 400 and 500, and they are part of
  what makes a seeded run reproduce, so an edit here changes every future run's numbers.
* **Nothing defaults twice.** A hyperparameter absent from `overrides` is not filled in with a
  value copied into this file, it is left to the agent class's own constructor default. Copying
  defaults is how two sources of truth begin.
* **No language-model seat.** A model seat needs a live backend, is seconds per decision rather
  than microseconds, and carries a journal that only the process that ran it holds. Batch tools
  that fan out across processes therefore cannot seat one, and the interface constructs it in its
  own branch where the server can be checked and the failure explained to a person.

`game` is anything the runner accepts. The active-inference agent is the one kind that reads
payoff structure off the game (`mpcr`, `cost`), so it is offered only by games that declare those,
and asking for it elsewhere is refused by name rather than by an attribute error.
"""

from __future__ import annotations

from typing import Any, Optional

from .classic import AllC, AllD, MajorityTFT, RandomAgent

#: Every kind this factory builds. The language model is deliberately absent, see the module note.
KINDS = ("qlearning", "dqn", "fep", "markov_brain", "classic")

#: Seed offsets per kind. Part of the reproducibility contract, not a tuning knob.
_SEED_OFFSET = {"qlearning": 100, "dqn": 200, "fep": 300, "classic": 400, "markov_brain": 500}

#: What each kind will accept as an override, and therefore what a sweep may vary.
#:
#: **An override this factory does not recognise is refused, not ignored.** A silently dropped
#: hyperparameter is the worst kind of defect a sweep can have: every cell runs, every cell is
#: recorded, the numbers do not move, and the finding reads "this parameter makes no difference".
#: Worse, the cells collapse onto one `config_hash`, because the parameter never reached the agent
#: whose attributes the recorder reads, so the ledger holds one design where the experimenter
#: believes there are five.
ACCEPTS: dict[str, tuple[str, ...]] = {
    "qlearning": ("alpha", "gamma", "epsilon_decay", "epsilon_min"),
    "dqn": ("hidden", "lr", "gamma", "buffer_size", "batch_size", "train_every",
            "target_sync_every", "epsilon_decay", "epsilon_min"),
    "fep": ("obs_noise", "drift", "precision", "reciprocity"),
    "markov_brain": ("n_hidden",),
    "classic": ("strategy", "p_cooperate"),
}


def _checked(kind: str, overrides: dict[str, Any]) -> dict[str, Any]:
    accepted = ACCEPTS[kind]
    unknown = sorted(set(overrides) - set(accepted))
    if unknown:
        raise ValueError(
            f"{kind} does not take {', '.join(unknown)}. It accepts {', '.join(accepted)}. "
            f"An override that is quietly dropped makes a sweep report that the parameter has no "
            f"effect, which is why this is an error rather than a warning.")
    return overrides


def classic_agent(strategy: str, index: int, n_agents: int, seed: int,
                  overrides: Optional[dict[str, Any]] = None):
    overrides = overrides or {}
    if strategy == "AllC":
        return AllC(f"AllC {index}")
    if strategy == "AllD":
        return AllD(f"AllD {index}")
    if strategy == "Random":
        return RandomAgent(f"Random {index}", p_cooperate=overrides.get("p_cooperate", 0.5),
                           seed=seed + _SEED_OFFSET["classic"] + index)
    if strategy == "MajorityTFT":
        return MajorityTFT(f"TFT {index}", n_agents=n_agents)
    raise ValueError(f"unknown classic strategy {strategy}")


def build(kind: str, index: int, game: Any, seed: int, classic_strategy: str = "AllD",
          reciprocity: float = 0.0, markov_hidden: int = 2, epsilon_decay: float = 0.9995,
          epsilon_min: float = 0.02, overrides: Optional[dict[str, Any]] = None):
    """One agent for seat `index`.

    `overrides` carries kind-specific hyperparameters. A key that is not present falls back to the
    agent class's own default, so leaving the mapping empty reproduces the platform's default
    behaviour exactly.
    """
    overrides = overrides or {}
    labels, actions = game.state_labels(), game.action_names

    if kind == "qlearning":
        from .qlearning import QLearningAgent
        _checked(kind, overrides)
        return QLearningAgent(
            name=f"Q-learner {index}", n_states=game.n_states, n_actions=game.n_actions,
            alpha=overrides.get("alpha", 0.1), gamma=overrides.get("gamma", 0.95),
            epsilon=1.0,
            epsilon_min=overrides.get("epsilon_min", epsilon_min),
            epsilon_decay=overrides.get("epsilon_decay", epsilon_decay),
            seed=seed + _SEED_OFFSET["qlearning"] + index,
            state_labels=labels, action_labels=actions,
        )

    if kind == "dqn":
        from .dqn import DQNAgent          # lazy: torch is imported only when one is asked for
        _checked(kind, overrides)
        kwargs = {key: overrides[key] for key in
                  ("hidden", "lr", "gamma", "buffer_size", "batch_size", "train_every",
                   "target_sync_every") if key in overrides}
        return DQNAgent(
            name=f"DeepQ {index}", n_states=game.n_states, n_actions=game.n_actions,
            epsilon=1.0,
            epsilon_min=overrides.get("epsilon_min", epsilon_min),
            epsilon_decay=overrides.get("epsilon_decay", epsilon_decay),
            seed=seed + _SEED_OFFSET["dqn"] + index,
            state_labels=labels, action_labels=actions, **kwargs,
        )

    if kind == "fep":
        from .fep import FEPAgent
        _checked(kind, overrides)
        if not hasattr(game, "mpcr") or not hasattr(game, "cost"):
            raise ValueError(
                f"the active-inference agent reads this game's payoff structure directly (mpcr and "
                f"cost), and {getattr(game, 'name', type(game).__name__)!r} declares neither, so it "
                f"has nothing to form a belief about here")
        kwargs = {key: overrides[key] for key in ("obs_noise", "drift", "precision")
                  if key in overrides}
        return FEPAgent(
            name=f"FEP {index}", n_agents=game.n_agents, mpcr=game.mpcr, cost=game.cost,
            reciprocity=overrides.get("reciprocity", reciprocity),
            seed=seed + _SEED_OFFSET["fep"] + index,
            start_state=game.start_state, **kwargs,
        )

    if kind == "markov_brain":
        from .markov_brain import MarkovBrainAgent
        _checked(kind, overrides)
        return MarkovBrainAgent(
            name=f"MarkovBrain {index}", n_states=game.n_states, n_actions=game.n_actions,
            n_hidden=int(overrides.get("n_hidden", markov_hidden)),
            seed=seed + _SEED_OFFSET["markov_brain"] + index,
            start_state=game.start_state,
        )

    if kind == "classic":
        _checked(kind, overrides)
        return classic_agent(str(overrides.get("strategy", classic_strategy)), index,
                             game.n_agents, seed, overrides)

    if kind == "llm":
        raise ValueError(
            "a language-model seat is not built here: it needs a live backend, costs seconds per "
            "decision, and keeps a decision journal that only the process running it holds. The "
            "interface constructs it in its own branch, where a missing server can be explained.")
    raise ValueError(f"unknown kind {kind!r}, expected one of {', '.join(KINDS)}")
