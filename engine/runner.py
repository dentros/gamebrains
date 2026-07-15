"""
The match runner — the online game loop.

Given a game and a roster of agents, it plays `rounds` rounds: every agent picks an action, the
game resolves them simultaneously, every agent learns from its own transition (`update`), and the
whole thing is logged (event-log) and optionally traced (live console). It is brain-agnostic: it
never knows how any agent learns, only the Agent contract. That is exactly what lets heterogeneous
brains (Q-learning + DQN + FEP + classic) share one match.

Returns per-round records as numpy arrays for the metrics layer.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .agent import Agent
from .console import LiveConsole
from .eventlog import EventLog
from .game import Game
from .registry import describe_registry


def run_match(
    game: Game,
    agents: Sequence[Agent],
    rounds: int,
    seed: int = 0,
    eventlog: Optional[EventLog] = None,
    console: Optional[LiveConsole] = None,
    log_every: int = 100,
    snapshot_every: int = 0,
    snapshot_agent: int = 0,
) -> dict[str, np.ndarray]:
    n = game.n_agents
    if len(agents) != n:
        raise ValueError(f"game expects {n} agents, roster has {len(agents)}")

    if eventlog is not None:
        eventlog.meta(
            type="meta",
            game=game.describe(),
            seed=seed,
            rounds=rounds,
            roster=[{"idx": i, "name": a.name, "kind": getattr(a, "kind", "?"),
                     "training_mode": getattr(a, "training_mode", "?")} for i, a in enumerate(agents)],
            registry=describe_registry(),
        )
    if console is not None:
        console.header(game.describe(), agents, seed, rounds)

    actions_hist = np.zeros((rounds, n), dtype=int)
    rewards_hist = np.zeros((rounds, n), dtype=float)
    coop_hist = np.zeros(rounds, dtype=int)

    obs = game.reset()
    for t in range(rounds):
        actions = [agents[i].act(obs[i]) for i in range(n)]
        result = game.step(actions)

        for i in range(n):
            agents[i].update(obs[i], actions[i], result.rewards[i], result.observations[i], result.done)

        actions_hist[t] = actions
        rewards_hist[t] = result.rewards
        coop_hist[t] = result.info.get("cooperators", int(sum(actions)))

        if eventlog is not None:
            eventlog.log({
                "type": "round",
                "round": t,
                "actions": list(actions),
                "cooperators": int(coop_hist[t]),
                "rewards": [round(r, 6) for r in result.rewards],
            })

        if console is not None and (t < 10 or (log_every and t % log_every == 0)):
            extra = ""
            tracked = agents[snapshot_agent]
            if getattr(tracked, "kind", "") in ("qlearning", "dqn"):
                extra = f"ε={getattr(tracked, 'epsilon', 0):.3f}"
            console.round_line(t, actions, int(coop_hist[t]), float(np.mean(result.rewards)), extra)

        if (console is not None and snapshot_every and t > 0 and t % snapshot_every == 0):
            console.brain_snapshot(agents[snapshot_agent], title=f"round {t}")

        if eventlog is not None and snapshot_every and t > 0 and t % snapshot_every == 0:
            _log_snapshots(eventlog, agents, t)

        obs = result.observations

    for a in agents:
        a.on_match_end()

    return {"actions": actions_hist, "rewards": rewards_hist, "cooperators": coop_hist}


def _log_snapshots(eventlog: EventLog, agents: Sequence[Agent], rnd: int) -> None:
    for i, a in enumerate(agents):
        eventlog.log({
            "type": "brain_snapshot",
            "round": rnd,
            "agent": i,
            "name": a.name,
            "brain": a.render_brain(),
        })
