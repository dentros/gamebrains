"""
The match runner — the online game loop.

Given a game and a roster of agents, it plays `rounds` rounds: every agent picks an action, the
game resolves them simultaneously, every agent learns from its own transition (`update`), and the
whole thing is logged (event-log) and optionally traced (live console). It is brain-agnostic: it
never knows how any agent learns, only the Agent contract. That is exactly what lets heterogeneous
brains (Q-learning + DQN + FEP + classic) share one match.

`rounds` is always the total round budget, never an episode count. A game that ends episodes
early (`StepResult.done`) is restarted via `game.reset()` and keeps playing until that budget
runs out, so one match may contain many episodes. This is what episodic games such as the
congestion family need: agents race to a terminal, the episode ends, and the next one begins.
Single-stage games are unaffected, since they only report `done` on the final round.

An episode boundary emits an `episode` event. Its per-agent reward totals and round count are
computed here, but anything game-specific (for a congestion game, which agents reached the
terminal) is supplied by the game itself under `StepResult.info["episode"]` and forwarded
verbatim, so the runner stays as game-agnostic as it is brain-agnostic.

Do not confuse two different ways an episode can end without a winner. If the *game* stops an
episode at its own round cap because nobody got there, that is a real outcome: the game reports
`done` and supplies its own (all-zero) payload, and it is recorded like any other episode. What
`partial_episode` below controls is the other case, where the *match* budget simply ran out
mid-episode, which is an artifact of how long we chose to run rather than anything the agents did.

Returns per-round records as numpy arrays for the metrics layer, plus the per-episode records.
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
    partial_episode: str = "drop",
) -> dict[str, Any]:
    n = game.n_agents
    if len(agents) != n:
        raise ValueError(f"game expects {n} agents, roster has {len(agents)}")
    if partial_episode not in ("drop", "record"):
        raise ValueError(f"partial_episode must be 'drop' or 'record', got {partial_episode!r}")

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

    # Before anything is played: let every agent resolve itself against this specific game, and
    # refuse the pairing now if it cannot. A strategy that means "always give way" has to look up
    # which action that is here, since the index differs per game.
    for a in agents:
        a.on_match_start(game)

    episodes: list[dict[str, Any]] = []
    ep_rewards = np.zeros(n, dtype=float)
    ep_rounds = 0

    obs = game.reset()
    for t in range(rounds):
        actions = [agents[i].act(obs[i]) for i in range(n)]
        result = game.step(actions)

        for i in range(n):
            agents[i].update(obs[i], actions[i], result.rewards[i], result.observations[i], result.done)

        actions_hist[t] = actions
        rewards_hist[t] = result.rewards
        coop_hist[t] = result.info.get("cooperators", int(sum(actions)))
        ep_rewards += result.rewards
        ep_rounds += 1

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

        if result.done:
            record = {
                "episode": len(episodes),
                "rounds": ep_rounds,
                "rewards": [round(float(r), 6) for r in ep_rewards],
                **(result.info.get("episode") or {}),
            }
            episodes.append(record)
            if eventlog is not None:
                eventlog.log({"type": "episode", **record})
            ep_rewards = np.zeros(n, dtype=float)
            ep_rounds = 0
            # Restart only when there is budget left to play. Resetting on the final round would
            # leave the game rewound behind our back, which callers that inspect it afterwards
            # (and every single-stage game, which reports done exactly there) do not expect.
            obs = game.reset() if t < rounds - 1 else result.observations
        else:
            obs = result.observations

    for a in agents:
        a.on_match_end()

    # The match budget can run out mid-episode. Whether that half-played contest counts is a
    # scientific choice, not ours to make silently: "drop" leaves the alternation series to
    # completed contests only, "record" keeps it as a no-winner episode (everyone scored zero,
    # nobody reached a terminal). Either way `truncated` marks it and `partial_episode_rounds`
    # reports how many rounds were involved, so the choice is never invisible downstream.
    if ep_rounds > 0 and partial_episode == "record":
        record = {
            "episode": len(episodes),
            "rounds": ep_rounds,
            "rewards": [round(float(r), 6) for r in ep_rewards],
            "truncated": True,
        }
        episodes.append(record)
        if eventlog is not None:
            eventlog.log({"type": "episode", **record})

    return {"actions": actions_hist, "rewards": rewards_hist, "cooperators": coop_hist,
            "episodes": episodes, "partial_episode_rounds": ep_rounds,
            "partial_episode_policy": partial_episode}


def _log_snapshots(eventlog: EventLog, agents: Sequence[Agent], rnd: int) -> None:
    for i, a in enumerate(agents):
        eventlog.log({
            "type": "brain_snapshot",
            "round": rnd,
            "agent": i,
            "name": a.name,
            "brain": a.render_brain(),
        })
