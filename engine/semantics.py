"""Pairing a roster with a game, and refusing the pairing when it cannot be trusted.

Games declare what their action indices and observations mean (see `engine/game.py`). Agents whose
behaviour depends on that meaning resolve it at `on_match_start`. That mechanism was correct and
documented, and it was still possible to write an agent that ignored it entirely: the runner called
the hook and accepted whatever came back, so an agent that hardcoded an index played on. The
declaration was advice.

This module is where the advice becomes a contract. Every entry point that drives agents against a
game calls `bind_agents` instead of looping over `on_match_start` itself, and two pairings are
refused before the first round:

  1. an agent that has not declared its `semantics` at all
  2. an agent that declares `"role-bound"` and then asks the game nothing

The second check is the one with teeth, and it works by counting: `Game._semantic_queries` rises
whenever `action_for` or `require_observation_kind` is called, so reading it either side of one
agent's hook says whether *that* agent resolved itself against *this* game.

## What this does and does not guarantee

It guarantees that a meaning-dependent agent cannot silently skip resolution, which is the failure
that actually happened here.

It does not guarantee that an agent declaring `"index-agnostic"` is telling the truth. Nothing
short of analysing the agent's own arithmetic could, and an agent may compute an index for reasons
no checker can distinguish from hardcoding one. What the declaration buys is that the claim is now
explicit, attributable and reviewable: skipping it is no longer possible, and getting it wrong is a
statement someone made rather than a step someone forgot. We think that is the honest boundary of
what a runtime check can do here, and we would rather name it than imply a stronger guarantee.
"""

from __future__ import annotations

from typing import Any, Sequence

INDEX_AGNOSTIC = "index-agnostic"
ROLE_BOUND = "role-bound"
VALID = (INDEX_AGNOSTIC, ROLE_BOUND)


class SemanticBindingError(RuntimeError):
    """A roster cannot be trusted against this game, so the match does not start.

    Deliberately not a warning. A match that runs with a misbound agent produces numbers that look
    exactly like valid ones, and those numbers reach an event log, a ledger record and eventually a
    figure. There is no downstream stage that can tell them apart, so the only place to stop is
    here.
    """


def bind_agents(game: Any, agents: Sequence[Any]) -> None:
    """Resolve every agent against this game, or refuse the pairing.

    Call this instead of looping over `on_match_start`. Every entry point that drives agents does:
    the match runner, the PettingZoo adapter (and through it the Gymnasium one), the benchmark
    harness and the bake-off wrappers.
    """
    for i, agent in enumerate(agents):
        declared = getattr(agent, "semantics", None)

        if declared not in VALID:
            raise SemanticBindingError(
                f"{_who(i, agent)} does not declare how it relates to the meaning of action "
                f"indices, so this match cannot be trusted to mean what it appears to mean. Set "
                f"the class attribute `semantics` to {INDEX_AGNOSTIC!r} if the agent works purely "
                f"over indices without interpreting them, or to {ROLE_BOUND!r} if its behaviour "
                f"depends on what they mean, in which case resolve them in on_match_start via "
                f"game.action_for(...) / game.require_observation_kind(...). "
                f"Got {declared!r}. See engine/agent.py and engine/game.py."
            )

        before = game._semantic_queries
        agent.on_match_start(game)
        asked = game._semantic_queries - before

        if declared == ROLE_BOUND and asked == 0:
            raise SemanticBindingError(
                f"{_who(i, agent)} declares itself {ROLE_BOUND!r} but asked "
                f"{type(game).__name__} nothing during on_match_start, so whatever action indices "
                f"it is about to play were decided without reference to what they mean in this "
                f"game. That is the failure this declaration exists to prevent: an agent labelled "
                f"cooperative playing the most aggressive move available, with every test still "
                f"passing. Call game.action_for(<role>) and/or "
                f"game.require_observation_kind(<kind>) in on_match_start, or declare "
                f"{INDEX_AGNOSTIC!r} if the agent genuinely does not care."
            )


def _who(i: int, agent: Any) -> str:
    return f"Roster seat {i} ({getattr(agent, 'name', '?')!r}, kind {getattr(agent, 'kind', '?')!r})"
