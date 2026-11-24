"""What a language model is told, as a named and versioned profile rather than a free-text box.

A prompt is not a presentation detail. It is the agent's information set, so two models compared
under different prompts are not being compared at all, and a result stated without saying what the
model could see is not interpretable. The first profile here shows why that matters: asked to act
on a bare count, `llama3.2:1b` read an observation of 1 out of 5 as evidence that the others had
cooperated, and cooperated itself into last place.

So the prompt is a profile, chosen by name, versioned, and recorded with the run:

  * **Every model tested sees the same thing**, which is what makes a comparison between models a
    comparison of models.
  * **The choice is queryable.** The profile name lands in `roster[i].params`, so it reaches
    `config_hash`, the Smart Filter, the analytics filters and the meta-analysis grouping, exactly
    like a learning rate. Two runs under different profiles are two experiments and the repository
    will not offer one as the other's answer.
  * **A new version is a new name, never an edit.** Changing what `informed-v1` means would
    silently redefine every result recorded under it. Add `informed-v2` instead and leave the old
    one where it is.

There is deliberately no free-text prompt argument. A caller who needs something these profiles do
not offer adds a profile here, which costs a name and a line and buys the same guarantee for
everyone who comes after. The one field left open is `persona`, because a persona is an
experimental manipulation rather than a description of the board, and it is recorded too.

    from gamebrains.agents.llm import LLMAgent
    from gamebrains.agents.llm_ollama import OllamaBackend

    agent = LLMAgent("LLM 0", OllamaBackend(), profile="informed-v1")
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class PromptProfile:
    """One settled answer to "what does the model get to know?".

    Frozen, because a profile that could be edited at runtime would make the recorded name a label
    rather than a specification.
    """

    name: str
    history: int
    """How many of the agent's own past rounds appear verbatim."""

    others_explicitly: bool
    """State how many of the *other* players conceded, rather than handing over a raw count that
    includes the agent itself. The count is public information the game already publishes; what
    this changes is whether the model has to work out that it is in the number."""

    payoff_rule: bool
    """Describe how payoffs are made, in a sentence, from what the game declares."""

    cumulative_payoff: bool
    """Say what the agent has earned so far, so that losing steadily is visible."""

    round_position: bool
    """Say which round this is out of how many."""

    decode_state: bool
    """Where a game can render its observation in words (`state_labels`), use that instead of the
    bare index. The congestion family is the case: the index encodes every player's position, and
    without this the model is told only that the number is not a count."""

    information: str
    """The declaration `Agent.information` reports for an agent using this profile."""

    neutral_labels: bool = False
    """Offer the two options as "Option A" and "Option B", which of them concedes drawn from the
    agent's seed, and leave out every word that names a side: the game's name, its action names,
    "concede" and "claim". What each option does is then said in the rule sentence
    (`neutral_rules`), since a label alone tells the model nothing. A word like "cooperate" is a
    cue a model has read a great deal about, so a profile that keeps it measures the word as much
    as the game."""

    pd_history: bool = False
    """A two-player matrix game told as such: the horizon and the four payoffs in a system message,
    and every past round as one line of both moves, "round i: you=..., opponent=...". The options
    carry the game's own action names. Refused by a game that is not a two-player matrix game."""

    reply: str = "json"
    """How the model answers: `json` (the decision schema) or `move` (one short sentence, then a
    last line `MOVE: <option>`), which is the format much of the language-model game literature
    asks for."""


#: What the measurements in the paper were taken under: the observation, a gloss on what it means,
#: and the agent's own last rounds. Kept exactly as it was, because results recorded under it must
#: keep meaning what they meant.
MINIMAL_V1 = PromptProfile(
    name="minimal-v1", history=6, others_explicitly=False, payoff_rule=False,
    cumulative_payoff=False, round_position=False, decode_state=False,
    information="observation+history",
)

#: Everything the game already publishes, said plainly. This is not a richer model or a better
#: prompt technique, it is the same board described without arithmetic the model has to do first.
INFORMED_V1 = PromptProfile(
    name="informed-v1", history=6, others_explicitly=True, payoff_rule=True,
    cumulative_payoff=True, round_position=True, decode_state=True,
    information="observation+history+others",
)

#: What `informed-v1` tells, with no word that names a side and the option order drawn per seed.
#: The rule sentence is its own (`neutral_rules`) rather than `payoff_sentence`, and states the
#: multiplier as the pool's, which is how a reader computes a share from it. It also follows a game
#: whose return changes during the match, and says so on the round it changes.
NEUTRAL_V1 = PromptProfile(
    name="neutral-v1", history=6, others_explicitly=True, payoff_rule=True,
    cumulative_payoff=True, round_position=True, decode_state=True,
    information="observation+history+others", neutral_labels=True,
)

#: A repeated two-player matrix game with its whole history. The same content under two reply
#: formats, since the format is itself a variable: `pd-history-v1` answers in the decision schema,
#: `pd-history-move-v1` in a sentence and a `MOVE:` line.
PD_HISTORY_V1 = PromptProfile(
    name="pd-history-v1", history=0, others_explicitly=True, payoff_rule=True,
    cumulative_payoff=False, round_position=True, decode_state=False,
    information="observation+history+others", pd_history=True, reply="json",
)
PD_HISTORY_MOVE_V1 = PromptProfile(
    name="pd-history-move-v1", history=0, others_explicitly=True, payoff_rule=True,
    cumulative_payoff=False, round_position=True, decode_state=False,
    information="observation+history+others", pd_history=True, reply="move",
)

PROFILES: dict[str, PromptProfile] = {
    p.name: p for p in (MINIMAL_V1, INFORMED_V1, NEUTRAL_V1, PD_HISTORY_V1, PD_HISTORY_MOVE_V1)}


def pd_system(game: Any, labels: dict[str, str], reply: str) -> str:
    """The system message of a `pd_history` profile: horizon, options, payoffs, what to reply.

    Raises for a game that does not declare four payoffs for two players, since a table written
    for a game this module does not know would be a claim about it."""
    if getattr(game, "n_agents", None) != 2 or not all(
            hasattr(game, k) for k in ("T", "R", "P", "S")):
        raise ValueError(f"a pd_history profile needs a two-player matrix game with payoffs T, R, "
                         f"P and S; {getattr(game, 'name', type(game).__name__)!r} is not one")
    c, d = labels["concede"], labels["claim"]
    rounds = getattr(game, "rounds", None)
    horizon = f"a {rounds}-round game" if rounds else "a repeated game"
    text = (f"You are playing {horizon} against the same opponent. Each round you both choose "
            f"{c} or {d} at the same time. Points per round: both {c}: {game.R:g} each. You {c} "
            f"and they {d}: you {game.S:g}, they {game.T:g}. You {d} and they {c}: you "
            f"{game.T:g}, they {game.S:g}. Both {d}: {game.P:g} each. Aim for the highest total "
            f"for yourself.")
    if reply == "move":
        return text + (f" Give one short sentence of reasoning, then end with a last line of "
                       f"exactly 'MOVE: {c}' or 'MOVE: {d}'.")
    return text + (f' Reply with a single JSON object and nothing else, with the fields "action" '
                   f'(one of ["{c}", "{d}"]), "rationale" (one short sentence on why) and '
                   f'"confidence" (a number between 0 and 1).')

#: The labels a neutral profile offers, in the order they are listed.
NEUTRAL_LABELS = ("Option A", "Option B")

DEFAULT_PROFILE = MINIMAL_V1.name


def get(name: str) -> PromptProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(
            f"unknown prompt profile {name!r}. Known profiles: {', '.join(sorted(PROFILES))}. "
            f"Add one in agents/llm_prompt.py rather than passing prompt text, so that every "
            f"model tested sees the same board and the choice is recorded with the run."
        ) from None


def payoff_sentence(game: Any) -> str:
    """One sentence about how this game pays, built from what the game declares.

    Returns an empty string for a game that declares nothing usable, which is the honest outcome:
    a sentence invented here would be a claim about a game this module does not know.
    """
    mpcr = getattr(game, "mpcr", None)
    cost = getattr(game, "cost", None)
    n = getattr(game, "n_agents", None)
    if mpcr is not None and cost is not None and n:
        return (f"Conceding costs you {cost:g}. Everything conceded is multiplied by {mpcr:g} "
                f"and shared equally among all {n} players, whether they conceded or not.")

    full = getattr(game, "full_reward", None)
    positions = getattr(game, "num_positions", None)
    if full is not None and positions:
        return (f"The first player to reach the end alone takes {full:g}. If several arrive "
                f"together the reward is divided, and if everyone arrives it can be nothing.")
    return ""


def state_sentence(game: Any, observation: int) -> str:
    """The observation rendered in the game's own words, when it can render it."""
    labels = getattr(game, "state_labels", None)
    if not callable(labels):
        return ""
    try:
        rendered = labels()
    except Exception:
        return ""
    if not rendered or not (0 <= observation < len(rendered)):
        return ""
    return str(rendered[observation])


def state_legend(game: Any) -> str:
    """The game's own explanation of how to read its labels, if it offers one.

    A decoded label without its convention is a number with more digits. Games that can explain
    themselves declare `state_label_legend`; games that cannot are simply not decoded.
    """
    legend = getattr(game, "state_label_legend", None)
    if not callable(legend):
        return ""
    try:
        return str(legend())
    except Exception:
        return ""


def neutral_label_map(roles: Sequence[str], seed: int) -> dict[str, str]:
    """Which neutral label each role gets, drawn from `seed`.

    The labels are always listed A then B, so drawing which role is A is what puts the two options
    in a random order. Seeded, so a run can say which order its model saw, and recorded with the
    decisions for the same reason.
    """
    if len(roles) > len(NEUTRAL_LABELS):
        raise ValueError(f"a neutral profile labels at most {len(NEUTRAL_LABELS)} options, "
                         f"got {len(roles)} roles")
    order = list(roles)
    random.Random(seed).shuffle(order)
    return {role: label for role, label in zip(order, NEUTRAL_LABELS)}


def neutral_rules(game: Any, labels: dict[str, str]) -> list[tuple[int, str]]:
    """What each option does and how it pays, as (first round, sentence), in neutral words.

    A game whose return changes during a match (the Public Goods Game's `mpcr_schedule`) gives one
    entry per change. An empty list for a game this module does not know, rather than a sentence
    invented about it.
    """
    concede, claim = labels.get("concede"), labels.get("claim")
    n = getattr(game, "n_agents", None)

    mpcr_at = getattr(game, "mpcr_at", None)
    cost = getattr(game, "cost", None)
    if callable(mpcr_at) and cost is not None and n and concede:
        starts = [1] + [start for start, _ in getattr(game, "mpcr_schedule", ())]
        return [(start,
                 f"Each player who chooses {concede} pays {cost:g} into a pool. The pool is "
                 f"multiplied by {mpcr_at(start) * n:g} and shared equally among all {n} players, "
                 f"whether they chose {concede} or not.")
                for start in starts]

    full = getattr(game, "full_reward", None)
    positions = getattr(game, "num_positions", None)
    if full is not None and positions and concede and claim:
        terminal = positions - 1
        return [(1, f"Each round, {claim} moves you one cell forward and {concede} keeps you where "
                    f"you are. The first player to reach cell {terminal} alone takes {full:g}. If "
                    f"several arrive together the reward is divided, and if everyone arrives it "
                    f"can be nothing. Then everyone starts again from cell 0.")]
    return []


def rule_for_round(rules: list[tuple[int, str]], round_number: int) -> str:
    """The rule in force in `round_number`, opened with a notice on the round it changed."""
    current, start = "", 0
    for first, sentence in rules:
        if round_number >= first:
            current, start = sentence, first
    if current and start > 1 and round_number == start:
        return f"From this round on the rule has changed. {current}"
    return current


def others_conceded(observation: int, n_agents: Optional[int],
                    own_last_was_concede: Optional[bool]) -> Optional[int]:
    """How many of the *other* players conceded, from a count that includes the agent.

    Returns None when the observation is the start marker or when the arithmetic cannot be done,
    rather than guessing, since an off-by-one here is exactly the error the profile exists to
    prevent. The same subtraction is made by the active-inference agent for the same reason.
    """
    if n_agents is None or observation is None:
        return None
    if observation > n_agents:                      # the start marker, nothing has happened yet
        return None
    if own_last_was_concede is None:                # first decision, no action of ours in the count
        return observation
    return max(0, observation - int(own_last_was_concede))
