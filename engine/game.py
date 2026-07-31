"""
The Game contract.

Every game in GameBrains is n-player-native (n >= 2). A game is a *repeated* stage
game: `reset()` starts a new match, and each `step(actions)` advances one round in which
all agents move simultaneously. This keeps the engine agnostic to the specific game so
that the N-player Prisoner's Dilemma (Public Goods), MBoE (Battle of the Exes), and future
games all plug into the same loop, metrics, and visualization.

Observations are integers so that tabular agents (Q-learning) can index a Q-table directly;
`n_states` bounds that index. Richer agents (DQN/FEP/LLM) may ignore the integer and read
the full round context from `StepResult.info` instead.

## Declaring what your actions and observations mean

An action index means nothing on its own. Action 1 is "cooperate" in the Public Goods Game and
"move toward the jar" in a congestion game, which are close to opposites: contributing helps the
group, grabbing the resource jams it. An agent that hardcodes `1` is therefore correct in one game
and silently backwards in the other. That is a real bug this platform shipped, found only when the
second game arrived.

So a game declares its own semantics, and agents ask for meaning rather than for numbers. Two
declarations, both optional, both just class attributes:

  `action_roles`      which of *your* action indices fills each named role
  `observation_kind`  what the integer you hand out actually means

Declare only what your game genuinely has. An agent that needs a role you did not declare will
refuse to play, loudly, which is the entire point: a clear failure at setup beats a silent
misinterpretation that quietly corrupts a whole experiment.

### The role vocabulary

Roles are context words, and a game may use whichever fit. Two levels, by design:

**Universal**, for any game with a tension between individual and collective payoff. Declare these
if they apply, and every role-aware agent written by anyone will work with your game:

  `concede`  the action that forgoes individual gain
  `claim`    the action that pursues it

**Family-specific**, so a game can also speak its own literature's language. These are aliases,
pointing at the same indices:

  social dilemmas       `cooperate` / `defect`
  congestion families   `yield` / `contest`

A coordination game such as Battle of the Sexes has no concede/claim axis at all: the question is
*which* option to meet on, not whether to give way. Such a game declares only its own roles
(`option_a` / `option_b`), and an always-concede agent correctly refuses it instead of pretending.

Adding a new word is allowed and expected. Prefer an existing one where it honestly fits, since
every new word is one more thing an agent author has to know about.

### The observation vocabulary

  `opaque`         (default) just an index. Promises nothing, so no agent may interpret it.
  `concede_count`  the integer is how many agents played `concede` last round, and the value
                   `n_agents + 1` means "first round, nothing has happened yet". Reciprocating
                   strategies need exactly this.
  `board_index`    an encoded full board state. Positional, not countable.

Default to `opaque` when unsure. It costs you only the agents that need to read the number, and it
never misleads one.

### Example

    class MyGame(Game):
        n_actions = 2
        action_names = ["Hold back", "Grab"]
        observation_kind = "concede_count"
        action_roles = {"concede": 0, "claim": 1,      # universal, so generic agents work
                        "yield": 0, "contest": 1}      # this family's own words

See `docs/adding-a-game.md` for the full walkthrough.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    """Outcome of one round (one `step`)."""

    observations: list[int]          # per-agent observation index for the *next* round
    rewards: list[float]             # per-agent payoff for this round
    done: bool                       # True when the match is over
    info: dict[str, Any] = field(default_factory=dict)  # rich per-round context for metrics/viz


class Game(ABC):
    """Abstract base class for an n-player repeated game."""

    #: number of players (>= 2)
    n_agents: int
    #: number of discrete actions available to each player
    n_actions: int
    #: human-readable action labels, len == n_actions (e.g. ["Defect", "Cooperate"])
    action_names: list[str]
    #: number of distinct observation indices an agent may receive
    n_states: int
    #: short identifier used in logs and filenames
    name: str
    #: what the integer in `observations` means -- see the module docstring's vocabulary.
    #: The default promises nothing, which is the safe answer.
    observation_kind: str = "opaque"
    #: semantic role -> this game's own action index. Empty means "no roles declared", so
    #: role-aware agents will decline this game rather than guess.
    action_roles: dict[str, int] = {}

    def action_for(self, role: str) -> int:
        """This game's action index for a semantic role, or a refusal that says what it does have.

        Agents call this instead of hardcoding an index, so the same strategy means the right
        thing in every game that declares the role, and visibly fails in games that do not.
        """
        try:
            return self.action_roles[role]
        except KeyError:
            declared = ", ".join(sorted(self.action_roles)) or "none"
            raise ValueError(
                f"{type(self).__name__} does not declare the {role!r} action role, so an agent "
                f"asking for it cannot play this game meaningfully. Roles it does declare: "
                f"{declared}. See engine/game.py's module docstring."
            ) from None

    def require_observation_kind(self, *kinds: str) -> None:
        """Refuse early if this game's observation is not one an agent knows how to read.

        Anything beyond an index into a table is a promise the game has to make explicitly;
        reading `concede_count` semantics out of an `opaque` index is how a strategy ends up
        confidently acting on a number that means something else entirely.
        """
        if self.observation_kind not in kinds:
            raise ValueError(
                f"{type(self).__name__} hands out {self.observation_kind!r} observations, but "
                f"this agent can only interpret {' or '.join(repr(k) for k in kinds)}. "
                f"See engine/game.py's module docstring."
            )

    @abstractmethod
    def reset(self) -> list[int]:
        """Begin a new match. Returns the initial per-agent observation indices."""
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: list[int]) -> StepResult:
        """Advance one round given every agent's action. See `StepResult`."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """Serializable description of the game configuration (goes into the event-log header)."""
        return {
            "name": self.name,
            "n_agents": self.n_agents,
            "n_actions": self.n_actions,
            "action_names": list(self.action_names),
            "n_states": self.n_states,
            # Part of the design, not decoration: a game that assigns its roles to different
            # actions is a different game, and every role-aware agent in the roster behaves
            # differently in it. It therefore belongs in what config_hash covers.
            "observation_kind": self.observation_kind,
            "action_roles": dict(self.action_roles),
        }
