"""
Game #2: the congestion family, with the Honey-Jar Game as its headline member.

`n` agents race along a corridor of `num_positions` cells toward a single high-reward terminal.
Each round every agent simultaneously chooses to stay or step forward. The episode ends the moment
anyone arrives. Arriving alone pays `full_reward`; arriving together pays a reduced share; under
the published main rule, everyone arriving at once pays nothing at all, the jar jammed.

This is one parametrized family rather than one game, because the source papers themselves treat
it that way: the Honey-Jar Game is described there as "a minimally dynamic, repeated
threshold-congestion game" that "stands on its own footing within the congestion and market-entry
family". The axes below are exactly the ones those papers vary, so a named preset reproduces a
published configuration and the parameters let you leave the published grid deliberately.

  Source papers, both by the platform's own authors (cite when publishing results from this file):
    Papadopoulos, Freire, Sanchez-Fibla, Psannis, "The Coordination Gap: Multi-Agent Alternation
      Metrics for Temporal Fairness in Repeated Games". arXiv:2603.05789,
      Zenodo 10.5281/zenodo.18528891
    Papadopoulos, Freire, Sanchez-Fibla, Psannis, "Temporal Fair Division in Multi-Agent Systems:
      From Precise Alternation Metrics to Scalable Coordination Proxies". arXiv:2605.14879;
      conference version MDAI 2025, doi:10.1007/978-3-032-03711-4_16
  Ported from the authors' own `2. ALT MEASURES TO MEGALO/src/environment.py`, with the learning
  rule removed: there it lived inside `Environment.step()`, which only works when every agent is
  the same type. Here the game is brain-agnostic and learning lives in each `Agent.update()`, so a
  Q-learner, a DQN and a fixed rule can share one match.

Naming: HJG throughout. "MBoE" is a historical name the authors no longer use, and the Battle of
the Exes is a *predecessor*, not a special case: BoE pays both players a positive but asymmetric
reward when they differ, while this game is winner-take-all. The two are not equivalent even at
n=2, so BoE belongs in its own module rather than as a preset here.

Deliberately NOT presets: the El Farol Bar Problem and the Minority Game. Both are cited neighbours
of this family, but they turn on machinery this module does not have (attendance history driving a
population of strategy tables, a below-capacity payoff rather than an exclusivity one). Adding
labels for them here would claim more than the code does.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from ..engine.game import Game, StepResult

STAY, MOVE = 0, 1

#: Reward denominators, as (base, exponent). The share for a partial tie is
#: `full_reward / base**exponent`, where "n" is the whole population and "k" the number who
#: actually arrived together. The n-family keeps the penalty independent of how many happened to
#: collide, which is what the papers chose a priori to avoid confounding the collision count with
#: the alternation signal they measure. The k-family is literal Rosenthal congestion: crowding your
#: own resource is what costs you.
_REWARD_RULES: dict[str, tuple[str, int]] = {
    "ILF": ("n", 1),   # Inverse Linear Fractional, r/n      -- published, the main rule
    "IQF": ("n", 2),   # Inverse Quadratic Fractional, r/n^2 -- published
    "ICF": ("n", 3),   # r/n^3    -- EXPLORATORY, not in either paper
    "KLF": ("k", 1),   # r/k      -- published, the k-variant robustness check
    "KQF": ("k", 2),   # r/k^2    -- published, the commons-style counterpart
    "KCF": ("k", 3),   # r/k^3    -- EXPLORATORY, not in either paper
}

#: Rules the source papers actually ran. Anything outside this set is a legitimate experiment but
#: should not be reported as reproducing a published result.
PUBLISHED_REWARD_RULES = frozenset({"ILF", "IQF", "KLF", "KQF"})

#: A Q-table is n_states x n_actions per agent, and n_states grows as
#: num_positions^n * 2^(n*memory_episodes). Refuse politely rather than exhaust memory: at n=10
#: with one episode of memory that product is already about 60 million.
_MAX_STATES = 5_000_000


class CongestionGame(Game):
    """One episode is a race: agents advance along the corridor until somebody reaches the end.

    Args:
        n_agents: number of players, >= 2.
        num_positions: cells in the corridor, >= 2. The terminal is the last one, so an agent
            needs `num_positions - 1` moves to arrive. **This is the one-shot / dynamic dial.**
            At 2 a single move ends the episode, so the game is one-shot (the papers' term for
            this is "ballistic", borrowed from the Battle of the Exes literature). At 3 or more
            there is at least one intermediate cell where an agent's approach is visible before it
            commits, which is what makes the game "minimally dynamic" and lets movement itself act
            as a coordination signal.
        reward_rule: a key of `_REWARD_RULES`, or "custom" with `reward_base`/`reward_exponent`.
        reward_base, reward_exponent: only for reward_rule="custom". The named rules are just
            presets over this pair, and `describe()` always reports the resolved pair, so a custom
            rule that happens to equal a named one is correctly treated as the same design.
        full_reward: the payoff for arriving alone.
        collapse_at_full: if every single agent arrives together, pay exactly zero instead of the
            formula's share. Published main behaviour is True. The published k-variant used
            False, where `full_reward / k` with k = n simply yields a small share; the `hjg_k`
            preset sets that combination for you.
        memory_episodes: how many past episodes' arrival vectors are visible in the state. 0 is
            the papers' Type-A (positions only), 1 is Type-B (positions plus who arrived last
            episode), and higher generalizes beyond what they ran. Costs a factor of 2^n states
            per episode remembered.
        episode_max_rounds: abort an episode after this many rounds with nobody arriving. A cap is
            not optional: if every agent stays put the corridor never advances and the episode
            would never end. Defaults to a generous ten times the corridor length, so hitting it
            means genuine mutual stalling rather than a tight leash. A timed-out episode IS
            reported, with nobody marked as having arrived, because "everyone refused" is a real
            outcome of this game.
    """

    def __init__(
        self,
        n_agents: int,
        num_positions: int = 3,
        reward_rule: str = "ILF",
        full_reward: float = 100.0,
        collapse_at_full: bool = True,
        memory_episodes: int = 0,
        episode_max_rounds: int | None = None,
        reward_base: str | None = None,
        reward_exponent: int | None = None,
    ) -> None:
        if n_agents < 2:
            raise ValueError(f"congestion games need at least 2 agents, got {n_agents}")
        if num_positions < 2:
            raise ValueError(
                f"num_positions must be at least 2 (2 = one-shot), got {num_positions}")
        if memory_episodes < 0:
            raise ValueError(f"memory_episodes cannot be negative, got {memory_episodes}")

        if reward_rule == "custom":
            if reward_base not in ("n", "k") or reward_exponent is None:
                raise ValueError(
                    "reward_rule='custom' needs reward_base ('n' or 'k') and reward_exponent")
            base, exponent = reward_base, int(reward_exponent)
        elif reward_rule in _REWARD_RULES:
            base, exponent = _REWARD_RULES[reward_rule]
        else:
            raise ValueError(
                f"unknown reward_rule {reward_rule!r}; "
                f"expected one of {sorted(_REWARD_RULES)} or 'custom'")
        if exponent < 1:
            raise ValueError(f"reward_exponent must be at least 1, got {exponent}")

        self.name = "congestion"
        self.n_agents = n_agents
        self.n_actions = 2
        self.action_names = ["Stay", "Move"]
        # The board index encodes positions (and any remembered arrivals), so it is not a count
        # of anything: an agent that tried to read it as one would be acting on a number that
        # means something else entirely.
        self.observation_kind = "board_index"
        # Holding back is what forgoes your own shot at the jar so someone else can get through,
        # which puts STAY on the `concede` side even though it is numerically 0 here and
        # cooperating is 1 in the Public Goods Game. Getting this backwards is exactly the bug
        # the role mechanism exists to prevent.
        self.action_roles = {"concede": STAY, "claim": MOVE,
                             "yield": STAY, "contest": MOVE}

        self.num_positions = num_positions
        self.reward_rule = reward_rule
        self.reward_base = base
        self.reward_exponent = exponent
        self.full_reward = float(full_reward)
        self.collapse_at_full = bool(collapse_at_full)
        self.memory_episodes = memory_episodes
        self.episode_max_rounds = (
            episode_max_rounds if episode_max_rounds is not None else 10 * (num_positions - 1))
        if self.episode_max_rounds < num_positions - 1:
            raise ValueError(
                f"episode_max_rounds={self.episode_max_rounds} is below the {num_positions - 1} "
                f"moves needed to reach the terminal, so no episode could ever be won")

        self.n_states = (num_positions ** n_agents) * (2 ** (n_agents * memory_episodes))
        if self.n_states > _MAX_STATES:
            raise ValueError(
                f"this configuration needs {self.n_states:,} states "
                f"({num_positions}^{n_agents} positions x 2^({n_agents}x{memory_episodes}) "
                f"memory), over the {_MAX_STATES:,} cap. Reduce memory_episodes, num_positions, "
                f"or n_agents: memory costs a factor of 2^n per episode remembered.")

        # Survives reset(): the whole point of Type-B is that an episode can see the previous
        # one. Cleared only by new_match().
        self._memory: deque[list[int]] = deque(
            [[0] * n_agents for _ in range(memory_episodes)], maxlen=memory_episodes or 1)
        self._positions = [0] * n_agents
        self._round_in_episode = 0

    # --- state ---------------------------------------------------------------------------

    def _encode(self) -> int:
        """Positions in base `num_positions`, with the remembered arrival vectors as high bits.

        Computed arithmetically rather than through the source's base-N string parsing, which
        silently caps num_positions at 10 because it relies on single decimal digits.
        """
        code = 0
        for p in self._positions:
            code = code * self.num_positions + p
        if self.memory_episodes:
            past = 0
            for vector in self._memory:          # oldest first, consistently
                for flag in vector:
                    past = past * 2 + flag
            code += past * (self.num_positions ** self.n_agents)
        return code

    def _observe(self) -> list[int]:
        """Every agent sees the same public board, so one index repeated per seat."""
        return [self._encode()] * self.n_agents

    def new_match(self) -> None:
        """Forget the cross-episode memory. `reset()` deliberately does not, since it starts the
        next episode of the same match and Type-B must carry over."""
        self._memory = deque(
            [[0] * self.n_agents for _ in range(self.memory_episodes)],
            maxlen=self.memory_episodes or 1)

    def reset(self) -> list[int]:
        self._positions = [0] * self.n_agents
        self._round_in_episode = 0
        return self._observe()

    # --- play ----------------------------------------------------------------------------

    def _share(self, k: int) -> float:
        if self.collapse_at_full and k == self.n_agents:
            return 0.0
        denominator = (self.n_agents if self.reward_base == "n" else k) ** self.reward_exponent
        return self.full_reward / denominator

    def step(self, actions: list[int]) -> StepResult:
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions, got {len(actions)}")

        self._round_in_episode += 1
        terminal = self.num_positions - 1
        arrived = [0] * self.n_agents
        for i, action in enumerate(actions):
            if action == MOVE and self._positions[i] < terminal:
                self._positions[i] += 1
                if self._positions[i] == terminal:
                    arrived[i] = 1

        k = sum(arrived)
        rewards = [0.0] * self.n_agents
        if k:
            share = self.full_reward if k == 1 else self._share(k)
            for i, did in enumerate(arrived):
                if did:
                    rewards[i] = share

        timed_out = not k and self._round_in_episode >= self.episode_max_rounds
        done = bool(k) or timed_out

        info: dict[str, Any] = {
            "positions": list(self._positions),
            "arrived": arrived,
            "terminal_occurrences": k,
            "round_in_episode": self._round_in_episode,
            # `cooperators` is what the runner logs per round for every game. Here the natural
            # reading is restraint: an agent that held back rather than pushing for the jar.
            "cooperators": sum(1 for a in actions if a == STAY),
        }
        if done:
            # Exactly the two series the ALT/RP metrics consume, per episode: who reached the
            # terminal (ties included), and how many did. Exclusive wins are then `sum(...) == 1`,
            # never `len(...) == 1`, since this is a flag per agent and not a list of winners.
            info["episode"] = {
                "top_agents": arrived,
                "terminal_occurrences": k,
                "timed_out": timed_out,
                "rounds": self._round_in_episode,
            }
            if self.memory_episodes:
                self._memory.append(arrived)

        return StepResult(observations=self._observe(), rewards=rewards, done=done, info=info)

    # --- description ---------------------------------------------------------------------

    @property
    def state_type(self) -> str:
        """The papers' label for this memory depth, so records stay readable in their terms."""
        if self.memory_episodes == 0:
            return "Type-A"
        return "Type-B" if self.memory_episodes == 1 else f"Type-B({self.memory_episodes})"

    @property
    def is_one_shot(self) -> bool:
        """A two-cell corridor is decided by a single simultaneous move. The papers call this the
        ballistic case; there is no approach to observe, so no room for movement to signal."""
        return self.num_positions == 2

    def describe(self) -> dict[str, Any]:
        """Self-contained enough to reconstruct the game, not just label it: every field that
        changes play is here, and the reward rule is reported as its resolved (base, exponent)
        rather than only its name. Written this way so it can become a shareable, addressable game
        object later without the description having to be redefined.

        Extends the base description rather than replacing it, so the declared semantics
        (observation kind, action roles) travel with every record without this game having to
        remember to repeat them.
        """
        described = super().describe()
        described.update({
            "family": "congestion",
            "num_positions": self.num_positions,
            "timing": "one_shot" if self.is_one_shot else "dynamic",
            "reward_rule": self.reward_rule,
            "reward_base": self.reward_base,
            "reward_exponent": self.reward_exponent,
            "reward_published": self.reward_rule in PUBLISHED_REWARD_RULES,
            "full_reward": self.full_reward,
            "collapse_at_full": self.collapse_at_full,
            "memory_episodes": self.memory_episodes,
            "state_type": self.state_type,
            "episode_max_rounds": self.episode_max_rounds,
        })
        return described


#: Named configurations. The first three reproduce published setups; `market_entry` is the closest
#: one-shot relative, which the Gap paper names as such while noting the two differences this
#: preset encodes: a single simultaneous entry decision, and a payoff that declines per entrant
#: instead of stepping to a fixed share.
PRESETS: dict[str, dict[str, Any]] = {
    "hjg": {
        "label": "Honey-Jar Game (ILF)",
        "note": "The published main configuration: dynamic corridor, fixed per-capita share for "
                "any partial tie, zero when everyone arrives at once.",
        "params": {"num_positions": 3, "reward_rule": "ILF", "collapse_at_full": True,
                   "memory_episodes": 0},
    },
    "hjg_iqf": {
        "label": "Honey-Jar Game (IQF)",
        "note": "As above with the quadratic share r/n^2, the harsher published tie penalty.",
        "params": {"num_positions": 3, "reward_rule": "IQF", "collapse_at_full": True,
                   "memory_episodes": 0},
    },
    "hjg_k": {
        "label": "Honey-Jar Game (k-variant)",
        "note": "The published robustness check. Per-claimant split r/k is literal Rosenthal "
                "congestion, and there is no zero floor: everyone arriving simply yields r/n.",
        "params": {"num_positions": 3, "reward_rule": "KLF", "collapse_at_full": False,
                   "memory_episodes": 0},
    },
    "hjg_memory": {
        "label": "Honey-Jar Game (Type-B)",
        "note": "The main configuration with one episode of memory: agents see who arrived last "
                "episode, which is what makes deliberate turn-taking learnable.",
        "params": {"num_positions": 3, "reward_rule": "ILF", "collapse_at_full": True,
                   "memory_episodes": 1},
    },
    "market_entry": {
        "label": "Market entry (one-shot)",
        "note": "A single simultaneous entry decision with a per-entrant declining payoff. The "
                "closest one-shot relative of HJG, kept separate because it drops the observable "
                "approach that makes the corridor version dynamic.",
        "params": {"num_positions": 2, "reward_rule": "KLF", "collapse_at_full": False,
                   "memory_episodes": 0},
    },
}


def from_preset(key: str, n_agents: int, **overrides: Any) -> CongestionGame:
    """Build a preset configuration, with any parameter overridable for a deliberate variation."""
    if key not in PRESETS:
        raise ValueError(f"unknown preset {key!r}; expected one of {sorted(PRESETS)}")
    return CongestionGame(n_agents=n_agents, **{**PRESETS[key]["params"], **overrides})
