# Adding a game to GameBrains

A game plugs into the platform by implementing one small interface. Everything else, the agents,
the metrics, the event log, the repository, the web UI, works from that interface without knowing
which game it is talking to.

This guide is the contract. If you follow it, every agent anyone has written will play your game
correctly, or refuse it clearly. If you skip the declaration section, agents will still run, but
some of them will act on numbers that mean something else in your game, and nothing will warn you.

## 1. The interface

```python
from gamebrains.engine.game import Game, StepResult

class MyGame(Game):
    name = "my_game"          # short id, appears in logs and filenames
    n_actions = 2
    action_names = ["Hold back", "Grab"]

    def __init__(self, n_agents: int, ...):
        self.n_agents = n_agents
        self.n_states = ...   # how many distinct observation indices you may hand out

    def reset(self) -> list[int]:
        """Start an episode. Returns one observation index per agent."""

    def step(self, actions: list[int]) -> StepResult:
        """One simultaneous move by everyone."""
```

`StepResult` carries `observations`, `rewards`, `done`, and a free-form `info` dict.

Requirements the platform relies on:

- **n-player native.** `n_agents >= 2` is a parameter, not a constant. A two-player-only game does
  not belong here.
- **Deterministic given a seed.** Same seed and config must reproduce byte-identical play. Take a
  seed and hold your own generator; never touch the global `random` or `np.random` state.
- **Observations are integers** in `range(n_states)`, so a tabular agent can index a table
  directly. Richer context goes in `info`, which agents that want it can read.

## 2. Declaring what your numbers mean

This is the part that is easy to skip and expensive to skip.

An action index means nothing on its own. Action `1` is "cooperate" in the Public Goods Game and
"move toward the jar" in the congestion game, which are close to opposites: contributing helps the
group, grabbing the contested resource jams it. An agent that hardcodes `1` is right in one game
and silently backwards in the other.

That is not hypothetical. It shipped here. `AllC` imported `COOPERATE` from the Public Goods Game,
so in the congestion game it played the most aggressive strategy available while still calling
itself cooperative. The tests passed, because the game ran fine. Only the meaning was wrong.

So: your game declares its own semantics, and agents ask for meaning instead of for numbers.

### Action roles

```python
action_roles = {"concede": 0, "claim": 1}
```

Two levels of vocabulary, on purpose.

**Universal.** Declare these if your game has any tension between individual and collective
payoff, and every role-aware agent, including ones written after your game, will work with it:

| Role | Meaning |
|---|---|
| `concede` | the action that forgoes individual gain |
| `claim` | the action that pursues it |

**Family-specific.** Aliases pointing at the same indices, so your game can also speak the
language of its own literature:

| Family | Roles |
|---|---|
| social dilemmas | `cooperate` / `defect` |
| congestion, anti-coordination | `yield` / `contest` |

A game may declare both:

```python
action_roles = {"concede": 0, "claim": 1,     # universal, so generic agents work
                "yield": 0,   "contest": 1}   # this family's own words
```

**Declare only what you genuinely have.** A coordination game such as Battle of the Sexes has no
concede/claim axis at all: the question is *which* option to meet on, not whether to give way. It
declares `{"option_a": 0, "option_b": 1}` and nothing else, and an always-concede agent then
refuses it with a clear message instead of pretending it fits.

Adding a new role word is allowed and expected. Prefer an existing one where it honestly fits,
since every new word is one more thing an agent author has to learn.

### Observation kind

```python
observation_kind = "concede_count"
```

| Value | Meaning |
|---|---|
| `opaque` | (default) just an index. Promises nothing, so no agent may interpret it. |
| `concede_count` | the integer is how many agents played `concede` last round, and `n_agents + 1` means "first round". Reciprocating strategies need exactly this. |
| `board_index` | an encoded full state. Positional, not countable. |

Default to `opaque` when unsure. It costs you only the agents that need to read the number, and it
never misleads one. Promising `concede_count` and handing out something else is the failure this
field exists to prevent.

### What agents do with it

```python
def on_match_start(self, game):
    self._action = game.action_for("concede")       # raises if not declared
    game.require_observation_kind("concede_count")  # raises if not readable
```

The runner calls `on_match_start` once before play. Both helpers raise a message naming what your
game *does* declare, so the failure is actionable at setup rather than a wrong number at round
40,000.

## 3. Episodes

If your game ends episodes early, set `done=True` and the runner restarts it with `reset()` until
the round budget runs out. One match then contains many episodes.

`rounds` passed to `run_match` is always a total round budget, never an episode count.

On the round where you set `done`, put whatever the episode meant into `info["episode"]`:

```python
info["episode"] = {"top_agents": [0, 1, 0],    # who reached the terminal, ties included
                   "terminal_occurrences": 1}
```

The runner adds per-agent reward totals and the round count, then emits one `episode` event. The
alternation metrics (ALT/RP) read exactly this series, since they ask *who* won and *when*. A
per-round log cannot answer that.

Two things to know:

- **A cap is not optional** if your episode can fail to end. If every agent can stall forever, the
  episode never terminates. Take an `episode_max_rounds` and end the episode when it is hit, with
  nobody marked as having arrived. That is a real outcome, not an error.
- `reset()` starts the *next episode of the same match*, so anything meant to carry across
  episodes (a memory of who won last time) must survive it. Provide a separate way to clear that
  if you need one.

## 4. `describe()`

```python
def describe(self):
    described = super().describe()          # name, n_agents, actions, roles, observation kind
    described.update({"my_param": self.my_param})
    return described
```

**Call `super()`.** The base fills in the declared semantics, and a game that rebuilds the dict
from scratch silently drops them.

Everything here goes into the experiment record and into `config_hash`, which is how the platform
decides whether two runs are the same experiment. So include every parameter that changes play,
and nothing that does not. See `docs/repository-schema.md` §4a for the three-tier rule.

Report resolved values, not just labels. The congestion game stores its reward rule as a
`(base, exponent)` pair as well as a name, so a custom rule spelling out the same formula is
correctly recognised as the same design rather than as something new.

## 5. Testing it

Assert against the payoff arithmetic you intend, not against whatever the code currently produces.
`tests/test_congestion.py` checks each reward case against the formulas in the source papers, so a
later refactor that changes the numbers fails loudly.

Worth covering:

- every payoff branch, by hand
- each edge of the parameter space you allow, and that impossible configurations are refused with
  a reason
- one end-to-end `run_match`, verifying the episode series looks right
- if you declare roles, that a role-aware agent picks the action you meant

Run one file with `python -m gamebrains.tests.test_my_game` from the project root.

## 6. Checklist

- [ ] `n_agents` is a parameter, `>= 2`
- [ ] deterministic under a seed, with its own generator
- [ ] observations are integers in `range(n_states)`
- [ ] `action_roles` declared, universal roles included if they apply
- [ ] `observation_kind` declared, `opaque` unless you really promise more
- [ ] `describe()` calls `super()` and reports every parameter that changes play
- [ ] episodic games: `info["episode"]` on the done round, and a cap that cannot hang
- [ ] tests assert the intended arithmetic
