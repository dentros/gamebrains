# Adding an agent to GameBrains

An agent plugs into the platform by implementing one small interface. Everything else, the games,
the runner, the metrics, the event log, the web UI, works from that interface without knowing what
is inside your agent or how it decides.

This guide is the contract, and it is the counterpart to `adding-a-game.md`. One part of it is
enforced at runtime: a match refuses to start if your agent has not said how it relates to the
meaning of action numbers. That check exists because of a real defect in this codebase, described in
section 3.

## 1. The interface

```python
from gamebrains.engine.agent import Agent

class MyAgent(Agent):
    kind = "myagent"                 # short tag; the UI picks a renderer from it
    training_mode = "online"         # "online" | "fixed" | "evolutionary"
    semantics = "index-agnostic"     # REQUIRED, see section 3

    def __init__(self, name: str, n_states: int, n_actions: int, seed: int = 0):
        self.name = name
        ...

    def act(self, observation: int) -> int:
        """Choose an action index given this round's observation index."""

    def update(self, observation, action, reward, next_observation, done) -> None:
        """Learn from one transition. No-op for fixed strategies."""
```

Requirements the platform relies on:

- **Deterministic given a seed.** Same seed and configuration must reproduce byte-identical play.
  Hold your own generator, seeded in `__init__`. Never touch the global `random` or `np.random`
  state, and note that this includes convenience calls such as `random.sample`, which reads the
  module-level generator even when every other line in the class is properly seeded. That exact
  line broke reproducibility here and no test caught it, because no test used the agent that had it.
- **Do not write to the game.** Your only output is the action you return.
- **`act` must return an index in `range(n_actions)`.** Anything else is a crash the game cannot
  interpret for you.

## 2. Transparency, which is the point of the platform

Two optional hooks are what make an agent visible rather than a black box. Implement them, or your
agent will run correctly and show nothing.

```python
    def inspect(self) -> dict:
        """The raw internal state: a Q-table, weights, a belief vector, a genome."""

    def render_brain(self) -> dict:
        """A small, serializable, stable-shaped payload the UI can draw."""
```

Keep `render_brain` small and keep its **shape** stable across versions, because a consumer reads
its keys. Key the payload by role rather than by display wording where the two can differ, and carry
the wording separately. A renderer that reads `probs["Cooperate"]` will silently display zero the
day that key becomes `probs["concede"]`, which is a failure with no error message.

## 3. Declaring how you relate to the meaning of actions

**This is the enforced part.** `semantics` is a class attribute with no default. A match refuses to
start if it is unset, or if you declare a dependence on meaning and then never resolve it.

Choose one:

### `"index-agnostic"`

Your agent works over action indices without interpreting them. A tabular learner, a network whose
output head is indexed by action, an evolved genome whose motor nodes map positionally: all of these
learn whichever column pays and never ask what a column stands for. Reversing a game's action
meanings changes what such an agent learns, not whether it is correct.

Declare this and you are done. You may play any game, and you need no `on_match_start`.

### `"role-bound"`

Your agent's behaviour depends on what the numbers mean. Anything defined as a *strategy*, in words,
is in this class: "always give way", "do what the majority did", "cooperate until betrayed".

You must resolve the meaning against the specific game before play:

```python
    semantics = "role-bound"

    def on_match_start(self, game) -> None:
        game.require_observation_kind("concede_count")   # if you read the observation
        self._concede = game.action_for("concede")       # never hardcode an index
        self._claim = game.action_for("claim")
```

Both calls may refuse, and a refusal is the correct outcome rather than a failure. A game with no
concede/claim axis, or one handing out board positions when you need a count of conceders, cannot
support your strategy, and declining it is better than acting on a number that means something else.
See `adding-a-game.md` for the vocabulary a game may declare.

### Why this is enforced rather than documented

`agents/classic.py` once imported `COOPERATE` from the Public Goods Game, pinning "cooperate" to
action 1. Action 1 in the congestion family is *move*, which grabs the contested resource. So the
agent labelled `AllC` played the most aggressive move available while still calling itself
cooperative. Nothing failed. The game ran, the metrics computed, the numbers looked ordinary. It was
found only because a second, structurally different game finally existed to expose it.

A misbound match produces numbers indistinguishable from valid ones, and there is no later stage
that can tell them apart, so the only place to stop is before the first round.

### What the check does not do

It cannot tell whether a declaration is *true*. An agent declaring `"index-agnostic"` while
hardcoding an index still gets through, and so does a `"role-bound"` agent that resolves the wrong
role. What the declaration buys is that skipping the question is impossible and getting it wrong is
an explicit, reviewable claim rather than an omission. Do not read `semantics` as a guarantee that
your agent is correct.

## 4. Wrapping another agent

If you write a wrapper (freezing a trained policy, logging, adapting), inherit the wrapped agent's
declaration rather than stating your own:

```python
    def __init__(self, inner, name):
        self._inner = inner
        self.semantics = inner.semantics      # never answer for it
```

A wrapper that declared for itself would launder a role-bound agent through as index-agnostic, which
is exactly the hole the check exists to close. Forward `on_match_start` too, or the wrapped strategy
never learns which action plays its role.

## 5. Testing it

Copy the shape of `tests/test_dqn.py`. At minimum:

- **Determinism.** Two agents with the same seed produce identical action sequences. Test this in
  the **near-greedy** regime (low epsilon), not at default exploration: at high epsilon most actions
  come from your own seeded generator and a broken agent compares as identical. That is how the
  defect in section 1 hid.
- **Different seeds differ.** Otherwise the determinism test is satisfiable by a frozen agent.
- **It learns**, if it claims to: some internal state changes after `update`.
- **Binding**, if role-bound: assert that the same strategy resolves to *opposite* indices in two
  games with opposed assignments, and that it refuses a game that cannot support it.

Run: `python -m gamebrains.tests.test_myagent`, then the whole suite with
`python -m gamebrains.tests.run_all`.

## 6. Checklist

- [ ] `semantics` declared, `"index-agnostic"` or `"role-bound"`
- [ ] role-bound agents resolve every role and observation kind in `on_match_start`
- [ ] no hardcoded action index anywhere in the class
- [ ] own seeded generator; no global `random` / `np.random`, including `random.sample`
- [ ] `act` returns an index in `range(n_actions)`
- [ ] `render_brain()` is small, stable in shape, and keyed by role where wording may vary
- [ ] determinism tested near-greedy, and different seeds shown to differ
- [ ] wrappers inherit `semantics` and forward `on_match_start`
