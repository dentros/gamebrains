"""A trained agent held still: it acts greedily and never learns again.

Used by every bake-off, in `experiments/run_bakeoff.py` and in the web interface. Each architecture
finishes its own training or evolution separately, then all of them are frozen into one evaluation
match, so the comparison measures final policies rather than how much each adapted while being
measured.

**The wrapper is the agent for every purpose except learning, so it answers as the agent does.**
There used to be two copies of this class, one per entry point, and each inherited `semantics` and
nothing else. That made a frozen agent misdescribe itself in the record of every bake-off:

- `information` fell back to the base default, `observation`, so a frozen FEP agent or Markov brain,
  which condition on memory, and a frozen AllD, which conditions on nothing, were all recorded as
  conditioning on the current observation;
- the hyperparameters `repository/record.py` reads off the live agent were not there to read, so two
  frozen Q-learners trained with different learning rates had the same roster description and
  therefore the same `config_hash`;
- the reproducibility tier fell back to `byte` whatever the wrapped agent declared.

Play was unaffected; the description of what played was wrong. Found by an external review of the
code (the first item) and by reading the recorder while fixing it (the other two).

`training_mode` is the one property answered for the wrapper rather than the wrapped agent, since
a frozen learner is genuinely `fixed` in the match it is in.
"""

from __future__ import annotations

from typing import Any

from ..engine.agent import Agent


class Frozen(Agent):
    """Acts greedily, never learns, and otherwise describes itself as the agent it wraps."""

    training_mode = "fixed"

    def __init__(self, agent: Agent, name: str) -> None:
        self._agent = agent
        self.name = name
        self.kind = agent.kind
        # Declarations are inherited, never restated. A wrapper answering for itself would let a
        # role-bound strategy through as index-agnostic, or record a memoryless view of an agent
        # that has a memory.
        self.semantics = agent.semantics
        self.information = getattr(agent, "information", Agent.information)

    def __getattr__(self, name: str) -> Any:
        # Reached only for attributes the wrapper does not have itself, which is how the recorder's
        # per-kind hyperparameters (alpha, reciprocity, n_hidden, evolved_by ...) are found.
        # Private and dunder names are not forwarded, so copying or pickling a half-built instance
        # cannot recurse into a missing `_agent`.
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "epsilon" and hasattr(self._agent, "epsilon"):
            return 0.0          # what it actually plays with; see `act`
        return getattr(self._agent, name)

    def act(self, observation: int) -> int:
        saved = getattr(self._agent, "epsilon", None)
        if saved is not None:
            self._agent.epsilon = 0.0
        try:
            return self._agent.act(observation)
        finally:
            if saved is not None:
                self._agent.epsilon = saved

    def on_match_start(self, game: Any) -> None:
        # Forwarded, or a frozen role-bound strategy never learns which action plays its role.
        self._agent.on_match_start(game)

    def update(self, *args: Any, **kwargs: Any) -> None:
        return None

    def inspect(self) -> dict:
        return self._agent.inspect()

    def render_brain(self) -> dict:
        return self._agent.render_brain()

    def reproducibility(self) -> str:
        return self._agent.reproducibility()

    def reproducibility_note(self) -> str:
        return self._agent.reproducibility_note()
