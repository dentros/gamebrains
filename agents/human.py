"""A seat a person plays, under the same contract as every other architecture.

The point of putting a human in the roster is not novelty. It is that every claim this platform
makes about comparing architectures is a claim about one interface, and a person is the hardest
thing to fit behind it: a person is slow, is not seeded, and cannot be handed an action index and
asked for a number. If the contract survives that, it is a real contract.

Three consequences follow, and each one reuses a mechanism that already exists rather than
inventing a parallel path for people.

**A person is asked in words, so this seat is `role-bound`.** It resolves `concede` and `claim`
against the game the same way `agents/llm.py` does, for the same reason: an index means nothing
without the game that defines it, and "press 1" is a question nobody can answer correctly. The
words the page shows come from the game's own declaration, so one unchanged seat plays both
families and a game declaring neither role is refused before the first round.

**A person is not reproducible, and the platform already has a word for that.** `reproducibility()`
returns `none`, which is the tier written for a served language model, and the recorder withholds
the byte-identical claim for any run containing one. Nothing here needed adding: a run with a
person in it is exactly as unrepeatable as a run with a model in it, and it is honest for both to
say so through the same field.

**A person takes seconds, and the runner is synchronous.** `act` blocks on a queue until a move
arrives, so the match runs in a thread and `engine/runner.run_match` is reused unchanged -- with
its semantics binding, its event log, its episode handling and its metrics. The alternative,
driving `game.step()` round by round from the web layer, would have duplicated the runner, and a
second copy of the runner is how the two quietly stop agreeing.

**There is no timeout substitution.** If nobody answers, the wait raises and the match stops. A
move invented on a person's behalf would enter the event log indistinguishable from one they made,
which is the same rule `agents/llm.py` applies when its repair budget runs out and it refuses to
supply a fallback action.
"""

from __future__ import annotations

import queue
from typing import Any, Optional

from ..engine.agent import Agent


class WaitingForPerson(RuntimeError):
    """Nobody supplied a move in time, so the match stops rather than inventing one."""


class HumanAgent(Agent):
    """One seat driven by a person, through a queue the interface fills."""

    kind = "human"
    #: Behaviour depends on what the actions mean, so the game must be asked. Declaring
    #: index-agnostic here would be a lie that `engine/semantics.bind_agents` cannot catch: it
    #: checks that a role-bound agent asked, not that an index-agnostic one truly did not care.
    semantics = "role-bound"
    #: What a person conditions on is whatever the page puts in front of them, which is the
    #: current observation plus the rounds already played. Declared to match what the interface
    #: actually shows; widening the display without widening this would make the record wrong.
    information = "observation+history"
    training_mode = "none"

    def __init__(self, name: str, wait_seconds: float = 600.0) -> None:
        self.name = name
        self.wait_seconds = wait_seconds
        #: Moves arrive here from whatever is driving the interface. Size one on purpose: a second
        #: move offered before the first is consumed is a double submit, and silently queueing it
        #: would play it a round later, against a board the person never saw.
        self._moves: queue.Queue[int] = queue.Queue(maxsize=1)
        #: What the seat is currently waiting to be answered, for the interface to render.
        self.pending: Optional[dict[str, Any]] = None
        self._concede: Optional[int] = None
        self._claim: Optional[int] = None
        self._labels: dict[int, str] = {}
        self._history: list[dict[str, Any]] = []
        self._round = 0

    # --- the contract -----------------------------------------------------------------------

    def on_match_start(self, game: Any) -> None:
        """Resolve the two moves in this game's own words, and keep its labels for the page."""
        self._concede = game.action_for("concede")
        self._claim = game.action_for("claim")
        names = list(getattr(game, "describe", dict)().get("action_names") or [])
        self._labels = {index: (names[index] if index < len(names) else str(index))
                        for index in (self._concede, self._claim)}

    def act(self, observation: int) -> int:
        """Show the board, then block until somebody answers."""
        if self._concede is None or self._claim is None:
            raise RuntimeError(
                "this seat has not been matched to a game yet, so it does not know what its "
                "moves mean. The runner calls on_match_start before play begins; a caller "
                "driving agents directly has to do the same")
        self.pending = {
            "round": self._round,
            "observation": observation,
            "choices": [
                {"action": self._concede, "role": "concede", "label": self._labels[self._concede]},
                {"action": self._claim, "role": "claim", "label": self._labels[self._claim]},
            ],
            "history": list(self._history[-10:]),
        }
        try:
            action = self._moves.get(timeout=self.wait_seconds)
        except queue.Empty as exc:
            raise WaitingForPerson(
                f"{self.name} was asked for a move at round {self._round} and none arrived within "
                f"{self.wait_seconds:.0f} seconds, so the match stopped. No move is substituted: "
                f"one invented here would be indistinguishable in the event log from a move the "
                f"person made") from exc
        finally:
            self.pending = None
        return action

    def update(self, observation: int, action: int, reward: float,
               next_observation: int, done: bool) -> None:
        """A person learns; this object does not. It only keeps what the page shows them."""
        self._history.append({
            "round": self._round,
            "action": action,
            "role": "concede" if action == self._concede else "claim",
            "label": self._labels.get(action, str(action)),
            "reward": round(float(reward), 4),
        })
        self._round += 1

    def inspect(self) -> dict[str, Any]:
        return {"rounds_played": self._round, "waiting": self.pending is not None,
                "history": list(self._history)}

    def render_brain(self) -> dict[str, Any]:
        """There is no internal state to draw, and saying so is better than drawing nothing.

        Every other seat answers this with a Q-table, a network, a belief or a circuit. A person's
        reasoning is not available to the platform at all, which is a real difference between this
        seat and the others rather than a gap in the implementation, so the panel says that
        instead of rendering an empty box.
        """
        total = sum(entry["reward"] for entry in self._history)
        conceded = sum(1 for entry in self._history if entry["role"] == "concede")
        return {
            "kind": self.kind,
            "opaque": True,
            "rounds_played": self._round,
            "conceded": conceded,
            "total_reward": round(total, 4),
            "note": "a person's reasoning is not visible to the platform, so there is no "
                    "internal state to show. What the run records is the same as for any seat: "
                    "the moves, and what they cost.",
        }

    # --- reproducibility --------------------------------------------------------------------

    def reproducibility(self) -> str:
        return "none"

    def reproducibility_note(self) -> str:
        return ("a person chose these moves, so the same configuration replayed will not "
                "reproduce them")

    # --- what the interface drives -----------------------------------------------------------

    def submit(self, action: int) -> None:
        """Hand the waiting seat its move. Raises if it is not waiting or the move is not legal."""
        if action not in (self._concede, self._claim):
            raise ValueError(f"{action!r} is not one of this game's two moves "
                             f"({self._concede} or {self._claim})")
        try:
            self._moves.put_nowait(action)
        except queue.Full as exc:
            raise RuntimeError("a move is already waiting to be played, so this one would be "
                               "played against a board you have not seen yet") from exc

    def submit_role(self, role: str) -> None:
        """The same, in the words the page shows, since that is what a person actually picked."""
        if role == "concede" and self._concede is not None:
            self.submit(self._concede)
        elif role == "claim" and self._claim is not None:
            self.submit(self._claim)
        else:
            raise ValueError(f"{role!r} is not a move this game declares")
