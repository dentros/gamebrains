"""An agent whose policy is a language model asked for a structured decision.

The sixth architecture in the zoo, and the first whose output is text. Everything difficult
about it follows from that one fact, and each difficulty is handled in the open rather than
smoothed over:

**It is role-bound, necessarily.** A model is asked to `concede` or to `claim`, never to emit
an action index, because an index means nothing without the game that defines it and a model
asked for "0 or 1" will answer plausibly and wrongly. The words come from the game's own
`action_roles` declaration (see `engine/game.py`), so the same agent plays the Public Goods
Game and the congestion family without a line of per-game code, and refuses a game that
declares neither role rather than guessing. That refusal is the designed outcome: a pure
coordination game has no concede axis, and an agent built around one has nothing to say there.

**Its answer can be malformed, and that is measured rather than hidden.** The decision is
requested as JSON against a schema. Where the server can constrain decoding it does, and where
it cannot the repair loop in `llm_backends.run_with_repair` feeds the validation errors back to
the model. Both paths record every attempt, so the cost of getting a usable answer out of a
model is a number this platform reports instead of an implementation detail it absorbs.

**It cannot support the reproducibility claim, and says so.** Every other agent here is
deterministic given its seed, which is what lets one `config_hash` promise one byte-identical
event log. No LLM backend can promise that, so a roster containing one carries the backend's
own `deterministic=False` up to the recorder, which withholds the claim for that run instead of
weakening it for every run. See `repository/record.py`.

**Exhausting the repair budget stops the match.** There is no fallback action, deliberately. A
substituted move would enter the event log indistinguishable from a decision the model made,
and every metric downstream would treat it as one.

    from gamebrains.agents.llm import LLMAgent
    from gamebrains.agents.llm_ollama import OllamaBackend

    agent = LLMAgent("LLM 0", OllamaBackend(model="llama3.2:1b"))
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..engine.agent import Agent
from .llm_backends import DEFAULT_MAX_ATTEMPTS, Backend, Response

#: The roles the agent asks a game for. Both must exist or the pairing is refused: an agent that
#: could only concede would not be making a decision.
DEFAULT_ROLES = ("concede", "claim")

#: How many past rounds go into the prompt. Small on purpose. A long history is expensive on every
#: single decision, and a 1B model does not use more of it, so the number is a cost we would be
#: paying for the appearance of context.
DEFAULT_HISTORY = 6

_OBSERVATION_GLOSS = {
    "concede_count": ("how many players took the conceding action in the previous round "
                      "(a value of n_agents + 1 means the match has not started yet)"),
    "board_index": "an encoded index of the whole board position, which is not a count",
    "opaque": "an index with no declared meaning, so read nothing into its value",
}


def decision_schema(roles: Sequence[str]) -> dict[str, Any]:
    """The JSON object a decision must be.

    Three fields rather than one, because a single enum is the case constrained decoding makes
    trivial and therefore the case that measures nothing. `rationale` is free text, which no
    server constrains, and `confidence` is a bounded number, which models routinely answer with a
    word. The gap between a constrained and an unconstrained run of this same schema is what
    `experiments/run_llm_contract.py` reports.
    """
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(roles)},
            "rationale": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["action", "rationale", "confidence"],
    }


class LLMAgent(Agent):
    """A language model in a seat, deciding by name and not by number.

    Args:
        name: display name.
        backend: anything satisfying `llm_backends.Backend`.
        roles: the action roles to offer the model, resolved against the game at match start.
        history: how many past rounds to put in the prompt.
        max_attempts: repair budget per decision.
        persona: an optional line describing who the model is playing as. Empty by default,
            since a persona is an experimental manipulation rather than a default setting.
    """

    kind = "llm"
    #: Its weights never change. The behaviour varies with the history in the prompt, exactly as a
    #: reciprocating classic strategy's does, and neither is learning.
    training_mode = "fixed"
    semantics = "role-bound"
    #: The prompt carries the last history rounds verbatim, which is a different thing from the
    #: others: raw past rather than a summary, and its own past only, since nothing in the prompt
    #: says what the other players did beyond the observation itself.
    information = "observation+history"

    def __init__(self, name: str, backend: Backend, roles: Sequence[str] = DEFAULT_ROLES,
                 history: int = DEFAULT_HISTORY, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                 persona: str = "") -> None:
        self.name = name
        self.backend = backend
        self.roles = tuple(roles)
        self.history = history
        self.max_attempts = max_attempts
        self.persona = persona

        self._role_actions: dict[str, int] = {}
        self._action_roles: dict[int, str] = {}
        self._game_context: dict[str, Any] = {}
        self._log: list[dict[str, Any]] = []
        self._journal: list[dict[str, Any]] = []
        self._last: Optional[Response] = None
        self._last_decision: dict[str, Any] = {}
        self._pending_action: Optional[int] = None

    # --- binding ----------------------------------------------------------------------

    def on_match_start(self, game: Any) -> None:
        """Resolve every role against this game, or decline the pairing.

        `game.action_for` raises with a message naming what the game does declare, which is the
        refusal path `engine/semantics.py` enforces. Nothing is caught here: a game with no
        concede axis is a game this agent should not play.
        """
        self._role_actions = {role: game.action_for(role) for role in self.roles}
        self._action_roles = {index: role for role, index in self._role_actions.items()}
        self._game_context = {
            "name": getattr(game, "name", type(game).__name__),
            "n_agents": getattr(game, "n_agents", None),
            "observation_kind": getattr(game, "observation_kind", "opaque"),
            "action_names": {role: getattr(game, "action_names", [])[index]
                             for role, index in self._role_actions.items()
                             if index < len(getattr(game, "action_names", []))},
        }
        self._log = []
        self._journal = []
        self._last = None
        self._last_decision = {}

    def _unbound(self) -> str:
        return (f"{self.name} has not been matched to a game yet, so it does not know which action "
                f"each role plays. The runner calls on_match_start before play begins.")

    # --- what the recorder needs to know ----------------------------------------------

    @property
    def model(self) -> str:
        return self.backend.capabilities().model

    @property
    def backend_name(self) -> str:
        return self.backend.capabilities().name

    @property
    def native_schema(self) -> bool:
        """Whether the server constrained decoding to the schema. Part of the design, not a
        runtime detail: the same model answering with and without that constraint is two
        experiments, so it belongs in what `config_hash` covers."""
        return self.backend.capabilities().native_schema

    def reproducibility(self) -> str:
        """Read from the backend rather than assumed, so a backend that can genuinely promise
        something is believed. The replay backend of `agents/llm_replay.py` is the one that can."""
        caps = self.backend.capabilities()
        if not caps.deterministic:
            return "none"
        return "replay" if caps.name == "replay" else "byte"

    def reproducibility_note(self) -> str:
        caps = self.backend.capabilities()
        return caps.note if self.reproducibility() != "byte" else ""

    def journal(self) -> list[dict[str, Any]]:
        """Every decision this agent made, in order, with the prompt that produced it.

        Recorded into the run's stored package so the run can be replayed exactly
        (`agents/llm_replay.py`). The prompt is kept in full rather than hashed alone, because a
        replay that cannot show what was asked is not evidence of anything, and because the hash is
        what the lookup uses while the text is what a reader checks it against.
        """
        return list(self._journal)

    # --- the prompt -------------------------------------------------------------------

    def build_prompt(self, observation: int) -> str:
        """The full text sent to the model for one decision.

        Public because the experiment script uses the same builder to generate its prompts. A
        study that measured a hand-written prompt would be reporting the reliability of a string
        that never plays a match.
        """
        ctx = self._game_context
        kind = ctx.get("observation_kind", "opaque")
        gloss = _OBSERVATION_GLOSS.get(kind, "an index with no declared meaning")

        names = ctx.get("action_names", {})
        options = "\n".join(
            f"- {role}: {names.get(role, role)}" for role in self.roles)

        lines = [
            f"You are one of {ctx.get('n_agents', '?')} players in a repeated game "
            f"called {ctx.get('name', 'the game')}.",
        ]
        if self.persona:
            lines.append(self.persona)
        lines += [
            "",
            "Your options this round:",
            options,
            "",
            f"The observation you are given is {observation}, which is {gloss}.",
        ]

        if self._log:
            lines += ["", "Your last rounds, most recent first:"]
            for entry in reversed(self._log[-self.history:]):
                lines.append(f"- you played {entry['role']} and received {entry['reward']:.2f}")
        else:
            lines += ["", "No rounds have been played yet."]

        lines += [
            "",
            "Reply with a single JSON object and nothing else, with the fields:",
            f'  "action": one of {list(self.roles)}',
            '  "rationale": one short sentence on why',
            '  "confidence": a number between 0 and 1',
        ]
        return "\n".join(lines)

    # --- the contract -----------------------------------------------------------------

    def act(self, observation: int) -> int:
        if not self._role_actions:
            raise RuntimeError(self._unbound())

        prompt = self.build_prompt(observation)
        schema = decision_schema(self.roles)
        # A SchemaViolation here is left to propagate. The alternative is a substituted action,
        # which would be recorded as a decision the model never made.
        response = self.backend.complete(prompt, schema, max_attempts=self.max_attempts)

        role = response.data["action"]
        self._last = response
        self._last_decision = {
            "role": role,
            "rationale": str(response.data.get("rationale", "")),
            "confidence": response.data.get("confidence"),
            "attempts": response.n_attempts,
            "repaired": response.repaired,
            "seconds": round(response.seconds, 3),
        }
        self._journal.append({
            "prompt": prompt,
            "data": dict(response.data),
            "attempts": response.n_attempts,
            "model": response.model,
        })
        self._pending_action = self._role_actions[role]
        return self._pending_action

    def update(self, observation: int, action: int, reward: float,
               next_observation: int, done: bool) -> None:
        """No learning happens. The transition is kept because the prompt is the memory."""
        entry = dict(self._last_decision)
        entry.update({"role": self._action_roles.get(action, "?"), "action": action,
                      "reward": float(reward)})
        self._log.append(entry)

    # --- transparency -----------------------------------------------------------------

    def inspect(self) -> dict[str, Any]:
        caps = self.backend.capabilities()
        return {
            "backend": caps.name,
            "model": caps.model,
            "capabilities": {"native_schema": caps.native_schema,
                             "activations": caps.activations,
                             "deterministic": caps.deterministic},
            "roles": dict(self._role_actions),
            "history": list(self._log),
            "last_attempts": [
                {"ok": a.ok, "error": a.error, "seconds": round(a.seconds, 3),
                 "prompt_chars": a.prompt_chars, "raw": a.raw}
                for a in (self._last.attempts if self._last else [])
            ],
        }

    def render_brain(self) -> dict[str, Any]:
        """The XAI panel's payload: what it decided, why it said it decided that, and what this
        backend cannot show. The last part matters as much as the first. A panel that silently
        omits activations for Ollama and shows them for a transformers backend teaches the reader
        that this model has none."""
        caps = self.backend.capabilities()
        decision = self._last_decision
        return {
            "kind": self.kind,
            "backend": caps.name,
            "model": caps.model,
            "role_labels": self._game_context.get("action_names", {}),
            "rationale": decision.get("rationale", ""),
            "role": decision.get("role", ""),
            "confidence": decision.get("confidence"),
            "attempts": decision.get("attempts", 0),
            "repaired": decision.get("repaired", False),
            "seconds": decision.get("seconds", 0.0),
            "native_schema": caps.native_schema,
            "activations": self._last.activations if self._last else None,
            "activations_available": caps.activations,
            "deterministic": caps.deterministic,
            "note": caps.note,
            "decisions": len(self._log),
        }
