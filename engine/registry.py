"""
Agent-type registry.

Maps a brain `kind` (e.g. "qlearning") to display + rendering metadata. The engine uses this
for the event-log header; a future "houses" UI can auto-generate one room per registered kind
(each room knows which renderer to use and how to theme itself). Registering here is the single
place a new brain announces itself to any frontend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentKind:
    kind: str                       # stable id, matches Agent.kind and render_brain()["kind"]
    display_name: str               # human label, e.g. "Q-Learning"
    renderer: str                   # frontend renderer id for render_brain() payloads
    training_mode: str              # "online" | "fixed" | "evolutionary"
    house: dict[str, Any] = field(default_factory=dict)  # future "house" theme metadata
    blurb: str = ""                 # one-line description


AGENT_REGISTRY: dict[str, AgentKind] = {}


def register_kind(kind: AgentKind) -> AgentKind:
    AGENT_REGISTRY[kind.kind] = kind
    return kind


def get_kind(kind: str) -> AgentKind | None:
    return AGENT_REGISTRY.get(kind)


def describe_registry() -> list[dict[str, Any]]:
    """Serializable snapshot of all registered kinds (for the event-log header)."""
    return [
        {
            "kind": k.kind,
            "display_name": k.display_name,
            "renderer": k.renderer,
            "training_mode": k.training_mode,
            "house": k.house,
            "blurb": k.blurb,
        }
        for k in AGENT_REGISTRY.values()
    ]


# --- Built-in kinds (brains register themselves on import, but we seed the known set here) ---
register_kind(AgentKind(
    kind="classic",
    display_name="Classic Strategy",
    renderer="rule",
    training_mode="fixed",
    house={"theme": "library", "emoji": "📜"},
    blurb="Fixed hand-written strategies (AllC, AllD, Random, Majority-TFT).",
))
register_kind(AgentKind(
    kind="qlearning",
    display_name="Q-Learning",
    renderer="qtable",
    training_mode="online",
    house={"theme": "grid", "emoji": "🧮"},
    blurb="Tabular Q-learning; its brain is an inspectable Q-table.",
))
register_kind(AgentKind(
    kind="dqn",
    display_name="Deep Q-Network",
    renderer="network",
    training_mode="online",
    house={"theme": "circuit", "emoji": "🕸️"},
    blurb="Q-learning with a neural-network function approximator.",
))
register_kind(AgentKind(
    kind="fep",
    display_name="Active Inference (FEP)",
    renderer="beliefs",
    training_mode="online",
    house={"theme": "observatory", "emoji": "🔮"},
    blurb="Free-energy-minimizing agent that models the others (theory of mind).",
))
register_kind(AgentKind(
    kind="markov_brain",
    display_name="Markov-Brain Animat",
    renderer="wiring",
    training_mode="evolutionary",
    house={"theme": "petri", "emoji": "🧬"},
    blurb="Evolved logic-gate network; small enough to measure integrated information (Φ).",
))
