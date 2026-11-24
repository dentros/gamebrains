"""A game written as data: a JSON document that picks one of the platform's game families and sets
its parameters, so a game can be made in the interface, saved, shared and loaded without anyone
writing or running code.

    {
      "schema": "gamebrains.game/1",
      "name": "Prisoner's Dilemma 5/3/1/0",
      "description": "The textbook payoffs.",
      "family": "matrix_2x2",
      "rounds": 100,
      "params": {"T": 5, "R": 3, "P": 1, "S": 0, "action_names": ["Cooperate", "Defect"]}
    }

**Nothing in a document is executed.** It can only choose among the families below and set the
parameters each one declares, so a document from anyone is safe to open, and every value is checked
twice: here for its shape and then by the game's own constructor, which keeps its own rules (the
Public Goods Game's social-dilemma condition, for instance). A key this module does not know is
refused rather than ignored, for the factory's reason: a setting accepted and dropped is recorded as
one the run used.

Families
  matrix_2x2     games/matrix_game.py: T, R, P, S, optional action_names; two players
  public_goods   games/public_goods.py: mpcr, cost, allow_non_dilemma, mpcr_schedule; n players
  congestion     games/congestion.py: a preset and any of its axes; n players

**Approval.** A saved game carries `approval`: who approved it, when, and the hash of the definition
it approved. It is given by the person who adds the game, the same way for every game, and is a
record only. A document whose definition no longer matches its approval's hash is not offered for
play until it is approved again.

`GameStore` keeps saved games as one JSON file each in a folder (`custom_games/` by default).
`conformance` builds the game and plays a short match with every agent kind, so the person adding a
game sees which kinds play it and, for any that cannot, the reason the kind gives.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA = "gamebrains.game/1"
FAMILIES = ("matrix_2x2", "public_goods", "congestion")
TOP_KEYS = {"schema", "name", "description", "family", "rounds", "params", "approval"}
PARAM_KEYS = {
    "matrix_2x2": {"T", "R", "P", "S", "action_names"},
    "public_goods": {"mpcr", "cost", "allow_non_dilemma", "mpcr_schedule"},
    "congestion": {"preset", "num_positions", "reward_rule", "collapse_at_full",
                   "memory_episodes", "episode_max_rounds", "full_reward"},
}
#: The number of players a document is checked and previewed with. The Match tab seats any number
#: the family's own rules allow.
DEFAULT_PLAYERS = {"matrix_2x2": 2, "public_goods": 4, "congestion": 3}
MAX_ROUNDS = 1_000_000
MAX_NAME = 80
MAX_DESCRIPTION = 600


class SpecError(ValueError):
    """A document that is not a game this platform can build, naming the field at fault."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field
        self.message = message


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(field, f"must be a number, got {value!r}")
    return value


def validate(raw: Any) -> dict[str, Any]:
    """The document as a clean dict, or SpecError naming what is wrong.

    Checks the shape here and then builds the game once, so the game's own rules have the last
    word. The returned dict holds only known keys, in a fixed order.
    """
    if not isinstance(raw, dict):
        raise SpecError("document", "must be a JSON object")
    unknown = sorted(set(raw) - TOP_KEYS)
    if unknown:
        raise SpecError("document", f"unknown keys {unknown}")
    if raw.get("schema") != SCHEMA:
        raise SpecError("schema", f"must be {SCHEMA!r}, got {raw.get('schema')!r}")
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip() or len(name) > MAX_NAME:
        raise SpecError("name", f"must be a non-empty text of at most {MAX_NAME} characters")
    description = raw.get("description", "")
    if not isinstance(description, str) or len(description) > MAX_DESCRIPTION:
        raise SpecError("description", f"must be text of at most {MAX_DESCRIPTION} characters")
    family = raw.get("family")
    if family not in FAMILIES:
        raise SpecError("family", f"must be one of {list(FAMILIES)}, got {family!r}")
    rounds = raw.get("rounds")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or not 1 <= rounds <= MAX_ROUNDS:
        raise SpecError("rounds", f"must be a whole number from 1 to {MAX_ROUNDS:,}")
    params = raw.get("params", {})
    if not isinstance(params, dict):
        raise SpecError("params", "must be a JSON object")
    unknown = sorted(set(params) - PARAM_KEYS[family])
    if unknown:
        raise SpecError("params", f"unknown for {family}: {unknown}; it takes "
                                  f"{sorted(PARAM_KEYS[family])}")

    clean_params = _check_params(family, params)
    spec: dict[str, Any] = {"schema": SCHEMA, "name": name.strip(),
                            "description": description.strip(), "family": family,
                            "rounds": rounds, "params": clean_params}
    try:
        build(spec, n_agents=DEFAULT_PLAYERS[family])
    except SpecError:
        raise
    except (ValueError, TypeError) as exc:
        raise SpecError("params", str(exc)) from None
    if "approval" in raw:
        spec["approval"] = _check_approval(raw["approval"])
    return spec


def _check_params(family: str, params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if family == "matrix_2x2":
        for key in ("T", "R", "P", "S"):
            if key not in params:
                raise SpecError(f"params.{key}", "is required for a 2x2 game")
            out[key] = _number(params[key], f"params.{key}")
        if "action_names" in params:
            names = params["action_names"]
            if not isinstance(names, list) or len(names) != 2 or \
                    not all(isinstance(a, str) for a in names):
                raise SpecError("params.action_names", "must be a list of two names")
            out["action_names"] = [a.strip() for a in names]
    elif family == "public_goods":
        for key in ("mpcr", "cost"):
            if key in params:
                out[key] = _number(params[key], f"params.{key}")
        if "allow_non_dilemma" in params:
            if not isinstance(params["allow_non_dilemma"], bool):
                raise SpecError("params.allow_non_dilemma", "must be true or false")
            out["allow_non_dilemma"] = params["allow_non_dilemma"]
        if params.get("mpcr_schedule"):
            schedule = params["mpcr_schedule"]
            if not isinstance(schedule, list) or not all(
                    isinstance(e, list) and len(e) == 2 for e in schedule):
                raise SpecError("params.mpcr_schedule", "must be a list of [round, mpcr] pairs")
            out["mpcr_schedule"] = [[e[0], _number(e[1], "params.mpcr_schedule")]
                                    for e in schedule]
    else:
        from .congestion import PRESETS
        preset = params.get("preset", "hjg")
        if preset not in PRESETS:
            raise SpecError("params.preset", f"must be one of {sorted(PRESETS)}, got {preset!r}")
        out["preset"] = preset
        for key in ("num_positions", "memory_episodes", "episode_max_rounds"):
            if key in params:
                value = params[key]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise SpecError(f"params.{key}", "must be a whole number")
                out[key] = value
        if "full_reward" in params:
            out["full_reward"] = _number(params["full_reward"], "params.full_reward")
        if "reward_rule" in params:
            if not isinstance(params["reward_rule"], str):
                raise SpecError("params.reward_rule", "must be a rule name")
            out["reward_rule"] = params["reward_rule"]
        if "collapse_at_full" in params:
            if not isinstance(params["collapse_at_full"], bool):
                raise SpecError("params.collapse_at_full", "must be true or false")
            out["collapse_at_full"] = params["collapse_at_full"]
    return out


def _check_approval(approval: Any) -> dict[str, str]:
    if not isinstance(approval, dict) or set(approval) != {"by", "at", "hash"} or \
            not all(isinstance(v, str) and v.strip() for v in approval.values()):
        raise SpecError("approval", "must hold exactly 'by', 'at' and 'hash', each a text")
    return {"by": approval["by"].strip(), "at": approval["at"], "hash": approval["hash"]}


def definition(spec: dict[str, Any]) -> dict[str, Any]:
    """Everything that defines the game, which is everything but the approval."""
    return {k: v for k, v in spec.items() if k != "approval"}


def spec_hash(spec: dict[str, Any]) -> str:
    canonical = json.dumps(definition(spec), sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def approve(spec: dict[str, Any], by: str, at: Optional[str] = None) -> dict[str, Any]:
    """The spec with its approval: who, when, and the hash of what they approved."""
    if not isinstance(by, str) or not by.strip():
        raise SpecError("approval.by", "the person approving must give a name")
    stamped = definition(spec)
    stamped["approval"] = {
        "by": by.strip(),
        "at": at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hash": spec_hash(stamped),
    }
    return stamped


def is_approved(spec: dict[str, Any]) -> bool:
    """An approval exists and covers this definition, not an earlier one."""
    approval = spec.get("approval")
    return bool(approval) and approval.get("hash") == spec_hash(spec)


def build(spec: dict[str, Any], n_agents: Optional[int] = None, rounds: Optional[int] = None):
    """The game a spec describes, for `n_agents` players and `rounds` rounds (the spec's own by
    default). The constructor of the family checks the values."""
    family, params = spec["family"], dict(spec.get("params", {}))
    rounds = rounds if rounds is not None else spec["rounds"]
    if family == "matrix_2x2":
        from .matrix_game import MatrixGame
        if n_agents not in (None, 2):
            raise SpecError("players", f"a 2x2 game is played by two players, not {n_agents}")
        return MatrixGame(rounds=rounds, **params)
    if family == "public_goods":
        from .public_goods import PublicGoodsGame
        schedule = params.pop("mpcr_schedule", None)
        return PublicGoodsGame(n_agents=n_agents or 4, rounds=rounds,
                               mpcr_schedule=[tuple(e) for e in schedule] if schedule else None,
                               **params)
    from .congestion import from_preset
    preset = params.pop("preset", "hjg")
    return from_preset(preset, n_agents or 3, **params)


def players(spec: dict[str, Any]) -> str:
    """How many may play, in words for a page."""
    return "2" if spec["family"] == "matrix_2x2" else f"{DEFAULT_PLAYERS[spec['family']]} or more"


def slug(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return (text or "game")[:48]


class GameStore:
    """Saved games, one JSON file each. A file that no longer validates is listed with its fault
    and never offered for play."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, spec: dict[str, Any]) -> str:
        if not is_approved(spec):
            raise SpecError("approval", "a game is saved only once it is approved")
        self.root.mkdir(parents=True, exist_ok=True)
        base = slug(spec["name"])
        name, k = base, 2
        while (self.root / f"{name}.json").exists():
            existing = self._read(self.root / f"{name}.json")
            if isinstance(existing, dict) and spec_hash(existing) == spec_hash(spec):
                break                                    # the same game again: overwrite in place
            name, k = f"{base}-{k}", k + 1
        (self.root / f"{name}.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False),
                                                encoding="utf-8")
        return name

    def _read(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def list(self) -> list[dict[str, Any]]:
        out = []
        if not self.root.exists():
            return out
        for path in sorted(self.root.glob("*.json")):
            raw = self._read(path)
            try:
                spec = validate(raw)
                fault = "" if is_approved(spec) else "not approved for this definition"
            except SpecError as exc:
                spec, fault = (raw if isinstance(raw, dict) else {}), str(exc)
            out.append({"slug": path.stem, "spec": spec, "fault": fault})
        return out

    def load(self, slug_: str) -> dict[str, Any]:
        """A saved, approved game, or SpecError saying why it cannot be played."""
        if not re.fullmatch(r"[a-z0-9-]{1,60}", slug_ or ""):
            raise SpecError("game", f"no saved game is called {slug_!r}")
        path = self.root / f"{slug_}.json"
        if not path.exists():
            raise SpecError("game", f"no saved game is called {slug_!r}")
        spec = validate(self._read(path))
        if not is_approved(spec):
            raise SpecError("approval", "this game's definition has changed since it was approved")
        return spec


def conformance(spec: dict[str, Any], rounds: int = 20) -> list[dict[str, Any]]:
    """Which agent kinds play this game, from a short match each, and why the others do not.

    Each kind sits beside a fixed strategy that plays any game declaring the two universal roles,
    so a refusal belongs to the kind under test. Where the family takes more than two players, one
    more match seats one of every kind that played, as the mixed check. The language-model seat
    needs a model server and is reported as not run here.
    """
    from ..agents import factory
    from ..engine.runner import run_match

    n = DEFAULT_PLAYERS[spec["family"]]
    results: list[dict[str, Any]] = []
    playing: list[str] = []
    seats = [(kind, None) for kind in factory.KINDS if kind != "classic"]
    seats += [("classic", s) for s in ("AllC", "AllD", "Random", "MajorityTFT")]
    for kind, strategy in seats:
        label = f"classic {strategy}" if strategy else kind
        try:
            game = build(spec, n_agents=n, rounds=rounds)
            agent = factory.build(kind, 0, game, 0, classic_strategy=strategy or "AllD")
            partners = [factory.build("classic", i, game, 0, classic_strategy="AllC")
                        for i in range(1, n)]
            run_match(game, [agent] + partners, rounds=rounds, seed=0)
            results.append({"kind": label, "status": "plays", "detail": ""})
            if not strategy:
                playing.append(kind)
        except Exception as exc:                      # the reason is the point of the check
            results.append({"kind": label, "status": "refuses",
                            "detail": f"{type(exc).__name__}: {exc}"})
    results.append({"kind": "llm", "status": "not run", "detail": "needs a model server"})
    if spec["family"] != "matrix_2x2" and len(playing) >= 2:
        try:
            game = build(spec, n_agents=len(playing), rounds=rounds)
            roster = [factory.build(kind, i, game, 0) for i, kind in enumerate(playing)]
            run_match(game, roster, rounds=rounds, seed=0)
            results.append({"kind": "mixed: " + ", ".join(playing), "status": "plays",
                            "detail": ""})
        except Exception as exc:
            results.append({"kind": "mixed: " + ", ".join(playing), "status": "refuses",
                            "detail": f"{type(exc).__name__}: {exc}"})
    return results
