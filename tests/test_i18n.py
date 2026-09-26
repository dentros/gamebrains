"""Every control a person can touch is named and explained, in both languages.

The interface grew field by field, and the explanations were written in one sitting afterwards.
That order is exactly how a field ends up unexplained: the next one added will be added by
somebody who is thinking about the feature rather than about the catalogue, and nothing on the
page looks wrong when a `\u24d8` is simply absent.

So the catalogue is checked against the templates rather than against itself. The tests below walk
the real HTML, collect every `name=` a user can set, and fail when one of them has no help text.
A new field therefore cannot reach a user unexplained: the suite says which key to write before
the feature is finished. This is the same rule the manuscript's guard applies to its own numbers,
for the same reason -- a claim nothing checks will drift.

Run:  "$GB/gamebrains/.venv/Scripts/python.exe" -m gamebrains.tests.test_i18n
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from ..webui import i18n

TEMPLATES = Path(__file__).resolve().parents[1] / "webui" / "templates"

#: Inputs that carry state rather than a decision: a form's `action` discriminator, the row
#: checkboxes that select records for comparison, the hidden copies the space map posts back to
#: itself. A person never chooses these, so there is nothing to explain, and demanding help text
#: for them would train whoever adds the next real field to add the exemption instead.
NOT_A_CHOICE = {
    "action",       # which submit button was pressed
    "rid",          # a record hash on a row checkbox
    "bundle",       # covered, but the file input repeats in the same form
    "op", "field", "value",   # the analytics filter rows, explained once by their labels
}

#: A field's help lives under `field.<name>` or `param.<name>`; a few read better under a name of
#: their own. Keys here rather than renaming the inputs, since the names are posted to routes.
ALIASES = {
    "row_kind": "field.row_kind",
    "row_count": "field.row_count",
    "row_classic_strategy": "param.strategy",
    "row_reciprocity": "param.reciprocity",
    "row_markov_hidden": "param.n_hidden",
    "row_genome_cid": "param.genome_cid",
    "row_epsilon_decay": "param.epsilon_pair",
    "row_epsilon_min": "param.epsilon_pair",
    "include_kind": "field.roster_has",
    "chart_x": "field.chart_x",
    "chart_y": "field.chart_y",
    "n_hidden": "param.n_hidden",
    "mpcr": "field.mpcr",
    "seed": "field.seed",
    "rounds": "field.rounds",
    "game": "field.game_kind",
    # The sweep calls the seat count `agents` where the match form calls it `n_agents`; both are
    # the same question to a person, so they share one explanation rather than drifting apart.
    "agents": "field.n_agents",
    "filter_op": "field.filter_operator",
}


def _named_inputs() -> dict[str, set[str]]:
    """Every `name=` a template offers, by file, with the hidden and repeated ones dropped."""
    found: dict[str, set[str]] = {}
    for path in sorted(TEMPLATES.glob("*.html")):
        text = path.read_text(encoding="utf-8")
        names = set()
        for tag in re.findall(r"<(?:input|select|textarea)\b[^>]*>", text, re.S):
            if 'type="hidden"' in tag:
                continue
            m = re.search(r'name="([a-zA-Z_][\w]*)(\[\])?"', tag)
            if m and m.group(1) not in NOT_A_CHOICE:
                names.add(m.group(1))
        if names:
            found[path.name] = names
    return found


def _key_for(name: str) -> str | None:
    """The catalogue key that explains a field, or None when nothing covers it."""
    if name in ALIASES and i18n.H.get(ALIASES[name]):
        return ALIASES[name]
    for candidate in (f"field.{name}", f"param.{name}"):
        if i18n.H.get(candidate):
            return candidate
    # `row_adv_<kind>_<param>` is generated per advanced parameter, so it resolves on the tail.
    if name.startswith("row_adv_"):
        tail = name.rsplit("_", 1)[-1]
        if i18n.H.get(f"param.{tail}"):
            return f"param.{tail}"
    return None


def test_every_field_is_explained() -> None:
    gaps = []
    for filename, names in _named_inputs().items():
        for name in sorted(names):
            if _key_for(name) is None:
                gaps.append(f"{filename}: {name}")
    assert not gaps, (
        "these controls have no help text, so they render without a badge. Add "
        "`field.<name>` or `param.<name>` to i18n.H:\n  " + "\n  ".join(gaps))


def test_both_languages_are_present() -> None:
    """A pair with an empty half would render a blank label or a badge with nothing in it."""
    broken = []
    for table, label in ((i18n.T, "T"), (i18n.H, "H")):
        for key, pair in table.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                broken.append(f"{label}[{key}] is not a (greek, english) pair")
            elif not pair[0].strip() or not pair[1].strip():
                broken.append(f"{label}[{key}] has an empty half")
    assert not broken, "\n  ".join(broken)


def test_greek_and_english_actually_differ() -> None:
    """Catches a pair left half-translated by copying one side into the other.

    A label that is a canonical name is identical in both languages on purpose, and `CANONICAL`
    is derived from that identity rather than listed, so this check is really asking a narrower
    question: is every such label one we would defend as a name? It fails when a label is
    identical *and* reads like prose, which is what a forgotten translation looks like.
    """
    prose = [key for key in i18n.CANONICAL
             if len(i18n.T[key][0].split()) > 3 or i18n.T[key][0].endswith(".")]
    assert not prose, (
        "these labels are identical in both languages but read like a sentence, so they look "
        f"like a translation somebody forgot rather than a name: {prose}")


def test_every_game_explains_itself() -> None:
    """A game the selector offers must say what it asks of its players.

    This is the check that makes the platform's growth safe. Games are the thing it expects to
    keep gaining, and the old help text for the selector described the two that existed when it
    was written -- which is accurate exactly until a third arrives. Now each game carries its own
    line, and adding one without a line fails here rather than shipping an unexplained option.
    """
    from ..webui.app import _GAME_CHOICES

    missing = [key for key, _ in _GAME_CHOICES if not i18n.GAMES.get(key)]
    assert not missing, (
        "these games are offered in the interface and explain nothing. Add them to i18n.GAMES "
        f"in the same commit that adds the game: {missing}")

    stale = [key for key in i18n.GAMES if key not in {k for k, _ in _GAME_CHOICES}]
    assert not stale, f"these described games are no longer offered: {stale}"


def test_a_game_description_names_no_other_game() -> None:
    """Each game's line is about that game, so it stays true when the roster of games changes."""
    labels = {"Public Goods", "Honey-Jar", "Market entry"}
    leaks = []
    for key, (el, en) in i18n.GAMES.items():
        for name in labels:
            if name.lower() in en.lower() and name.lower() not in key.replace("_", " "):
                leaks.append(f"{key} mentions {name}")
    assert not leaks, "; ".join(leaks)


def test_help_reads_as_a_sentence() -> None:
    """A help line is prose, not a field name repeated.

    Two failures this catches: a one-word entry that says nothing a label did not already say,
    and an entry that forgot its full stop, which reads as truncated inside the bubble.
    """
    thin = []
    for key, (el, en) in i18n.H.items():
        for lang, text in (("el", el), ("en", en)):
            if len(text.split()) < 5:
                thin.append(f"{key} [{lang}] is too short to explain anything")
            elif not text.rstrip().endswith((".", ";", "!", "?")):
                thin.append(f"{key} [{lang}] does not end in a full stop")
    assert not thin, "\n  ".join(thin)


def test_every_page_renders_in_both_languages() -> None:
    """The catalogue can be complete and a template still raise on a missing Jinja global."""
    from ..webui.app import app

    client = app.test_client()
    for route in ("/", "/evolve", "/analytics", "/meta", "/sweep", "/coverage", "/spacemap"):
        for lang in i18n.LANGS:
            response = client.get(f"{route}?lang={lang}")
            assert response.status_code == 200, f"{route} [{lang}] returned {response.status_code}"
            html = response.get_data(as_text=True)
            assert 'class="info ' in html, f"{route} [{lang}] rendered no help badges at all"


def test_an_unknown_language_falls_back_rather_than_failing() -> None:
    assert i18n.normalise("de") == i18n.DEFAULT_LANG
    assert i18n.normalise(None) == i18n.DEFAULT_LANG
    assert i18n.normalise("en") == "en"


def test_a_missing_key_is_visible_rather_than_silent() -> None:
    """`t` returns the key itself, so a gap shows up on the page instead of rendering blank."""
    assert i18n.t("field.does_not_exist", "el") == "field.does_not_exist"
    assert i18n.h("field.does_not_exist", "el") is None


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL {name}\n       {exc}")
    print(f"\n{'all i18n checks pass' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
