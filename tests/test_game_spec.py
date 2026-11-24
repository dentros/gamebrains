"""Games written as data (games/spec.py) and the builder that makes them (webui /games).

The first half checks the document: what it accepts, what it refuses and with which field named,
that its hash is stable, and that an approval covers exactly the definition it was given for. The
second half drives the builder through Flask's test client: the Prisoner's Dilemma is made in the
wizard, approved, saved, offered in the Match tab and played there.

Run: python -m gamebrains.tests.test_game_spec
"""

from __future__ import annotations

import copy
import io
import json
import tempfile
from pathlib import Path

from ..games import spec as game_spec
from ..games.matrix_game import MatrixGame

PD = {"schema": "gamebrains.game/1", "name": "Prisoner's Dilemma 5/3/1/0",
      "description": "The textbook payoffs.", "family": "matrix_2x2", "rounds": 14,
      "params": {"T": 5, "R": 3, "P": 1, "S": 0, "action_names": ["Cooperate", "Defect"]}}


def _refused(doc) -> game_spec.SpecError:
    try:
        game_spec.validate(doc)
    except game_spec.SpecError as exc:
        return exc
    raise AssertionError(f"accepted {doc}")


def test_a_document_is_checked_and_its_game_has_the_last_word():
    spec = game_spec.validate(PD)
    game = game_spec.build(spec)
    assert isinstance(game, MatrixGame) and game.payoffs == {"T": 5, "R": 3, "P": 1, "S": 0}
    assert game.action_names == ["Cooperate", "Defect"]
    assert _refused({**PD, "turbo": True}).field == "document"
    assert "unknown for matrix_2x2" in str(_refused({**PD, "params": {**PD["params"], "Q": 1}}))
    assert _refused({**PD, "params": {"T": "5", "R": 3, "P": 1, "S": 0}}).field == "params.T"
    assert _refused({**PD, "family": "chess"}).field == "family"
    assert _refused({**PD, "schema": "other/1"}).field == "schema"
    pgg = {**PD, "family": "public_goods", "params": {"mpcr": 1.5}}
    assert "social-dilemma condition" in str(_refused(pgg)), "the game's own rule must decide"
    assert game_spec.validate({**pgg, "params": {"mpcr": 0.5, "allow_non_dilemma": True,
                                                 "mpcr_schedule": [[8, 1.25]]}})


def test_the_hash_covers_the_definition_and_nothing_else():
    a = game_spec.validate(PD)
    reordered = json.loads(json.dumps(PD, sort_keys=True))
    assert game_spec.spec_hash(a) == game_spec.spec_hash(game_spec.validate(reordered))
    changed = game_spec.validate({**PD, "params": {**PD["params"], "S": -1}})
    assert game_spec.spec_hash(a) != game_spec.spec_hash(changed)
    approved = game_spec.approve(a, "N. P.")
    assert game_spec.spec_hash(approved) == game_spec.spec_hash(a), "approval is not definition"


def test_an_approval_covers_exactly_what_was_approved():
    approved = game_spec.approve(game_spec.validate(PD), "N. P.")
    assert game_spec.is_approved(approved)
    assert set(approved["approval"]) == {"by", "at", "hash"}
    edited = copy.deepcopy(approved)
    edited["params"]["T"] = 6
    assert not game_spec.is_approved(edited)
    try:
        game_spec.approve(approved, "  ")
    except game_spec.SpecError:
        pass
    else:
        raise AssertionError("an approval without a name was accepted")


def test_the_store_keeps_approved_games_and_names_its_faults():
    with tempfile.TemporaryDirectory() as root:
        store = game_spec.GameStore(root)
        spec = game_spec.validate(PD)
        try:
            store.save(spec)
        except game_spec.SpecError:
            pass
        else:
            raise AssertionError("an unapproved game was saved")
        approved = game_spec.approve(spec, "N. P.")
        slug = store.save(approved)
        assert store.save(approved) == slug, "the same game saved twice is one file"
        other = game_spec.approve(game_spec.validate({**PD, "params": {**PD["params"], "S": -1}}),
                                  "N. P.")
        assert store.save(other) == f"{slug}-2", "another game of the same name gets its own file"
        assert store.load(slug)["params"]["T"] == 5
        path = Path(root) / f"{slug}.json"
        tampered = json.loads(path.read_text(encoding="utf-8"))
        tampered["params"]["T"] = 9
        path.write_text(json.dumps(tampered), encoding="utf-8")
        listed = {g["slug"]: g["fault"] for g in store.list()}
        assert "not approved" in listed[slug] and listed[f"{slug}-2"] == ""
        try:
            store.load(slug)
        except game_spec.SpecError as exc:
            assert "changed since it was approved" in str(exc)
        else:
            raise AssertionError("a game edited after approval was offered for play")
        for bad in ("../x", "", "A"):
            try:
                store.load(bad)
            except game_spec.SpecError:
                continue
            raise AssertionError(f"loaded {bad!r}")


def test_the_check_says_who_plays_and_why_the_others_do_not():
    checks = {c["kind"]: c for c in game_spec.conformance(game_spec.validate(PD), rounds=12)}
    for kind in ("qlearning", "dqn", "dac", "markov_brain", "markov_gates", "classic AllC",
                 "classic MajorityTFT"):
        assert checks[kind]["status"] == "plays", (kind, checks[kind])
    assert checks["fep"]["status"] == "refuses" and "declares neither" in checks["fep"]["detail"]
    assert checks["llm"]["status"] == "not run"
    pgg = game_spec.validate({**PD, "family": "public_goods", "params": {"mpcr": 0.5}})
    pgg_checks = game_spec.conformance(pgg, rounds=12)
    assert all(c["status"] == "plays" for c in pgg_checks if c["kind"] != "llm"), pgg_checks
    assert any(c["kind"].startswith("mixed:") for c in pgg_checks)


def _client(root: str):
    from ..webui.app import app
    app.config["GAMEBRAINS_CUSTOM_GAMES"] = root
    return app, app.test_client()


BUILDER_FORM = {"game_name": PD["name"], "game_description": PD["description"],
                "game_family": "matrix_2x2", "payoff_T": "5", "payoff_R": "3", "payoff_P": "1",
                "payoff_S": "0", "action_coop": "Cooperate", "action_defect": "Defect",
                "rounds": "14"}


def test_the_prisoners_dilemma_is_made_in_the_wizard_and_played_in_a_match():
    with tempfile.TemporaryDirectory() as root:
        app, client = _client(root)
        try:
            for lang in ("en", "el"):
                for route in ("/games", "/games/new"):
                    assert client.get(f"{route}?lang={lang}").status_code == 200, (route, lang)
            checked = client.post("/games/new", data={**BUILDER_FORM, "action": "check"})
            page = checked.get_data(as_text=True)
            assert checked.status_code == 200 and "declares neither" in page
            unapproved = client.post("/games/new", data={**BUILDER_FORM, "action": "save"})
            assert unapproved.status_code == 400 and not list(Path(root).glob("*.json"))
            saved = client.post("/games/new", data={**BUILDER_FORM, "action": "save",
                                                    "approver": "N. P.",
                                                    "approve_confirm": "on"})
            assert saved.status_code == 302, saved.status_code
            files = list(Path(root).glob("*.json"))
            assert len(files) == 1
            stored = json.loads(files[0].read_text(encoding="utf-8"))
            assert game_spec.is_approved(stored) and stored["approval"]["by"] == "N. P."
            key = f"custom:{files[0].stem}"
            assert key in client.get("/").get_data(as_text=True), "the Match tab must offer it"
            played = client.post("/run", data={
                "game_kind": key, "rounds": "14", "seed": "0", "log_lines": "5",
                "row_kind[]": ["qlearning", "classic"], "row_count[]": ["1", "1"],
                "row_classic_strategy[]": ["", "MajorityTFT"], "row_reciprocity[]": ["", ""],
                "row_markov_hidden[]": ["", ""], "row_epsilon_decay[]": ["", ""],
                "row_epsilon_min[]": ["", ""], "row_genome_origin[]": ["", ""],
                "row_genome_cid[]": ["", ""]})
            assert played.status_code == 200, played.get_data(as_text=True)[:400]
            three = client.post("/run", data={
                "game_kind": key, "rounds": "5", "seed": "0", "log_lines": "5",
                "row_kind[]": ["classic"], "row_count[]": ["3"],
                "row_classic_strategy[]": ["AllC"], "row_reciprocity[]": [""],
                "row_markov_hidden[]": [""], "row_epsilon_decay[]": [""],
                "row_epsilon_min[]": [""], "row_genome_origin[]": [""], "row_genome_cid[]": [""]})
            assert three.status_code == 400 and "two players" in three.get_data(as_text=True), three.status_code
            download = client.get(f"/games/{files[0].stem}.json")
            assert json.loads(download.get_data(as_text=True)) == stored
        finally:
            app.config.pop("GAMEBRAINS_CUSTOM_GAMES", None)


def test_an_imported_game_is_checked_and_approved_again_here():
    with tempfile.TemporaryDirectory() as root:
        app, client = _client(root)
        try:
            foreign = game_spec.approve(game_spec.validate(PD), "someone else")
            page = client.post("/games/import", data={"spec_file": (
                io.BytesIO(json.dumps(foreign).encode("utf-8")), "pd.json")},
                content_type="multipart/form-data")
            html = page.get_data(as_text=True)
            assert page.status_code == 200 and 'value="Cooperate"' in html
            assert "someone else" not in html, "an approval in a file is not carried over"
            bad = client.post("/games/import", data={"spec_file": (
                io.BytesIO(b'{"schema": "gamebrains.game/1", "eval": 1}'), "x.json")},
                content_type="multipart/form-data")
            assert bad.status_code == 400 and not list(Path(root).glob("*.json"))
        finally:
            app.config.pop("GAMEBRAINS_CUSTOM_GAMES", None)


if __name__ == "__main__":
    test_a_document_is_checked_and_its_game_has_the_last_word()
    print("OK: a document is checked for shape, and the game's own rules have the last word")
    test_the_hash_covers_the_definition_and_nothing_else()
    print("OK: the hash covers the definition, in any key order, and not the approval")
    test_an_approval_covers_exactly_what_was_approved()
    print("OK: an approval covers exactly the definition it was given for")
    test_the_store_keeps_approved_games_and_names_its_faults()
    print("OK: the store keeps approved games and names the fault of any other")
    test_the_check_says_who_plays_and_why_the_others_do_not()
    print("OK: the check says which kinds play the game and why the others do not")
    test_the_prisoners_dilemma_is_made_in_the_wizard_and_played_in_a_match()
    print("OK: the Prisoner's Dilemma is made in the wizard, approved, saved and played")
    test_an_imported_game_is_checked_and_approved_again_here()
    print("OK: an imported game is checked, and approved again by whoever adds it")
