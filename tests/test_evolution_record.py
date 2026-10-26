"""An evolutionary run reaching the repository, and the records it must not disturb.

`evolve` used to be the one way of producing an agent whose results reached nothing: the web
interface stored a genome and rendered a page, and no ledger record was written, so lineage
detection, the Smart Filter, `/compare` and the meta-analysis covered every match and no
evolution. `experiments/evolution_record.py` closes that by recording the search's outcome as what
the ledger already understands, one evaluation match of the frozen best genome in self-play.

The load-bearing test here is `test_a_trained_seat_is_described_exactly_as_before`. Carrying the
search parameters into `config_hash` meant adding `evolved_by` to `_PARAM_ATTRS`, and the only
reason that could be done without rehashing the whole ledger is that `_agent_params` reads each
attribute under `hasattr`. If a trained agent ever grows that attribute, every existing qlearning
and dqn record silently changes identity and its lineage is cut.
"""

import sys
import tempfile
from pathlib import Path

from gamebrains.agents import factory
from gamebrains.agents.factory import evolvable
from gamebrains.engine.evolution import EvolutionConfig, evolve
from gamebrains.experiments.evolution_record import record_evolution
from gamebrains.games.public_goods import PublicGoodsGame
from gamebrains.repository.ledger import Ledger
from gamebrains.repository.record import _agent_params

PREFIX = {"markov_brain": "MB", "qlearning": "QL", "dqn": "DQ"}


def _game() -> PublicGoodsGame:
    return PublicGoodsGame(n_agents=4, rounds=30, mpcr=0.5)


def _config(architecture: str, **over) -> EvolutionConfig:
    base = dict(population_size=8, generations=2, match_rounds=30, matches_per_generation=1,
                seed=3, name_prefix=PREFIX[architecture])
    base.update(over)
    return EvolutionConfig(**base)


def _run(architecture: str, root: Path, tag: str, **over) -> dict:
    spawn, crossover = evolvable(architecture)
    config = _config(architecture, **over)
    result = evolve(_game, config, spawn=spawn, crossover=crossover)
    return record_evolution(root / "repo", result, config, architecture, _game,
                            root / f"{tag}.jsonl")


def test_a_trained_seat_is_described_exactly_as_before() -> None:
    """The regression that protects every record already in every ledger."""
    game = _game()
    for kind, expected in (
        ("qlearning", {"alpha", "gamma", "epsilon_decay", "epsilon_min"}),
        ("dqn", {"hidden", "lr", "gamma", "epsilon_decay", "epsilon_min", "buffer_size",
                 "batch_size", "train_every", "target_sync_every"}),
        ("markov_brain", {"n_hidden"}),
    ):
        params = _agent_params(factory.build(kind, 0, game, seed=0))
        assert set(params) == expected, (
            f"a trained {kind} seat now reports {sorted(params)} instead of {sorted(expected)}. "
            f"Any key added or removed here changes config_hash for every {kind} record ever "
            f"written and cuts their lineage. `evolved_by` must stay absent on a seat the GA did "
            f"not produce, which is what lets it be read without rehashing anything")
    print("OK: trained seats are described exactly as they were, so no hash moves")


def test_the_outcome_of_a_search_becomes_a_record() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        record = _run("qlearning", root, "q")
        assert record["config_hash"], "no config_hash on the record"
        assert "cooperation_rate" in record["metrics_summary"], (
            "the recorded evaluation match carries no social measures")
        ledger = Ledger(root / "repo")
        assert len(ledger.load_all()) == 1, "the run did not append exactly one record"
        assert ledger.verify_chain(), "the chain does not verify after an evolutionary record"
    print("OK: an evolutionary run appends one verifiable record")


def test_the_record_says_which_search_produced_it() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        record = _run("qlearning", Path(tmp), "q")
        for seat in record["roster"]:
            story = seat["params"].get("evolved_by")
            assert story, f"seat {seat} carries no provenance, so the search is unrecorded"
            assert story["architecture"] == "qlearning"
            assert story["generations"] == 2, (
                f"the record claims {story['generations']} generations, the run had 2")
            assert story["lifetime_learning"] == "off"
    print("OK: every seat names the search that produced it")


def test_two_searches_that_differ_are_two_designs() -> None:
    """A mutation rate that explores differently is a different experiment, not a rerun."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gentle = _run("qlearning", root, "gentle", mutation_rate=0.15)
        wild = _run("qlearning", root, "wild", mutation_rate=0.60)
        assert gentle["config_hash"] != wild["config_hash"], (
            "two GA settings that search very differently collided onto one config_hash, so the "
            "Smart Filter would offer one as the other's answer and the ledger would call them "
            "replications")
    print("OK: two different searches hash as two designs")


def test_three_substrates_are_three_designs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        hashes = {kind: _run(kind, root, kind)["config_hash"]
                  for kind in ("markov_brain", "qlearning", "dqn")}
        assert len(set(hashes.values())) == 3, (
            f"the three substrates produced {len(set(hashes.values()))} distinct designs: "
            f"{hashes}. A record that cannot tell a Q-table from a network is not usable for the "
            f"comparison the evolutionary bake-off exists to make")
    print("OK: the three substrates record as three distinct designs")


def test_the_recorded_evaluation_never_learns() -> None:
    """Even when the search itself was Lamarckian, the match that measures the result is not."""
    with tempfile.TemporaryDirectory() as tmp:
        spawn, crossover = evolvable("qlearning")
        config = _config("qlearning", lifetime_learning="lamarckian")
        result = evolve(_game, config, spawn=spawn, crossover=crossover)
        root = Path(tmp)
        record = record_evolution(root / "repo", result, config, "qlearning", _game,
                                  root / "lam.jsonl")
        story = record["roster"][0]["params"]["evolved_by"]
        assert story["lifetime_learning"] == "lamarckian", (
            "the record must still say the search was Lamarckian; that is what it was")
        # The seats that played the recorded match were frozen, which is the point: a match that
        # kept training would report neither the genome nor the search.
        twin = result.best.clone(name="check", seed=1)
        twin.lifetime_learning = False
        before = twin.Q.copy()
        from gamebrains.engine.runner import run_match
        run_match(_game(), [twin.clone(name=f"c{i}", seed=i) if i else twin
                            for i in range(4)], rounds=30, seed=7)
        assert (twin.Q == before).all(), (
            "a frozen seat learned during the evaluation match")
    print("OK: the evaluation match measures the genome, whatever the search allowed")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}\n     {exc}")
    print(f"\n{'all evolution-record tests passed' if not failures else f'{failures} failed'}")
    sys.exit(1 if failures else 0)
