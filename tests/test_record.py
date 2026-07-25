"""
Tests for repository/record.py -- the glue between a completed match and a ledger record.

Regression target: `game.describe()` embeds `rounds` (a per-run parameter) alongside the
experimental-design fields (mpcr, cost, ...). `record_experiment` must strip it before it reaches
`config_hash`, or two runs of the *same design* at different lengths get different config_hash
values and `extends` lineage detection silently never fires (see `record.py`'s
`_config_game_desc` docstring). This was caught by hand on a real `run_pgg.py` invocation before
being reduced to this test.
"""

import shutil
import tempfile
from pathlib import Path

from gamebrains.agents.classic import AllD
from gamebrains.agents.qlearning import QLearningAgent
from gamebrains.games.public_goods import PublicGoodsGame
from gamebrains.repository.ledger import Ledger
from gamebrains.repository.record import protocol_from_run, record_experiment


def _tmp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="gamebrains_record_test_"))


def _write_minimal_log(path: Path) -> None:
    path.write_text('{"type": "meta", "seed": 0}\n', encoding="utf-8")


def test_config_hash_independent_of_rounds_and_extends_detected():
    tmp = _tmp_dir()
    try:
        log_path = tmp / "log.jsonl"
        _write_minimal_log(log_path)

        short_game = PublicGoodsGame(n_agents=4, rounds=50)
        roster = [AllD(f"AllD {i}") for i in range(4)]
        short = record_experiment(tmp, short_game, roster, log_path, rounds=50, seed=0, metrics={})

        long_game = PublicGoodsGame(n_agents=4, rounds=100)
        long = record_experiment(tmp, long_game, roster, log_path, rounds=100, seed=0, metrics={})

        assert short["config_hash"] == long["config_hash"]  # same design, different length
        assert long["lineage"]["extends"] is not None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_config_hash_changes_with_agent_hyperparameters():
    """Bug found 2026-07-16 via real usage: `_roster_description` used to record only
    {kind, training_mode}, so two rosters differing ONLY in hyperparameters (e.g. Q-learning's
    alpha) silently collided onto the same config_hash -- the ledger could not tell them apart at
    all. Same design, same seed/rounds, different alpha must now produce different config_hash."""
    tmp = _tmp_dir()
    try:
        log_path = tmp / "log.jsonl"
        _write_minimal_log(log_path)
        game = PublicGoodsGame(n_agents=2, rounds=50)

        roster_a = [QLearningAgent(f"Q{i}", n_states=game.n_states, n_actions=game.n_actions,
                                   alpha=0.1) for i in range(2)]
        record_a = record_experiment(tmp, game, roster_a, log_path, rounds=50, seed=0, metrics={})

        roster_b = [QLearningAgent(f"Q{i}", n_states=game.n_states, n_actions=game.n_actions,
                                   alpha=0.5) for i in range(2)]
        record_b = record_experiment(tmp, game, roster_b, log_path, rounds=50, seed=0, metrics={})

        assert record_a["config_hash"] != record_b["config_hash"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_config_hash_separates_runs_scored_under_different_protocols():
    """Two runs identical in game, roster and seed but scored under different conventions are
    different experiments. If they shared a config_hash, the Smart Filter would advertise one as
    a reusable result for the other and the ledger would call them replications."""
    tmp = _tmp_dir()
    try:
        log_path = tmp / "log.jsonl"
        _write_minimal_log(log_path)
        game = PublicGoodsGame(n_agents=3, rounds=50)
        roster = [AllD(f"AllD{i}") for i in range(3)]

        dropped = record_experiment(tmp, game, roster, log_path, rounds=50, seed=0, metrics={},
                                    protocol={"partial_episode": "drop"})
        kept = record_experiment(tmp, game, roster, log_path, rounds=50, seed=0, metrics={},
                                 protocol={"partial_episode": "record"})

        assert dropped["config_hash"] != kept["config_hash"]
        assert dropped["protocol"] == {"partial_episode": "drop"}

        # The exact-match lookup has to agree with how the records were hashed, or it would offer
        # the "drop" run as an already-computed answer for a "record" request.
        ledger = Ledger(tmp)
        desc, roster_desc = dropped["game"], dropped["roster"]
        hit = ledger.find_exact(desc, roster_desc, dropped["code_version"], rounds=50,
                                master_seed=0, protocol={"partial_episode": "drop"})
        assert hit is not None and hit["config_hash"] == dropped["config_hash"]

        other = ledger.find_exact(desc, roster_desc, dropped["code_version"], rounds=50,
                                  master_seed=0, protocol={"partial_episode": "record"})
        assert other is not None and other["config_hash"] == kept["config_hash"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_protocol_is_read_back_off_the_run_not_restated():
    """`protocol_from_run` exists so the recorded conventions cannot drift from the ones that
    actually produced the numbers."""
    assert protocol_from_run({"partial_episode_policy": "record"}) == {"partial_episode": "record"}
    # A caller that predates the field still records the documented default rather than nothing.
    assert protocol_from_run({}) == {"partial_episode": "drop"}


if __name__ == "__main__":
    test_config_hash_independent_of_rounds_and_extends_detected()
    print("OK: config_hash independent of rounds, extends detected across real run_pgg.py-shaped calls")
    test_config_hash_changes_with_agent_hyperparameters()
    print("OK: config_hash changes when agent hyperparameters change")
    test_config_hash_separates_runs_scored_under_different_protocols()
    print("OK: config_hash separates runs scored under different protocols")
    test_protocol_is_read_back_off_the_run_not_restated()
    print("OK: protocol is read back off the run rather than restated")
