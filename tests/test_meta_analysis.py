"""Pooling across seeds and across installations, checked against arithmetic done by hand.

Every expected number below is computed independently of the module, because a statistics helper
tested against its own output is a tautology. The interval uses the Student t multiplier for the
sample size, and that choice is itself asserted: a normal 1.96 at n=4 would understate a seed
sweep's interval by nearly half, which is exactly the direction that flatters a result.

Run: python -m gamebrains.tests.test_meta_analysis
"""

from __future__ import annotations

import math
import statistics

from ..repository.meta_analysis import (
    compare_sources, design_key, group_by_design, heterogeneity, summarise,
)


def _record(config_hash: str, seed: int, value: float, source: str = "this installation",
            protocol: dict | None = None, reproducible: bool = True) -> dict:
    return {
        "config_hash": config_hash,
        "content_cid": f"cid-{config_hash}-{seed}",
        "protocol": protocol if protocol is not None else {"partial_episode": "drop"},
        "game": {"name": "public_goods", "n_agents": 4},
        "roster": [{"kind": "qlearning"}, {"kind": "qlearning"},
                   {"kind": "dqn"}, {"kind": "classic"}],
        "horizon": {"rounds": 1500},
        "seeds": {"master": seed},
        "metrics_summary": {"cooperation_rate": value},
        "reproducibility": {"byte_identical_claimed": reproducible,
                            "nondeterministic_agents": []},
        "source": {"label": source, "local": source == "this installation", "verified": True},
    }


def test_the_interval_uses_the_right_multiplier_for_the_sample_size() -> None:
    values = [0.40, 0.44, 0.52, 0.48]
    got = summarise(values)

    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    expected_half = 3.182 * sd / math.sqrt(4)          # t(0.975, df=3), from a table

    assert abs(got["mean"] - mean) < 1e-12
    assert abs(got["sd"] - sd) < 1e-12
    assert abs(got["ci95"] - expected_half) < 1e-9
    assert got["n"] == 4 and abs(got["low"] - (mean - expected_half)) < 1e-9

    normal_half = 1.96 * sd / math.sqrt(4)
    assert got["ci95"] > normal_half * 1.5, (
        "at n=4 the t multiplier is much larger than 1.96, and using the normal one would "
        "understate the interval")
    print(f"OK: mean {got['mean']:.4f} +- {got['ci95']:.4f} at n=4, t rather than normal")


def test_one_run_gets_no_interval_rather_than_a_zero_one() -> None:
    """A single observation cannot report spread. Printing zero would be a claim that the quantity
    does not vary, made from data that could not have seen variation."""
    got = summarise([0.42])
    assert got["n"] == 1 and got["mean"] == 0.42
    assert got["sd"] is None and got["ci95"] is None
    assert "spread" in got["note"]
    assert summarise([])["n"] == 0
    print("OK: n=1 reports a mean and declines to report dispersion")


def test_designs_group_by_configuration_and_split_by_protocol() -> None:
    """Runs differing only in seed are one experiment. Runs scored under different conventions are
    not, and pooling them would average two different quantities (see normalize.build_config)."""
    records = [
        _record("aaa", seed=0, value=0.40), _record("aaa", seed=1, value=0.50),
        _record("aaa", seed=2, value=0.45),
        _record("bbb", seed=0, value=0.10),
        _record("aaa", seed=3, value=0.90, protocol={"partial_episode": "keep"}),
    ]
    rows = group_by_design(records, "cooperation_rate")

    assert len(rows) == 3, [r["config_hash"] + " " + r["protocol"] for r in rows]
    biggest = rows[0]
    assert biggest["config_hash"] == "aaa" and biggest["summary"]["n"] == 3
    assert biggest["seeds"] == [0, 1, 2]
    assert biggest["roster"] == "1x classic, 1x dqn, 2x qlearning"

    odd_protocol = [r for r in rows if "keep" in r["protocol"]]
    assert len(odd_protocol) == 1 and odd_protocol[0]["summary"]["n"] == 1, (
        "a differently scored run must not join the pool")
    print("OK: seeds pool, protocols do not")


def test_a_withheld_metric_is_skipped_rather_than_counted_as_zero() -> None:
    """`metrics_summary` carries a refusal as None or as its own reason in words (see
    record._scalar_metrics_summary). Reading either as a number would put a fabricated value into
    an average."""
    records = [
        _record("aaa", seed=0, value=0.40),
        {**_record("aaa", seed=1, value=0.0),
         "metrics_summary": {"cooperation_rate": None}},
        {**_record("aaa", seed=2, value=0.0),
         "metrics_summary": {"cooperation_rate": "withheld: precondition not met"}},
    ]
    rows = group_by_design(records, "cooperation_rate")
    assert rows[0]["summary"]["n"] == 1 and rows[0]["values"] == [0.40]
    print("OK: withheld metrics are skipped, not averaged in as zero")


def test_heterogeneity_agrees_with_the_formula_and_declines_when_it_cannot() -> None:
    by_source = {
        "lab A": summarise([0.40, 0.44, 0.42]),
        "lab B": summarise([0.41, 0.43, 0.42]),
    }
    got = heterogeneity(by_source)

    weights, means = [], []
    for summary in by_source.values():
        se = summary["sd"] / math.sqrt(summary["n"])
        weights.append(1 / se ** 2)
        means.append(summary["mean"])
    pooled = sum(w * m for w, m in zip(weights, means)) / sum(weights)
    q = sum(w * (m - pooled) ** 2 for w, m in zip(weights, means))

    assert abs(got["pooled"] - pooled) < 1e-12
    assert abs(got["q"] - q) < 1e-12
    assert got["i2"] == 0.0, "two sources agreeing to this degree leave no excess variance"
    assert got["df"] == 1

    # Disagreement has to move it, or the statistic is decorative.
    apart = heterogeneity({"lab A": summarise([0.10, 0.12, 0.11]),
                           "lab B": summarise([0.80, 0.82, 0.81])})
    assert apart["q"] > got["q"] and apart["i2"] > 90
    print(f"OK: agreement gives I^2 {got['i2']:.0f}%, disagreement {apart['i2']:.0f}%")

    single = heterogeneity({"lab A": summarise([0.4]), "lab B": summarise([0.6])})
    assert single["q"] is None and single["usable"] == 0
    assert "two sources" in single["note"]
    print("OK: sources of one run each are excluded and the absence is stated")


def test_shared_designs_are_the_ones_worth_comparing() -> None:
    records = [
        _record("aaa", 0, 0.40), _record("aaa", 1, 0.44),
        _record("aaa", 0, 0.41, source="Lab A"), _record("aaa", 1, 0.45, source="Lab A"),
        _record("bbb", 0, 0.10),
    ]
    rows = group_by_design(records, "cooperation_rate")
    shared = compare_sources(rows)

    assert len(shared) == 1 and shared[0]["config_hash"] == "aaa"
    assert set(shared[0]["sources"]) == {"this installation", "Lab A"}
    assert shared[0]["heterogeneity"]["usable"] == 2
    print("OK: only designs two installations both ran are compared")


def test_a_group_reports_whether_every_run_in_it_could_be_reproduced() -> None:
    """A pooled mean over runs that decline the reproduction claim is still a number, and a reader
    has to be told which kind it is."""
    rows = group_by_design(
        [_record("aaa", 0, 0.40), _record("aaa", 1, 0.44, reproducible=False)],
        "cooperation_rate")
    assert rows[0]["reproducible"] is False
    assert group_by_design([_record("aaa", 0, 0.40)], "cooperation_rate")[0]["reproducible"] is True
    print("OK: a pool says whether everything in it claimed reproducibility")


def test_design_key_reads_the_two_tiers_that_define_an_experiment() -> None:
    assert design_key({"config_hash": "aaa", "protocol": {"b": 2, "a": 1}}) == ("aaa", "a=1,b=2")
    assert design_key({"config_hash": "aaa"}) == ("aaa", "")


if __name__ == "__main__":
    test_the_interval_uses_the_right_multiplier_for_the_sample_size()
    test_one_run_gets_no_interval_rather_than_a_zero_one()
    test_designs_group_by_configuration_and_split_by_protocol()
    test_a_withheld_metric_is_skipped_rather_than_counted_as_zero()
    test_heterogeneity_agrees_with_the_formula_and_declines_when_it_cannot()
    test_shared_designs_are_the_ones_worth_comparing()
    test_a_group_reports_whether_every_run_in_it_could_be_reproduced()
    test_design_key_reads_the_two_tiers_that_define_an_experiment()
    print("\nall meta-analysis tests passed")
