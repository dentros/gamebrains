"""
Tests for the Analytics page's pure query/filter logic (webui/app.py's `_analytics_row`,
`_row_field_candidates`, `_row_matches_filters`, `_row_matches_kind_filter`).

These operate on plain dicts shaped like a real Ledger record (see repository/ledger.py's
`append`), so they're exercised directly here without needing a live Flask request or an actual
ledger on disk -- the same style as test_equilibrium.py/test_record.py testing their modules'
pure functions in isolation.
"""

from gamebrains.webui.app import (
    _analytics_row,
    _row_field_candidates,
    _row_matches_filters,
    _row_matches_kind_filter,
)


def _fake_record(n_agents, mpcr, rounds, roster, cooperation_rate=0.5, efficiency=0.5):
    return {
        "config_hash": "deadbeef" * 8, "content_cid": "cafebabe" * 8,
        "timestamp": "2026-07-17T12:00:00+00:00",
        "game": {"n_agents": n_agents, "mpcr": mpcr, "cost": 1.0},
        "roster": roster,
        "horizon": {"rounds": rounds}, "seeds": {"master": 0},
        "metrics_summary": {"cooperation_rate": cooperation_rate, "efficiency": efficiency},
        "lineage": {},
    }


def test_row_field_candidates_reads_simple_metric_and_roster_params():
    roster = [
        {"kind": "qlearning", "training_mode": "online", "params": {"alpha": 0.1, "gamma": 0.95}},
        {"kind": "qlearning", "training_mode": "online", "params": {"alpha": 0.5, "gamma": 0.95}},
        {"kind": "classic", "training_mode": "fixed", "params": {"strategy": "AllD"}},
    ]
    record = _fake_record(3, 0.5, 1500, roster, cooperation_rate=0.42)
    row = _analytics_row(record)

    assert _row_field_candidates(row, "n_agents") == [3]
    assert _row_field_candidates(row, "metric:cooperation_rate") == [0.42]
    # both qlearning agents' alpha values are returned -- "any agent of this kind" semantics
    assert sorted(_row_field_candidates(row, "roster:qlearning:alpha")) == [0.1, 0.5]
    # classic's "strategy" param is a string, not numeric -- correctly excluded
    assert _row_field_candidates(row, "roster:classic:strategy") == []
    # a kind absent from the roster yields no candidates, not an error
    assert _row_field_candidates(row, "roster:dqn:lr") == []


def test_row_matches_filters_uses_any_agent_of_kind_semantics():
    roster = [
        {"kind": "qlearning", "training_mode": "online", "params": {"alpha": 0.1}},
        {"kind": "qlearning", "training_mode": "online", "params": {"alpha": 0.9}},
    ]
    row = _analytics_row(_fake_record(2, 0.6, 1000, roster))

    # one of the two qlearning agents has alpha >= 0.8 -> the filter passes
    assert _row_matches_filters(row, [{"field": "roster:qlearning:alpha", "op": ">=", "value": 0.8}])
    # no agent has alpha >= 0.95 -> the filter correctly rejects
    assert not _row_matches_filters(row, [{"field": "roster:qlearning:alpha", "op": ">=", "value": 0.95}])
    # multiple filters combine with AND
    assert _row_matches_filters(row, [
        {"field": "n_agents", "op": "==", "value": 2},
        {"field": "roster:qlearning:alpha", "op": ">=", "value": 0.8},
    ])
    assert not _row_matches_filters(row, [
        {"field": "n_agents", "op": "==", "value": 5},
        {"field": "roster:qlearning:alpha", "op": ">=", "value": 0.8},
    ])


def test_row_matches_kind_filter():
    roster = [{"kind": "fep", "training_mode": "online", "params": {}}]
    row = _analytics_row(_fake_record(2, 0.6, 1000, roster))

    assert _row_matches_kind_filter(row, set())  # no kind filter -> everything matches
    assert _row_matches_kind_filter(row, {"fep"})
    assert not _row_matches_kind_filter(row, {"dqn"})


if __name__ == "__main__":
    test_row_field_candidates_reads_simple_metric_and_roster_params()
    print("OK: field candidates read simple metrics and roster hyperparameters correctly")
    test_row_matches_filters_uses_any_agent_of_kind_semantics()
    print("OK: filters use 'any agent of this kind' semantics and combine with AND")
    test_row_matches_kind_filter()
    print("OK: kind-inclusion filter matches correctly")
