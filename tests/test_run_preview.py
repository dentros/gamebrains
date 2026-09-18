"""The three views that read the repository instead of running a match.

Driven through Flask's own test client against a temporary repository, so what is exercised is the
route a browser hits rather than the helper functions underneath it. The repository root is
redirected for the duration: these tests must never read or write the developer's real ledger.

Run: python -m gamebrains.tests.test_run_preview
"""

from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path

from ..agents.classic import AllD
from ..games.public_goods import PublicGoodsGame
from ..repository.bundle import export_bundle
from ..repository.ledger import Ledger, _record_hash
from ..repository.record import record_experiment
from ..webui import app as webui


def _id(record: dict) -> str:
    """A record's own hash, which is what these views key on.

    Not `content_cid`: content addressing stores one object for identical bytes, so two runs whose
    packages match share a CID. A first version of these views keyed on it, and two seeds of one
    design opened the same page.
    """
    return _record_hash(record)


class _Repo:
    """A temporary repository, with the web app pointed at it for the duration."""

    def __enter__(self) -> Path:
        self.dir = Path(tempfile.mkdtemp(prefix="gamebrains_webui_"))
        self.previous = webui._REPO_ROOT
        webui._REPO_ROOT = self.dir
        webui.app.config["TESTING"] = True
        return self.dir

    def __exit__(self, *exc) -> None:
        webui._REPO_ROOT = self.previous
        shutil.rmtree(self.dir, ignore_errors=True)


def _record_runs(root: Path, seeds=(0, 1), rounds: int = 40) -> list[dict]:
    log = root / "log.jsonl"
    log.write_text(
        '{"type": "meta", "seed": 0}\n'
        '{"type": "round", "round": 0, "actions": [1, 0, 1], "rewards": [1.0, 2.0, 1.0], '
        '"cooperators": 2}\n'
        '{"type": "round", "round": 1, "actions": [0, 0, 1], "rewards": [0.5, 0.5, 0.2], '
        '"cooperators": 1}\n'
        '{"type": "brain_snapshot", "round": 1, "agent": 0, "name": "AllD 0", '
        '"brain": {"kind": "classic", "strategy": "AllD", "rule": "Always claim."}}\n',
        encoding="utf-8")
    game = PublicGoodsGame(n_agents=3, rounds=rounds)
    roster = [AllD(f"AllD {i}") for i in range(3)]
    return [record_experiment(root, game, roster, log, rounds=rounds, seed=seed,
                              metrics={"cooperation_rate": 0.30 + 0.1 * seed,
                                       "efficiency": 0.5,
                                       "transfer_entropy_bits": "withheld: precondition not met"})
            for seed in seeds]


def test_a_stored_run_opens_with_its_metrics_chain_status_and_brains() -> None:
    with _Repo() as root:
        records = _record_runs(root)
        client = webui.app.test_client()
        page = client.get(f"/run/{_id(records[0])}").get_data(as_text=True)

        assert "Stored experiment" in page
        assert "signature valid" in page
        assert "0.3000" in page, "the recorded metric must be shown as recorded"
        assert "withheld: precondition not met" in page, (
            "a measure that declined to report itself shows its reason, not a number")
        assert "Always claim." in page, "the stored brain snapshot should render as a card"
        assert "Cooperation rate by round" in page
        print("OK: a stored run renders from the ledger and its event log alone")


def test_an_unknown_record_says_so_rather_than_failing() -> None:
    with _Repo():
        response = webui.app.test_client().get("/run/" + "0" * 64)
        assert response.status_code == 404
        assert "No record" in response.get_data(as_text=True)
        print("OK: an unknown record id gives a 404 that says what was looked for")


def test_two_runs_sharing_a_content_cid_still_open_separately() -> None:
    """The defect these ids exist to prevent. Both runs below store the same event log, so the
    content store holds one object and both records point at it. Keyed on that CID, the second
    run's link showed the first run's numbers."""
    with _Repo() as root:
        records = _record_runs(root, seeds=(0, 1))
        assert records[0]["content_cid"] == records[1]["content_cid"], (
            "this test is only meaningful while the two runs really do share a package")
        assert _id(records[0]) != _id(records[1])

        client = webui.app.test_client()
        first = client.get(f"/run/{_id(records[0])}").get_data(as_text=True)
        second = client.get(f"/run/{_id(records[1])}").get_data(as_text=True)
        assert "0.3000" in first and "0.4000" not in first
        assert "0.4000" in second and "0.3000" not in second
        print("OK: two records sharing one stored package open as the runs they are")


def test_two_runs_compare_on_the_measures_they_both_report() -> None:
    with _Repo() as root:
        records = _record_runs(root)
        client = webui.app.test_client()
        page = client.get(f"/compare?rid={_id(records[0])}"
                          f"&rid={_id(records[1])}").get_data(as_text=True)

        assert "Comparing 2 runs" in page
        assert "0.3000" in page and "0.4000" in page
        assert "0.1000" in page, "the spread between the two must be shown"
        print("OK: two stored runs compare, with the gap between them stated")


def test_the_analytics_table_opens_and_selects_runs() -> None:
    with _Repo() as root:
        records = _record_runs(root)
        page = webui.app.test_client().get("/analytics").get_data(as_text=True)
        assert f"/run/{_id(records[0])}" in page
        assert 'name="rid"' in page and "Compare selected" in page
        print("OK: analytics rows open a run and can be selected for comparison")


def test_meta_pools_by_design_and_reports_the_seeds_it_pooled() -> None:
    with _Repo() as root:
        _record_runs(root, seeds=(0, 1, 2))
        page = webui.app.test_client().get("/meta?metric=cooperation_rate").get_data(as_text=True)

        assert "Designs" in page and "this installation" in page
        # Seeds 0, 1 and 2 recorded 0.30, 0.40 and 0.50, so the row must read mean 0.4000,
        # sd 0.1000 and a t interval of 0.4 +- 4.303 * 0.1 / sqrt(3) = 0.1516 to 0.6484.
        assert "0.4000" in page and "0.1000" in page
        assert "0.1516" in page and "0.6484" in page, (
            "the interval has to be the t one for n=3, not a normal approximation")
        print("OK: three seeds of one design pool into one row with a t interval")


def test_the_import_form_refuses_the_wrong_key_in_the_page() -> None:
    """The refusal has to reach the person, not just the log. A bundle that silently failed to
    import would leave the page looking as though nothing had been attempted."""
    with _Repo() as root:
        other = Path(tempfile.mkdtemp(prefix="gamebrains_other_"))
        try:
            _record_runs(other)
            bundle = other / "bundle.zip"
            export_bundle(other, bundle)

            wrong_key = Ledger(root).public_key_hex
            client = webui.app.test_client()
            page = client.post("/meta", data={
                "action": "import",
                "trusted_key": wrong_key,
                "bundle": (io.BytesIO(bundle.read_bytes()), "bundle.zip"),
            }, content_type="multipart/form-data").get_data(as_text=True)

            assert "refused" in page and "different key" in page
            assert not (root / "imports").exists()
            print("OK: an import under the wrong key is refused on the page, and nothing is kept")
        finally:
            shutil.rmtree(other, ignore_errors=True)


def test_an_imported_chain_shows_up_as_its_own_source() -> None:
    with _Repo() as root:
        other = Path(tempfile.mkdtemp(prefix="gamebrains_other_"))
        try:
            _record_runs(other)
            key = Ledger(other).public_key_hex
            bundle = other / "bundle.zip"
            export_bundle(other, bundle, title="Lab A")

            client = webui.app.test_client()
            page = client.post("/meta", data={
                "action": "import",
                "trusted_key": key,
                "label": "Lab A",
                "bundle": (io.BytesIO(bundle.read_bytes()), "bundle.zip"),
            }, content_type="multipart/form-data").get_data(as_text=True)

            assert "Imported 2 records from Lab A" in page
            assert "verified" in page
            listing = client.get("/meta?metric=cooperation_rate").get_data(as_text=True)
            assert "Lab A" in listing
            print("OK: an imported chain appears as its own source and its runs pool alongside")
        finally:
            shutil.rmtree(other, ignore_errors=True)


if __name__ == "__main__":
    test_a_stored_run_opens_with_its_metrics_chain_status_and_brains()
    test_an_unknown_record_says_so_rather_than_failing()
    test_two_runs_sharing_a_content_cid_still_open_separately()
    test_two_runs_compare_on_the_measures_they_both_report()
    test_the_analytics_table_opens_and_selects_runs()
    test_meta_pools_by_design_and_reports_the_seeds_it_pooled()
    test_the_import_form_refuses_the_wrong_key_in_the_page()
    test_an_imported_chain_shows_up_as_its_own_source()
    print("\nall repository-view tests passed")
