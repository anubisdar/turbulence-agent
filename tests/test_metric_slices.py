"""Metrics split by what the search actually met.

An aggregate acceptance rate of 91% hid the thing that mattered: every
rejection sat on an unresolved route, and every one of those turned out to
be the validator being wrong. A single number cannot show a pattern that
lives in a subset.

The slices are the ways a search genuinely differs - whether anything was
known, whether the two sources agreed, and whether it ran to completion or
hit a budget - rather than an arbitrary partition.
"""

import sqlite3

import pytest

from app.runs import RunRecord, init_runs, record_run, summary


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    init_runs(c)
    return c


def slices(conn):
    return {r["slice"]: r for r in summary(conn)["by_outcome"]}


class TestTheSlicesSeparateRealDifferences:
    def test_an_unresolved_search_is_its_own_slice(self, conn):
        record_run(conn, RunRecord(request_id="a", reading="unresolved"))
        assert "nothing was known" in slices(conn)

    def test_a_reading_with_agreement_is_its_own_slice(self, conn):
        record_run(conn, RunRecord(request_id="a", reading="light",
                                   sources_disagree=0))
        assert "a reading, sources agreed" in slices(conn)

    def test_disagreement_is_its_own_slice(self, conn):
        record_run(conn, RunRecord(request_id="a", reading="moderate",
                                   sources_disagree=1))
        assert "sources disagreed" in slices(conn)

    def test_a_failed_source_takes_precedence_over_the_reading(self, conn):
        """A degraded search is a different kind of thing from a quiet one,
        even though both usually read unresolved. Counting it as "nothing
        was known" would hide the failure inside the ordinary case."""
        record_run(conn, RunRecord(request_id="a", reading="unresolved",
                                   degraded=1))
        assert "a source failed" in slices(conn)
        assert "nothing was known" not in slices(conn)

    def test_a_truncated_search_takes_precedence_over_the_reading(self, conn):
        record_run(conn, RunRecord(request_id="a", reading="unresolved",
                                   truncated=1))
        assert "stopped on a budget" in slices(conn)

    def test_every_search_lands_in_exactly_one_slice(self, conn):
        for i, record in enumerate([
                RunRecord(request_id="a", reading="unresolved"),
                RunRecord(request_id="b", reading="light"),
                RunRecord(request_id="c", reading="moderate",
                          sources_disagree=1),
                RunRecord(request_id="d", degraded=1),
                RunRecord(request_id="e", truncated=1)]):
            record_run(conn, record)
        assert sum(r["searches"] for r in summary(conn)["by_outcome"]) == 5


class TestAcceptanceIsPerSlice:
    """The number the aggregate hid."""

    def test_a_slice_with_a_low_rate_is_visible(self, conn):
        for i in range(40):
            record_run(conn, RunRecord(
                request_id=f"u{i}", reading="unresolved", llm_called=1,
                llm_accepted=0 if i % 4 == 0 else 1))
        for i in range(60):
            record_run(conn, RunRecord(
                request_id=f"r{i}", reading="light", llm_called=1,
                llm_accepted=1))

        found = slices(conn)
        assert found["nothing was known"]["acceptance"] == 0.75
        assert found["a reading, sources agreed"]["acceptance"] == 1.0

    def test_acceptance_counts_only_searches_that_called_the_model(self, conn):
        """The explainer is off by default. Counting searches that never
        asked for one would understate the rate."""
        record_run(conn, RunRecord(request_id="a", reading="light",
                                   llm_called=1, llm_accepted=1))
        for i in range(9):
            record_run(conn, RunRecord(request_id=f"b{i}", reading="light"))
        found = slices(conn)["a reading, sources agreed"]
        assert found["searches"] == 10
        assert found["explained"] == 1
        assert found["acceptance"] == 1.0

    def test_a_slice_that_never_called_the_model_has_no_rate(self, conn):
        """None rather than zero: no rate is a different fact from a rate
        of nought, and showing 0% would be a lie."""
        record_run(conn, RunRecord(request_id="a", reading="light"))
        assert slices(conn)["a reading, sources agreed"]["acceptance"] is None


class TestCostAndLatencyPerSlice:
    def test_a_degraded_search_shows_its_lower_call_count(self, conn):
        """Fewer corridors were built, so fewer calls were spent. Averaged
        with the rest it would look like the cost of a normal search."""
        for i in range(5):
            record_run(conn, RunRecord(request_id=f"d{i}", degraded=1,
                                       api_calls=3, elapsed=9.0))
        for i in range(5):
            record_run(conn, RunRecord(request_id=f"r{i}", reading="light",
                                       api_calls=8, elapsed=14.0))
        found = slices(conn)
        assert found["a source failed"]["mean_calls"] == 3.0
        assert found["a reading, sources agreed"]["mean_calls"] == 8.0

    def test_an_empty_window_returns_no_slices_rather_than_failing(self, conn):
        assert summary(conn)["by_outcome"] == []
