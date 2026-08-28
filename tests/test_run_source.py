"""Live and replayed searches are told apart in the record.

Both were written to search_runs with nothing to distinguish them. A
status page window holding 977 replayed searches and six real ones
therefore described the replay: 99% of the searches it reported on had
never touched a real API, and the panel disagreed with its own caption.

The value existed the whole time. `source` was computed twice in the
service - once for the API payload and once for the trace - and never
passed to record_run. Nothing was broken; the field simply never reached
the one place that persists it.
"""

import sqlite3

import pytest

from app.runs import RunRecord, init_runs, record_run, summary


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    init_runs(c)
    return c


def test_the_column_exists(conn):
    columns = {row[1] for row in conn.execute("PRAGMA table_info(search_runs)")}
    assert "source" in columns


def test_a_run_defaults_to_live(conn):
    """A caller that says nothing is running for real. The default has to
    be the safe direction: mislabelling a live search as synthetic would
    hide it from the page entirely."""
    record_run(conn, RunRecord(request_id="a", reading="light"))
    assert summary(conn)["totals"]["searches"] == 1
    assert summary(conn)["synthetic_runs"] == 0


def test_replayed_runs_are_not_counted_as_behaviour(conn):
    for i in range(50):
        record_run(conn, RunRecord(request_id=f"f{i}", source="fixtures",
                                   reading="unresolved", degraded=1))
    for i in range(3):
        record_run(conn, RunRecord(request_id=f"r{i}", source="live",
                                   reading="light"))
    found = summary(conn)
    assert found["totals"]["searches"] == 3
    assert found["synthetic_runs"] == 50


def test_the_synthetic_count_is_reported_rather_than_hidden(conn):
    """Dropping them silently would trade one wrong number for another.
    How much of a window was synthetic is worth being able to answer."""
    record_run(conn, RunRecord(request_id="f", source="fixtures"))
    assert summary(conn)["synthetic_runs"] == 1


def test_slices_exclude_replayed_runs(conn):
    """The panel that surfaced the problem. Fifty degraded replay runs
    would otherwise fill the 'a source failed' slice and make every other
    row look like noise."""
    for i in range(50):
        record_run(conn, RunRecord(request_id=f"f{i}", source="fixtures",
                                   degraded=1, reading="unresolved"))
    record_run(conn, RunRecord(request_id="r", source="live",
                               reading="moderate", sources_disagree=1))
    slices = {r["slice"]: r["searches"] for r in summary(conn)["by_outcome"]}
    assert slices == {"sources disagreed": 1}


def test_rows_written_before_the_column_existed_count_as_live(conn):
    """COALESCE, because the rows that caused this predate the fix. They
    were real searches and dropping them would be a second wrong answer."""
    record_run(conn, RunRecord(request_id="old", reading="light"))
    conn.execute("UPDATE search_runs SET source = NULL")
    conn.commit()
    assert summary(conn)["totals"]["searches"] == 1
    assert summary(conn)["synthetic_runs"] == 0
