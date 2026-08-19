# install-to: app
"""
One row per search, so the agent's behaviour can be asked questions.

Logs answer "what happened in that search". They cannot answer "what
fraction of searches resolve", "how often does the explainer get rejected",
or "where does the time go" without scanning and parsing text. This table
exists for the second kind of question.

WHAT IS DELIBERATELY NOT STORED. No client IP. The threat model concluded
that an address plus a route is closer to personal data than either alone,
and a dashboard is not a reason to reverse that. The country is resolved at
write time and the address discarded, because a two-letter code answers the
only question the dashboard asks of it.

Trip content follows the same rule as the logs: origin and destination are
stored only when TURBULENCE_LOG_TRIP_CONTENT is set, and the route column is
null otherwise. Everything else here is operational shape - counts,
durations, outcomes - which carries no itinerary.

RETENTION. Rows older than the window are deleted on write. A dashboard
answering "how is it behaving lately" does not need last spring, and an
unbounded table on a 30 GB volume is a slow leak.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

RETENTION_DAYS = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS search_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id          TEXT NOT NULL,
    started_at          TEXT NOT NULL,

    -- trip, only when trip content logging is enabled
    origin              TEXT,
    dest                TEXT,
    country             TEXT,

    -- outcome
    reading             TEXT,
    observed_reading    TEXT,
    forecast_reading    TEXT,
    sources_disagree    INTEGER DEFAULT 0,
    winner              TEXT,
    stop_reason         TEXT,
    truncated           INTEGER DEFAULT 0,
    degraded            INTEGER DEFAULT 0,
    degraded_reason     TEXT,

    -- shape of the search
    nodes_generated     INTEGER DEFAULT 0,
    corridors_kept      INTEGER DEFAULT 0,
    coverage_fraction   REAL,

    -- cost
    api_calls           INTEGER DEFAULT 0,
    aeroapi_calls       INTEGER DEFAULT 0,
    awc_calls           INTEGER DEFAULT 0,

    -- where the time went, seconds
    elapsed             REAL,
    aeroapi_seconds     REAL,
    awc_seconds         REAL,
    retrieval_seconds   REAL,
    explainer_seconds   REAL,
    scoring_seconds     REAL,

    -- models
    embedding_calls     INTEGER DEFAULT 0,
    embedding_cpu_ms    REAL,
    llm_called          INTEGER DEFAULT 0,
    llm_model           TEXT,
    llm_accepted        INTEGER,
    llm_reject_reason   TEXT,
    llm_tokens_in       INTEGER,
    llm_tokens_out      INTEGER
);
CREATE INDEX IF NOT EXISTS idx_runs_started ON search_runs(started_at);
"""


@dataclass
class RunRecord:
    """Everything worth keeping about one search."""
    request_id: str
    started_at: str = ""

    origin: str | None = None
    dest: str | None = None
    country: str | None = None

    reading: str | None = None
    observed_reading: str | None = None
    forecast_reading: str | None = None
    sources_disagree: int = 0
    winner: str | None = None
    stop_reason: str | None = None
    truncated: int = 0
    degraded: int = 0
    degraded_reason: str | None = None

    nodes_generated: int = 0
    corridors_kept: int = 0
    coverage_fraction: float | None = None

    api_calls: int = 0
    aeroapi_calls: int = 0
    awc_calls: int = 0

    elapsed: float | None = None
    aeroapi_seconds: float | None = None
    awc_seconds: float | None = None
    retrieval_seconds: float | None = None
    explainer_seconds: float | None = None
    scoring_seconds: float | None = None

    embedding_calls: int = 0
    embedding_cpu_ms: float | None = None
    llm_called: int = 0
    llm_model: str | None = None
    llm_accepted: int | None = None
    llm_reject_reason: str | None = None
    llm_tokens_in: int | None = None
    llm_tokens_out: int | None = None


# ------------------------------------------------------------------ timing


@dataclass
class Timings:
    """Accumulates where a search spent its time.

    Wall clock for external calls, since waiting is what they cost. CPU time
    for the embedding model, since that is work this machine actually does
    and wall clock would include scheduler noise.
    """
    aeroapi_seconds: float = 0.0
    awc_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    explainer_seconds: float = 0.0
    scoring_seconds: float = 0.0
    aeroapi_calls: int = 0
    awc_calls: int = 0
    embedding_calls: int = 0
    embedding_cpu_ms: float = 0.0

    @contextmanager
    def track(self, bucket: str, count: bool = True) -> Iterator[None]:
        """Time a block into one bucket."""
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            setattr(self, f"{bucket}_seconds",
                    getattr(self, f"{bucket}_seconds", 0.0) + elapsed)
            if count and hasattr(self, f"{bucket}_calls"):
                setattr(self, f"{bucket}_calls",
                        getattr(self, f"{bucket}_calls") + 1)

    @contextmanager
    def track_embedding(self) -> Iterator[None]:
        """CPU time for the local model.

        `process_time` rather than `perf_counter`: this is the one piece of
        real computation the machine performs, and the question it answers
        is how much CPU it costs, not how long it was scheduled over.
        """
        started = time.process_time()
        try:
            yield
        finally:
            self.embedding_cpu_ms += (time.process_time() - started) * 1000
            self.embedding_calls += 1


# ------------------------------------------------------------------ storage


def init_runs(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def resolve_country(ip: str | None,
                    db_path: str = "/usr/share/GeoIP/GeoLite2-Country.mmdb"
                    ) -> str | None:
    """A two-letter country code, or None.

    The address is used and discarded. Nothing here returns it, and nothing
    stores it: the dashboard asks which countries reach the site, which a
    country code answers without identifying anyone.
    """
    if not ip:
        return None
    try:
        import maxminddb
    except ImportError:
        return None
    if not os.path.exists(db_path):
        return None
    try:
        with maxminddb.open_database(db_path) as reader:
            found = reader.get(ip.split(",")[0].strip())
        if isinstance(found, dict):
            return (found.get("country") or {}).get("iso_code")
    except (ValueError, OSError):
        return None
    return None


def record_run(conn: sqlite3.Connection, run: RunRecord,
               retention_days: int = RETENTION_DAYS) -> None:
    """Write a run and prune anything past the window.

    A failure here must never fail a search. The record exists to describe
    what happened; losing one row is a gap in a chart, and raising would
    turn it into a lost answer.
    """
    if not run.started_at:
        run.started_at = datetime.now(timezone.utc).isoformat()

    fields = asdict(run)
    columns = ", ".join(fields)
    placeholders = ", ".join("?" for _ in fields)
    try:
        conn.execute(
            f"INSERT INTO search_runs ({columns}) VALUES ({placeholders})",
            list(fields.values()))
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(days=retention_days)).isoformat()
        conn.execute("DELETE FROM search_runs WHERE started_at < ?", (cutoff,))
        conn.commit()
    except sqlite3.Error:
        pass


def from_payload(payload: dict[str, Any], request_id: str,
                 timings: Timings | None = None,
                 country: str | None = None) -> RunRecord:
    """Build a record from a finished search payload."""
    request = payload.get("request") or {}
    outcome = payload.get("outcome") or {}
    wx = outcome.get("turbulence") or {}
    explanation = payload.get("explanation") or {}
    corridors = payload.get("corridors") or []

    log_trip = os.environ.get("TURBULENCE_LOG_TRIP_CONTENT",
                              "").lower() in ("1", "true", "yes")

    reasons = outcome.get("degraded_reasons") or []
    rejected = explanation.get("rejected") or []

    run = RunRecord(
        request_id=request_id,
        origin=request.get("origin") if log_trip else None,
        dest=request.get("dest") if log_trip else None,
        country=country,
        reading=outcome.get("reading"),
        observed_reading=(wx.get("observed") or {}).get("reading"),
        forecast_reading=(wx.get("forecast") or {}).get("reading"),
        sources_disagree=int(bool(wx.get("disagree"))),
        winner=outcome.get("winner"),
        stop_reason=outcome.get("stop"),
        truncated=int(bool(outcome.get("truncated"))),
        degraded=int(bool(outcome.get("degraded"))),
        degraded_reason=str(reasons[0])[:200] if reasons else None,
        nodes_generated=outcome.get("nodes_generated") or 0,
        corridors_kept=sum(1 for c in corridors if c.get("kept")),
        coverage_fraction=wx.get("coverage_fraction"),
        api_calls=outcome.get("calls_used") or 0,
        elapsed=outcome.get("elapsed_seconds"),
        llm_called=int(bool(explanation.get("enabled"))),
        llm_model=explanation.get("model"),
        llm_accepted=(1 if explanation.get("source") == "model" else 0)
        if explanation.get("enabled") else None,
        llm_reject_reason=str(rejected[0])[:200] if rejected else None,
        llm_tokens_in=explanation.get("tokens_in"),
        llm_tokens_out=explanation.get("tokens_out"),
    )

    if timings:
        run.aeroapi_calls = timings.aeroapi_calls
        run.awc_calls = timings.awc_calls
        run.aeroapi_seconds = round(timings.aeroapi_seconds, 4)
        run.awc_seconds = round(timings.awc_seconds, 4)
        run.retrieval_seconds = round(timings.retrieval_seconds, 4)
        run.explainer_seconds = round(timings.explainer_seconds, 4)
        run.scoring_seconds = round(timings.scoring_seconds, 4)
        run.embedding_calls = timings.embedding_calls
        run.embedding_cpu_ms = round(timings.embedding_cpu_ms, 2)

    return run


# ------------------------------------------------------------------ queries


def _rows(conn: sqlite3.Connection, sql: str, params=()) -> list[dict]:
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def summary(conn: sqlite3.Connection, days: int = RETENTION_DAYS
            ) -> dict[str, Any]:
    """Everything the status page needs, in one pass per panel."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    totals = _rows(conn, """
        SELECT COUNT(*) AS searches,
               SUM(reading IS NOT NULL AND reading != 'unresolved') AS resolved,
               SUM(api_calls) AS api_calls,
               SUM(degraded) AS degraded,
               SUM(truncated) AS truncated
        FROM search_runs WHERE started_at >= ?""", (since,))[0]

    elapsed = [r["elapsed"] for r in _rows(
        conn, "SELECT elapsed FROM search_runs WHERE started_at >= ? "
              "AND elapsed IS NOT NULL ORDER BY elapsed", (since,))]
    median = elapsed[len(elapsed) // 2] if elapsed else None

    daily = _rows(conn, """
        SELECT substr(started_at, 1, 10) AS day,
               COUNT(*) AS total,
               SUM(reading IS NOT NULL AND reading != 'unresolved') AS resolved
        FROM search_runs WHERE started_at >= ?
        GROUP BY day ORDER BY day""", (since,))

    sources = _rows(conn, """
        SELECT CASE
                 WHEN observed_reading != 'unresolved'
                  AND forecast_reading != 'unresolved' THEN 'both'
                 WHEN forecast_reading != 'unresolved' THEN 'forecast only'
                 WHEN observed_reading != 'unresolved' THEN 'pilots only'
                 ELSE 'neither' END AS which,
               COUNT(*) AS n
        FROM search_runs WHERE started_at >= ?
        GROUP BY which ORDER BY n DESC""", (since,))

    timing = _rows(conn, """
        SELECT AVG(aeroapi_seconds) AS aeroapi,
               AVG(awc_seconds) AS awc,
               AVG(retrieval_seconds) AS retrieval,
               AVG(explainer_seconds) AS explainer,
               AVG(scoring_seconds) AS scoring
        FROM search_runs WHERE started_at >= ? AND elapsed IS NOT NULL""",
        (since,))[0]

    models = _rows(conn, """
        SELECT SUM(llm_called) AS llm_calls,
               SUM(llm_accepted) AS llm_accepted,
               SUM(llm_tokens_in) AS tokens_in,
               SUM(llm_tokens_out) AS tokens_out,
               AVG(explainer_seconds) AS llm_seconds,
               SUM(embedding_calls) AS embedding_calls,
               AVG(embedding_cpu_ms) AS embedding_cpu_ms,
               SUM(api_calls) AS api_calls
        FROM search_runs WHERE started_at >= ?""", (since,))[0]

    rejections = _rows(conn, """
        SELECT llm_reject_reason AS reason, COUNT(*) AS n
        FROM search_runs
        WHERE started_at >= ? AND llm_reject_reason IS NOT NULL
        GROUP BY reason ORDER BY n DESC LIMIT 6""", (since,))

    degraded = _rows(conn, """
        SELECT degraded_reason AS reason, COUNT(*) AS n
        FROM search_runs
        WHERE started_at >= ? AND degraded_reason IS NOT NULL
        GROUP BY reason ORDER BY n DESC LIMIT 6""", (since,))

    countries = _rows(conn, """
        SELECT country, COUNT(*) AS n FROM search_runs
        WHERE started_at >= ? AND country IS NOT NULL
        GROUP BY country ORDER BY n DESC LIMIT 8""", (since,))

    return {
        "window_days": days,
        "totals": totals,
        "median_seconds": median,
        "daily": daily,
        "sources": sources,
        "timing": timing,
        "models": models,
        "rejections": rejections,
        "degraded": degraded,
        "countries": countries,
    }
