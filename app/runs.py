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
from collections import namedtuple
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
    region              TEXT,
    asn_number          TEXT,
    asn_name            TEXT,
    challenge           TEXT,
    fact_problems       INTEGER,

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


#: Columns added after the table first shipped. Anything here is applied to
#: an existing table on start; anything in SCHEMA alone reaches only a fresh
#: one.
_EXPECTED_COLUMNS = {
    "source": "TEXT",
    "region": "TEXT",
    "asn_number": "TEXT",
    "asn_name": "TEXT",
    "challenge": "TEXT",
    "fact_problems": "INTEGER",
}


@dataclass
class RunRecord:
    """Everything worth keeping about one search."""
    request_id: str
    started_at: str = ""

    origin: str | None = None
    dest: str | None = None
    country: str | None = None
    region: str | None = None
    asn_number: str | None = None
    asn_name: str | None = None
    #: How the request got past the challenge. Refusals never reach here,
    #: because a refused request never becomes a search - those are counted
    #: from the log instead.
    #: How the search ran: "live" against the real APIs, or "fixtures"
    #: against recorded ones. Both were written to this table
    #: indistinguishably until 28 August, so a status page over that
    #: period reported replay runs as production behaviour.
    source: str = "live"
    challenge: str | None = None
    #: Facts that failed their shape check before reaching the model.
    #: Zero on every ordinary search; anything else means a provider sent
    #: something unexpected, or something tried to reach the model through
    #: one of the two fields this system does not compute.
    fact_problems: int = 0

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
    """Create the table, and add any column a newer build expects.

    `CREATE TABLE IF NOT EXISTS` leaves an existing table alone, so a
    deployment that adds a column finds the old shape still in place and
    fails on the first query that mentions it - a 500 on the status page
    rather than anything visible at install time.

    SQLite has no `ADD COLUMN IF NOT EXISTS`, so the existing columns are
    read and the difference applied. Adding a column is cheap, backfills
    NULL, and is safe to run on every start.
    """
    conn.executescript(SCHEMA)

    existing = {row[1] for row in conn.execute(
        "PRAGMA table_info(search_runs)")}
    for column, kind in _EXPECTED_COLUMNS.items():
        if column not in existing:
            conn.execute(
                f"ALTER TABLE search_runs ADD COLUMN {column} {kind}")
    conn.commit()


#: Where the MaxMind databases live once update-geoip.sh has run.
GEOIP_DIR = os.environ.get("TURBULENCE_GEOIP_DIR", "/usr/share/GeoIP")

#: A resolved network, deliberately coarse. See resolve_origin.
Origin = namedtuple(
    "Origin", "country country_name region asn_number asn_name")
UNKNOWN_ORIGIN = Origin(None, None, None, None, None)


def _lookup(ip: str, filename: str) -> dict | None:
    try:
        import maxminddb
    except ImportError:
        return None
    path = os.path.join(GEOIP_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        with maxminddb.open_database(path) as reader:
            found = reader.get(ip)
        return found if isinstance(found, dict) else None
    except (ValueError, OSError):
        return None


def resolve_origin(ip: str | None) -> Origin:
    """Where a request came from, at a deliberately coarse resolution.

    The address is used and discarded. Nothing here returns it and nothing
    stores it, which is the rule the threat model settled on: an address
    plus a route is closer to personal data than either alone.

    REGION, NOT CITY. City was considered and rejected twice over. On free
    GeoLite2 it is frequently the registrant's address rather than the
    user's, so the chart would be confidently wrong; and city plus a
    timestamp on a site with this little traffic identifies a person, which
    is the thing the address was withheld to avoid. A US state is accurate
    and coarse enough to stay on the right side of that.

    THE NETWORK IS THE USEFUL PART. Country answers nothing when every
    other country is blocked at the edge. What actually distinguishes
    traffic is whose network it arrives from: a request from AS16509
    (Amazon) is automation, one from AS7922 (Comcast) is a person on a
    domestic line. That is the question the panel is really asking.
    """
    if not ip:
        return UNKNOWN_ORIGIN

    address = ip.split(",")[0].strip()
    if not address:
        return UNKNOWN_ORIGIN

    country = country_name = region = asn_number = asn_name = None

    # The databases carry the country's name alongside its code, so nothing
    # here needs a hand-maintained table of 250 entries that would drift.
    # The code is what gets stored - it is stable and compact - and the name
    # is what gets displayed.
    city = _lookup(address, "GeoLite2-City.mmdb")
    if city:
        found = city.get("country") or {}
        country = found.get("iso_code")
        country_name = (found.get("names") or {}).get("en")
        subdivisions = city.get("subdivisions") or []
        if subdivisions:
            names = (subdivisions[0].get("names") or {})
            region = names.get("en") or subdivisions[0].get("iso_code")

    if country is None:
        found = (_lookup(address, "GeoLite2-Country.mmdb") or {})
        found = found.get("country") or {}
        country = found.get("iso_code")
        country_name = country_name or (found.get("names") or {}).get("en")

    asn = _lookup(address, "GeoLite2-ASN.mmdb")
    if asn:
        number = asn.get("autonomous_system_number")
        asn_number = f"AS{number}" if number is not None else None
        asn_name = asn.get("autonomous_system_organization")

    return Origin(country, country_name or country, region,
                  asn_number, asn_name)


def resolve_country(ip: str | None,
                    db_path: str | None = None) -> str | None:
    """Kept for callers that only want the country."""
    return resolve_origin(ip).country


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
                 country: str | None = None,
                 origin_info: "Origin | None" = None,
                 source: str = "live",
                 challenge: str | None = None) -> RunRecord:
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
        country=(origin_info.country if origin_info else country),
        region=(origin_info.region if origin_info else None),
        asn_number=(origin_info.asn_number if origin_info else None),
        asn_name=(origin_info.asn_name if origin_info else None),
        challenge=challenge,
        fact_problems=len(
            ((payload.get("explanation") or {}).get("fact_problems")) or []),
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
    """Everything the status page needs, in one pass per panel.

    Live runs only. Replay and fixture searches were written to the same
    table with nothing to tell them apart, so a window holding 977
    replayed searches and six real ones described the replay. They are
    counted separately rather than dropped, because "how much of this
    window was synthetic" is itself worth being able to answer.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    synthetic = _rows(conn, """
        SELECT COUNT(*) AS n FROM search_runs
        WHERE started_at >= ? AND COALESCE(source, 'live') != 'live'
        """, (since,))[0]["n"]

    totals = _rows(conn, """
        SELECT COUNT(*) AS searches,
               SUM(reading IS NOT NULL AND reading != 'unresolved') AS resolved,
               SUM(api_calls) AS api_calls,
               SUM(degraded) AS degraded,
               SUM(truncated) AS truncated
        FROM search_runs WHERE started_at >= ?
          AND COALESCE(source, 'live') = 'live'""", (since,))[0]

    elapsed = [r["elapsed"] for r in _rows(
        conn, "SELECT elapsed FROM search_runs WHERE started_at >= ? "
              "AND COALESCE(source, 'live') = 'live' "
              "AND elapsed IS NOT NULL ORDER BY elapsed", (since,))]
    median = elapsed[len(elapsed) // 2] if elapsed else None

    daily = _rows(conn, """
        SELECT substr(started_at, 1, 10) AS day,
               COUNT(*) AS total,
               SUM(reading IS NOT NULL AND reading != 'unresolved') AS resolved
        FROM search_runs WHERE started_at >= ?
          AND COALESCE(source, 'live') = 'live'
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
          AND COALESCE(source, 'live') = 'live'
        GROUP BY which ORDER BY n DESC""", (since,))

    timing = _rows(conn, """
        SELECT AVG(aeroapi_seconds) AS aeroapi,
               AVG(awc_seconds) AS awc,
               AVG(retrieval_seconds) AS retrieval,
               AVG(explainer_seconds) AS explainer,
               AVG(scoring_seconds) AS scoring
        FROM search_runs WHERE started_at >= ?
          AND COALESCE(source, 'live') = 'live' AND elapsed IS NOT NULL""",
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
        FROM search_runs WHERE started_at >= ?
          AND COALESCE(source, 'live') = 'live'""", (since,))[0]

    rejections = _rows(conn, """
        SELECT llm_reject_reason AS reason, COUNT(*) AS n
        FROM search_runs
        WHERE started_at >= ?
          AND COALESCE(source, 'live') = 'live' AND llm_reject_reason IS NOT NULL
        GROUP BY reason ORDER BY n DESC LIMIT 6""", (since,))

    degraded = _rows(conn, """
        SELECT degraded_reason AS reason, COUNT(*) AS n
        FROM search_runs
        WHERE started_at >= ?
          AND COALESCE(source, 'live') = 'live' AND degraded_reason IS NOT NULL
        GROUP BY reason ORDER BY n DESC LIMIT 6""", (since,))

    regions = _rows(conn, """
        SELECT COALESCE(region, country, 'unknown') AS region,
               COUNT(*) AS n
        FROM search_runs WHERE started_at >= ?
          AND COALESCE(source, 'live') = 'live'
        GROUP BY region ORDER BY n DESC LIMIT 8""", (since,))

    networks = _rows(conn, """
        SELECT COALESCE(asn_name, 'unknown') AS network,
               COALESCE(asn_number, '') AS asn,
               COUNT(*) AS n
        FROM search_runs WHERE started_at >= ?
          AND COALESCE(source, 'live') = 'live'
        GROUP BY network, asn ORDER BY n DESC LIMIT 8""", (since,))

    challenges = _rows(conn, """
        SELECT COALESCE(challenge, 'not recorded') AS outcome,
               COUNT(*) AS n
        FROM search_runs WHERE started_at >= ?
          AND COALESCE(source, 'live') = 'live'
        GROUP BY outcome ORDER BY n DESC""", (since,))

    # Zero on every ordinary search, so a non-zero value is the whole
    # signal. Counted rather than sampled: this is rare enough that every
    # occurrence is worth seeing.
    fact_problems = _rows(conn, """
        SELECT COUNT(*) AS searches, SUM(fact_problems) AS problems
        FROM search_runs
        WHERE started_at >= ?
          AND COALESCE(source, 'live') = 'live' AND COALESCE(fact_problems, 0) > 0""",
        (since,))[0]

    # Sliced, not just totalled. An aggregate acceptance rate of 91% hid
    # the thing that mattered: every rejection was on an unresolved route,
    # and every one of them was the validator being wrong. A single number
    # cannot show a pattern that lives in a subset.
    #
    # The slices are the ways a search genuinely differs: whether anything
    # was known, whether the two sources agreed, and whether the search ran
    # to completion or hit a budget.
    by_outcome = _rows(conn, """
        SELECT
            CASE
                WHEN degraded THEN 'a source failed (pilot reports, forecast)'
                WHEN truncated THEN 'stopped on a budget'
                WHEN reading = 'unresolved' THEN 'nothing was known'
                WHEN sources_disagree THEN 'sources disagreed'
                ELSE 'a reading, sources agreed'
            END AS slice,
            COUNT(*) AS searches,
            SUM(llm_called) AS explained,
            SUM(CASE WHEN llm_called AND llm_accepted THEN 1 ELSE 0 END)
                AS accepted,
            ROUND(AVG(elapsed), 1) AS median_seconds,
            ROUND(AVG(api_calls), 1) AS mean_calls,
            SUM(COALESCE(fact_problems, 0)) AS fact_problems
        FROM search_runs
        WHERE started_at >= ?
          AND COALESCE(source, 'live') = 'live'
        GROUP BY slice ORDER BY searches DESC""", (since,))

    for row in by_outcome:
        explained = row.get("explained") or 0
        row["acceptance"] = (round((row.get("accepted") or 0) / explained, 3)
                             if explained else None)

    return {
        "window_days": days,
        "synthetic_runs": synthetic,
        "challenges": challenges,
        "by_outcome": by_outcome,
        "fact_problems": {
            "searches": fact_problems["searches"] or 0,
            "problems": fact_problems["problems"] or 0,
        },
        "totals": totals,
        "median_seconds": median,
        "daily": daily,
        "sources": sources,
        "timing": timing,
        "models": models,
        "rejections": rejections,
        "degraded": degraded,
        "regions": regions,
        "networks": networks,
    }
