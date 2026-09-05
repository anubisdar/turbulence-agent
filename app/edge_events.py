# install-to: app
"""
Edge events, persisted so the status page has one window rather than four.

WHY THIS EXISTS. Three panels on the status page were reconstructed from
logs at read time: firewall detections and challenge refusals from the
journal, edge blocks from the Caddy access log. Each carried its own
window - 24 hours for two of them, and for the third a byte-bounded tail
that meant different spans depending on traffic and reset entirely on log
rotation. Next to a database that keeps 30 exact days, the effect was data
that appeared to vanish overnight.

The counts also could not be exact. A tail read sees what is in the file
now, and a log that rolls at 20 MB drops history without saying so.

WHAT CHANGED. A timer ingests these events into the same database the
searches live in, on the same 30-day retention. The status page then reads
one source with one window, and makes no subprocess calls at all.

ADDRESSES ARE NOT STORED. Country and network are resolved during ingest
and the address is discarded, exactly as for searches. The deduplication
key is a hash, which cannot be reversed into an address.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

#: Matches the search record, so every panel covers the same span.
RETENTION_DAYS = 30

SCHEMA = """
CREATE TABLE IF NOT EXISTS edge_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at TEXT NOT NULL,
    kind        TEXT NOT NULL,
    detail      TEXT,
    country     TEXT,
    asn_name    TEXT,
    path        TEXT,
    score       INTEGER,
    probe       INTEGER NOT NULL DEFAULT 0,
    dedup_key   TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_edge_when ON edge_events(occurred_at);
CREATE INDEX IF NOT EXISTS idx_edge_kind ON edge_events(kind, occurred_at);
"""

#: The three things the edge produces that never become searches.
WAF = "waf"
EDGE_BLOCK = "edge_block"
CHALLENGE_REFUSAL = "challenge_refusal"


def init_edge_events(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # CREATE TABLE IF NOT EXISTS leaves an existing table alone, so a
    # database written before `probe` existed never gains the column from
    # the schema above. Added here instead, idempotently, because the
    # alternative is a status page that reads a column that is not there.
    columns = {row[1] for row in conn.execute(
        "PRAGMA table_info(edge_events)")}
    if "probe" not in columns:
        conn.execute("ALTER TABLE edge_events "
                     "ADD COLUMN probe INTEGER NOT NULL DEFAULT 0")
    conn.commit()


def dedup_key(*parts: Any) -> str:
    """A stable identity for an event, so re-reading a window is harmless.

    The ingest deliberately re-reads more than it needs - clock skew and a
    slow writer both mean the last run's boundary is not trustworthy - so
    every insert has to be idempotent. Hashed rather than stored raw
    because one of the parts is usually an address.
    """
    joined = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:32]


def record_events(conn: sqlite3.Connection,
                  events: Iterable[dict],
                  retention_days: int = RETENTION_DAYS) -> int:
    """Insert events, ignoring ones already seen. Returns the number added."""
    rows = [(e.get("occurred_at"), e.get("kind"), e.get("detail"),
             e.get("country"), e.get("asn_name"), e.get("path"),
             e.get("score"), 1 if e.get("probe") else 0,
             e["dedup_key"]) for e in events]
    if not rows:
        return 0

    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO edge_events "
        "(occurred_at, kind, detail, country, asn_name, path, score, "
        " probe, dedup_key) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    cutoff = (datetime.now(timezone.utc)
              - timedelta(days=retention_days)).isoformat()
    conn.execute("DELETE FROM edge_events WHERE occurred_at < ?", (cutoff,))
    conn.commit()
    return conn.total_changes - before


def _rows(conn: sqlite3.Connection, sql: str, args: tuple) -> list[dict]:
    conn.row_factory = sqlite3.Row
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def summary(conn: sqlite3.Connection,
            days: int = RETENTION_DAYS) -> dict:
    """Everything the three edge panels need, over the same window as the
    rest of the page.

    Every count here excludes the operator's own probes. check_edge.sh
    fires four attack vectors an hour from a known address, and each one
    trips several rule families, so on a site this quiet the self-check
    was most of the firewall panel - 625 of 673 detections were United
    States, which is to say were the check itself. The probes are marked
    at ingest rather than dropped, and returned under `self_check` so the
    page can say how many were set aside. Hiding them would replace one
    wrong number with another.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    def top(kind: str, column: str, limit: int = 6) -> list[dict]:
        return _rows(conn, f"""
            SELECT COALESCE({column}, 'unknown') AS label, COUNT(*) AS n
            FROM edge_events
            WHERE kind = ? AND occurred_at >= ? AND probe = 0
            GROUP BY label ORDER BY n DESC LIMIT {limit}""", (kind, since))

    waf_totals = _rows(conn, """
        SELECT COUNT(*) AS detections,
               SUM(CASE WHEN score >= 5 THEN 1 ELSE 0 END) AS would_block,
               MAX(score) AS worst_score
        FROM edge_events
        WHERE kind = ? AND occurred_at >= ? AND probe = 0""",
        (WAF, since))[0]

    # Counted from the deduplication key rather than the address, which is
    # never stored. Distinct keys undercount a source that sent identical
    # requests in the same second, which is a price worth paying.
    waf_sources = _rows(conn, """
        SELECT COUNT(DISTINCT country || '/' || COALESCE(asn_name, '')) AS n
        FROM edge_events
        WHERE kind = ? AND occurred_at >= ? AND probe = 0""",
        (WAF, since))[0]["n"]

    # What was set aside, so the page can account for it rather than
    # appear to have lost traffic between one deploy and the next.
    self_check = {row["label"]: row["n"] for row in _rows(conn, """
        SELECT kind AS label, COUNT(*) AS n
        FROM edge_events
        WHERE occurred_at >= ? AND probe = 1
        GROUP BY kind""", (since,))}

    return {
        "window_days": days,
        "waf": {
            "detections": waf_totals["detections"] or 0,
            "would_block": waf_totals["would_block"] or 0,
            "worst_score": waf_totals["worst_score"] or 0,
            "networks_seen": waf_sources or 0,
            "attacks": top(WAF, "detail"),
            "countries": top(WAF, "country"),
            "paths": top(WAF, "path"),
        },
        "blocked": top(EDGE_BLOCK, "detail"),
        "blocked_countries": top(EDGE_BLOCK, "country", 8),
        "refusals": top(CHALLENGE_REFUSAL, "detail"),
        "self_check": {
            "waf": self_check.get(WAF, 0),
            "edge_block": self_check.get(EDGE_BLOCK, 0),
            "total": sum(self_check.values()),
        },
    }
