"""The operator's own probes, and keeping them out of the counts.

check_edge.sh fires four attack vectors an hour and sets a distinctive
user agent, and its header says the probes carry it "so they can be
excluded from the status page rather than inflating its detection
counts". Nothing read it. Each vector trips several rule families, so the
firewall panel reached 673 detections of which 625 were United States -
the health check watching itself, presented as attack traffic.

The exclusion is asserted here rather than left to the comment that
described it for months without it existing.

Marked, not dropped. `probe = 1` stays in the table so who-came.sh and
any later question can still see it; only the panels set it aside, and
`self_check` in the summary says how much was set aside so the page can
account for the difference rather than appear to have lost traffic
between one deploy and the next.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.edge_events import (
    EDGE_BLOCK,
    WAF,
    init_edge_events,
    record_events,
    summary,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import ingest_edge_events as ing  # noqa: E402


#: RFC 5737 documentation ranges, not real addresses. The operator's own
#: address is what the live filter matches on, and this repository is
#: public - a test fixture is not a reason to publish where somebody
#: lives.
PROBE_IP = "192.0.2.1"          # TEST-NET-1, stands in for the operator
OUTSIDE_IP = "198.51.100.7"     # TEST-NET-2, stands in for a scanner
PROBE_UA = "turbulence-edge-check/1.0 (daily health probe)"


class _Origin:
    """Stands in for app.runs.Origin without touching the databases."""

    def __init__(self, ip: str) -> None:
        probe = ip == PROBE_IP
        self.country = "US" if probe else "LU"
        self.country_name = "United States" if probe else "Luxembourg"
        self.asn_name = "Verizon Business" if probe else "Visual Online S.A."


@pytest.fixture(autouse=True)
def _no_geoip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ing, "resolve_origin", _Origin)


@pytest.fixture
def now() -> float:
    return time.time()


@pytest.fixture
def since() -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=30)


def access_line(ts: float, ip: str, agent: str, uri: str,
                status: int = 403) -> str:
    """One Caddy access log entry. Headers are lists, as Caddy writes
    them."""
    return json.dumps({
        "ts": ts, "status": status,
        "request": {"remote_ip": ip, "uri": uri,
                    "headers": {"User-Agent": [agent]}},
        "resp_headers": {},
    })


def waf_line(ts: float, ip: str, uri: str, unique_id: str, tag: str,
             score: int) -> str:
    """One Coraza journal line. Note what is absent: no user agent.

    That absence is the reason probe_addresses exists - a firewall
    detection cannot be recognised as a probe from its own log line.
    """
    return json.dumps({
        "logger": "http.handlers.waf", "ts": ts,
        "msg": (f'Coraza: Warning. [unique_id "{unique_id}"] '
                f'[uri "{uri}"] [client "{ip}"] [tag "{tag}"] '
                f'Total Score: {score}'),
    })


def write_log(tmp_path: Path, lines: list[str]) -> str:
    path = tmp_path / "access.log"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


class TestProbeAddresses:
    """Which addresses presented the probe agent inside the window."""

    def test_the_probe_agent_is_recognised(self, tmp_path, now, since):
        path = write_log(tmp_path, [
            access_line(now - 60, PROBE_IP, PROBE_UA,
                        "/?file=../../etc/passwd"),
            access_line(now - 60, OUTSIDE_IP, "curl/8.0", "/"),
        ])
        assert ing.probe_addresses(path, since) == {PROBE_IP}

    def test_entries_outside_the_window_are_ignored(self, tmp_path, now,
                                                    since):
        """A probe from yesterday must not excuse today's traffic from the
        same address."""
        path = write_log(tmp_path, [
            access_line(now - 86400, PROBE_IP, PROBE_UA, "/"),
        ])
        assert ing.probe_addresses(path, since) == set()

    def test_a_missing_log_yields_no_probes(self, tmp_path, since):
        """Fails towards counting everything as real traffic, which is the
        error that gets noticed."""
        missing = str(tmp_path / "absent.log")
        assert ing.probe_addresses(missing, since) == set()

    def test_malformed_lines_do_not_stop_the_scan(self, tmp_path, now, since):
        path = write_log(tmp_path, [
            "not json at all",
            "",
            access_line(now - 60, PROBE_IP, PROBE_UA, "/"),
        ])
        assert ing.probe_addresses(path, since) == {PROBE_IP}


class TestWafClassification:
    """A firewall detection has no agent of its own to go on."""

    def test_a_probe_detection_is_marked(self, now):
        lines = [waf_line(now - 60, PROBE_IP, "/?file=x", "P1",
                          "attack-lfi", 40)]
        events = ing.read_waf(lines, {}, {PROBE_IP})
        assert len(events) == 1
        assert events[0]["probe"] is True

    def test_outside_traffic_is_not_marked(self, now):
        lines = [waf_line(now - 60, OUTSIDE_IP, "/?id=1", "R1",
                          "attack-sqli", 20)]
        events = ing.read_waf(lines, {}, {PROBE_IP})
        assert events[0]["probe"] is False

    def test_every_family_of_one_probe_request_is_marked(self, now):
        """One request tripping several rules becomes several rows, and
        the fan-out is why the panel read 673 rather than about 130."""
        lines = [
            waf_line(now - 60, PROBE_IP, "/?file=x", "P1", "attack-lfi", 40),
            waf_line(now - 60, PROBE_IP, "/?file=x", "P1", "attack-rce", 40),
            waf_line(now - 60, PROBE_IP, "/?file=x", "P1", "attack-generic",
                     40),
        ]
        events = ing.read_waf(lines, {}, {PROBE_IP})
        assert len(events) == 3
        assert all(e["probe"] for e in events)

    def test_no_probe_set_leaves_everything_unmarked(self, now):
        """--no-probe-filter, and the behaviour before this change."""
        lines = [waf_line(now - 60, PROBE_IP, "/?file=x", "P1",
                          "attack-lfi", 40)]
        assert ing.read_waf(lines, {}, set())[0]["probe"] is False

    def test_the_dedup_key_does_not_depend_on_the_flag(self, now):
        """Re-reading a window ingested before the flag existed must not
        insert a second copy of every event."""
        lines = [waf_line(now - 60, PROBE_IP, "/?file=x", "P1",
                          "attack-lfi", 40)]
        marked = ing.read_waf(lines, {}, {PROBE_IP})[0]
        plain = ing.read_waf(lines, {}, set())[0]
        assert marked["dedup_key"] == plain["dedup_key"]


class TestEdgeBlockClassification:
    """Here the agent is on the record, so it is read directly."""

    def test_the_agent_alone_is_enough(self, tmp_path, now, since):
        path = write_log(tmp_path, [
            access_line(now - 60, PROBE_IP, PROBE_UA, "/?file=x"),
        ])
        events = ing.read_edge_blocks(path, since, {}, set())
        assert len(events) == 1
        assert events[0]["probe"] is True

    def test_the_scanner_vector_is_caught_by_address(self, tmp_path, now,
                                                     since):
        """The scanner probe sends a scanner's agent on purpose, so the
        pattern cannot match it and the address set has to."""
        path = write_log(tmp_path, [
            access_line(now - 60, PROBE_IP, "Nikto/2.1.6", "/"),
        ])
        assert ing.read_edge_blocks(path, since, {}, set())[0]["probe"] \
            is False
        assert ing.read_edge_blocks(path, since, {}, {PROBE_IP})[0]["probe"] \
            is True

    def test_outside_traffic_is_not_marked(self, tmp_path, now, since):
        path = write_log(tmp_path, [
            access_line(now - 60, OUTSIDE_IP, "Mozilla/5.0", "/"),
        ])
        assert ing.read_edge_blocks(path, since, {}, {PROBE_IP})[0]["probe"] \
            is False

    def test_a_rate_limit_is_still_recorded(self, tmp_path, now, since):
        path = write_log(tmp_path, [
            access_line(now - 60, OUTSIDE_IP, "curl/8.0", "/", status=429),
        ])
        events = ing.read_edge_blocks(path, since, {}, set())
        assert events[0]["detail"] == "rate limit"


class TestSummaryExcludesProbes:
    """The panels, which is where the wrong number was showing."""

    @pytest.fixture
    def conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(":memory:")
        init_edge_events(c)
        return c

    @staticmethod
    def event(kind, detail, country, probe, key, score=40):
        return {"occurred_at": datetime.now(timezone.utc).isoformat(),
                "kind": kind, "detail": detail, "country": country,
                "asn_name": "n", "path": "/", "score": score,
                "probe": probe, "dedup_key": key}

    def test_probe_detections_are_not_counted(self, conn):
        record_events(conn, [
            self.event(WAF, "path traversal", "United States", True, "p1"),
            self.event(WAF, "path traversal", "United States", True, "p2"),
            self.event(WAF, "SQL injection", "Bulgaria", False, "r1"),
        ])
        data = summary(conn)
        assert data["waf"]["detections"] == 1
        assert [r["label"] for r in data["waf"]["countries"]] == ["Bulgaria"]
        assert [r["label"] for r in data["waf"]["attacks"]] \
            == ["SQL injection"]

    def test_what_was_set_aside_is_reported(self, conn):
        """Otherwise the panel drops between deploys with no explanation,
        which looks like lost data rather than a corrected count."""
        record_events(conn, [
            self.event(WAF, "path traversal", "United States", True, "p1"),
            self.event(EDGE_BLOCK, "web application firewall",
                       "United States", True, "p2"),
            self.event(WAF, "SQL injection", "Bulgaria", False, "r1"),
        ])
        data = summary(conn)
        assert data["self_check"] == {"waf": 1, "edge_block": 1, "total": 2}

    def test_probe_rows_are_kept_in_the_table(self, conn):
        """Marked, not dropped. who-came.sh still has to be able to
        separate the operator's traffic rather than lose it."""
        record_events(conn, [
            self.event(WAF, "path traversal", "United States", True, "p1"),
        ])
        kept = conn.execute(
            "SELECT COUNT(*) FROM edge_events WHERE probe = 1").fetchone()[0]
        assert kept == 1

    def test_worst_score_ignores_the_probes(self, conn):
        """The probes score 40 by design, so leaving them in makes the
        worst score a property of the health check."""
        record_events(conn, [
            self.event(WAF, "path traversal", "United States", True, "p1",
                       score=40),
            self.event(WAF, "SQL injection", "Bulgaria", False, "r1",
                       score=5),
        ])
        assert summary(conn)["waf"]["worst_score"] == 5

    def test_blocked_countries_exclude_probes(self, conn):
        record_events(conn, [
            self.event(EDGE_BLOCK, "web application firewall",
                       "United States", True, "p1"),
            self.event(EDGE_BLOCK, "geo filter", "Bulgaria", False, "r1"),
        ])
        labels = [r["label"] for r in summary(conn)["blocked_countries"]]
        assert labels == ["Bulgaria"]


class TestMigration:
    """A database written before the column existed."""

    OLD_SCHEMA = """
        CREATE TABLE edge_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            occurred_at TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT,
            country TEXT, asn_name TEXT, path TEXT, score INTEGER,
            dedup_key TEXT NOT NULL UNIQUE);"""

    def _old_db(self) -> sqlite3.Connection:
        c = sqlite3.connect(":memory:")
        c.executescript(self.OLD_SCHEMA)
        c.execute("INSERT INTO edge_events (occurred_at, kind, detail,"
                  " country, score, dedup_key) VALUES (?,?,?,?,?,?)",
                  (datetime.now(timezone.utc).isoformat(), WAF,
                   "path traversal", "United States", 40, "old"))
        c.commit()
        return c

    def test_the_column_is_added(self):
        conn = self._old_db()
        init_edge_events(conn)
        columns = {r[1] for r in conn.execute(
            "PRAGMA table_info(edge_events)")}
        assert "probe" in columns

    def test_existing_rows_default_to_real_traffic(self):
        """Rows ingested before the flag existed cannot be reclassified -
        the address they came from was never stored. They stay counted,
        which overstates the panel until they age out of the window. That
        is the safe direction and it is why the deploy note says to
        backfill deliberately or wait."""
        conn = self._old_db()
        init_edge_events(conn)
        assert conn.execute(
            "SELECT probe FROM edge_events").fetchone()[0] == 0
        assert summary(conn)["waf"]["detections"] == 1

    def test_running_it_twice_is_harmless(self):
        conn = self._old_db()
        init_edge_events(conn)
        init_edge_events(conn)
        init_edge_events(conn)
        columns = [r[1] for r in conn.execute(
            "PRAGMA table_info(edge_events)")]
        assert columns.count("probe") == 1
