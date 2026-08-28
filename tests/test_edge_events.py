"""Edge events: firewall detections, edge blocks and challenge refusals.

None of these is produced by the application. They happen in Caddy, or
before a search starts, and were previously reconstructed from logs while
rendering the status page - two panels on a 24 hour window and one on a
byte-bounded tail of a file that rolls at 20 MB. Everything else on that
page came from a database keeping 30 exact days.

The result was data that appeared to vanish overnight, and counts that
could not be exact. A timer now ingests them into the same database on the
same retention.
"""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from app.edge_events import (
    CHALLENGE_REFUSAL,
    EDGE_BLOCK,
    RETENTION_DAYS,
    WAF,
    dedup_key,
    init_edge_events,
    record_events,
    summary,
)

NOW = datetime.now(timezone.utc)


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    init_edge_events(c)
    return c


def event(kind=WAF, days_ago=0, detail="SQL injection", country="China",
          path="/", score=5, key=None):
    return {
        "occurred_at": (NOW - timedelta(days=days_ago)).isoformat(),
        "kind": kind, "detail": detail, "country": country,
        "asn_name": "Some Network", "path": path, "score": score,
        "dedup_key": key or dedup_key(kind, days_ago, detail, path),
    }


class TestIngestIsIdempotent:
    """The ingest re-reads far more than the timer interval on purpose:
    clock skew and a slow writer both mean the previous boundary is not
    trustworthy. Every insert therefore has to be repeatable."""

    def test_the_same_events_insert_once(self, conn):
        events = [event(days_ago=i) for i in range(5)]
        assert record_events(conn, events) == 5
        assert record_events(conn, events) == 0

    def test_an_overlapping_window_adds_only_the_new(self, conn):
        record_events(conn, [event(days_ago=i) for i in range(3)])
        added = record_events(conn, [event(days_ago=i) for i in range(5)])
        assert added == 2

    def test_the_key_does_not_contain_the_address(self):
        """It is a hash. The address is resolved to a country during ingest
        and then discarded, as it is for searches."""
        key = dedup_key(WAF, "45.155.205.7", "SQL injection")
        assert "45.155.205.7" not in key
        assert len(key) == 32

    def test_empty_input_is_harmless(self, conn):
        assert record_events(conn, []) == 0


class TestOneWindow:
    def test_the_window_matches_the_search_record(self, conn):
        assert RETENTION_DAYS == 30
        assert summary(conn)["window_days"] == 30

    def test_events_inside_the_window_are_counted(self, conn):
        record_events(conn, [event(days_ago=d) for d in (0, 5, 14, 29)])
        assert summary(conn)["waf"]["detections"] == 4

    def test_events_beyond_retention_are_pruned_on_write(self, conn):
        """Not merely excluded from the query - removed, so the table does
        not grow without bound."""
        record_events(conn, [event(days_ago=0), event(days_ago=45)])
        remaining = conn.execute(
            "SELECT COUNT(*) FROM edge_events").fetchone()[0]
        assert remaining == 1

    def test_a_panel_no_longer_resets_daily(self, conn):
        """The behaviour that prompted this: a detection from three days
        ago used to fall outside a 24 hour window and disappear."""
        record_events(conn, [event(days_ago=3)])
        assert summary(conn)["waf"]["detections"] == 1


class TestSummaryShape:
    def test_would_block_counts_the_threshold(self, conn):
        """The rule set refuses at an anomaly score of five."""
        record_events(conn, [
            event(days_ago=0, score=4, key="a"),
            event(days_ago=1, score=5, key="b"),
            event(days_ago=2, score=40, key="c")])
        waf = summary(conn)["waf"]
        assert waf["detections"] == 3
        assert waf["would_block"] == 2
        assert waf["worst_score"] == 40

    def test_the_three_kinds_are_reported_separately(self, conn):
        record_events(conn, [
            event(kind=WAF, key="w"),
            event(kind=EDGE_BLOCK, detail="geo filter", key="e"),
            event(kind=CHALLENGE_REFUSAL, detail="presented no token",
                  key="c")])
        data = summary(conn)
        assert data["waf"]["detections"] == 1
        assert data["blocked"] == [{"label": "geo filter", "n": 1}]
        assert data["refusals"] == [{"label": "presented no token", "n": 1}]

    def test_an_edge_block_does_not_count_as_a_detection(self, conn):
        record_events(conn, [event(kind=EDGE_BLOCK, key="e")])
        assert summary(conn)["waf"]["detections"] == 0

    def test_an_empty_store_reports_zero_rather_than_failing(self, conn):
        data = summary(conn)
        assert data["waf"]["detections"] == 0
        assert data["waf"]["worst_score"] == 0
        assert data["blocked"] == []


class TestIngestReadsRealLogShapes:
    """Parsed from the formats actually produced by this deployment, not
    from an idea of them."""

    def _module(self):
        import importlib.util
        from pathlib import Path

        path = (Path(__file__).resolve().parents[1] / "scripts"
                / "ingest_edge_events.py")
        if not path.exists():
            pytest.skip("ingest_edge_events.py not installed")
        spec = importlib.util.spec_from_file_location("ingest", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _waf_line(self, rule="942100", tag="attack-sqli", uri="/?id=1",
                  uid="abc", ip="45.155.205.7", score=None):
        import json

        msg = (f'[client "{ip}"] Coraza: Warning. matched [id "{rule}"] '
               f'[tag "application-multi"] [tag "{tag}"] [tag "OWASP_CRS"] '
               f'[uri "{uri}"] [unique_id "{uid}"]')
        if score is not None:
            msg = (f'[client "{ip}"] Coraza: Warning. Inbound Anomaly Score '
                   f'Exceeded (Total Score: {score}) [id "949110"] '
                   f'[tag "anomaly-evaluation"] [uri "{uri}"] '
                   f'[unique_id "{uid}"]')
        return json.dumps({"level": "error", "ts": 1787516236.7,
                           "logger": "http.handlers.waf", "msg": msg})

    def test_one_request_with_four_rules_is_one_detection(self):
        """The four test vectors run against this deployment produced
        thirteen log lines between them. Counting lines would be wrong by a
        factor of three."""
        module = self._module()
        lines = [
            self._waf_line(rule="930100", tag="attack-lfi", uid="same"),
            self._waf_line(rule="930110", tag="attack-lfi", uid="same"),
            self._waf_line(rule="930120", tag="attack-lfi", uid="same"),
            self._waf_line(uid="same", score=40),
        ]
        events = module.read_waf(lines, {})
        assert len({e["dedup_key"] for e in events}) == 1
        assert events[0]["score"] == 40

    def test_two_families_on_one_request_are_both_recorded(self):
        module = self._module()
        events = module.read_waf([
            self._waf_line(tag="attack-lfi", uid="x"),
            self._waf_line(tag="attack-rce", uid="x"),
            self._waf_line(uid="x", score=40)], {})
        assert {e["detail"] for e in events} == {
            "path traversal", "remote command execution"}

    def test_a_bare_path_keeps_its_query(self):
        """Stripping it makes every injection look like "/", because the
        payload is in the query and the path is bare."""
        module = self._module()
        assert module._label_path("/?q=%3Cscript%3E").startswith("/?q=<script>")

    def test_a_real_path_drops_its_query(self):
        module = self._module()
        assert module._label_path("/.env?a=1") == "/.env"

    def test_refusals_distinguish_no_token_from_a_bad_one(self):
        """One saw the page and failed; the other never saw it, which is
        what an automated client looks like."""
        module = self._module()
        events = module.read_refusals([
            "WARNING req=aaa turbulence-agent.api challenge outcome=no_token",
            "WARNING req=bbb turbulence-agent.api challenge outcome=rejected",
            "INFO req=ccc turbulence-agent.api challenge outcome=session",
        ])
        assert {e["detail"] for e in events} == {
            "presented no token", "presented one that failed"}

    def test_non_firewall_journal_lines_are_ignored(self):
        import json

        module = self._module()
        lines = [
            json.dumps({"logger": "http.log.access", "msg": "handled"}),
            json.dumps({"logger": "tls", "msg": "certificate obtained"}),
            "not json at all",
            self._waf_line(uid="only"),
        ]
        assert len({e["dedup_key"] for e in module.read_waf(lines, {})}) == 1

    def test_a_403_is_attributed_by_its_marker_not_its_status(self, tmp_path):
        """Both the geo filter and the firewall refuse with 403, so the
        status stopped being enough the moment the engine moved to
        blocking. Before this, four firewall refusals from a US address
        were reported as blocked by a filter that allows the US.

        The geo handler appends blocked_by="geo"; a 403 without it came
        from the firewall.
        """
        import json

        module = self._module()
        log = tmp_path / "access.log"
        now = datetime.now(timezone.utc).timestamp()
        log.write_text("\n".join(json.dumps(e) for e in [
            {"ts": now, "status": 403, "blocked_by": "geo",
             "request": {"remote_ip": "1.2.3.4", "uri": "/"}},
            {"ts": now, "status": 403,
             "request": {"remote_ip": "1.2.3.5", "uri": "/?q=x"}},
            {"ts": now, "status": 429,
             "request": {"remote_ip": "1.2.3.6", "uri": "/"}},
            {"ts": now, "status": 200,
             "request": {"remote_ip": "1.2.3.7", "uri": "/"}},
        ]) + "\n")
        events = module.read_edge_blocks(
            str(log), datetime.now(timezone.utc) - timedelta(hours=1), {})
        assert {e["detail"] for e in events} == {
            "geo filter", "web application firewall", "rate limit"}

    def test_the_marker_is_read_from_response_headers_too(self, tmp_path):
        """Caddy's log_append can surface under resp_headers depending on
        version, and there it arrives as a list."""
        import json

        module = self._module()
        log = tmp_path / "access.log"
        now = datetime.now(timezone.utc).timestamp()
        log.write_text(json.dumps(
            {"ts": now, "status": 403,
             "resp_headers": {"blocked_by": ["geo"]},
             "request": {"remote_ip": "1.2.3.4", "uri": "/"}}) + "\n")
        events = module.read_edge_blocks(
            str(log), datetime.now(timezone.utc) - timedelta(hours=1), {})
        assert events[0]["detail"] == "geo filter"

    def test_a_missing_access_log_is_not_an_error(self, tmp_path):
        module = self._module()
        assert module.read_edge_blocks(
            str(tmp_path / "absent.log"),
            datetime.now(timezone.utc), {}) == []
