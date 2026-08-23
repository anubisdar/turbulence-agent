"""Tests for the class of bug that keeps getting through.

Six defects reached a running system in one session, and they shared a
shape rather than a subject:

  A method was called that did not exist.        (generator.cache_report)
  A name was used where it was not bound.        (timings in reputation)
  A field was defined and read but never set.    (client_ip)
  A metric was defined and charted, never fed.   (track_embedding)
  A timing was subtracted but never measured.    (retrieval_seconds)
  Routes were written and then shipped stale.    (/api/status)

Every one passed the existing suite, because the existing suite tests
behaviour and a metric that silently reads zero has no behaviour to be
wrong about. These tests check that things are *connected*, which is a
different question from whether they are correct.

The single most valuable check here is the first one. `pyflakes` finds an
undefined name in under a second and would have caught one of the six
before it left the machine.
"""

import ast
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"


def _modules() -> list[Path]:
    return sorted(p for p in APP.rglob("*.py")
                  if "__pycache__" not in str(p))


class TestNothingIsUndefined:
    """An undefined name is a bug that runs fine until the line executes.

    `timings` sat unbound inside `run_reputation_search` through a full
    install and reached production, where it surfaced as "Safety record
    lookup failed" on every search.
    """

    def test_pyflakes_reports_no_undefined_names(self):
        pyflakes = pytest.importorskip(
            "pyflakes", reason="pip install pyflakes to enable this check")
        result = subprocess.run(
            [sys.executable, "-m", "pyflakes", str(APP)],
            capture_output=True, text=True)
        undefined = [line for line in result.stdout.splitlines()
                     if "undefined name" in line]
        assert not undefined, (
            "undefined names found:\n  " + "\n  ".join(undefined))

    def test_every_module_compiles(self):
        """Catches a truncated or half-written file, which a stale drop
        produces more often than a syntax error does."""
        broken = []
        for path in _modules():
            try:
                ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as e:
                broken.append(f"{path.name}: {e}")
        assert not broken, "\n".join(broken)


class TestCollaboratorsHaveWhatIsCalledOnThem:
    """`service.py` called `generator.cache_report()` when the installed
    generator had no such method. Fifty-four tests failed with the same
    AttributeError, all of them downstream of one missing definition."""

    def test_the_generator_has_what_the_service_calls(self):
        from app.reasoning.generator import CorridorGenerator
        from app.web import service

        source = inspect.getsource(service)
        called = {node.func.attr
                  for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name)
                  and node.func.value.id == "generator"}
        missing = sorted(m for m in called
                         if not hasattr(CorridorGenerator, m))
        assert not missing, (
            f"service calls these on the generator, which does not have "
            f"them: {missing}")

    def test_the_timings_sink_has_what_the_clients_call(self):
        from app.runs import Timings
        for attr in ("track", "track_embedding"):
            assert hasattr(Timings, attr), f"Timings is missing {attr}"

    def test_the_run_record_accepts_every_field_the_service_writes(self):
        """A renamed column turns into a silent TypeError inside a handler
        that swallows failures, so the row simply never appears."""
        from app.runs import RunRecord, from_payload
        record = from_payload({"request": {}, "outcome": {}, "corridors": []},
                              "req")
        assert isinstance(record, RunRecord)


class TestNoFieldIsDeclaredAndNeverFilled:
    """The generalisation of the three specific defects below.

    A dataclass field with a default is valid when nothing ever assigns it,
    so a field added in anticipation of a producer that never arrives sits
    at its default forever and no test objects. This has happened three
    times: `client_ip`, `observed_worst_at`, and the timing metrics.

    Rather than testing each one, this walks the dataclasses that carry
    reportable state and asserts that something, somewhere, assigns every
    field. A field nobody writes is either dead or waiting for a producer,
    and both are worth knowing about.
    """

    #: Fields that are legitimately never assigned by our own code, with the
    #: reason. Anything not listed here must have a writer.
    ALLOWED = {
        # set by the caller rather than by us
        "Evidence.agreement",
    }

    def _assigned_names(self) -> set[str]:
        """Every name written anywhere in the application.

        A dataclass field declaration is itself an annotated assignment, so
        collecting those would count every field as written by virtue of
        existing. Declarations in a class body are skipped; what remains is
        somewhere that actually produces a value.
        """
        assigned: set[str] = set()

        def record(target):
            if isinstance(target, ast.Name):
                assigned.add(target.id)
            elif isinstance(target, ast.Attribute):
                assigned.add(target.attr)

        for path in _modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))

            declarations = {
                id(node)
                for cls in ast.walk(tree) if isinstance(cls, ast.ClassDef)
                for node in cls.body if isinstance(node, ast.AnnAssign)
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        record(target)
                elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                    if id(node) not in declarations:
                        record(node.target)
                elif isinstance(node, ast.keyword) and node.arg:
                    assigned.add(node.arg)
        return assigned

    def test_every_evidence_field_has_a_writer(self):
        from app.reasoning.critic import Evidence

        assigned = self._assigned_names()
        orphans = sorted(
            name for name in Evidence.__dataclass_fields__
            if name not in assigned and f"Evidence.{name}" not in self.ALLOWED)
        assert not orphans, (
            f"these Evidence fields are declared and never assigned "
            f"anywhere, so they hold their default forever: {orphans}")

    def test_every_run_record_field_has_a_writer(self):
        from app.runs import RunRecord

        assigned = self._assigned_names()
        orphans = sorted(
            name for name in RunRecord.__dataclass_fields__
            if name not in assigned
            and f"RunRecord.{name}" not in self.ALLOWED)
        assert not orphans, (
            f"these RunRecord columns are never written, so the status page "
            f"will chart a permanent zero: {orphans}")


class TestFieldsThatAreReadAreAlsoWritten:
    """`client_ip` existed on the request, was read by the country lookup,
    and was never populated from the incoming request. The page reported
    "the GeoIP database may not be readable", which was not the problem."""

    def test_the_client_ip_reaches_the_service(self, monkeypatch):
        from fastapi.testclient import TestClient

        import app.web.api as api

        seen = {}

        def capture(req, api_key, db_path):
            seen["client_ip"] = req.client_ip
            return {"request": {}, "outcome": {}, "corridors": [],
                    "narration": [], "notes": [], "generator_notes": []}

        monkeypatch.setattr(api, "run_corridor_search", capture)
        TestClient(api.app).post(
            "/api/search/corridors",
            json={"origin": "KPIT", "dest": "KBOS", "use_fixtures": True},
            headers={"X-Forwarded-For": "203.0.113.9"})
        assert seen.get("client_ip") == "203.0.113.9", (
            "the forwarded address never reached the service, so no country "
            "can ever be resolved")

    def test_the_explainer_returns_token_counts_when_the_model_answers(self):
        """These come back on every API response and were being discarded
        with the rest of the object, leaving spend unmeasurable."""
        from app.reasoning.explainer import Explanation, explain

        class Fake:
            last_usage = {"tokens_in": 412, "tokens_out": 96}

            def complete(self, system, user):
                return (
                    "A turbulence forecast covers the route you are flying "
                    "and it calls for moderate conditions at cruise. No "
                    "pilot has reported what the air was actually like, so "
                    "the forecast is the only source here. It describes a "
                    "broad area over several hours rather than a "
                    "measurement.")

        payload = {
            "request": {"origin": "KPIT", "dest": "KBOS"},
            "outcome": {"reading": "moderate", "turbulence": {
                "reading": "moderate",
                "observed": {"reading": "unresolved", "count": 0},
                "forecast": {"reading": "moderate", "count": 1},
                "disagree": False, "summary": "fallback"}},
            "corridors": [],
        }
        out = explain(payload, client=Fake())
        assert isinstance(out, Explanation)
        assert out.source == "model"
        assert out.tokens_in == 412 and out.tokens_out == 96


class TestMetricsAreActuallyFed:
    """A metric defined, charted and never populated reads as a quiet zero.
    `track_embedding` existed on the timing sink, had a row on the status
    page, and nothing in the codebase ever called it."""

    def test_something_calls_track_embedding(self):
        called_in = [p.name for p in _modules()
                     if "track_embedding" in p.read_text(encoding="utf-8")
                     and p.name != "runs.py"]
        assert called_in, (
            "nothing outside runs.py calls track_embedding, so the embedding "
            "CPU figure will always be zero")

    def test_something_sets_retrieval_seconds(self):
        setters = [p.name for p in _modules()
                   if "retrieval_seconds" in p.read_text(encoding="utf-8")
                   and p.name != "runs.py"]
        assert setters, "retrieval_seconds is never measured"

    def test_the_clients_accept_a_timing_sink(self):
        """Without this the two largest bars on the chart stay at zero and
        everything unmeasured lands in the remainder bucket, which once made
        the page claim the deterministic core was its slowest part."""
        from app.sources.aeroapi import AeroAPIClient
        from app.sources.gairmet import GairmetClient
        for cls in (AeroAPIClient, GairmetClient):
            assert "timings" in cls.__dataclass_fields__, \
                f"{cls.__name__} cannot report where its time went"

    def test_the_service_hands_the_sink_to_the_clients(self):
        from app.web import service
        source = inspect.getsource(service)
        assert "client.timings = timings" in source
        assert "gairmet_client.timings = timings" in source


class TestTheRouteTableIsWhatWeThinkItIs:
    """`/api/status` and `/status` were written, then a stale copy of the
    file shipped without them. The page returned 404 and the cause was two
    layers away from the symptom."""

    EXPECTED = {
        "/",
        "/api/health",
        "/api/fixes",
        "/api/search/corridors",
        "/api/search/reputation",
        "/api/status",
        "/status",
    }

    def test_every_expected_route_is_registered(self):
        from app.web.api import app
        registered = {r.path for r in app.routes if hasattr(r, "path")}
        missing = sorted(self.EXPECTED - registered)
        assert not missing, f"routes missing from the app: {missing}"

    def test_the_status_page_file_is_installed(self):
        """The route can exist while the file it serves does not, which
        returns the same 404 for a different reason."""
        from app.web.api import STATIC_DIR
        assert (STATIC_DIR / "status.html").exists(), (
            "the status route is registered but its page is not installed")

    def test_the_index_page_is_installed(self):
        from app.web.api import STATIC_DIR
        assert (STATIC_DIR / "index.html").exists()


#: The conftest guard raises this when a test reaches for the network. The
#: reputation path loads a sentence transformer, which checks for the model
#: online, so it cannot run in an offline test environment. Recognised
#: rather than worked around: the point of the guard is that tests do not
#: silently make live calls.
_NETWORK_BLOCKED = "attempted a live network call"


def _skip_if_the_network_guard_fired(response_text: str):
    """Skip when the failure is the offline guard rather than the code.

    Matched narrowly on purpose. A 500 from any other cause is exactly what
    this test exists to catch - the reputation path broke in production on
    an unbound name and returned a 500 that looked just like this one.
    """
    if _NETWORK_BLOCKED in response_text:
        pytest.skip("the reputation path loads an embedding model, which "
                    "the offline test guard blocks; exercised against a "
                    "running instance by scripts/validate_turbulence.py")


def _require_retrieval_index():
    """Skip when the NTSB index is not on this machine.

    The index is 20 MB and is deployed rather than committed, so a developer
    checkout may not have it. Its absence must not be reported as a broken
    endpoint, and its presence must not let a broken endpoint pass.
    """
    import os
    import sqlite3

    from app.web.api import _db_path

    path = _db_path()
    if not os.path.exists(path):
        pytest.skip(f"no retrieval index at {path}")

    # Every table the search touches, not just the first one. A partial
    # index is worse than none: it lets the test proceed and then fail for
    # a reason that has nothing to do with the code under test.
    required = {"case_aircraft", "chunks", "cases"}
    with sqlite3.connect(path) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        missing = required - tables
        if missing:
            pytest.skip(f"the retrieval index is missing {sorted(missing)}; "
                        f"run scripts/ingest_ntsb.py")
        populated = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if not populated:
        pytest.skip("the retrieval index has no chunks; "
                    "run scripts/embed_chunks.py")


class TestEveryEndpointAnswers:
    """A smoke pass. Half the value of the six defects was that nobody
    called the affected path until it was in production."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from app.web.api import app
        return TestClient(app)

    @pytest.mark.parametrize("path", [
        "/api/health", "/api/fixes", "/api/status", "/status", "/",
    ])
    def test_get_endpoints_do_not_error(self, client, path):
        response = client.get(path)
        assert response.status_code < 500, (
            f"{path} returned {response.status_code}: {response.text[:160]}")

    def test_a_corridor_search_completes(self, client):
        """A 400 saying the fixtures are not installed is a correct answer
        on a machine without them; a 500 is never correct."""
        response = client.post(
            "/api/search/corridors",
            json={"origin": "KPIT", "dest": "KBOS", "use_fixtures": True})
        if response.status_code == 400 and "probe" in response.text:
            pytest.skip("no captured fixtures on this machine")
        assert response.status_code == 200, response.text

    def test_a_reputation_search_completes(self, client):
        """The path that broke in production: an unbound name inside the
        function that does the encoding.

        Skipped where the retrieval index is absent, since a missing table
        is an environment fact rather than a defect. Where the index is
        present - which is everywhere the agent actually runs - a 500 here
        is the exact failure this test exists to catch.
        """
        _require_retrieval_index()

        # The response carries a reference rather than a cause, which is the
        # remediation for I1 working as designed. So the cause is read from
        # the log, which is where it was deliberately put.
        import io

        from app.logging_setup import configure

        buf = io.StringIO()
        configure(level="DEBUG", use_syslog=False, stream=buf, force=True)
        response = client.get("/api/search/reputation",
                              params={"aircraft_type": "737-800",
                                      "query": "turbulence injuries"})
        if response.status_code >= 500:
            _skip_if_the_network_guard_fired(buf.getvalue())
        assert response.status_code < 500, (
            f"{response.text[:120]} / {buf.getvalue()[-300:]}")

    def test_the_reputation_path_inside_a_search(self, client):
        """Reached through a different caller than the endpoint above, which
        is why one of them worked while the other did not."""
        response = client.post(
            "/api/search/corridors",
            json={"origin": "KPIT", "dest": "KBOS", "use_fixtures": True,
                  "include_reputation": True})
        if response.status_code == 400 and "probe" in response.text:
            pytest.skip("no captured fixtures on this machine")
        assert response.status_code == 200, response.text
        reputation = response.json().get("reputation") or {}
        if not reputation.get("available"):
            reason = str(reputation.get("reason", ""))
            _skip_if_the_network_guard_fired(reason)
            assert "Error" not in reason and "error" not in reason, (
                f"the safety record failed rather than being unavailable: "
                f"{reason}")


class TestOriginResolution:
    """Country answers nothing when every other country is blocked at the
    edge. Region and network are the questions the panel is actually
    asking, and the address itself is still never stored."""

    def test_an_unknown_address_resolves_to_nothing(self):
        from app.runs import UNKNOWN_ORIGIN, resolve_origin
        assert resolve_origin(None) == UNKNOWN_ORIGIN
        assert resolve_origin("") == UNKNOWN_ORIGIN
        assert resolve_origin("not-an-ip") == UNKNOWN_ORIGIN

    def test_the_first_forwarded_hop_is_used(self):
        """X-Forwarded-For accumulates; the client is the first entry."""
        from app.runs import resolve_origin
        assert resolve_origin("203.0.113.9, 10.0.0.1") == \
            resolve_origin("203.0.113.9")

    def test_nothing_returned_resembles_an_address(self):
        from app.runs import resolve_origin
        for value in resolve_origin("203.0.113.9"):
            if value is None:
                continue
            assert not re.match(r"^\d+\.\d+\.\d+\.\d+$", str(value))

    def test_the_record_carries_region_and_network(self):
        from app.runs import RunRecord
        fields = set(RunRecord.__dataclass_fields__)
        assert {"region", "asn_number", "asn_name"} <= fields

    def test_the_record_still_has_no_address_column(self):
        """The rule the threat model settled on, unchanged by this."""
        from app.runs import RunRecord
        fields = set(RunRecord.__dataclass_fields__)
        for forbidden in ("ip", "client_ip", "remote_addr", "address",
                          "city", "latitude", "longitude", "postal"):
            assert forbidden not in fields

    def test_no_city_is_ever_resolved(self):
        """Region, not city: on the free database a city is often the
        registrant's address, and city plus a timestamp identifies a person
        on a site this quiet."""
        import inspect

        from app import runs
        source = inspect.getsource(runs.resolve_origin)
        assert '"city"' not in source
        assert "subdivisions" in source

    def test_the_summary_reports_regions_and_networks(self):
        import sqlite3

        from app.runs import init_runs, summary
        conn = sqlite3.connect(":memory:")
        init_runs(conn)
        data = summary(conn)
        assert "regions" in data
        assert "networks" in data
        assert "countries" not in data


class TestBlockedReportClassification:
    """Blocked requests never reach the application, so the only thing
    known about them is what Caddy logged. The user agent is the most
    honest signal of intent, and a scanner usually says so."""

    @pytest.mark.parametrize("agent,expected", [
        ("Mozilla/5.0 (compatible; Nmap Scripting Engine; https://nmap.org)",
         "security scanner"),
        ("masscan/1.3 (https://github.com/robertdavidgraham/masscan)",
         "security scanner"),
        ("Mozilla/5.0 (compatible; CensysInspect/1.1; +https://censys.io/)",
         "security scanner"),
        ("python-requests/2.31.0", "library default"),
        ("curl/8.5.0", "library default"),
        ("Go-http-client/1.1", "library default"),
        ("Mozilla/5.0 (compatible; GPTBot/1.2; +https://openai.com/gptbot)",
         "AI crawler"),
        ("Mozilla/5.0 (compatible; Googlebot/2.1)", "search crawler"),
        ("Mozilla/5.0 (Windows NT 10.0) Chrome/126.0 Safari/537.36",
         "browser"),
        ("", "no user agent"),
        ("   ", "no user agent"),
        ("SomethingUnrecognised/1.0", "other"),
    ])
    def test_agents_are_classified(self, agent, expected):
        import importlib.util
        from pathlib import Path

        path = (Path(__file__).resolve().parents[1] / "scripts"
                / "blocked_report.py")
        if not path.exists():
            pytest.skip("blocked_report.py not installed")
        spec = importlib.util.spec_from_file_location("blocked_report", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.classify(agent) == expected

    def test_a_scanner_pretending_to_be_a_browser_is_still_caught(self):
        """Every scanner in the wild prefixes Mozilla/5.0. Matching the
        browser pattern first would classify all of them as people."""
        import importlib.util
        from pathlib import Path

        path = (Path(__file__).resolve().parents[1] / "scripts"
                / "blocked_report.py")
        if not path.exists():
            pytest.skip("blocked_report.py not installed")
        spec = importlib.util.spec_from_file_location("blocked_report", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.classify(
            "Mozilla/5.0 (compatible; Nmap Scripting Engine)") \
            == "security scanner"


class TestEdgeCountsReportWhyTheyAreEmpty:
    """A permission failure reading the Caddy log used to return an empty
    result, which the page rendered as "not wired up yet". That is the same
    mistake as reading missing weather data as calm air, in a different
    place: a failure indistinguishable from an absence."""

    def _blocks(self, monkeypatch, path):
        monkeypatch.setenv("TURBULENCE_CADDY_LOG", path)
        from app.web.api import _edge_blocks
        return _edge_blocks()

    def test_a_missing_log_says_so(self, monkeypatch):
        result = self._blocks(monkeypatch, "/nonexistent/access.log")
        assert result["readable"] is False
        assert "no access log" in result["note"]

    def test_a_console_format_log_says_so(self, monkeypatch, tmp_path):
        """The format Caddy ships with drops the fields this needs."""
        log = tmp_path / "access.log"
        log.write_text("2026/08/22 22:50:01 INFO handled request\n" * 20)
        result = self._blocks(monkeypatch, str(log))
        assert result["readable"] is False
        assert "console format" in result["note"]

    def test_a_json_log_is_counted(self, monkeypatch, tmp_path):
        import json

        log = tmp_path / "access.log"
        log.write_text("\n".join(
            json.dumps({"status": 403,
                        "request": {"remote_ip": "45.155.205.7"}})
            for _ in range(7)) + "\n")
        result = self._blocks(monkeypatch, str(log))
        assert result["readable"] is True
        assert result["note"] is None
        assert {"reason": "geo filter", "n": 7} in result["counts"]

    def test_rate_limited_requests_are_counted_separately(
            self, monkeypatch, tmp_path):
        import json

        log = tmp_path / "access.log"
        log.write_text("\n".join([
            json.dumps({"status": 403, "request": {"remote_ip": "1.2.3.4"}}),
            json.dumps({"status": 429, "request": {"remote_ip": "1.2.3.4"}}),
            json.dumps({"status": 200, "request": {"remote_ip": "1.2.3.4"}}),
        ]) + "\n")
        result = self._blocks(monkeypatch, str(log))
        reasons = {r["reason"]: r["n"] for r in result["counts"]}
        assert reasons == {"geo filter": 1, "rate limit": 1}

    def test_a_successful_request_is_never_counted_as_blocked(
            self, monkeypatch, tmp_path):
        import json

        log = tmp_path / "access.log"
        log.write_text("\n".join(
            json.dumps({"status": 200, "request": {"remote_ip": "1.2.3.4"}})
            for _ in range(5)) + "\n")
        result = self._blocks(monkeypatch, str(log))
        assert result["counts"] == []

    def test_the_status_endpoint_carries_the_note(self, monkeypatch):
        from fastapi.testclient import TestClient

        from app.web.api import app
        monkeypatch.setenv("TURBULENCE_CADDY_LOG", "/nonexistent/x.log")
        body = TestClient(app).get("/api/status")
        if body.status_code != 200:
            pytest.skip("status endpoint unavailable in this environment")
        data = body.json()
        assert "blocked_note" in data
        assert "blocked_countries" in data


class TestCountryNamesRatherThanCodes:
    """The databases carry the country's name beside its code, so nothing
    needs a hand-maintained table of 250 entries that would drift. The code
    is stored - stable and compact - and the name is displayed."""

    def test_the_origin_carries_both(self):
        from app.runs import Origin
        assert "country" in Origin._fields
        assert "country_name" in Origin._fields

    def test_an_unresolvable_address_gives_neither(self):
        from app.runs import resolve_origin
        origin = resolve_origin("not-an-ip")
        assert origin.country is None
        assert origin.country_name is None

    def test_the_name_falls_back_to_the_code(self):
        """A Country-only database has no names, so the code is better than
        nothing and better than an empty cell."""
        from app.runs import Origin
        origin = Origin("NL", "NL", None, None, None)
        assert (origin.country_name or origin.country) == "NL"

    def test_the_record_stores_the_code_not_the_name(self):
        """Codes are stable; display names are not, and a column holding
        both would be ambiguous to group by."""
        from app.runs import RunRecord
        assert "country" in RunRecord.__dataclass_fields__
        assert "country_name" not in RunRecord.__dataclass_fields__

    def test_the_edge_panel_asks_for_the_name(self):
        import inspect

        from app.web import api
        source = inspect.getsource(api._edge_blocks)
        assert "country_name" in source
