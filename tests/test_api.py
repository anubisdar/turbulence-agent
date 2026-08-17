"""Tests for the HTTP API.

Runs against fixture mode throughout, so no network and no API key. The
fixtures are the payload shapes captured from a live probe.
"""

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.web.api import app
from app.web.service import FixtureTransport, SearchRequest, ServiceError

KPIT = (40.4914167, -80.2326944)
KBOS = (42.3629, -71.0064)


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory):
    """A captured-probe directory, written once for the module."""
    d = tmp_path_factory.mktemp("aeroapi_probe")

    (d / "airport_pair_flights.json").write_text(json.dumps({"flights": [
        {"segments": [{
            "ident": "JBU1286", "fa_flight_id": "JBU1286-1786169312-airline-551p",
            "status": "Arrived / Gate Arrival", "aircraft_type": "BCS3",
            "actual_off": "2026-08-10T12:52:35Z",
            "route": "EWC WOMBT TOSTR PONCT JFUND2",
            "route_distance": 495, "filed_altitude": 350,
            "origin": {"code": "KPIT"}, "destination": {"code": "KBOS"}}]},
        {"segments": [{
            "ident": "RPA5678", "fa_flight_id": "RPA5678-1786256240-airline-137p",
            "status": "Arrived / Delayed", "aircraft_type": "E75S",
            "actual_off": "2026-08-09T10:00:00Z",
            "route": "TYROO PSB J49 HNK PONCT JFUND2",
            "route_distance": 495, "filed_altitude": 310,
            "origin": {"code": "KPIT"}, "destination": {"code": "KBOS"}}]},
    ]}))

    (d / "flight_route.json").write_text(json.dumps({"fixes": [
        {"name": "KPIT", "latitude": 40.4914167, "longitude": -80.2326944,
         "type": "Origin Airport"},
        {"name": "EWC", "latitude": 40.7997, "longitude": -80.2144,
         "type": "VOR-DME (NAVAID)"},
        {"name": "WOMBT", "latitude": 41.0333, "longitude": -78.9,
         "type": "Waypoint"},
        {"name": "TOSTR", "latitude": 41.6, "longitude": -76.5,
         "type": "Waypoint"},
        {"name": "PONCT", "latitude": 42.2, "longitude": -72.9,
         "type": "Reporting Point"},
        {"name": "KBOS", "latitude": 42.3629, "longitude": -71.0064,
         "type": "Destination Airport"}]}))

    (d / "flight_route_RPA5678.json").write_text(json.dumps({"fixes": [
        {"name": "KPIT", "latitude": 40.4914167, "longitude": -80.2326944,
         "type": "Origin Airport"},
        {"name": "TYROO", "latitude": 40.62, "longitude": -79.55,
         "type": "Waypoint"},
        {"name": "PSB", "latitude": 40.9163, "longitude": -77.9927,
         "type": "VOR-TAC (NAVAID)"},
        {"name": "HNK", "latitude": 42.0619, "longitude": -75.9694,
         "type": "VOR-DME (NAVAID)"},
        {"name": "PONCT", "latitude": 42.2, "longitude": -72.9,
         "type": "Reporting Point"},
        {"name": "KBOS", "latitude": 42.3629, "longitude": -71.0064,
         "type": "Destination Airport"}]}))

    positions = []
    for i in range(154):
        f = i / 153.0
        positions.append({
            "altitude": max(1, int(350 * min(1.0, f * 4, (1 - f) * 4 + 0.15))),
            "groundspeed": 470,
            "latitude": KPIT[0] + (KBOS[0] - KPIT[0]) * f + 0.35 * math.sin(f * math.pi),
            "longitude": KPIT[1] + (KBOS[1] - KPIT[1]) * f,
            "timestamp": f"2026-08-10T13:{i % 60:02d}:00Z", "update_type": "A"})
    (d / "flight_track.json").write_text(json.dumps({"positions": positions}))

    (d / "airport_routes.json").write_text(json.dumps({"routes": [
        {"route": "EWC WOMBT TOSTR PONCT JFUND2", "count": 48,
         "filed_altitude_min": 310, "filed_altitude_max": 390,
         "route_distance": "557 sm"},
        {"route": "TYROO PSB J49 HNK PONCT JFUND2", "count": 39,
         "filed_altitude_min": 250, "filed_altitude_max": 450,
         "route_distance": "550 sm"}]}))
    return d


@pytest.fixture
def client(fixture_dir, tmp_path, monkeypatch):
    """A client whose fixtures and database are both temporary."""
    monkeypatch.setattr("app.web.service.FIXTURE_DIR", fixture_dir)
    monkeypatch.setattr(
        "app.web.service.FixtureTransport.__init__",
        lambda self, directory=fixture_dir: (
            setattr(self, "directory", Path(directory)),
            setattr(self, "calls", []))[0] or None)
    monkeypatch.setenv("TURBULENCE_DB", str(tmp_path / "test.db"))
    monkeypatch.delenv("AEROAPI_KEY", raising=False)
    return TestClient(app)


def do_search(client, **kw):
    body = {"origin": "KPIT", "dest": "KBOS", "use_fixtures": True}
    body.update(kw)
    r = client.post("/api/search/corridors", json=body)
    assert r.status_code == 200, r.text
    return r.json()


class TestHealth:
    def test_reports_what_is_configured(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert "aeroapi_key_configured" in body
        assert "fixtures_available" in body


class TestCorridorSearch:
    def test_all_four_sources_appear(self, client):
        ids = {c["id"] for c in do_search(client)["corridors"]}
        assert {"track", "filed", "alternate", "gc"} <= ids

    def test_every_corridor_carries_a_decision_and_a_score(self, client):
        for c in do_search(client)["corridors"]:
            assert c["decision"]
            assert isinstance(c["score"], float)
            assert set(c["components"]) == {
                "provenance", "geometry", "agreement", "coverage"}

    def test_pruned_corridors_carry_their_reason(self, client):
        pruned = [c for c in do_search(client)["corridors"] if not c["kept"]]
        assert pruned
        assert all(c["reason"] for c in pruned)

    def test_the_winner_is_flagged(self, client):
        data = do_search(client)
        winners = [c for c in data["corridors"] if c["is_winner"]]
        assert len(winners) == 1
        assert winners[0]["id"] == data["outcome"]["winner"]

    def test_the_trace_and_notes_come_through(self, client):
        data = do_search(client)
        assert data["trace"]
        assert data["generator_notes"]

    def test_reading_is_unresolved_without_weather(self, client):
        """Evidence is not attached yet, so this must not read as smooth."""
        data = do_search(client)
        assert data["outcome"]["reading"] == "unresolved"
        assert any("not smooth" in n.lower() for n in data["notes"])

    def test_the_call_budget_is_respected(self, client):
        data = do_search(client, max_tool_calls=2)
        assert data["outcome"]["calls_used"] <= 2

    def test_the_graph_controller_agrees_with_the_loop(self, client):
        plain = do_search(client, use_graph=False)
        graph = do_search(client, use_graph=True)
        assert plain["trace"] == graph["trace"]
        assert plain["outcome"]["winner"] == graph["outcome"]["winner"]


class TestGeoJson:
    def test_coordinates_are_lon_lat_not_lat_lon(self, client):
        """Reversing this puts Pennsylvania in the Indian Ocean."""
        data = do_search(client)
        for feature in data["geojson"]["features"]:
            coords = (feature["geometry"]["coordinates"][0]
                      if feature["geometry"]["type"] == "Polygon"
                      else feature["geometry"]["coordinates"])
            for lon, lat in coords:
                assert -85 < lon < -65, "longitude out of range for KPIT-KBOS"
                assert 38 < lat < 46, "latitude out of range for KPIT-KBOS"

    def test_each_corridor_yields_a_polygon_and_a_centreline(self, client):
        data = do_search(client)
        kinds = {}
        for f in data["geojson"]["features"]:
            kinds.setdefault(f["properties"]["id"], set()).add(
                f["properties"]["kind"])
        assert all(v == {"corridor", "centerline"} for v in kinds.values())

    def test_polygon_rings_are_closed(self, client):
        for f in do_search(client)["geojson"]["features"]:
            if f["geometry"]["type"] == "Polygon":
                ring = f["geometry"]["coordinates"][0]
                assert ring[0] == ring[-1]

    def test_properties_carry_what_the_map_needs_to_style(self, client):
        for f in do_search(client)["geojson"]["features"]:
            p = f["properties"]
            for key in ("id", "decision", "score", "kept", "is_winner",
                        "provenance", "kind"):
                assert key in p


class TestOverlaps:
    def test_pairs_are_reported_for_depth_one_corridors(self, client):
        overlaps = do_search(client)["overlaps"]
        assert overlaps
        assert all("/" not in o["a"] and "/" not in o["b"] for o in overlaps)

    def test_sorted_by_fraction(self, client):
        fractions = [o["fraction"] for o in do_search(client)["overlaps"]]
        assert fractions == sorted(fractions, reverse=True)

    def test_dominance_range_is_flagged(self, client):
        for o in do_search(client)["overlaps"]:
            assert o["dominance_range"] == (o["fraction"] >= 0.80)


class TestFixCache:
    def test_the_cache_grows_during_a_search(self, client):
        data = do_search(client)
        assert data["fix_cache"]["after"] > data["fix_cache"]["before"]

    def test_the_fixes_endpoint_reports_the_cache(self, client):
        do_search(client)
        body = client.get("/api/fixes").json()
        assert body["total"] > 0


class TestValidation:
    @pytest.mark.parametrize("body", [
        {"origin": "K", "dest": "KBOS"},
        {"origin": "KPIT", "dest": "KBOS", "beam_width": 0},
        {"origin": "KPIT", "dest": "KBOS", "max_tool_calls": 999},
        {"origin": "KPIT", "dest": "KBOS", "width_nm": 0.5},
    ])
    def test_bad_input_is_rejected(self, client, body):
        body["use_fixtures"] = True
        assert client.post("/api/search/corridors", json=body).status_code == 422

    def test_live_mode_without_a_key_fails_clearly(self, client):
        r = client.post("/api/search/corridors",
                        json={"origin": "KPIT", "dest": "KBOS",
                              "use_fixtures": False})
        assert r.status_code == 400
        assert "AEROAPI_KEY" in r.json()["detail"]


class TestFixtureTransport:
    def test_per_flight_routes_win_over_the_default(self, fixture_dir):
        t = FixtureTransport(fixture_dir)
        _, donor, _ = t("/flights/RPA5678-1786256240-airline-137p/route", {})
        names = {f["name"] for f in donor["fixes"]}
        assert "TYROO" in names

    def test_the_default_route_serves_other_flights(self, fixture_dir):
        t = FixtureTransport(fixture_dir)
        _, body, _ = t("/flights/JBU1286-1786169312-airline-551p/route", {})
        names = {f["name"] for f in body["fixes"]}
        assert "WOMBT" in names

    def test_an_unknown_path_is_not_invented(self, fixture_dir):
        status, body, _ = FixtureTransport(fixture_dir)("/nonsense", {})
        assert status == 404
        assert body is None

    def test_availability_is_reported(self, fixture_dir, tmp_path):
        assert FixtureTransport(fixture_dir).available
        assert not FixtureTransport(tmp_path / "empty").available


class TestStaticPage:
    def test_the_page_is_served_when_built(self, client):
        r = client.get("/")
        if r.status_code == 404:
            pytest.skip("static page not installed")
        assert "text/html" in r.headers["content-type"]
        assert "leaflet" in r.text.lower()

    def test_score_weights_in_the_page_match_the_critic(self, client):
        """The decomposition bar sizes segments by criterion weight. If the
        page and the critic disagree, the graphic lies about the scoring."""
        r = client.get("/")
        if r.status_code == 404:
            pytest.skip("static page not installed")
        import re
        from app.reasoning.critic import WEIGHTS
        for name, weight in WEIGHTS.items():
            # the page aligns these declarations, so allow any run of spaces
            pattern = rf"\['{name}',\s*{weight:.2f}\s*,"
            assert re.search(pattern, r.text), f"{name} {weight}"


class TestResponseContract:
    """Fields the page reads. Renaming one silently empties part of the UI."""

    CORRIDOR_FIELDS = ["id", "kept", "score", "components", "is_winner",
                       "provenance", "length_nm", "altitude_min_ft",
                       "altitude_max_ft", "reason", "decision"]
    OUTCOME_FIELDS = ["reading", "nodes_generated", "calls_used",
                      "depth_reached", "truncated", "stop", "elapsed_seconds"]

    def test_corridor_fields(self, client):
        for c in do_search(client)["corridors"]:
            for field in self.CORRIDOR_FIELDS:
                assert field in c, field

    def test_outcome_fields(self, client):
        outcome = do_search(client)["outcome"]
        for field in self.OUTCOME_FIELDS:
            assert field in outcome, field

    def test_request_echo_fields(self, client):
        req = do_search(client)["request"]
        for field in ("origin", "dest", "source", "controller"):
            assert field in req, field

    def test_feature_properties(self, client):
        for f in do_search(client)["geojson"]["features"]:
            for field in ("id", "kind", "kept", "is_winner", "score"):
                assert field in f["properties"], field


class TestDepartureTime:
    """A 07:00 and a 19:00 departure fly different air, so the reference
    flight is chosen by time of day rather than by recency."""

    def test_the_time_is_echoed_back(self, client):
        data = do_search(client, departure_date="2026-08-14",
                         departure_time="07:30")
        assert data["request"]["departure_time"] == "07:30"
        assert data["request"]["departure_date"] == "2026-08-14"

    def test_selection_by_time_is_reported(self, client):
        data = do_search(client, departure_time="10:00")
        assert any("nearest 10:00" in n for n in data["generator_notes"])

    @pytest.mark.parametrize("bad", ["7:30", "0730", "25:00", "noon"])
    def test_malformed_times_are_rejected(self, client, bad):
        r = client.post("/api/search/corridors",
                        json={"origin": "KPIT", "dest": "KBOS",
                              "use_fixtures": True, "departure_time": bad})
        assert r.status_code == 422

    def test_no_time_still_works(self, client):
        assert do_search(client)["request"]["departure_time"] is None


class TestAircraftBridge:
    """AeroAPI reports ICAO designators; NTSB files make/model strings."""

    def test_the_reference_aircraft_is_resolved(self, client):
        craft = do_search(client)["aircraft"]
        assert craft["icao_designator"] == "BCS3"
        assert craft["variant"] == "A220-300"
        assert craft["resolved"] is True

    def test_an_unknown_designator_is_not_guessed(self):
        from app.retrieval.aircraft_types import resolve_icao
        assert not resolve_icao("ZZZZ").usable

    def test_max_and_next_generation_do_not_collide(self):
        from app.retrieval.aircraft_types import resolve_icao
        assert resolve_icao("B38M").variant == "737 MAX 8"
        assert resolve_icao("B738").variant == "737-800"

    def test_every_mapped_designator_resolves(self):
        from app.retrieval.aircraft_types import ICAO_DESIGNATORS, resolve_icao
        unresolved = [d for d in ICAO_DESIGNATORS if not resolve_icao(d).usable]
        assert unresolved == []


class TestReputationToggle:
    def test_off_by_default(self, client):
        assert do_search(client)["reputation"] is None

    def test_on_returns_a_verdict_or_a_stated_reason(self, client):
        rep = do_search(client, include_reputation=True)["reputation"]
        assert rep is not None
        assert "available" in rep
        if not rep["available"]:
            assert rep["reason"], "an unavailable record must say why"

    def test_an_unresolvable_type_states_the_absence(self):
        from app.web.service import _reputation_for
        rep = _reputation_for({"icao_designator": "ZZZZ", "resolved": False},
                              "nonexistent.db")
        assert not rep["available"]
        assert "not an absence of events" in rep["reason"]


class TestNarration:
    """The narration exists because the agent's best properties are invisible
    in its output. It is derived from the finished payload, so it can describe
    what happened but never influence it."""

    def test_beats_are_produced(self, client):
        beats = do_search(client)["narration"]
        assert len(beats) > 8

    def test_every_beat_is_shaped_for_the_panel(self, client):
        for b in do_search(client)["narration"]:
            for field in ("role", "concept", "text", "kind", "pause_ms"):
                assert field in b, field
            assert b["kind"] in ("info", "caution")
            assert b["pause_ms"] > 0

    def test_the_tot_roles_are_all_represented(self, client):
        roles = {b["role"] for b in do_search(client)["narration"]}
        assert {"Thought generator", "Critic", "Controller"} <= roles

    def test_branching_evaluation_and_pruning_are_all_narrated(self, client):
        concepts = {b["concept"] for b in do_search(client)["narration"]}
        assert {"Branching", "Evaluation", "Pruning", "Termination"} <= concepts

    def test_determinism_is_stated_explicitly(self, client):
        text = " ".join(b["text"] for b in do_search(client)["narration"])
        assert "No language model produced any number" in text

    def test_the_coverage_guardrail_is_narrated(self, client):
        """The mitigation is invisible in the output, so it has to be said."""
        beats = do_search(client)["narration"]
        text = " ".join(b["text"] for b in beats)
        assert "Coverage is never allowed to prune" in text

    def test_an_unresolved_reading_is_not_dressed_up(self, client):
        text = " ".join(b["text"] for b in do_search(client)["narration"])
        assert "Unresolved is not smooth" in text

    def test_pruned_branches_are_narrated_with_a_reason(self, client):
        data = do_search(client)
        pruned = [c for c in data["corridors"] if not c["kept"]]
        if not pruned:
            pytest.skip("nothing was pruned in this run")
        prune_beats = [b for b in data["narration"] if b["concept"] == "Pruning"]
        assert prune_beats
        assert all(b["kind"] == "caution" for b in prune_beats)

    def test_narration_is_a_pure_function_of_the_payload(self, client):
        """Calling it twice on the same payload gives the same beats."""
        from app.web.narrate import narrate
        payload = do_search(client)
        assert narrate(payload) == narrate(payload)

    def test_it_survives_a_payload_with_nothing_in_it(self):
        from app.web.narrate import narrate
        beats = narrate({"request": {}, "outcome": {}, "corridors": [],
                         "overlaps": []})
        assert isinstance(beats, list)
        assert beats


class TestTurbulenceToggle:
    """Turbulence lookup is on by default. Switching it off must leave the
    reading unresolved rather than implying calm air."""

    def test_on_by_default(self, client):
        assert do_search(client)["request"]["include_turbulence"] is True

    def test_off_is_echoed_and_explained(self, client):
        data = do_search(client, include_turbulence=False)
        assert data["request"]["include_turbulence"] is False
        assert data["outcome"]["reading"] == "unresolved"
        assert any("switched off" in n for n in data["notes"])

    def test_off_never_reports_smooth(self, client):
        data = do_search(client, include_turbulence=False)
        assert data["outcome"]["reading"] != "smooth"

    def test_the_summary_is_present_either_way(self, client):
        for flag in (True, False):
            wx = do_search(client, include_turbulence=flag)["outcome"]["turbulence"]
            assert wx is not None
            assert "available" in wx


class TestTurbulenceSummaryShape:
    """Fields the verdict strip reads. Renaming one empties part of the UI."""

    def test_the_two_sources_are_reported_separately(self, client):
        wx = do_search(client)["outcome"]["turbulence"]
        assert "observed" in wx and "forecast" in wx
        for side in ("observed", "forecast"):
            assert "reading" in wx[side]
            assert "count" in wx[side]

    def test_disagreement_is_a_flag_not_an_average(self, client):
        wx = do_search(client)["outcome"]["turbulence"]
        assert "disagree" in wx
        assert isinstance(wx["disagree"], bool)

    def test_corridors_carry_their_own_readings(self, client):
        for c in do_search(client)["corridors"]:
            if c.get("kept"):
                for field in ("reading", "observed_reading",
                              "forecast_reading", "sources_disagree"):
                    assert field in c, field


class TestNarrationCoversEvidence:
    def test_a_disagreement_is_narrated_as_a_guardrail(self):
        from app.web.narrate import narrate
        payload = {
            "request": {"origin": "KPIT", "dest": "KBOS", "beam_width": 2,
                        "depth_limit": 2, "max_tool_calls": 14},
            "outcome": {"stop": "depth_limit", "nodes_generated": 8,
                        "calls_used": 8, "depth_reached": 2,
                        "winner": "track/high", "reading": "moderate",
                        "elapsed_seconds": 3.2,
                        "turbulence": {
                            "available": True, "reading": "moderate",
                            "observed": {"reading": "light", "count": 2,
                                         "mean_age_minutes": 27.5},
                            "forecast": {"reading": "moderate", "count": 1},
                            "disagree": True, "coverage_fraction": 0.2}},
            "corridors": [], "overlaps": [],
            "fix_cache": {"before": 27, "after": 27, "by_type": {}},
        }
        text = " ".join(b["text"] for b in narrate(payload))
        assert "the two sources disagree" in text.lower()
        assert "neither source supports" in text
        cautions = [b for b in narrate(payload)
                    if b["kind"] == "caution" and "disagree" in b["text"]]
        assert cautions, "a disagreement must be flagged, not stated quietly"


class TestNarrationCoversAnEmptySearch:
    """Silence about a search that happened is the same gap as silence about
    a result. A reader must be able to tell "we looked and found nothing"
    from "we never looked"."""

    def _payload(self, **outcome):
        base = {
            "request": {"origin": "KPIT", "dest": "KBOS", "beam_width": 2,
                        "depth_limit": 2, "max_tool_calls": 8},
            "corridors": [], "overlaps": [],
            "fix_cache": {"before": 27, "after": 27, "by_type": {}},
            "outcome": {
                "stop": "depth_limit", "nodes_generated": 8, "calls_used": 8,
                "depth_reached": 2, "winner": "track/high",
                "reading": "unresolved", "elapsed_seconds": 12.5,
                "turbulence": {
                    "available": False, "reading": "unresolved",
                    "observed": {"reading": "unresolved", "count": 0},
                    "forecast": {"reading": "unresolved", "count": 0},
                    "disagree": False,
                    "summary": "This is a short hop of about 431 nautical "
                               "miles. Nothing is known either way."},
            },
        }
        base["outcome"].update(outcome)
        return base

    def test_a_search_that_found_nothing_is_still_narrated(self):
        from app.web.narrate import narrate
        beats = narrate(self._payload())
        evidence = [b for b in beats if b["concept"] == "Evidence"]
        assert evidence, "an empty search must still produce an Evidence beat"
        assert "found neither" in evidence[0]["text"]
        assert "Both sources were queried" in evidence[0]["text"]

    def test_looking_and_finding_nothing_is_flagged(self):
        from app.web.narrate import narrate
        evidence = [b for b in narrate(self._payload())
                    if b["concept"] == "Evidence"]
        assert evidence[0]["kind"] == "caution"

    def test_switched_off_is_distinguished_from_found_nothing(self):
        from app.web.narrate import narrate
        payload = self._payload()
        payload["request"]["include_turbulence"] = False
        text = " ".join(b["text"] for b in narrate(payload)
                        if b["concept"] == "Evidence")
        assert "switched off" in text
        assert "found neither" not in text

    def test_the_explanation_is_not_repeated(self):
        """The plain summary belongs on one beat, not echoed on the next."""
        from app.web.narrate import narrate
        beats = narrate(self._payload())
        shown = [b for b in beats
                 if b.get("detail") and "short hop" in str(b["detail"])]
        assert len(shown) == 1

    def test_no_empty_search_beat_ever_implies_calm(self):
        from app.web.narrate import narrate
        for off in (True, False):
            payload = self._payload()
            payload["request"]["include_turbulence"] = not off
            for b in narrate(payload):
                low = b["text"].lower()
                assert "smooth" not in low or "not smooth" in low
                assert "calm" not in low


class TestExplainerToggle:
    """The written explanation is an enhancement. Off by default, and its
    absence never costs accuracy."""

    def test_off_by_default(self, client):
        data = do_search(client)
        assert data["request"]["include_explanation"] is False
        assert data["explanation"]["enabled"] is False
        assert data["explanation"]["source"] == "deterministic"

    def test_the_deterministic_text_is_always_present(self, client):
        for flag in (True, False):
            ex = do_search(client, include_explanation=flag)["explanation"]
            assert ex["text"], "there must always be something to read"

    def test_a_missing_key_is_explained_not_silent(self, client, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        ex = do_search(client, include_explanation=True)["explanation"]
        assert ex["source"] == "deterministic"
        assert any("ANTHROPIC_API_KEY" in str(r) for r in ex["rejected"])

    def test_the_response_says_which_path_produced_the_text(self, client):
        ex = do_search(client)["explanation"]
        assert ex["source"] in ("model", "deterministic")

    def test_the_flag_is_echoed_back(self, client):
        assert do_search(client, include_explanation=True)[
            "request"]["include_explanation"] is True


class TestExplainerNarration:
    def _payload(self, explanation):
        return {
            "request": {"origin": "KIAD", "dest": "KLAX", "beam_width": 2,
                        "depth_limit": 2, "max_tool_calls": 8},
            "outcome": {"stop": "depth_limit", "nodes_generated": 8,
                        "calls_used": 8, "depth_reached": 2,
                        "winner": "track/high", "reading": "moderate",
                        "elapsed_seconds": 15.0,
                        "turbulence": {
                            "available": True, "reading": "moderate",
                            "observed": {"reading": "unresolved", "count": 0},
                            "forecast": {"reading": "moderate", "count": 1},
                            "disagree": False, "coverage_fraction": 0.0}},
            "corridors": [], "overlaps": [],
            "fix_cache": {"before": 27, "after": 27, "by_type": {}},
            "explanation": explanation,
        }

    def test_an_accepted_explanation_says_it_was_checked(self):
        from app.web.narrate import narrate
        beats = narrate(self._payload({
            "enabled": True, "source": "model", "model": "claude-sonnet-5",
            "rejected": []}))
        text = " ".join(b["text"] for b in beats)
        assert "allowed only to restate them" in text
        assert "would have been discarded" in text

    def test_a_rejected_explanation_is_shown_as_a_guardrail(self):
        """The guardrail firing is more interesting than it never firing."""
        from app.web.narrate import narrate
        beats = narrate(self._payload({
            "enabled": True, "source": "deterministic",
            "rejected": ["contains reassurance: 'should be fine'"]}))
        flagged = [b for b in beats if b["kind"] == "caution"
                   and "discarded" in b["text"]]
        assert flagged
        assert "reassurance" in str(flagged[0]["detail"])

    def test_nothing_is_narrated_when_the_toggle_is_off(self):
        from app.web.narrate import narrate
        beats = narrate(self._payload({"enabled": False,
                                       "source": "deterministic",
                                       "rejected": []}))
        assert not [b for b in beats if b["concept"] == "Explanation"]


class TestGeoJsonAcrossTheDateLine:
    """The geometry can be right and the picture still wrong. A corridor
    stepping from 179 to -179 reads to a map library as a 358 degree move,
    and it draws the line back across the whole world."""

    def _corridor(self, a, b):
        from app.reasoning.geometry import build_corridor, great_circle
        return build_corridor(great_circle(a, b, 24))

    KSEA = (47.4502, -122.3088)
    RJTT = (35.5533, 139.7811)

    def test_the_centreline_is_continuous(self):
        from app.web.service import _path_to_geojson
        lons = [c[0] for c in
                _path_to_geojson(self._corridor(self.KSEA, self.RJTT))]
        steps = [abs(b - a) for a, b in zip(lons, lons[1:])]
        assert max(steps) < 180, "a step near 360 draws the wrong way round"

    def test_the_ring_is_continuous(self):
        from app.web.service import _ring_to_geojson
        lons = [c[0] for c in
                _ring_to_geojson(self._corridor(self.KSEA, self.RJTT))]
        steps = [abs(b - a) for a, b in zip(lons, lons[1:])]
        assert max(steps) < 180

    def test_a_domestic_corridor_is_unchanged(self):
        """The fix must not move anything that already drew correctly."""
        from app.web.service import _path_to_geojson
        lons = [c[0] for c in _path_to_geojson(
            self._corridor((40.4914, -80.2327), (42.3629, -71.0064)))]
        assert all(-85 < lon < -65 for lon in lons)

    def test_latitudes_stay_in_range(self):
        """Only longitude is unwrapped; latitude has no wrap to handle."""
        from app.web.service import _path_to_geojson
        for lon, lat in _path_to_geojson(self._corridor(self.KSEA, self.RJTT)):
            assert -90 <= lat <= 90

    def test_the_ring_still_closes(self):
        from app.web.service import _ring_to_geojson
        ring = _ring_to_geojson(self._corridor(self.KSEA, self.RJTT))
        assert ring[0] == ring[-1]


class TestTimeLimit:
    """25 seconds was chosen when a warm search took about a second. Real
    AeroAPI latency runs 12 to 25 seconds, so the limit was firing on
    ordinary searches and returning a truncated corridor that read as a
    different answer to the same question."""

    def test_the_default_clears_the_observed_latency(self):
        from app.reasoning.controller import DEFAULT_MAX_SECONDS
        assert DEFAULT_MAX_SECONDS >= 45, \
            "the ceiling must sit clear of a normal slow search"

    def test_it_is_echoed_back(self, client):
        assert do_search(client)["request"]["max_seconds"] == 60.0

    def test_it_can_be_set_per_search(self, client):
        assert do_search(client, max_seconds=120.0)[
            "request"]["max_seconds"] == 120.0

    @pytest.mark.parametrize("bad", [0.0, 1.0, 600.0, -5.0])
    def test_implausible_limits_are_rejected(self, client, bad):
        r = client.post("/api/search/corridors",
                        json={"origin": "KPIT", "dest": "KBOS",
                              "use_fixtures": True, "max_seconds": bad})
        assert r.status_code == 422

    def test_a_tight_limit_still_truncates_rather_than_failing(self, client):
        """The mechanism was never wrong, only the number."""
        data = do_search(client, max_seconds=5.0)
        assert "truncated" in data["outcome"]


class TestDegradedIsReported:
    def test_the_flag_is_in_the_response(self, client):
        outcome = do_search(client)["outcome"]
        assert "degraded" in outcome
        assert "degraded_reasons" in outcome

    def test_degraded_is_a_boolean_with_reasons_when_true(self, client):
        """Not asserted false: whether a fixture search degrades depends on
        whether the weather host is reachable from the test machine, which
        is environment rather than behaviour. What must hold is that the
        flag and its reasons agree."""
        outcome = do_search(client)["outcome"]
        assert isinstance(outcome["degraded"], bool)
        if outcome["degraded"]:
            assert outcome["degraded_reasons"]
        else:
            assert outcome["degraded_reasons"] == []

    def test_it_is_distinct_from_truncated(self, client):
        """Different causes, different words. A budget stopping a search and
        a source failing during one both give a partial answer."""
        outcome = do_search(client)["outcome"]
        assert outcome["degraded"] is not None
        assert outcome["truncated"] is not None

    def test_the_narration_separates_the_two(self):
        from app.web.narrate import narrate
        payload = {
            "request": {"origin": "KPIT", "dest": "KBOS", "beam_width": 2,
                        "depth_limit": 2, "max_tool_calls": 8},
            "outcome": {"stop": "depth_limit", "nodes_generated": 1,
                        "calls_used": 3, "depth_reached": 1, "winner": "gc",
                        "reading": "light", "elapsed_seconds": 20.2,
                        "truncated": False, "degraded": True,
                        "degraded_reasons": [
                            "Could not list flights on this pair: rate "
                            "limited twice"],
                        "turbulence": {
                            "reading": "light",
                            "observed": {"reading": "light", "count": 2},
                            "forecast": {"reading": "unresolved", "count": 0},
                            "disagree": False}},
            "corridors": [], "overlaps": [],
            "fix_cache": {"before": 461, "after": 461, "by_type": {}},
        }
        beats = narrate(payload)
        degraded = [b for b in beats if "data source failed" in b["text"]]
        assert degraded
        assert "different thing from a budget running out" in degraded[0]["text"]
        assert "rate limited" in str(degraded[0]["detail"])
