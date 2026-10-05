"""Tests for the corridor generator.

No network. Payload shapes are the ones AeroAPI actually returned during the
probe, including the detail that filed route strings name enroute fixes but
not the airports.
"""

import dataclasses
import math
import sqlite3

import pytest

from app.reasoning.controller import Budget
from app.reasoning.critic import Evidence, Provenance, Severity
from app.reasoning.evidence import GatherResult
from app.reasoning.generator import (
    MAX_USEFUL_DEPTH,
    MIN_SPLIT_LENGTH_NM,
    CorridorGenerator,
    _match_flight_number,
    _split_points,
)
from app.reasoning.geometry import great_circle, path_length_nm
from app.sources.aeroapi import AeroAPIClient
from app.sources.fixes import cache_stats, init_fixes

KPIT = (40.4914167, -80.2326944)
KBOS = (42.3629, -71.0064)
ALT_ROUTE = "TYROO PSB J49 HNK PONCT JFUND2"

PAIR = {"flights": [
    {"segments": [{"ident": "JBU1286", "fa_flight_id": "JBU1286-x",
                   "status": "Arrived", "aircraft_type": "BCS3",
                   "actual_off": "2026-08-10T12:52:35Z",
                   "route": "EWC WOMBT TOSTR PONCT JFUND2",
                   "filed_altitude": 350,
                   "origin": {"code": "KPIT"},
                   "destination": {"code": "KBOS"}}]},
    {"segments": [{"ident": "RPA5678", "fa_flight_id": "RPA5678-y",
                   "status": "Arrived", "aircraft_type": "E75S",
                   "actual_off": "2026-08-09T10:00:00Z",
                   "route": ALT_ROUTE, "filed_altitude": 310,
                   "origin": {"code": "KPIT"},
                   "destination": {"code": "KBOS"}}]},
]}

ROUTE_JBU = {"fixes": [
    {"name": "KPIT", "latitude": 40.4914167, "longitude": -80.2326944,
     "type": "Origin Airport"},
    {"name": "EWC", "latitude": 40.7997, "longitude": -80.2144, "type": "VOR"},
    {"name": "WOMBT", "latitude": 41.0333, "longitude": -78.9, "type": "Fix"},
    {"name": "TOSTR", "latitude": 41.6, "longitude": -76.5, "type": "Fix"},
    {"name": "PONCT", "latitude": 42.2, "longitude": -72.9, "type": "Fix"},
    {"name": "KBOS", "latitude": 42.3629, "longitude": -71.0064,
     "type": "Destination Airport"},
]}

# The donor flight's own filed route, carrying the alternate routing's fixes.
ROUTE_RPA = {"fixes": [
    {"name": "KPIT", "latitude": 40.4914167, "longitude": -80.2326944,
     "type": "Origin Airport"},
    {"name": "TYROO", "latitude": 40.62, "longitude": -79.55, "type": "Fix"},
    {"name": "PSB", "latitude": 40.9163, "longitude": -77.9927, "type": "VOR"},
    {"name": "HNK", "latitude": 42.0619, "longitude": -75.9694, "type": "VOR"},
    {"name": "PONCT", "latitude": 42.2, "longitude": -72.9, "type": "Fix"},
    {"name": "KBOS", "latitude": 42.3629, "longitude": -71.0064,
     "type": "Destination Airport"},
]}


def _track_point(i, n=60):
    f = i / (n - 1)
    return {"latitude": KPIT[0] + (KBOS[0] - KPIT[0]) * f + 0.35 * math.sin(f * math.pi),
            "longitude": KPIT[1] + (KBOS[1] - KPIT[1]) * f,
            "altitude": 350, "timestamp": f"2026-08-10T13:{i % 60:02d}:00Z",
            "update_type": "A"}


TRACK = {"positions": [_track_point(i) for i in range(60)]}

ROUTES = {"routes": [
    {"route": "EWC WOMBT TOSTR PONCT JFUND2", "count": 48,
     "filed_altitude_min": 310, "filed_altitude_max": 390,
     "route_distance": "557 sm"},
    {"route": ALT_ROUTE, "count": 40,
     "filed_altitude_min": 250, "filed_altitude_max": 450,
     "route_distance": "550 sm"},
]}


def make_gen(overrides=None, conn=None):
    """A generator wired to canned payloads. `overrides` replaces a path."""
    routes = {
        "/flights/to/KBOS": PAIR,
        "/flights/JBU1286-x/route": ROUTE_JBU,
        "/flights/RPA5678-y/route": ROUTE_RPA,
        "/track": TRACK,
        "/routes/KBOS": ROUTES,
    }
    routes.update(overrides or {})

    def transport(path, params):
        for key, payload in routes.items():
            if key.startswith("/flights/") and path.startswith(key):
                return (200, payload, "") if payload is not None else (404, None, "")
            if path.endswith(key):
                return (200, payload, "") if payload is not None else (404, None, "")
        return 404, None, "not found"

    conn = conn or sqlite3.connect(":memory:")
    init_fixes(conn)
    client = AeroAPIClient(api_key="t", transport=transport,
                           spacing_seconds=0, sleep=lambda s: None)
    return CorridorGenerator(client=client, conn=conn,
                             origin="KPIT", dest="KBOS"), conn


class TestDepthOne:
    def test_all_four_sources_are_generated(self):
        gen, _ = make_gen()
        out = gen(None, 1, Budget(max_tool_calls=12))
        assert {c.id for c in out} == {"track", "filed", "alternate", "gc"}

    def test_provenance_is_assigned_per_source(self):
        gen, _ = make_gen()
        prov = {c.id: c.provenance for c in gen(None, 1, Budget(max_tool_calls=12))}
        assert prov["track"] is Provenance.ACTUAL_TRACK
        assert prov["filed"] is Provenance.FILED_ROUTE
        assert prov["alternate"] is Provenance.PUBLISHED_AIRWAY
        assert prov["gc"] is Provenance.GREAT_CIRCLE

    def test_evidence_is_empty_because_weather_is_a_separate_step(self):
        gen, _ = make_gen()
        for c in gen(None, 1, Budget(max_tool_calls=12)):
            assert c.evidence.coverage_fraction is None
            assert c.evidence.agreement is None

    def test_the_great_circle_survives_an_exhausted_budget(self):
        """The geometric floor needs no API call once airports are cached,
        so the search can never come back empty for want of budget."""
        gen1, conn = make_gen()
        gen1(None, 1, Budget(max_tool_calls=12))      # warms the cache

        gen2, _ = make_gen(conn=conn)
        out = gen2(None, 1, Budget(max_tool_calls=0))
        assert "gc" in {c.id for c in out}

    def test_every_corridor_is_scored_against_the_same_baseline(self):
        gen, _ = make_gen()
        out = gen(None, 1, Budget(max_tool_calls=12))
        baselines = {c.geometry.great_circle_nm for c in out}
        assert len(baselines) == 1
        assert baselines.pop() > 0


class TestAirportAnchoring:
    """A filed route names enroute fixes, not airports. `TYROO PSB J49 HNK
    PONCT` starts 35 nm from the field, so a corridor built straight from the
    string is shorter than the great circle and gets rejected as impossible."""

    def test_the_alternate_corridor_reaches_both_airports(self):
        gen, _ = make_gen()
        gen(None, 1, Budget(max_tool_calls=12))
        pts = gen.shapes["alternate"].points
        assert pts[0] == pytest.approx(KPIT, abs=1e-6)
        assert pts[-1] == pytest.approx(KBOS, abs=1e-6)

    def test_the_alternate_is_longer_than_the_great_circle(self):
        gen, _ = make_gen()
        out = {c.id: c for c in gen(None, 1, Budget(max_tool_calls=12))}
        gc_nm = out["alternate"].geometry.great_circle_nm
        assert out["alternate"].geometry.length_nm > gc_nm

    def test_without_anchoring_it_would_have_been_too_short(self):
        """Guards the regression directly: the bare fix list is not a corridor."""
        bare = [(f["latitude"], f["longitude"]) for f in ROUTE_RPA["fixes"]
                if f["name"] in ("TYROO", "PSB", "HNK", "PONCT")]
        assert path_length_nm(bare) < path_length_nm(great_circle(KPIT, KBOS))


class TestFixCacheWarming:
    def test_the_filed_route_populates_the_cache(self):
        gen, conn = make_gen()
        gen(None, 1, Budget(max_tool_calls=12))
        assert cache_stats(conn)["total"] >= 6

    def test_a_donor_flight_supplies_the_alternate_routings_fixes(self):
        gen, conn = make_gen()
        gen(None, 1, Budget(max_tool_calls=12))
        names = {r[0] for r in conn.execute("SELECT name FROM route_fixes")}
        assert {"TYROO", "PSB", "HNK"} <= names
        assert any("filed the alternate routing" in n for n in gen.notes)

    def test_a_warm_cache_needs_no_donor_call(self):
        gen1, conn = make_gen()
        gen1(None, 1, Budget(max_tool_calls=12))
        first = gen1.client.calls_made

        gen2, _ = make_gen(conn=conn)
        gen2(None, 1, Budget(max_tool_calls=12))
        assert gen2.client.calls_made < first


class TestDegradedInputs:
    def test_no_departed_flight_leaves_only_geometry_sources(self):
        scheduled_only = {"flights": [{"segments": [
            {"ident": "X", "fa_flight_id": "x-1", "actual_off": None,
             "route": "EWC PONCT", "origin": {"code": "KPIT"},
             "destination": {"code": "KBOS"}}]}]}
        gen, _ = make_gen({"/flights/to/KBOS": scheduled_only})
        out = gen(None, 1, Budget(max_tool_calls=12))
        assert "track" not in {c.id for c in out}
        assert any("not the same as their being smooth" in n for n in gen.notes)

    def test_an_empty_pair_yields_nothing_and_says_why(self):
        gen, _ = make_gen({"/flights/to/KBOS": {"flights": []},
                           "/routes/KBOS": {"routes": []}})
        out = gen(None, 1, Budget(max_tool_calls=12))
        assert out == []
        assert any("no turbulence conclusion follows" in n for n in gen.notes)

    def test_a_spent_budget_stops_generation_without_inventing_corridors(self):
        gen, _ = make_gen()
        out = gen(None, 1, Budget(max_tool_calls=1))
        assert all(c.provenance is not Provenance.ACTUAL_TRACK for c in out)
        assert any("budget exhausted" in n.lower() for n in gen.notes)

    def test_calls_never_exceed_the_budget(self):
        gen, _ = make_gen()
        budget = Budget(max_tool_calls=2)
        gen(None, 1, budget)
        assert budget.calls_used <= 2


class TestFlightNumberMatching:
    """Unit tests for the ident/flight-number matcher, no client involved."""

    def test_an_exact_ident_match(self):
        from app.sources.aeroapi import FlightSegment
        seg = FlightSegment(
            ident="JBU1286", fa_flight_id="x", aircraft_type=None,
            status=None, actual_off="2026-08-10T12:52:35Z",
            scheduled_out=None, route=None, filed_altitude_ft=None,
            reported_distance=None)
        assert _match_flight_number([seg], "JBU1286") is seg

    def test_an_iata_prefix_matches_an_icao_ident(self):
        """AeroAPI's ident carries the ICAO prefix; a passenger types the
        shorter IATA one for the same airline."""
        from app.sources.aeroapi import FlightSegment
        seg = FlightSegment(
            ident="UAL1234", fa_flight_id="x", aircraft_type=None,
            status=None, actual_off="2026-08-10T12:52:35Z",
            scheduled_out=None, route=None, filed_altitude_ft=None,
            reported_distance=None)
        assert _match_flight_number([seg], "UA1234") is seg

    def test_a_different_flight_number_does_not_match(self):
        from app.sources.aeroapi import FlightSegment
        seg = FlightSegment(
            ident="UAL1234", fa_flight_id="x", aircraft_type=None,
            status=None, actual_off="2026-08-10T12:52:35Z",
            scheduled_out=None, route=None, filed_altitude_ft=None,
            reported_distance=None)
        assert _match_flight_number([seg], "UA9999") is None

    def test_no_segments_is_no_match(self):
        assert _match_flight_number([], "UA1234") is None

    def test_a_codeshare_suffix_letter_still_matches_on_the_digits(self):
        """A flight number can carry a trailing letter (BA249A). The suffix
        is not part of what identifies the flight, so it's ignored on both
        sides rather than breaking the match."""
        from app.sources.aeroapi import FlightSegment
        seg = FlightSegment(
            ident="BAW249", fa_flight_id="x", aircraft_type=None,
            status=None, actual_off="2026-08-10T12:52:35Z",
            scheduled_out=None, route=None, filed_altitude_ft=None,
            reported_distance=None)
        assert _match_flight_number([seg], "BA249A") is seg


class TestFlightNumberPin:
    """`_get_flight` prefers a matching flight number over the
    nearest-to-target-time pick, and explains itself either way."""

    def test_a_match_pins_the_reference_flight(self):
        gen, _ = make_gen()
        gen = dataclasses.replace(gen, flight_number="JBU1286")
        flight = gen._get_flight(Budget(max_tool_calls=12))
        assert flight is not None
        assert flight.ident == "JBU1286"
        assert any("pinned to JBU1286" in n for n in gen.notes)

    def test_no_match_falls_back_to_nearest_departure_time(self):
        gen, _ = make_gen()
        gen = dataclasses.replace(gen, flight_number="DL9999",
                                  target_time="12:00")
        flight = gen._get_flight(Budget(max_tool_calls=12))
        # Falls back to _pick_reference's normal behaviour rather than
        # coming back empty because the pin missed.
        assert flight is not None
        assert flight.ident in {"JBU1286", "RPA5678"}
        assert any("No recent" in n and "DL9999" in n for n in gen.notes)

    def test_a_match_that_has_not_flown_yet_falls_back_with_its_own_note(self):
        scheduled_and_flown = {"flights": [
            {"segments": [{"ident": "JBU1286", "fa_flight_id": "x-1",
                          "actual_off": None,
                          "route": "EWC PONCT", "origin": {"code": "KPIT"},
                          "destination": {"code": "KBOS"}}]},
            {"segments": [{"ident": "RPA5678", "fa_flight_id": "RPA5678-y",
                          "actual_off": "2026-08-09T10:00:00Z",
                          "route": ALT_ROUTE, "filed_altitude": 310,
                          "origin": {"code": "KPIT"},
                          "destination": {"code": "KBOS"}}]},
        ]}
        gen, _ = make_gen({"/flights/to/KBOS": scheduled_and_flown})
        gen = dataclasses.replace(gen, flight_number="JBU1286")
        flight = gen._get_flight(Budget(max_tool_calls=12))
        assert flight is not None
        assert flight.ident == "RPA5678"
        assert any("hasn't flown yet" in n for n in gen.notes)

    def test_a_flight_missing_from_the_pair_listing_is_found_by_ident(self):
        # Regression for JBU2454 KDCA-KBOS: the pair listing held only
        # upcoming departures, so nothing in it had flown and the reference
        # flight (and its aircraft type) came back empty even though the
        # flight itself had already departed and has a type.
        upcoming_only = {"flights": [
            {"segments": [{"ident": "JBU2454", "fa_flight_id": "up-1",
                          "aircraft_type": "BCS3", "actual_off": None,
                          "origin": {"code": "KPIT"},
                          "destination": {"code": "KBOS"}}]},
        ]}
        by_ident = {"flights": [
            {"ident": "JBU2454", "fa_flight_id": "JBU2454-x",
             "aircraft_type": "BCS3",
             "actual_off": "2026-10-04T15:13:09Z",
             "origin": {"code": "KPIT"}, "destination": {"code": "KBOS"}},
            {"ident": "JBU2454", "fa_flight_id": "up-1",
             "aircraft_type": "BCS3", "actual_off": None,
             "scheduled_out": "2026-10-05T15:00:00Z",
             "origin": {"code": "KPIT"}, "destination": {"code": "KBOS"}},
        ]}
        gen, _ = make_gen({"/flights/to/KBOS": upcoming_only,
                           "/flights/JBU2454": by_ident})
        gen = dataclasses.replace(gen, flight_number="JBU2454")
        flight = gen._get_flight(Budget(max_tool_calls=12))
        assert flight is not None
        assert flight.fa_flight_id == "JBU2454-x"
        assert flight.aircraft_type == "BCS3"
        assert any("looked up directly" in n for n in gen.notes)

    def test_by_ident_result_on_another_pair_is_not_used(self):
        # A reused flight number must not lend its track to a different
        # route.
        elsewhere = {"flights": [
            {"ident": "JBU2454", "fa_flight_id": "other-1",
             "aircraft_type": "BCS3",
             "actual_off": "2026-10-04T15:13:09Z",
             "origin": {"code": "KJFK"}, "destination": {"code": "KMCO"}},
        ]}
        gen, _ = make_gen({"/flights/JBU2454": elsewhere})
        gen = dataclasses.replace(gen, flight_number="JBU2454")
        flight = gen._get_flight(Budget(max_tool_calls=12))
        assert flight is not None
        assert flight.fa_flight_id != "other-1"

    def test_by_ident_failure_falls_back_as_before(self):
        gen, _ = make_gen()  # no /flights/DL9999 payload -> 404
        gen = dataclasses.replace(gen, flight_number="DL9999")
        flight = gen._get_flight(Budget(max_tool_calls=12))
        assert flight is not None
        assert any("No recent" in n and "DL9999" in n for n in gen.notes)

    def test_no_flight_number_behaves_exactly_as_before(self):
        gen, _ = make_gen()
        flight = gen._get_flight(Budget(max_tool_calls=12))
        assert flight is not None
        assert not any("pinned" in n.lower() for n in gen.notes)


class TestDepthTwo:
    def _parent(self, gen):
        out = {c.id: c for c in gen(None, 1, Budget(max_tool_calls=12))}
        return out["track"]

    def test_altitude_branches_are_produced(self):
        gen, _ = make_gen()
        children = gen(self._parent(gen), 2, Budget(max_tool_calls=12))
        assert len(children) == 2

    def test_branches_differ_only_in_altitude(self):
        gen, _ = make_gen()
        parent = self._parent(gen)
        children = gen(parent, 2, Budget(max_tool_calls=12))
        bands = {(gen.shapes[c.id].altitude_min_ft,
                  gen.shapes[c.id].altitude_max_ft) for c in children}
        assert len(bands) == 2
        for c in children:
            assert gen.shapes[c.id].points == gen.shapes[parent.id].points

    def test_branch_source_is_recorded(self):
        gen, _ = make_gen()
        gen(self._parent(gen), 2, Budget(max_tool_calls=12))
        assert any("Altitude branches" in n for n in gen.notes)

    def test_children_carry_the_parents_provenance(self):
        gen, _ = make_gen()
        parent = self._parent(gen)
        for c in gen(parent, 2, Budget(max_tool_calls=12)):
            assert c.provenance is parent.provenance
            assert c.parent_id == parent.id

    def test_no_altitude_information_means_no_branch(self):
        gen, _ = make_gen({"/routes/KBOS": {"routes": []}})
        out = {c.id: c for c in gen(None, 1, Budget(max_tool_calls=12))}
        assert gen(out["gc"], 2, Budget(max_tool_calls=12)) == []

    def test_nothing_beyond_the_useful_depth(self):
        gen, _ = make_gen()
        parent = self._parent(gen)
        assert gen(parent, MAX_USEFUL_DEPTH + 1, Budget(max_tool_calls=12)) == []


class TestOverlapWiring:
    def test_shapes_are_retained_for_every_corridor(self):
        gen, _ = make_gen()
        out = gen(None, 1, Budget(max_tool_calls=12))
        for c in out:
            assert c.id in gen.shapes

    def test_the_overlap_fn_compares_real_geometry(self):
        gen, _ = make_gen()
        out = {c.id: c for c in gen(None, 1, Budget(max_tool_calls=12))}
        fn = gen.overlap_fn
        assert fn(out["gc"], out["gc"]) == pytest.approx(1.0, abs=0.01)
        assert 0.0 <= fn(out["gc"], out["alternate"]) <= 1.0


class TestCruiseBand:
    """A flown track runs from the ground up. Banding a corridor from every
    position gives a floor below zero, which would match low-level advisories
    that have nothing to do with the cruise segment."""

    PROFILE = [-100, 1000, 5000, 12000, 24000, 33000, 35000, 35000,
               34000, 20000, 3000, 0]

    def test_ground_positions_are_excluded(self):
        from app.reasoning.generator import cruise_band
        assert cruise_band(self.PROFILE) == (33000, 35000)

    def test_the_floor_is_never_negative(self):
        from app.reasoning.generator import cruise_band
        band = cruise_band(self.PROFILE)
        assert band[0] > 0

    def test_a_track_that_never_climbed_has_no_band(self):
        from app.reasoning.generator import cruise_band
        assert cruise_band([-100, 500, 2000]) is None

    def test_no_altitudes_at_all(self):
        from app.reasoning.generator import cruise_band
        assert cruise_band([]) is None

    def test_the_corridor_uses_the_cruise_band(self):
        gen, _ = make_gen()
        gen(None, 1, Budget(max_tool_calls=12))
        shape = gen.shapes["track"]
        assert shape.altitude_min_ft is None or shape.altitude_min_ft > 0

    def test_the_exclusion_is_reported(self):
        gen, _ = make_gen()
        gen(None, 1, Budget(max_tool_calls=12))
        assert any("cruise" in n.lower() for n in gen.notes)


#: A threshold no corridor can reach, so these tests exercise depth and
#: evidence rather than early stopping.
NO_EARLY_STOP = 1.1


class TestEvidenceWiring:
    """Evidence is attached to survivors after pruning, not to every
    candidate before it."""

    def _sources(self, severity="MOD", reports=True):
        from datetime import datetime, timedelta, timezone
        from app.sources.gairmet import GairmetClient
        now = datetime(2026, 8, 16, 12, 30, tzinfo=timezone.utc)

        forecast = {
            "hazard": "TURB-HI", "severity": severity, "top": "400",
            "base": "300", "validTime": "2026-08-16T12:00:00.000Z",
            "expireTime": 1786892400,
            "coords": [{"lat": "43.5", "lon": "-79.0"},
                       {"lat": "43.5", "lon": "-72.0"},
                       {"lat": "39.5", "lon": "-72.0"},
                       {"lat": "39.5", "lon": "-79.0"}],
        }

        class PR:
            def __init__(s, lat, lon, alt, sev):
                s.latitude, s.longitude, s.altitude_ft = lat, lon, alt
                s.turbulence_severity = sev
                s.observation_time = now - timedelta(minutes=20)

        fetch = ((lambda bbox, hours: [PR(41.5, -77.0, 34000, "light")])
                 if reports else (lambda bbox, hours: []))
        return fetch, GairmetClient(
            transport=lambda p, q: (200, [forecast], "")), now

    def _gen(self, **kw):
        fetch, client, now = self._sources(**kw)
        gen, _ = make_gen()
        gen.fetch_pireps = fetch
        gen.gairmet_client = client
        gen.when = now
        return gen

    def test_a_reading_is_produced_once_sources_are_wired(self):
        from app.reasoning.controller import Budget, search
        from app.reasoning.critic import Severity
        gen = self._gen()
        res = search(gen, beam_width=2, depth_limit=1,
                     confidence_threshold=NO_EARLY_STOP,
                     budget=Budget(max_tool_calls=14),
                     overlap_fn=gen.overlap_fn,
                     enrich=gen.gather_for_survivors)
        assert res.reading is not Severity.UNRESOLVED

    def test_without_sources_the_reading_stays_unresolved(self):
        """No turbulence layer must never mean smooth air."""
        from app.reasoning.controller import Budget, search
        from app.reasoning.critic import Severity
        gen, _ = make_gen()
        res = search(gen, beam_width=2, depth_limit=1,
                     confidence_threshold=NO_EARLY_STOP,
                     budget=Budget(max_tool_calls=14),
                     overlap_fn=gen.overlap_fn,
                     enrich=gen.gather_for_survivors)
        assert res.reading is Severity.UNRESOLVED

    def test_evidence_is_only_gathered_for_survivors(self):
        """Fetching for a corridor about to be pruned spends a call on an
        answer nobody reads."""
        from app.reasoning.controller import Budget, search
        gen = self._gen()
        res = search(gen, beam_width=1, depth_limit=1,
                     confidence_threshold=NO_EARLY_STOP,
                     budget=Budget(max_tool_calls=14),
                     overlap_fn=gen.overlap_fn,
                     enrich=gen.gather_for_survivors)
        assert len(gen.evidence) <= 1
        assert set(gen.evidence) <= {c.id for c in res.survivors}

    def test_the_gather_counts_against_the_budget(self):
        from app.reasoning.controller import Budget, search
        gen = self._gen()
        budget = Budget(max_tool_calls=14)
        search(gen, beam_width=2, depth_limit=1,
               confidence_threshold=NO_EARLY_STOP, budget=budget,
               overlap_fn=gen.overlap_fn, enrich=gen.gather_for_survivors)
        assert budget.calls_used > gen.client.calls_made

    def test_an_exhausted_budget_leaves_the_reading_unresolved(self):
        from app.reasoning.controller import Budget, search
        from app.reasoning.critic import Severity
        gen = self._gen()
        res = search(gen, beam_width=2, depth_limit=1,
                     confidence_threshold=NO_EARLY_STOP,
                     budget=Budget(max_tool_calls=4),
                     overlap_fn=gen.overlap_fn,
                     enrich=gen.gather_for_survivors)
        assert res.reading is Severity.UNRESOLVED

    def test_altitude_branches_gather_their_own_evidence(self):
        """Same lateral corridor, different air. A report at FL340 is not
        evidence about FL315."""
        from app.reasoning.controller import Budget, search
        gen = self._gen()
        search(gen, beam_width=2, depth_limit=2,
               confidence_threshold=NO_EARLY_STOP,
               budget=Budget(max_tool_calls=24), overlap_fn=gen.overlap_fn,
               enrich=gen.gather_for_survivors)
        children = [k for k in gen.evidence if "/" in k]
        assert len(children) >= 2, "each surviving band needs its own evidence"
        # The bands differ, so their evidence may differ too.
        bands = {gen.shapes[k].altitude_min_ft for k in children}
        assert len(bands) > 1

    def test_the_graph_agrees_with_the_loop(self):
        from app.reasoning.controller import Budget, search
        from app.reasoning.graph import search_graph
        kw = dict(beam_width=2, depth_limit=2,
                  confidence_threshold=NO_EARLY_STOP,
                  overlap_fn=None)
        a_gen = self._gen()
        a = search(a_gen, budget=Budget(max_tool_calls=20),
                   enrich=a_gen.gather_for_survivors, **kw)
        b_gen = self._gen()
        b = search_graph(b_gen, budget=Budget(max_tool_calls=20),
                         enrich=b_gen.gather_for_survivors, **kw)
        assert a.reading is b.reading
        assert a.trace() == b.trace()


class TestAirportLookupFallback:
    """The great-circle corridor needs no external data to compute, so it
    should not be the source that fails first. It used to depend on airport
    coordinates arriving via a filed route, which breaks on any pair where
    no usable route comes back."""

    def _gen_without_cached_airports(self, route="EWC PONCT"):
        """A pair whose reference flight yields a route with no airports,
        which is what a wrong or foreign reference flight looks like."""
        pair = {"flights": [{"segments": [{
            "ident": "JZA8807", "fa_flight_id": "JZA8807-x",
            "status": "Arrived", "aircraft_type": "DH8D",
            "actual_off": "2026-08-16T23:00:00Z",
            "route": route, "filed_altitude": 240,
            "origin": {"code": "KSEA"},
            "destination": {"code": "RJTT"}}]}]}
        thin_route = {"fixes": [
            {"name": "EWC", "latitude": 40.7997, "longitude": -80.2144,
             "type": "VOR"},
            {"name": "PONCT", "latitude": 42.2, "longitude": -72.9,
             "type": "Fix"},
        ]}
        airports = {
            "KSEA": {"latitude": 47.4502, "longitude": -122.3088,
                     "name": "Seattle-Tacoma Intl"},
            "RJTT": {"latitude": 35.5533, "longitude": 139.7811,
                     "name": "Tokyo Haneda"},
        }

        def transport(path, params):
            if path.endswith("/flights/to/RJTT"):
                return 200, pair, ""
            if path.startswith("/flights/JZA8807-x/route"):
                return 200, thin_route, ""
            if path.endswith("/track"):
                return 200, {"positions": []}, ""
            if path.endswith("/routes/RJTT"):
                return 200, {"routes": []}, ""
            for code, body in airports.items():
                if path == f"/airports/{code}":
                    return 200, body, ""
            return 404, None, ""

        import sqlite3
        from app.sources.aeroapi import AeroAPIClient
        from app.sources.fixes import init_fixes
        conn = sqlite3.connect(":memory:")
        init_fixes(conn)
        client = AeroAPIClient(api_key="t", transport=transport,
                               spacing_seconds=0, sleep=lambda s: None)
        return CorridorGenerator(client=client, conn=conn,
                                 origin="KSEA", dest="RJTT")

    def test_a_corridor_is_still_produced(self):
        """The failure this fixes: zero corridors on a pair whose filed
        route names no airports."""
        gen = self._gen_without_cached_airports()
        out = gen(None, 1, Budget(max_tool_calls=12))
        assert out, "the great circle must survive a useless filed route"
        assert "gc" in {c.id for c in out}

    def test_the_airports_are_located_and_cached(self):
        gen = self._gen_without_cached_airports()
        gen(None, 1, Budget(max_tool_calls=12))
        names = {r[0] for r in
                 gen.conn.execute("SELECT name FROM route_fixes")}
        assert {"KSEA", "RJTT"} <= names
        assert any("Located" in n for n in gen.notes)

    def test_the_lookup_costs_budget(self):
        gen = self._gen_without_cached_airports()
        budget = Budget(max_tool_calls=12)
        gen(None, 1, budget)
        assert budget.calls_used >= 2

    def test_a_warm_cache_needs_no_lookup(self):
        gen = self._gen_without_cached_airports()
        gen(None, 1, Budget(max_tool_calls=12))
        first = gen.client.calls_made

        gen2 = self._gen_without_cached_airports()
        gen2.conn = gen.conn          # reuse the warmed cache
        gen2(None, 1, Budget(max_tool_calls=12))
        assert gen2.client.calls_made < first

    def test_an_unknown_airport_fails_with_a_reason(self):
        gen = self._gen_without_cached_airports()
        gen.dest = "ZZZZ"
        out = gen(None, 1, Budget(max_tool_calls=12))
        assert out == []
        assert any("ZZZZ" in n for n in gen.notes)

    def test_an_exhausted_budget_does_not_invent_a_position(self):
        gen = self._gen_without_cached_airports()
        out = gen(None, 1, Budget(max_tool_calls=2))
        assert all(c.id != "gc" for c in out)


class TestNoNonstopService:
    """A pair with no nonstop is a different absence from one where nothing
    has flown lately, and the note should say which."""

    def _gen(self, segments):
        import sqlite3
        from app.sources.aeroapi import AeroAPIClient
        from app.sources.fixes import init_fixes
        pair = {"flights": [{"segments": segments}]}

        def transport(path, params):
            if path.endswith("/flights/to/RJTT"):
                return 200, pair, ""
            if path == "/airports/KSAN":
                return 200, {"latitude": 32.7336, "longitude": -117.1897}, ""
            if path == "/airports/RJTT":
                return 200, {"latitude": 35.5533, "longitude": 139.7811}, ""
            if path.endswith("/routes/RJTT"):
                return 200, {"routes": []}, ""
            return 404, None, ""

        conn = sqlite3.connect(":memory:")
        init_fixes(conn)
        return CorridorGenerator(
            client=AeroAPIClient(api_key="t", transport=transport,
                                 spacing_seconds=0, sleep=lambda s: None),
            conn=conn, origin="KSAN", dest="RJTT")

    CONNECTION = [
        {"ident": "SKW4002", "fa_flight_id": "s-1", "aircraft_type": "E75L",
         "actual_off": "2026-08-16T22:48:35Z",
         "origin": {"code": "KSAN"}, "destination": {"code": "KLAX"}},
        {"ident": "ANA125", "fa_flight_id": "a-1", "aircraft_type": "B789",
         "actual_off": "2026-08-17T00:42:55Z",
         "origin": {"code": "KLAX"}, "destination": {"code": "RJTT"}},
    ]

    def test_the_absence_of_nonstop_service_is_stated(self):
        gen = self._gen(self.CONNECTION)
        gen(None, 1, Budget(max_tool_calls=12))
        assert any("No nonstop flights operate" in n for n in gen.notes)

    def test_the_geometric_corridor_is_labelled_as_such(self):
        """A great circle between two airports nobody flies directly is not
        a route anyone takes, and the note says so."""
        gen = self._gen(self.CONNECTION)
        gen(None, 1, Budget(max_tool_calls=12))
        assert any("not a route an aircraft takes" in n for n in gen.notes)

    def test_no_feeder_leg_becomes_the_reference_flight(self):
        gen = self._gen(self.CONNECTION)
        gen(None, 1, Budget(max_tool_calls=12))
        assert gen._flight is None

    def test_a_great_circle_is_still_offered(self):
        gen = self._gen(self.CONNECTION)
        out = gen(None, 1, Budget(max_tool_calls=12))
        assert "gc" in {c.id for c in out}


class TestConnectionAwareSearch:
    """A pair with no nonstop service, but a real connecting itinerary
    whose legs AeroAPI reported distances for, gets searched on the
    long-haul leg instead of falling back to a bare geometric line between
    two airports nobody flies directly - see
    AeroAPIClient.route_or_long_haul_leg and CorridorGenerator._get_flight.
    """

    CONNECTION_WITH_DISTANCE = [
        {"ident": "SKW4002", "fa_flight_id": "skw-1", "aircraft_type": "E75L",
         "route_distance": 100, "actual_off": "2026-08-16T22:48:35Z",
         "origin": {"code": "KSAN"}, "destination": {"code": "KLAX"}},
        {"ident": "ANA125", "fa_flight_id": "ana-1", "aircraft_type": "B789",
         "route_distance": 5100, "actual_off": "2026-08-17T00:42:55Z",
         "origin": {"code": "KLAX"}, "destination": {"code": "RJTT"}},
    ]

    ANA_ROUTE = {"fixes": [
        {"name": "KLAX", "latitude": 33.9425, "longitude": -118.4081,
         "type": "Origin Airport"},
        {"name": "MDPT", "latitude": 40.0, "longitude": 179.0, "type": "Fix"},
        {"name": "RJTT", "latitude": 35.5533, "longitude": 139.7811,
         "type": "Destination Airport"},
    ]}

    def _gen(self, segments=None, extra_routes=None):
        import sqlite3
        from app.sources.aeroapi import AeroAPIClient
        from app.sources.fixes import init_fixes
        pair = {"flights": [{"segments": segments if segments is not None
                                        else self.CONNECTION_WITH_DISTANCE}]}
        routes = {
            "/flights/to/RJTT": pair,
            "/airports/KSAN": {"latitude": 32.7336, "longitude": -117.1897},
            "/airports/KLAX": {"latitude": 33.9425, "longitude": -118.4081},
            "/airports/RJTT": {"latitude": 35.5533, "longitude": 139.7811},
            "/flights/ana-1/route": self.ANA_ROUTE,
            "/routes/RJTT": {"routes": []},
        }
        routes.update(extra_routes or {})

        def transport(path, params):
            for suffix, payload in routes.items():
                if path.endswith(suffix):
                    return 200, payload, ""
            return 404, None, ""

        conn = sqlite3.connect(":memory:")
        init_fixes(conn)
        return CorridorGenerator(
            client=AeroAPIClient(api_key="t", transport=transport,
                                 spacing_seconds=0, sleep=lambda s: None),
            conn=conn, origin="KSAN", dest="RJTT")

    def test_the_substitution_is_recorded(self):
        gen = self._gen()
        gen(None, 1, Budget(max_tool_calls=12))
        assert gen.route_substitution == ("KSAN", "RJTT", "KLAX", "RJTT")

    def test_the_long_haul_leg_becomes_the_reference_flight(self):
        """Not SKW4002, the short feeder into LAX - ANA125, the leg that
        actually flies to Tokyo."""
        gen = self._gen()
        gen(None, 1, Budget(max_tool_calls=12))
        assert gen._flight is not None
        assert gen._flight.ident == "ANA125"

    def test_the_generator_s_own_origin_and_dest_move_to_the_leg(self):
        gen = self._gen()
        gen(None, 1, Budget(max_tool_calls=12))
        assert (gen.origin, gen.dest) == ("KLAX", "RJTT")

    def test_the_substitution_is_explained_in_a_note(self):
        gen = self._gen()
        gen(None, 1, Budget(max_tool_calls=12))
        assert any("Searching the long-haul leg instead" in n
                   and "KSAN" in n and "KLAX to RJTT" in n
                   for n in gen.notes)

    def test_a_corridor_is_still_produced_on_the_substituted_pair(self):
        gen = self._gen()
        out = gen(None, 1, Budget(max_tool_calls=12))
        assert out, "expected at least one corridor on the substituted leg"

    def test_a_genuine_nonstop_pair_is_never_substituted(self):
        """The overwhelming common case: a pair with its own nonstop
        service is untouched by any of this."""
        segments = [{"ident": "NH106", "fa_flight_id": "nh-1",
                    "aircraft_type": "B77W", "route_distance": 5460,
                    "actual_off": "2026-08-16T20:00:00Z",
                    "origin": {"code": "KSAN"}, "destination": {"code": "RJTT"}}]
        gen = self._gen(segments=segments)
        gen(None, 1, Budget(max_tool_calls=12))
        assert gen.route_substitution is None
        assert (gen.origin, gen.dest) == ("KSAN", "RJTT")

    def test_no_ranked_leg_means_no_substitution(self):
        """The general no-nonstop-service class this feature extends:
        when nothing among the connecting itineraries lets the long-haul
        leg be told apart from a feeder, this behaves exactly as it did
        before the feature existed - see TestNoNonstopService."""
        undistanced = [
            {"ident": "SKW4002", "fa_flight_id": "skw-1",
             "actual_off": "2026-08-16T22:48:35Z",
             "origin": {"code": "KSAN"}, "destination": {"code": "KLAX"}},
            {"ident": "ANA125", "fa_flight_id": "ana-1",
             "actual_off": "2026-08-17T00:42:55Z",
             "origin": {"code": "KLAX"}, "destination": {"code": "RJTT"}},
        ]
        gen = self._gen(segments=undistanced)
        gen(None, 1, Budget(max_tool_calls=12))
        assert gen.route_substitution is None
        assert any("No nonstop flights operate" in n for n in gen.notes)


class TestDegradedSearches:
    """A search that lost a data source explored less of the tree. That is
    a different thing from one a budget cut short, and both differ from a
    search that simply pruned corridors."""

    def _rate_limited(self):
        import sqlite3
        from app.sources.aeroapi import AeroAPIClient
        from app.sources.fixes import init_fixes
        conn = sqlite3.connect(":memory:")
        init_fixes(conn)
        return CorridorGenerator(
            client=AeroAPIClient(api_key="t",
                                 transport=lambda p, q: (429, None, "slow"),
                                 spacing_seconds=0, sleep=lambda s: None),
            conn=conn, origin="KPIT", dest="KBOS")

    def test_a_rate_limited_search_is_marked_degraded(self):
        gen = self._rate_limited()
        gen(None, 1, Budget(max_tool_calls=8))
        assert gen.degraded
        assert any("rate limited" in n for n in gen.degraded)

    def test_a_healthy_search_is_not_marked_degraded(self):
        gen, _ = make_gen()
        gen(None, 1, Budget(max_tool_calls=12))
        assert gen.degraded == []

    @pytest.mark.parametrize("note,expected", [
        ("Could not list flights on this pair: rate limited twice", True),
        ("Turbulence forecasts could not be fetched (HTTP 403).", True),
        ("Tool budget exhausted before the flown track was fetched.", True),
        ("ZZZZ is not an airport AeroAPI recognises.", True),
        ("Cached 19 route fix(es) from JBU1286.", False),
        ("Cruise band from the flown track: FL313 to FL350.", False),
        ("Airway segment(s) J49 approximated as straight legs.", False),
        ("No pilot reports were filed anywhere near this route.", False),
    ])
    def test_real_notes_are_classified_correctly(self, note, expected):
        """An earlier version matched "could not fetch" and missed "could
        not be fetched", which is what comes out of the weather layer."""
        from app.reasoning.generator import _DEGRADED_MARKERS
        flagged = any(m in note.lower() for m in _DEGRADED_MARKERS)
        assert flagged is expected

    def test_notes_reach_the_log(self):
        import io
        from app.logging_setup import configure
        buf = io.StringIO()
        configure(level="DEBUG", use_syslog=False, stream=buf, force=True)
        gen = self._rate_limited()
        gen(None, 1, Budget(max_tool_calls=8))
        logged = buf.getvalue()
        assert "generator degraded" in logged
        assert "rate limited" in logged

    def test_an_ordinary_note_logs_without_the_degraded_marker(self):
        import io
        from app.logging_setup import configure
        buf = io.StringIO()
        configure(level="DEBUG", use_syslog=False, stream=buf, force=True)
        gen, _ = make_gen()
        gen(None, 1, Budget(max_tool_calls=12))
        logged = buf.getvalue()
        assert "generator note=" in logged
        assert "generator degraded" not in logged


class TestSplitPoints:
    """The pure helper behind a longitudinal split: cut a path in two by
    cumulative distance, not by how many points happen to represent it."""

    def test_splits_at_a_shared_vertex(self):
        points = [(0.0, 0.0), (0.0, 2.0), (0.0, 4.0)]
        first, second = _split_points(points)
        assert first[-1] == pytest.approx(second[0], abs=1e-6)
        assert first[0] == (0.0, 0.0)
        assert second[-1] == (0.0, 4.0)

    def test_interpolates_inside_a_leg(self):
        points = [(0.0, 0.0), (0.0, 10.0)]
        first, second = _split_points(points)
        # The only leg is 10 degrees of longitude at the equator; the split
        # point should land near its midpoint.
        assert first[-1][1] == pytest.approx(5.0, abs=0.5)
        assert first[-1] == second[0]

    def test_the_halves_are_roughly_equal_length(self):
        points = [(40.0, -80.0 + i * 0.5) for i in range(41)]
        first, second = _split_points(points)
        len_first = path_length_nm(first)
        len_second = path_length_nm(second)
        assert len_first == pytest.approx(len_second, rel=0.05)

    def test_too_few_points_returns_nothing(self):
        assert _split_points([]) == ([], [])
        assert _split_points([(0.0, 0.0)]) == ([], [])

    def test_zero_length_path_still_splits_by_index(self):
        points = [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
        first, second = _split_points(points)
        assert first and second


class TestDepthThree:
    """A conditional longitudinal split: fires only when the parent's own
    evidence says the corridor is not uniform, and never spends a call to
    evaluate its children - see `_gather_for`."""

    #: A long, straight synthetic corridor, decoupled from the KPIT-KBOS
    #: fixtures above so the split trigger can be tested in isolation from
    #: everything depth 1 and 2 already cover.
    LONG_PATH = [(40.0, -80.0 + i * 0.5) for i in range(41)]

    def _parent_with_evidence(self, gen, *, coverage=None, mixed=False,
                              matched=None, points=None, depth=2):
        points = points if points is not None else self.LONG_PATH
        cid = "probe"
        shape = gen._register(cid, points)
        gc_nm = path_length_nm(great_circle(points[0], points[-1], 24))
        parent = gen._corridor(cid, Provenance.ACTUAL_TRACK, points, gc_nm,
                               altitude_min=30000, altitude_max=36000,
                               depth=depth, label="probe")
        gen.evidence[cid] = GatherResult(
            evidence=Evidence(coverage_fraction=coverage),
            notes=[], matched_advisories=matched or [],
            observed_mixed=mixed, raw_reports=[], raw_advisories=[],
        )
        return parent

    def test_no_evidence_means_no_split(self):
        gen, _ = make_gen()
        shape = gen._register("probe", self.LONG_PATH)
        parent = gen._corridor("probe", Provenance.ACTUAL_TRACK,
                               self.LONG_PATH, 1000.0, depth=2)
        assert gen._longitudinal_branches(parent) == []

    def test_a_short_corridor_never_splits(self):
        gen, _ = make_gen()
        short_path = [(40.0, -80.0 + i * 0.02) for i in range(11)]
        parent = self._parent_with_evidence(gen, coverage=0.4,
                                            points=short_path)
        assert path_length_nm(short_path) < MIN_SPLIT_LENGTH_NM
        assert gen._longitudinal_branches(parent) == []

    def test_full_coverage_does_not_split(self):
        gen, _ = make_gen()
        parent = self._parent_with_evidence(gen, coverage=1.0)
        assert gen._longitudinal_branches(parent) == []

    def test_no_coverage_at_all_does_not_split(self):
        """Nobody having looked at any of it is uniform silence, not a
        reason to split - splitting would just produce two more unresolved
        children."""
        gen, _ = make_gen()
        parent = self._parent_with_evidence(gen, coverage=None)
        assert gen._longitudinal_branches(parent) == []

    def test_partial_coverage_splits(self):
        gen, _ = make_gen()
        parent = self._parent_with_evidence(gen, coverage=0.4)
        children = gen._longitudinal_branches(parent)
        assert len(children) == 2
        assert {c.label for c in children} == {"first half", "second half"}

    def test_mixed_reports_split_even_with_full_coverage(self):
        gen, _ = make_gen()
        parent = self._parent_with_evidence(gen, coverage=1.0, mixed=True)
        assert len(gen._longitudinal_branches(parent)) == 2

    def test_forecast_covering_only_one_half_splits(self):
        from types import SimpleNamespace
        gen, _ = make_gen()
        # A ring hugging the western end of the path only - roughly
        # KPIT-side longitudes, well clear of the eastern half.
        western_ring = [(39.0, -80.5), (41.0, -80.5),
                        (41.0, -75.0), (39.0, -75.0)]
        parent = self._parent_with_evidence(
            gen, coverage=None,
            matched=[SimpleNamespace(ring=western_ring)])
        children = gen._longitudinal_branches(parent)
        assert len(children) == 2

    def test_forecast_covering_the_whole_route_does_not_split(self):
        from types import SimpleNamespace
        gen, _ = make_gen()
        wide_ring = [(35.0, -85.0), (45.0, -85.0),
                    (45.0, -60.0), (35.0, -60.0)]
        parent = self._parent_with_evidence(
            gen, coverage=None,
            matched=[SimpleNamespace(ring=wide_ring)])
        assert gen._longitudinal_branches(parent) == []

    def test_no_signal_at_all_does_not_split(self):
        gen, _ = make_gen()
        parent = self._parent_with_evidence(gen)
        assert gen._longitudinal_branches(parent) == []

    def test_children_share_the_split_point(self):
        gen, _ = make_gen()
        parent = self._parent_with_evidence(gen, coverage=0.5)
        children = {c.label: c for c in gen._longitudinal_branches(parent)}
        first_pts = gen.shapes[children["first half"].id].points
        second_pts = gen.shapes[children["second half"].id].points
        assert first_pts[-1] == pytest.approx(second_pts[0], abs=1e-6)

    def test_children_inherit_provenance_and_altitude_band(self):
        gen, _ = make_gen()
        parent = self._parent_with_evidence(gen, coverage=0.5)
        for c in gen._longitudinal_branches(parent):
            assert c.provenance is parent.provenance
            assert c.parent_id == parent.id
            shape = gen.shapes[c.id]
            assert shape.altitude_min_ft == 30000
            assert shape.altitude_max_ft == 36000

    def test_split_children_are_tracked_for_free_reevaluation(self):
        gen, _ = make_gen()
        parent = self._parent_with_evidence(gen, coverage=0.5)
        children = gen._longitudinal_branches(parent)
        assert {c.id for c in children} <= gen._split_children

    def test_the_note_names_the_reason(self):
        gen, _ = make_gen()
        parent = self._parent_with_evidence(gen, coverage=0.4)
        gen._longitudinal_branches(parent)
        assert any("Splitting" in n and "partial" in n for n in gen.notes)

    def test_nothing_beyond_depth_three(self):
        gen, _ = make_gen()
        parent = self._parent_with_evidence(gen, coverage=0.4)
        assert gen(parent, MAX_USEFUL_DEPTH + 1, Budget(max_tool_calls=12)) == []

    def test_depth_three_is_routed_to_the_split(self):
        """__call__ dispatches depth 2 to altitude branches and depth 3 to
        the longitudinal split - a regression here would silently run the
        wrong branch logic at the wrong depth."""
        gen, _ = make_gen()
        parent = self._parent_with_evidence(gen, coverage=0.4)
        children = gen(parent, 3, Budget(max_tool_calls=12))
        assert {c.label for c in children} == {"first half", "second half"}


class TestDepthThreeReusesEvidence:
    """The whole cost claim rests on this: a split child's evidence comes
    from data already fetched for its parent, never a second fetch."""

    def _counting_sources(self):
        from datetime import datetime, timedelta, timezone
        from app.sources.gairmet import GairmetClient
        now = datetime(2026, 8, 16, 12, 30, tzinfo=timezone.utc)
        calls = {"pireps": 0, "gairmet": 0}

        class PR:
            def __init__(s, lat, lon, alt, sev):
                s.latitude, s.longitude, s.altitude_ft = lat, lon, alt
                s.turbulence_severity = sev
                s.observation_time = now - timedelta(minutes=20)

        # Two reports on the real flown track (see _track_point above, at
        # f=0.25 and f=0.75), with different severities, so coverage is
        # partial and the readings disagree - either signal alone would
        # trigger the split. Altitude is inside the "high" band
        # (hi-2000, hi) = (43000, 45000) that the beam-width-1 search keeps
        # at depth 2 for this fixture's filed range of FL250-FL450 - the
        # "low" branch is pruned, so a report at cruise (35000) would fall
        # outside the surviving corridor's altitude band and count as
        # observed nowhere.
        def fetch(bbox, hours):
            calls["pireps"] += 1
            return [PR(41.2068, -77.9261, 44000, "light"),
                   PR(42.1425, -73.3130, 44000, "severe")]

        def gairmet_fetch():
            calls["gairmet"] += 1
            return []

        client = GairmetClient(transport=lambda p, q: (200, [], ""))
        client.fetch = gairmet_fetch
        return fetch, client, now, calls

    def test_split_children_cost_no_extra_calls(self):
        from app.reasoning.controller import Budget, search
        gen, _ = make_gen()
        fetch, client, now, calls = self._counting_sources()
        gen.fetch_pireps = fetch
        gen.gairmet_client = client
        gen.when = now

        budget = Budget(max_tool_calls=30)
        res = search(gen, beam_width=1, depth_limit=3,
                     confidence_threshold=1.1, budget=budget,
                     overlap_fn=gen.overlap_fn,
                     enrich=gen.gather_for_survivors)

        split_ids = {cid for cid in gen.evidence if cid in gen._split_children}
        assert split_ids, "the split should have fired on this fixture"

        # One PIREP fetch and one GAIRMET fetch per non-split corridor that
        # was actually evaluated (depth 1's survivor, depth 2's altitude
        # bands) - the split children must not add to either count.
        non_split_evidence = len(gen.evidence) - len(split_ids)
        assert calls["pireps"] == non_split_evidence
        assert calls["gairmet"] == non_split_evidence

    def test_split_children_carry_their_own_filtered_evidence(self):
        """Reuse still means re-deriving the reading against the smaller
        shape, not copying the parent's - the whole point is that the two
        halves can disagree.

        Beam width 2 here (not 1, as in the cost test above) so that both
        split children survive to have their evidence gathered - with a
        beam of 1 only the higher-scoring half would ever reach
        `gen.evidence`, which says nothing about whether the two halves
        are derived independently.
        """
        from app.reasoning.controller import Budget, search
        gen, _ = make_gen()
        fetch, client, now, calls = self._counting_sources()
        gen.fetch_pireps = fetch
        gen.gairmet_client = client
        gen.when = now

        search(gen, beam_width=2, depth_limit=3, confidence_threshold=1.1,
              budget=Budget(max_tool_calls=30), overlap_fn=gen.overlap_fn,
              enrich=gen.gather_for_survivors)

        split_ids = [cid for cid in gen.evidence if cid in gen._split_children]
        assert len(split_ids) == 2
        readings = {gen.evidence[cid].evidence.reading for cid in split_ids}
        # One report near each half; each half should see only its own.
        assert Severity.LIGHT in readings or Severity.SEVERE in readings
