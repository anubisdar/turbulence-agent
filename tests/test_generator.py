"""Tests for the corridor generator.

No network. Payload shapes are the ones AeroAPI actually returned during the
probe, including the detail that filed route strings name enroute fixes but
not the airports.
"""

import math
import sqlite3

import pytest

from app.reasoning.controller import Budget
from app.reasoning.critic import Provenance
from app.reasoning.generator import MAX_USEFUL_DEPTH, CorridorGenerator
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
