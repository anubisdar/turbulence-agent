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
                   "filed_altitude": 350}]},
    {"segments": [{"ident": "RPA5678", "fa_flight_id": "RPA5678-y",
                   "status": "Arrived", "aircraft_type": "E75S",
                   "actual_off": "2026-08-09T10:00:00Z",
                   "route": ALT_ROUTE, "filed_altitude": 310}]},
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
             "route": "EWC PONCT"}]}]}
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
