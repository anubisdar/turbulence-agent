"""Tests for the route fix cache.

Token classification is checked against every identifier that appeared in
real AeroAPI responses for KPIT-KBOS, so the patterns are validated against
filed routes rather than invented ones.
"""

import sqlite3

import pytest

from app.sources.fixes import (
    RouteResolution,
    TokenKind,
    cache_stats,
    classify,
    init_fixes,
    lookup,
    resolve_route,
    tokenize,
    upsert_fixes,
)

# Verbatim from data/aeroapi_probe/airport_routes.json
REAL_ROUTES = [
    "EWC WOMBT TOSTR PONCT JFUND2",
    "TYROO PSB J49 HNK PONCT JFUND2",
    "EWC JHW JEWLR GOATR BTV TEMPR ENE OOSHN5",
    "EWC JHW Q82 PONCT JFUND2",
    "CKB COBBE MAULS Q34 RBV Q419 JFK ROBUC3",
    "EWC JHW Q82 VIEEW Q82 MEMMS Q82 PONCT JFUND2",
]

# Shape returned by /flights/{id}/route
SAMPLE_FIXES = [
    {"name": "KPIT", "latitude": 40.4914167, "longitude": -80.2326944,
     "type": "Origin Airport"},
    {"name": "EWC", "latitude": 40.7997, "longitude": -80.2144, "type": "VOR"},
    {"name": "WOMBT", "latitude": 41.0333, "longitude": -78.9, "type": "Fix"},
    {"name": "TOSTR", "latitude": 41.6, "longitude": -76.5, "type": "Fix"},
    {"name": "PONCT", "latitude": 42.2, "longitude": -72.9, "type": "Fix"},
    {"name": "KBOS", "latitude": 42.3629, "longitude": -71.0064,
     "type": "Destination Airport"},
]


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    init_fixes(conn)
    yield conn
    conn.close()


class TestClassification:
    @pytest.mark.parametrize("token", ["J49", "Q82", "Q34", "Q419", "J174"])
    def test_airways(self, token):
        assert classify(token) is TokenKind.AIRWAY

    @pytest.mark.parametrize("token", ["JFUND2", "OOSHN5", "ROBUC3", "BOSRP5"])
    def test_procedures(self, token):
        assert classify(token) is TokenKind.PROCEDURE

    @pytest.mark.parametrize("token", ["EWC", "PSB", "HNK", "JFK", "ENE"])
    def test_navaids_are_points(self, token):
        assert classify(token) is TokenKind.POINT

    @pytest.mark.parametrize("token", ["WOMBT", "TOSTR", "PONCT", "TYROO",
                                       "GOATR", "MEMMS", "DRUNK"])
    def test_fixes_are_points(self, token):
        assert classify(token) is TokenKind.POINT

    @pytest.mark.parametrize("token", ["KPIT", "KBOS"])
    def test_airports_are_points(self, token):
        assert classify(token) is TokenKind.POINT

    def test_every_token_in_the_real_routes_is_recognised(self):
        """No unknowns across the six routings AeroAPI returned."""
        for route in REAL_ROUTES:
            for t in tokenize(route):
                assert classify(t) is not TokenKind.UNKNOWN, t

    def test_garbage_is_unknown(self):
        assert classify("") is TokenKind.UNKNOWN
        assert classify("!!!") is TokenKind.UNKNOWN
        assert classify("12345678") is TokenKind.UNKNOWN

    def test_a_slashless_oceanic_token_is_still_a_position(self):
        """`4030N08015W` was written into an earlier test as an example of
        garbage. It is a real oceanic waypoint with the slash omitted, which
        is the form some flight plans use."""
        from app.sources.fixes import TokenKind, parse_oceanic
        assert classify("4030N08015W") is TokenKind.OCEANIC
        assert parse_oceanic("4030N08015W") == pytest.approx((40.5, -80.25))


class TestTokenize:
    def test_splits_on_whitespace(self):
        assert tokenize("EWC WOMBT PONCT") == ["EWC", "WOMBT", "PONCT"]

    def test_splits_procedure_transitions_on_the_dot(self):
        assert tokenize("JFUND2.PONCT") == ["JFUND2", "PONCT"]

    def test_uppercases_and_drops_blanks(self):
        assert tokenize("  ewc   wombt  ") == ["EWC", "WOMBT"]

    def test_empty_route(self):
        assert tokenize("") == []
        assert tokenize(None) == []


class TestCache:
    def test_fixes_are_stored(self, db):
        assert upsert_fixes(db, SAMPLE_FIXES) == len(SAMPLE_FIXES)
        assert cache_stats(db)["total"] == len(SAMPLE_FIXES)

    def test_reinsert_increments_seen_count_not_row_count(self, db):
        upsert_fixes(db, SAMPLE_FIXES)
        upsert_fixes(db, SAMPLE_FIXES)
        assert cache_stats(db)["total"] == len(SAMPLE_FIXES)
        row = db.execute(
            "SELECT seen_count FROM route_fixes WHERE name='EWC'").fetchone()
        assert row[0] == 2

    def test_fixes_without_coordinates_are_skipped(self, db):
        assert upsert_fixes(db, [{"name": "NOPOS"}]) == 0

    def test_provenance_is_recorded(self, db):
        upsert_fixes(db, SAMPLE_FIXES, source="test-source")
        row = db.execute(
            "SELECT source FROM route_fixes WHERE name='EWC'").fetchone()
        assert row[0] == "test-source"

    def test_lookup_reports_missing_names(self, db):
        upsert_fixes(db, SAMPLE_FIXES)
        found, missing = lookup(db, ["EWC", "NOTREAL"])
        assert "EWC" in found
        assert missing == ["NOTREAL"]


class TestRouteResolution:
    def test_a_fully_cached_route_resolves(self, db):
        upsert_fixes(db, SAMPLE_FIXES)
        res = resolve_route(db, "EWC WOMBT TOSTR PONCT JFUND2")
        assert res.resolved
        assert [p[0] for p in res.points] == ["EWC", "WOMBT", "TOSTR", "PONCT"]

    def test_point_order_is_preserved(self, db):
        upsert_fixes(db, SAMPLE_FIXES)
        res = resolve_route(db, "PONCT TOSTR WOMBT EWC")
        assert [p[0] for p in res.points] == ["PONCT", "TOSTR", "WOMBT", "EWC"]

    def test_airways_are_dropped_and_recorded(self, db):
        upsert_fixes(db, SAMPLE_FIXES)
        res = resolve_route(db, "EWC J49 PONCT")
        assert res.airways_dropped == ["J49"]
        assert any("straight legs" in n for n in res.notes())

    def test_procedures_are_dropped_and_recorded(self, db):
        upsert_fixes(db, SAMPLE_FIXES)
        res = resolve_route(db, "EWC PONCT JFUND2")
        assert res.procedures_dropped == ["JFUND2"]
        assert any("Terminal procedure" in n for n in res.notes())


class TestGapsAreNotShortcuts:
    """A route with an unresolvable point is unknown, not shorter."""

    def test_a_missing_point_blocks_resolution(self, db):
        upsert_fixes(db, SAMPLE_FIXES)
        res = resolve_route(db, "EWC MYSTR PONCT")
        assert not res.resolved
        assert res.missing == ["MYSTR"]

    def test_the_gap_is_reported_explicitly(self, db):
        upsert_fixes(db, SAMPLE_FIXES)
        res = resolve_route(db, "EWC MYSTR PONCT")
        assert any("not treated as a shortcut" in n for n in res.notes())

    def test_coverage_reflects_what_was_found(self, db):
        upsert_fixes(db, SAMPLE_FIXES)
        res = resolve_route(db, "EWC MYSTR PONCT")
        assert res.coverage == pytest.approx(2 / 3)

    def test_an_empty_cache_resolves_nothing(self, db):
        res = resolve_route(db, "EWC WOMBT PONCT")
        assert not res.resolved
        assert len(res.missing) == 3

    def test_a_single_point_is_not_a_corridor(self, db):
        upsert_fixes(db, SAMPLE_FIXES)
        res = resolve_route(db, "EWC")
        assert not res.resolved


class TestCacheWarming:
    def test_one_route_call_populates_fixes_for_later_routes(self, db):
        """The whole point: fixes cached from one flight resolve another."""
        upsert_fixes(db, SAMPLE_FIXES)
        # A different filed routing that reuses cached points
        res = resolve_route(db, "EWC Q82 PONCT JFUND2")
        assert res.resolved
        assert [p[0] for p in res.points] == ["EWC", "PONCT"]

    def test_stats_summarise_the_cache(self, db):
        upsert_fixes(db, SAMPLE_FIXES)
        stats = cache_stats(db)
        assert stats["total"] == 6
        assert "VOR" in stats["by_type"]


class TestTypeNormalisation:
    """AeroAPI returns both 'Waypoint' and 'WAYPOINT' for the same thing,
    which splits the cache statistics. Acronyms must survive the fold."""

    @pytest.mark.parametrize("raw,expected", [
        ("WAYPOINT", "Waypoint"),
        ("Waypoint", "Waypoint"),
        ("VOR", "VOR"),
        ("VOR-DME (NAVAID)", "VOR-DME (NAVAID)"),
        ("VOR-TAC (NAVAID)", "VOR-TAC (NAVAID)"),
        ("Reporting Point", "Reporting Point"),
        ("Origin Airport", "Origin Airport"),
        (None, None),
    ])
    def test_casing(self, raw, expected):
        from app.sources.fixes import _normalize_type
        assert _normalize_type(raw) == expected

    def test_variants_land_in_one_bucket(self, db):
        upsert_fixes(db, [
            {"name": "A", "latitude": 1, "longitude": 1, "type": "Waypoint"},
            {"name": "B", "latitude": 2, "longitude": 2, "type": "WAYPOINT"},
        ])
        assert cache_stats(db)["by_type"] == {"Waypoint": 2}


class TestOceanicWaypoints:
    """Over water there is nothing to name a fix after, so routes carry the
    coordinates in the token. These were classified UNKNOWN and discarded,
    which is why every transpacific routing failed to resolve."""

    @pytest.mark.parametrize("token,expected", [
        ("5700N/15000W", (57.0, -150.0)),
        ("3600N/15000E", (36.0, 150.0)),
        ("5000N/17000E", (50.0, 170.0)),
        ("4800N/18000E", (48.0, 180.0)),
        ("5100N/14000W", (51.0, -140.0)),
        ("3000S/06000W", (-30.0, -60.0)),
    ])
    def test_real_tokens_parse(self, token, expected):
        from app.sources.fixes import parse_oceanic
        assert parse_oceanic(token) == pytest.approx(expected)

    def test_minutes_are_the_last_two_digits(self):
        """`5230N` is 52 degrees 30 minutes, which is 52.5."""
        from app.sources.fixes import parse_oceanic
        lat, lon = parse_oceanic("5230N/04030W")
        assert lat == pytest.approx(52.5)
        assert lon == pytest.approx(-40.5)

    def test_they_are_classified_as_oceanic(self):
        from app.sources.fixes import TokenKind
        assert classify("5700N/15000W") is TokenKind.OCEANIC
        assert classify("5700N/15000W") is not TokenKind.UNKNOWN

    def test_an_oceanic_route_designator_is_not_a_point(self):
        """`OTR13` names a route, not a position."""
        from app.sources.fixes import TokenKind, parse_oceanic
        assert classify("OTR13") is not TokenKind.OCEANIC
        assert parse_oceanic("OTR13") is None

    def test_impossible_coordinates_are_rejected(self):
        from app.sources.fixes import parse_oceanic
        assert parse_oceanic("9900N/15000W") is None      # latitude > 90
        assert parse_oceanic("5700N/19000W") is None      # longitude > 180

    def test_they_need_no_cache_lookup(self, db):
        """The position is in the token, so nothing can be missing."""
        res = resolve_route(db, "5100N/14000W 5400N/15000W")
        assert res.missing == []
        assert len(res.points) == 2

    def test_a_transpacific_route_resolves(self, db):
        upsert_fixes(db, [
            {"name": "KSEA", "latitude": 47.4502, "longitude": -122.3088,
             "type": "Airport"},
            {"name": "RJTT", "latitude": 35.5533, "longitude": 139.7811,
             "type": "Airport"},
            {"name": "NATES", "latitude": 52.0, "longitude": -160.0,
             "type": "Fix"},
        ])
        res = resolve_route(
            db, "KSEA BANGR9 NATES 5100N/14000W 5400N/15000W J523 RJTT")
        assert res.resolved
        assert res.coverage == 1.0

    def test_order_is_preserved_around_them(self, db):
        """An oceanic point sits between named fixes, not after them."""
        upsert_fixes(db, [
            {"name": "KSEA", "latitude": 47.4502, "longitude": -122.3088,
             "type": "Airport"},
            {"name": "RJTT", "latitude": 35.5533, "longitude": 139.7811,
             "type": "Airport"},
        ])
        res = resolve_route(db, "KSEA 5100N/14000W RJTT")
        assert [p[0] for p in res.points] == ["KSEA", "5100N/14000W", "RJTT"]

    def test_resolution_is_reported(self, db):
        res = resolve_route(db, "5100N/14000W 5400N/15000W")
        assert any("oceanic waypoint" in n for n in res.notes())

    def test_a_domestic_route_is_unaffected(self, db):
        upsert_fixes(db, SAMPLE_FIXES)
        res = resolve_route(db, "EWC WOMBT TOSTR PONCT JFUND2")
        assert res.resolved
        assert res.oceanic_points == []
