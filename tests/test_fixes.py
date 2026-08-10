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
        assert classify("4030N08015W") is TokenKind.UNKNOWN
        assert classify("") is TokenKind.UNKNOWN


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
