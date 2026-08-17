"""Tests for the AeroAPI client.

Fixtures are the real payload shapes captured from a live Personal-tier key
during the probe, not invented ones. Nothing here touches the network.
"""

import pytest

from app.sources.aeroapi import (
    AeroAPIClient,
    AeroAPIError,
    AlternateRouting,
    RateLimited,
    TierRestricted,
    parse_distance_to_nm,
)

# ---- payloads captured from data/aeroapi_probe/ -------------------------

PAIR_FLIGHTS = {"flights": [
    {"segments": [{
        "ident": "RPA5678", "fa_flight_id": "RPA5678-1786256240-airline-137p",
        "status": "Scheduled", "aircraft_type": "E75S", "actual_off": None,
        "scheduled_out": "2026-08-11T14:59:00Z",
        "route": "TYROO PSB J49 HNK PONCT JFUND2",
        "route_distance": 495, "filed_altitude": 310,
        "origin": {"code": "KPIT"}, "destination": {"code": "KBOS"},
    }]},
    {"segments": [{
        "ident": "JBU1286", "fa_flight_id": "JBU1286-1786169312-airline-551p",
        "status": "Arrived / Gate Arrival", "aircraft_type": "BCS3",
        "actual_off": "2026-08-10T12:52:35Z",
        "scheduled_out": "2026-08-10T12:30:00Z",
        "route": "EWC WOMBT TOSTR PONCT JFUND2",
        "route_distance": 495, "filed_altitude": 350,
        "origin": {"code": "KPIT"}, "destination": {"code": "KBOS"},
    }]},
    {"segments": [{
        "ident": "AAL99", "fa_flight_id": "AAL99-x-airline-1",
        "status": "Arrived / Delayed", "aircraft_type": "BCS3",
        "actual_off": "2026-08-09T09:00:00Z",
        "route": "EWC JHW Q82 PONCT JFUND2",
        "route_distance": 495, "filed_altitude": 330,
        "origin": {"code": "KPIT"}, "destination": {"code": "KBOS"},
    }]},
    {"segments": [{"ident": "NOID", "status": "Cancelled"}]},   # no fa_flight_id
]}

FLIGHT_ROUTE = {"fixes": [
    {"name": "KPIT", "latitude": 40.4914167, "longitude": -80.2326944,
     "distance_from_origin": 0, "type": "Origin Airport"},
    {"name": "EWC", "latitude": 40.7997, "longitude": -80.2144,
     "distance_from_origin": 19, "type": "VOR"},
    {"name": "WOMBT", "latitude": 41.0333, "longitude": -78.9,
     "distance_from_origin": 80, "type": "Fix"},
    {"name": "NOPOS", "type": "Fix"},                          # no coordinates
    {"name": "KBOS", "latitude": 42.3629, "longitude": -71.0064,
     "distance_from_origin": 495, "type": "Destination Airport"},
]}

FLIGHT_TRACK = {"positions": [
    {"altitude": 10, "altitude_change": "C", "groundspeed": 141,
     "heading": 272, "latitude": 40.48932, "longitude": -80.22598,
     "timestamp": "2026-08-10T12:52:35Z", "update_type": "A"},
    {"altitude": 350, "groundspeed": 470, "latitude": 41.2,
     "longitude": -77.5, "timestamp": "2026-08-10T13:20:00Z",
     "update_type": "A"},
    {"altitude": None, "latitude": None, "longitude": None,
     "timestamp": "2026-08-10T13:30:00Z"},                     # unusable
]}

AIRPORT_ROUTES = {"routes": [
    {"route": "EWC WOMBT TOSTR PONCT JFUND2", "count": 48,
     "filed_altitude_min": 310, "filed_altitude_max": 390,
     "route_distance": "557 sm", "last_departure_time": "2026-08-11T11:20:00Z"},
    {"route": "TYROO PSB J49 HNK PONCT JFUND2", "count": 40,
     "filed_altitude_min": 250, "filed_altitude_max": 450,
     "route_distance": "550 sm"},
    {"route": "CKB COBBE MAULS Q34 RBV Q419 JFK ROBUC3", "count": 1,
     "filed_altitude_min": 370, "filed_altitude_max": 370,
     "route_distance": "782 sm"},
]}


def client(routes: dict, spacing=0.0):
    """A client whose transport serves canned payloads by path suffix."""
    calls = []

    def transport(path, params):
        calls.append((path, params))
        for suffix, payload in routes.items():
            if path.endswith(suffix):
                if isinstance(payload, int):
                    return payload, None, f'{{"title":"error {payload}"}}'
                if isinstance(payload, tuple):
                    return payload[0], None, payload[1]
                return 200, payload, ""
        return 404, None, "not found"

    c = AeroAPIClient(api_key="test", transport=transport,
                      spacing_seconds=spacing, sleep=lambda s: None)
    c._calls = calls
    return c


class TestDistanceNormalisation:
    @pytest.mark.parametrize("raw,expected", [
        ("557 sm", 484.0), ("550 sm", 477.9), ("480 nm", 480.0),
        ("100 km", 54.0), (495, 495.0), (None, None), ("garbage", None),
    ])
    def test_units(self, raw, expected):
        got = parse_distance_to_nm(raw)
        if expected is None:
            assert got is None
        else:
            assert got == pytest.approx(expected, abs=1.0)


class TestFlightsBetween:
    def test_segments_are_flattened_out_of_itineraries(self):
        c = client({"/flights/to/KBOS": PAIR_FLIGHTS})
        segs = c.flights_between("KPIT", "KBOS")
        assert [s.ident for s in segs] == ["RPA5678", "JBU1286", "AAL99"]

    def test_entries_without_a_flight_id_are_skipped(self):
        c = client({"/flights/to/KBOS": PAIR_FLIGHTS})
        assert all(s.fa_flight_id for s in c.flights_between("KPIT", "KBOS"))

    def test_filed_altitude_is_converted_to_feet(self):
        c = client({"/flights/to/KBOS": PAIR_FLIGHTS})
        seg = next(s for s in c.flights_between("KPIT", "KBOS")
                   if s.ident == "JBU1286")
        assert seg.filed_altitude_ft == 35000

    def test_has_flown_distinguishes_departed_from_scheduled(self):
        c = client({"/flights/to/KBOS": PAIR_FLIGHTS})
        segs = {s.ident: s for s in c.flights_between("KPIT", "KBOS")}
        assert not segs["RPA5678"].has_flown
        assert segs["JBU1286"].has_flown


class TestMostRecentlyFlown:
    def test_picks_the_latest_departure(self):
        c = client({"/flights/to/KBOS": PAIR_FLIGHTS})
        assert c.most_recently_flown("KPIT", "KBOS").ident == "JBU1286"

    def test_ignores_scheduled_flights(self):
        c = client({"/flights/to/KBOS": PAIR_FLIGHTS})
        assert c.most_recently_flown("KPIT", "KBOS").has_flown

    def test_returns_none_when_nothing_has_flown(self):
        only_scheduled = {"flights": [{"segments": [{
            "ident": "X", "fa_flight_id": "x-1", "actual_off": None}]}]}
        c = client({"/flights/to/KBOS": only_scheduled})
        assert c.most_recently_flown("KPIT", "KBOS") is None


class TestRouteFixes:
    def test_fixes_carry_coordinates(self):
        c = client({"/route": FLIGHT_ROUTE})
        fixes = c.route_fixes("JBU1286-x")
        assert [f.name for f in fixes] == ["KPIT", "EWC", "WOMBT", "KBOS"]

    def test_fixes_without_coordinates_are_dropped(self):
        c = client({"/route": FLIGHT_ROUTE})
        assert "NOPOS" not in [f.name for f in c.route_fixes("x")]

    def test_cache_rows_match_what_the_fix_store_expects(self):
        c = client({"/route": FLIGHT_ROUTE})
        row = c.route_fixes("x")[0].as_cache_row()
        assert set(row) == {"name", "latitude", "longitude", "type"}


class TestTrack:
    def test_altitude_is_converted_from_hundreds_of_feet(self):
        c = client({"/track": FLIGHT_TRACK})
        pos = c.track("x")
        assert pos[0].altitude_ft == 1000
        assert pos[1].altitude_ft == 35000

    def test_positions_without_coordinates_are_dropped(self):
        c = client({"/track": FLIGHT_TRACK})
        assert len(c.track("x")) == 2

    def test_timestamps_are_preserved(self):
        c = client({"/track": FLIGHT_TRACK})
        assert c.track("x")[0].timestamp == "2026-08-10T12:52:35Z"


class TestAlternateRoutings:
    def test_sorted_by_how_often_they_are_filed(self):
        c = client({"/routes/KBOS": AIRPORT_ROUTES})
        routings = c.alternate_routings("KPIT", "KBOS")
        assert [r.count for r in routings] == [48, 40, 1]

    def test_statute_miles_are_converted(self):
        c = client({"/routes/KBOS": AIRPORT_ROUTES})
        top = c.alternate_routings("KPIT", "KBOS")[0]
        assert top.reported_distance_nm == pytest.approx(484, abs=1)

    def test_altitude_band_is_converted_to_feet(self):
        c = client({"/routes/KBOS": AIRPORT_ROUTES})
        top = c.alternate_routings("KPIT", "KBOS")[0]
        assert top.filed_altitude_min_ft == 31000
        assert top.filed_altitude_max_ft == 39000

    def test_the_two_dominant_routings_are_distinct(self):
        """92% of filed traffic on this pair, and genuinely different paths."""
        c = client({"/routes/KBOS": AIRPORT_ROUTES})
        routings = c.alternate_routings("KPIT", "KBOS")
        assert routings[0].route != routings[1].route
        assert routings[0].count + routings[1].count > 0.9 * sum(
            r.count for r in routings)


class TestErrorHandling:
    def test_tier_restriction_is_distinguished_from_a_bad_key(self):
        body = '{"title":"Invalid API key","detail":"Alerts and Historical ' \
               'data are only available on Standard and Premium tiers."}'
        c = client({"/history/flights/X": (401, body)})
        with pytest.raises(TierRestricted):
            c.request("/history/flights/X")

    def test_a_genuine_key_rejection_raises_plainly(self):
        c = client({"/airports/KPIT": (401, '{"title":"Invalid API key"}')})
        with pytest.raises(AeroAPIError) as exc:
            c.request("/airports/KPIT")
        assert not isinstance(exc.value, TierRestricted)

    def test_rate_limit_is_retried_once(self):
        attempts = {"n": 0}

        def transport(path, params):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return 429, None, "slow down"
            return 200, {"ok": True}, ""

        c = AeroAPIClient(api_key="k", transport=transport,
                          spacing_seconds=0, sleep=lambda s: None)
        assert c.request("/x") == {"ok": True}
        assert attempts["n"] == 2

    def test_a_second_rate_limit_gives_up(self):
        c = client({"/x": 429})
        with pytest.raises(RateLimited):
            c.request("/x")

    def test_other_errors_surface(self):
        c = client({"/x": 500})
        with pytest.raises(AeroAPIError):
            c.request("/x")


class TestCallAccounting:
    def test_every_request_is_counted(self):
        c = client({"/flights/to/KBOS": PAIR_FLIGHTS, "/route": FLIGHT_ROUTE})
        c.flights_between("KPIT", "KBOS")
        c.route_fixes("x")
        assert c.calls_made == 2

    def test_a_retry_counts_as_a_second_call(self):
        """Retries cost money too, so the budget must see them."""
        attempts = {"n": 0}

        def transport(path, params):
            attempts["n"] += 1
            return (429, None, "") if attempts["n"] == 1 else (200, {}, "")

        c = AeroAPIClient(api_key="k", transport=transport,
                          spacing_seconds=0, sleep=lambda s: None)
        c.request("/x")
        assert c.calls_made == 2

    def test_the_call_log_records_paths(self):
        c = client({"/flights/to/KBOS": PAIR_FLIGHTS})
        c.flights_between("KPIT", "KBOS")
        assert c.call_log == ["/airports/KPIT/flights/to/KBOS"]



CONNECTION = {"flights": [
    # The real shape of a KSAN to RJTT query: no nonstop exists, so every
    # itinerary connects. Flattening these into one pool and picking by
    # departure time returns a regional feeder leg as the reference flight.
    {"segments": [
        {"ident": "SKW4002", "fa_flight_id": "skw-1", "aircraft_type": "E75L",
         "actual_off": "2026-08-16T22:48:35Z",
         "origin": {"code": "KSAN"}, "destination": {"code": "KLAX"}},
        {"ident": "ANA125", "fa_flight_id": "ana-1", "aircraft_type": "B789",
         "actual_off": "2026-08-17T00:42:55Z",
         "origin": {"code": "KLAX"}, "destination": {"code": "RJTT"}},
    ]},
    {"segments": [
        {"ident": "DAL2508", "fa_flight_id": "dal-1", "aircraft_type": "B738",
         "actual_off": "2026-08-16T18:25:39Z",
         "origin": {"code": "KSAN"}, "destination": {"code": "KSEA"}},
        {"ident": "ANA117", "fa_flight_id": "ana-2", "aircraft_type": "B789",
         "actual_off": "2026-08-17T00:16:09Z",
         "origin": {"code": "KSEA"}, "destination": {"code": "RJTT"}},
    ]},
]}


class TestNonstopFiltering:
    """An itinerary can be a connection. Picking any segment by departure
    time returned a Dash 8 turboprop as the reference flight for Seattle to
    Tokyo, and a filed route that started at Los Angeles for a search from
    San Diego."""

    def test_connecting_segments_are_excluded(self):
        c = client({"/flights/to/RJTT": CONNECTION})
        assert c.flights_between("KSAN", "RJTT") == []

    def test_a_pair_with_no_nonstop_returns_nothing(self):
        """The honest answer. Returning someone else's flight is worse."""
        c = client({"/flights/to/RJTT": CONNECTION})
        assert c.most_recently_flown("KSAN", "RJTT") is None

    def test_a_genuine_nonstop_survives(self):
        payload = {"flights": [{"segments": [
            {"ident": "NH106", "fa_flight_id": "nh-1", "aircraft_type": "B77W",
             "actual_off": "2026-08-16T20:00:00Z",
             "origin": {"code": "KSAN"}, "destination": {"code": "RJTT"}}]}]}
        c = client({"/flights/to/RJTT": payload})
        assert [s.ident for s in c.flights_between("KSAN", "RJTT")] == ["NH106"]

    def test_the_middle_leg_of_a_connection_is_not_taken(self):
        """ANA125 flies to Tokyo but departs from Los Angeles."""
        c = client({"/flights/to/RJTT": CONNECTION})
        assert "ANA125" not in [s.ident
                                for s in c.flights_between("KSAN", "RJTT")]

    def test_connections_are_available_when_asked_for(self):
        c = client({"/flights/to/RJTT": CONNECTION})
        idents = [s.ident for s in
                  c.flights_between("KSAN", "RJTT", nonstop_only=False)]
        assert "ANA125" in idents

    def test_matching_is_case_insensitive(self):
        payload = {"flights": [{"segments": [
            {"ident": "NH106", "fa_flight_id": "nh-1", "aircraft_type": "B77W",
             "actual_off": "2026-08-16T20:00:00Z",
             "origin": {"code": "ksan"}, "destination": {"code": "rjtt"}}]}]}
        c = client({"/flights/to/RJTT": payload})
        assert len(c.flights_between("KSAN", "RJTT")) == 1
