"""Tests for the G-AIRMET turbulence client.

Fixtures are the payload shapes the live endpoint actually returned during
the probe, not invented ones. Nothing here touches the network.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.sources.gairmet import (
    MAX_FORECAST_DISTANCE,
    TURBULENCE_HAZARDS,
    GairmetClient,
    GairmetFetchError,
    TurbulenceSeverity,
    parse_advisory,
    parse_ring,
    parse_severity,
    select_for_time,
    turbulence_only,
)

# Verbatim from data/awc_probe/gairmet_turb-hi.json
REAL_TURB = {
    "tag": "1W", "forecastHour": 3, "validTime": "2026-08-16T12:00:00.000Z",
    "hazard": "TURB-HI", "geometryType": "AREA", "latlonpairs": 10,
    "frequency": None, "severity": "MOD", "due_to": None, "status": "AMD",
    "top": "400", "base": "300", "fzltop": None, "fzlbase": None,
    "level": None, "receiptTime": 1786882373, "issueTime": 1786882320,
    "expireTime": 1786892400, "product": "TANGO", "geom": "AREA",
    "coords": [
        {"lat": "46.55", "lon": "-95.03"}, {"lat": "45.14", "lon": "-94.40"},
        {"lat": "43.30", "lon": "-95.27"}, {"lat": "41.80", "lon": "-98.32"},
        {"lat": "38.46", "lon": "-98.51"}, {"lat": "38.17", "lon": "-100.79"},
        {"lat": "39.48", "lon": "-102.87"}, {"lat": "43.59", "lon": "-101.90"},
        {"lat": "45.48", "lon": "-99.87"}, {"lat": "46.55", "lon": "-95.03"},
    ],
}

# The other hazards that arrive in the same bulletin.
OTHER_HAZARDS = [
    {"hazard": "FZLVL", "product": "TANGO", "top": "120", "base": "SFC",
     "coords": [{"lat": "40", "lon": "-90"}]},
    {"hazard": "IFR", "product": "SIERRA", "coords": []},
    {"hazard": "MT_OBSC", "product": "SIERRA", "coords": []},
    {"hazard": "ICE", "product": "ZULU", "coords": []},
    {"hazard": "LLWS", "product": "TANGO", "coords": []},
]

VALID_AT = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)


class TestSeverity:
    @pytest.mark.parametrize("code,expected", [
        ("MOD", TurbulenceSeverity.MODERATE),
        ("LGT", TurbulenceSeverity.LIGHT),
        ("SEV", TurbulenceSeverity.SEVERE),
        ("NEG", TurbulenceSeverity.NONE),
        ("EXTRM", TurbulenceSeverity.EXTREME),
    ])
    def test_known_codes(self, code, expected):
        assert parse_severity(code) == expected

    def test_compound_codes_take_the_worse_reading(self):
        """A passenger cares about the worst it might get, not the average."""
        assert parse_severity("LGT-MOD") == TurbulenceSeverity.MODERATE
        assert parse_severity("MOD OCNL SEV") == TurbulenceSeverity.SEVERE

    def test_an_unreadable_code_is_unknown_not_mild(self):
        assert parse_severity("WOBBLY") is None
        assert parse_severity(None) is None
        assert parse_severity("") is None

    def test_smooth_is_a_reading_not_an_absence(self):
        """NONE means a forecast of smooth air. It is not the same as having
        no forecast, and the two must not collapse."""
        assert parse_severity("SMTH") == TurbulenceSeverity.NONE
        assert parse_severity("SMTH") is not None


class TestRingParsing:
    def test_the_real_ring_parses(self):
        ring = parse_ring(REAL_TURB["coords"])
        assert len(ring) == 9          # 10 points, closing one dropped
        assert ring[0] == (46.55, -95.03)

    def test_coordinates_are_lat_lon(self):
        """Everything in this project uses (lat, lon). Reversing it here
        would put a Minnesota advisory in the Indian Ocean."""
        for lat, lon in parse_ring(REAL_TURB["coords"]):
            assert 24 < lat < 50, "latitude out of CONUS range"
            assert -125 < lon < -66, "longitude out of CONUS range"

    def test_the_closing_point_is_dropped(self):
        ring = parse_ring(REAL_TURB["coords"])
        assert ring[0] != ring[-1]

    def test_string_values_are_converted(self):
        ring = parse_ring([{"lat": "40.5", "lon": "-90.25"}])
        assert ring == [(40.5, -90.25)]

    def test_unusable_points_are_skipped_not_guessed(self):
        ring = parse_ring([{"lat": "40", "lon": "-90"},
                           {"lat": None, "lon": "-91"},
                           {"lat": "junk", "lon": "-92"}])
        assert ring == [(40.0, -90.0)]

    def test_empty_input(self):
        assert parse_ring(None) == []
        assert parse_ring([]) == []


class TestAdvisoryParsing:
    @pytest.fixture
    def advisory(self):
        return parse_advisory(REAL_TURB)

    def test_flight_levels_become_feet(self, advisory):
        """`top: '400'` is FL400, which is 40,000 ft, not 400."""
        assert advisory.base_ft == 30000
        assert advisory.top_ft == 40000

    def test_surface_base(self):
        a = parse_advisory({**REAL_TURB, "base": "SFC"})
        assert a.base_ft == 0

    def test_severity_is_mapped(self, advisory):
        assert advisory.severity == TurbulenceSeverity.MODERATE

    def test_both_timestamp_formats_are_handled(self, advisory):
        """validTime is an ISO string, expireTime is unix seconds."""
        assert advisory.valid_time == VALID_AT
        assert advisory.expire_time is not None
        assert advisory.expire_time > advisory.valid_time

    def test_forecast_hour_is_read(self, advisory):
        assert advisory.forecast_hour == 3

    def test_a_complete_advisory_is_usable(self, advisory):
        assert advisory.usable

    def test_a_missing_altitude_band_is_not_usable(self):
        a = parse_advisory({**REAL_TURB, "top": None, "base": None})
        assert not a.usable

    def test_a_missing_hazard_yields_nothing(self):
        assert parse_advisory({**REAL_TURB, "hazard": ""}) is None


class TestThreeDimensionalContainment:
    """A forecast for FL300-FL400 says nothing about a flight at FL410."""

    @pytest.fixture
    def advisory(self):
        return parse_advisory(REAL_TURB)

    @pytest.mark.parametrize("alt,inside", [
        (30000, True), (35000, True), (40000, True),
        (29000, False), (41000, False), (10000, False),
    ])
    def test_altitude_containment(self, advisory, alt, inside):
        assert advisory.covers_altitude(alt) is inside

    def test_a_corridor_band_overlapping_the_advisory(self, advisory):
        assert advisory.overlaps_band(31300, 35000)

    def test_a_corridor_band_above_the_advisory(self, advisory):
        assert not advisory.overlaps_band(41000, 43000)

    def test_a_corridor_band_below_the_advisory(self, advisory):
        assert not advisory.overlaps_band(10000, 20000)

    def test_bands_touching_at_the_edge_overlap(self, advisory):
        assert advisory.overlaps_band(40000, 45000)

    def test_an_unbanded_corridor_falls_back_to_two_dimensions(self, advisory):
        assert advisory.overlaps_band(None, None)


class TestValidityWindow:
    """Forecasts step every three hours. A corridor at 18:00Z must not be
    scored against a forecast valid at 06:00Z."""

    @pytest.fixture
    def advisory(self):
        return parse_advisory(REAL_TURB)

    def test_inside_its_own_window(self, advisory):
        assert advisory.valid_at(VALID_AT + timedelta(minutes=30))

    def test_within_one_forecast_step(self, advisory):
        assert advisory.valid_at(VALID_AT + timedelta(hours=2))

    def test_far_outside_the_window(self, advisory):
        assert not advisory.valid_at(VALID_AT + timedelta(hours=10))

    def test_selection_filters_by_time(self, advisory):
        near = select_for_time([advisory], VALID_AT + timedelta(hours=1))
        far = select_for_time([advisory], VALID_AT + timedelta(hours=12))
        assert len(near) == 1
        assert far == []

    def test_selection_drops_unusable_advisories(self):
        broken = parse_advisory({**REAL_TURB, "top": None, "base": None})
        assert select_for_time([broken], VALID_AT) == []


class TestHazardFiltering:
    """The endpoint ignores its own `type` parameter and returns every
    hazard in the bulletin. Without filtering here, corridors would be
    scored against icing and freezing-level advisories."""

    def test_only_turbulence_survives(self):
        bulletin = OTHER_HAZARDS + [REAL_TURB]
        out = turbulence_only(bulletin)
        assert len(out) == 1
        assert out[0].hazard == "TURB-HI"

    def test_both_turbulence_products_are_kept(self):
        low = {**REAL_TURB, "hazard": "TURB-LO", "top": "180", "base": "SFC"}
        out = turbulence_only([REAL_TURB, low])
        assert {a.hazard for a in out} == TURBULENCE_HAZARDS

    def test_product_tango_alone_does_not_mean_turbulence(self):
        """Turbulence arrives as product TANGO, but so do freezing level and
        low-level wind shear. Filtering on product would sweep them in."""
        tango_not_turb = [h for h in OTHER_HAZARDS
                          if h.get("product") == "TANGO"]
        assert tango_not_turb, "fixture should contain non-turbulence TANGO"
        assert turbulence_only(tango_not_turb) == []

    def test_an_empty_bulletin(self):
        assert turbulence_only([]) == []
        assert turbulence_only(None) == []


class TestClient:
    def _client(self, status=200, body=None):
        return GairmetClient(transport=lambda p, q: (status, body, ""))

    def test_a_full_bulletin_is_filtered(self):
        c = self._client(body=OTHER_HAZARDS + [REAL_TURB])
        out = c.fetch()
        assert len(out) == 1
        assert c.calls_made == 1

    def test_no_content_is_an_empty_list_not_an_error(self):
        """204 means no active advisories, which is a real answer."""
        assert self._client(status=204, body=[]).fetch() == []

    def test_a_feature_collection_payload(self):
        c = self._client(body={"type": "FeatureCollection",
                               "features": [REAL_TURB]})
        assert len(c.fetch()) == 1

    def test_a_server_error_raises(self):
        with pytest.raises(GairmetFetchError):
            self._client(status=500, body=None).fetch()

    def test_an_unparseable_payload_raises(self):
        with pytest.raises(GairmetFetchError):
            self._client(body="not a list or dict").fetch()

    def test_calls_are_counted_for_budgeting(self):
        c = self._client(body=[REAL_TURB])
        c.fetch()
        c.fetch()
        assert c.calls_made == 2


class TestAbsenceIsNotSmooth:
    """The rule this whole project turns on, applied to forecasts."""

    def test_an_empty_result_carries_no_severity(self):
        out = GairmetClient(transport=lambda p, q: (204, [], "")).fetch()
        assert out == []
        assert not any(getattr(a, "severity", None) for a in out)

    def test_a_forecast_of_smooth_air_is_distinguishable_from_no_forecast(self):
        smooth = parse_advisory({**REAL_TURB, "severity": "SMTH"})
        assert smooth.severity == TurbulenceSeverity.NONE
        assert smooth.severity is not None
