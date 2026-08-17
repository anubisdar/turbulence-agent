"""Tests for route geometry.

Distances are checked against real airport coordinates rather than synthetic
ones, so a projection or unit error shows up as a wrong number of nautical
miles instead of passing quietly.
"""

import math

import pytest
from pyproj import Geod

from app.reasoning.geometry import (
    DEFAULT_WIDTH_NM,
    MIN_ENROUTE_LEG_NM,
    NM_TO_M,
    build_corridor,
    corridor_overlap_fn,
    great_circle,
    leg_length_nm,
    max_dogleg_deg,
    overlap_fraction,
    path_length_nm,
    simplify_track,
)

GEOD = Geod(ellps="WGS84")

KPIT = (40.4914167, -80.2326944)
KBOS = (42.3629, -71.0064)
KJFK = (40.6398, -73.7789)

# The filed routing AeroAPI returned for JBU1286, with fix positions.
FILED_PIT_BOS = [
    KPIT,
    (40.7997, -80.2144),    # EWC, 19 nm due north of the field
    (41.0333, -78.9000),    # WOMBT
    (41.6000, -76.5000),    # TOSTR
    (42.2000, -72.9000),    # PONCT
    KBOS,
]


def offset(point, bearing_deg, distance_nm):
    """A point a given distance and bearing from another."""
    lon, lat, _ = GEOD.fwd(point[1], point[0], bearing_deg,
                           distance_nm * NM_TO_M)
    return (lat, lon)


class TestGreatCircle:
    def test_pit_to_bos_is_about_430_nm(self):
        """Published great-circle distance is roughly 430 nm."""
        assert path_length_nm(great_circle(KPIT, KBOS)) == pytest.approx(431, abs=3)

    def test_endpoints_are_included(self):
        pts = great_circle(KPIT, KBOS, 10)
        assert pts[0] == KPIT
        assert pts[-1] == KBOS

    def test_point_count_is_respected(self):
        assert len(great_circle(KPIT, KBOS, 24)) == 24

    def test_a_geodesic_barely_turns(self):
        assert max_dogleg_deg(great_circle(KPIT, KBOS, 24), min_leg_nm=0) < 1.0

    def test_more_points_do_not_change_the_length(self):
        coarse = path_length_nm(great_circle(KPIT, KBOS, 4))
        fine = path_length_nm(great_circle(KPIT, KBOS, 64))
        assert fine == pytest.approx(coarse, rel=0.001)


class TestPathLength:
    def test_a_filed_route_is_longer_than_the_geodesic(self):
        gc = path_length_nm(great_circle(KPIT, KBOS))
        filed = path_length_nm(FILED_PIT_BOS)
        assert filed > gc
        assert filed / gc < 1.2

    def test_single_point_has_no_length(self):
        assert path_length_nm([KPIT]) == 0.0


class TestDoglegExcludesTerminalLegs:
    """Departure and arrival fixes sit near their airports, so the first and
    last legs turn sharply by design. Counting those as enroute doglegs would
    eliminate perfectly ordinary filed routes."""

    def test_the_first_leg_of_the_real_route_is_short(self):
        first = leg_length_nm(FILED_PIT_BOS[0], FILED_PIT_BOS[1])
        assert first < MIN_ENROUTE_LEG_NM

    def test_naive_measurement_sees_a_large_turn(self):
        assert max_dogleg_deg(FILED_PIT_BOS, min_leg_nm=0) > 60

    def test_enroute_measurement_sees_a_gentle_one(self):
        assert max_dogleg_deg(FILED_PIT_BOS) < 20

    def test_a_genuine_enroute_dogleg_is_still_caught(self):
        """A hard turn between two long legs must not be excused."""
        a = KPIT
        b = offset(a, 90, 200)
        c = offset(b, 200, 200)
        assert max_dogleg_deg([a, b, c]) > 60


class TestCorridorWidth:
    def test_the_buffer_is_nautical_miles_not_degrees(self):
        """A projection or unit slip shows up here as a wildly wrong width."""
        pts = great_circle(KPIT, KBOS, 24)
        corridor = build_corridor(pts, width_nm=25.0)
        mid = pts[len(pts) // 2]
        assert corridor.contains(*offset(mid, 0, 20))
        assert not corridor.contains(*offset(mid, 0, 40))

    def test_points_on_the_path_are_inside(self):
        pts = great_circle(KPIT, KBOS, 24)
        corridor = build_corridor(pts)
        for p in pts[1:-1]:
            assert corridor.contains(*p)

    def test_a_distant_point_is_outside(self):
        corridor = build_corridor(great_circle(KPIT, KBOS, 24))
        assert not corridor.contains(25.7617, -80.1918)   # Miami

    def test_a_wider_corridor_admits_more(self):
        pts = great_circle(KPIT, KBOS, 24)
        mid = pts[len(pts) // 2]
        far = offset(mid, 0, 40)
        assert not build_corridor(pts, width_nm=25).contains(*far)
        assert build_corridor(pts, width_nm=60).contains(*far)

    def test_two_points_are_enough(self):
        assert build_corridor([KPIT, KBOS]).length_nm > 400

    def test_one_point_is_not_a_corridor(self):
        with pytest.raises(ValueError):
            build_corridor([KPIT])


class TestThreeDimensionalContainment:
    """A turbulence advisory for FL240-FL390 does not apply at FL410."""

    @pytest.fixture
    def banded(self):
        return build_corridor(great_circle(KPIT, KBOS, 24),
                              altitude_min_ft=24000, altitude_max_ft=39000)

    def test_inside_the_band(self, banded):
        mid = great_circle(KPIT, KBOS, 24)[12]
        assert banded.contains(*mid, altitude_ft=34000)

    def test_above_the_band(self, banded):
        mid = great_circle(KPIT, KBOS, 24)[12]
        assert not banded.contains(*mid, altitude_ft=41000)

    def test_below_the_band(self, banded):
        mid = great_circle(KPIT, KBOS, 24)[12]
        assert not banded.contains(*mid, altitude_ft=18000)

    def test_no_altitude_supplied_falls_back_to_two_dimensions(self, banded):
        mid = great_circle(KPIT, KBOS, 24)[12]
        assert banded.contains(*mid)

    def test_an_unbanded_corridor_accepts_any_altitude(self):
        corridor = build_corridor(great_circle(KPIT, KBOS, 24))
        mid = great_circle(KPIT, KBOS, 24)[12]
        assert corridor.contains(*mid, altitude_ft=41000)


class TestOverlap:
    def test_a_corridor_fully_overlaps_itself(self):
        c = build_corridor(great_circle(KPIT, KBOS, 24))
        assert overlap_fraction(c, c) == pytest.approx(1.0, abs=0.01)

    def test_unrelated_corridors_do_not_overlap(self):
        a = build_corridor(great_circle(KPIT, KBOS, 24))
        b = build_corridor(great_circle((25.79, -80.29), (32.90, -97.04), 24))
        assert overlap_fraction(a, b) == 0.0

    def test_the_filed_route_partly_overlaps_the_geodesic(self):
        gc = build_corridor(great_circle(KPIT, KBOS, 24))
        filed = build_corridor(FILED_PIT_BOS)
        frac = overlap_fraction(gc, filed)
        assert 0.3 < frac < 1.0

    def test_overlap_is_measured_against_the_smaller_corridor(self):
        """A short corridor inside a long one is fully redundant."""
        long_ = build_corridor(great_circle(KPIT, KBOS, 24))
        pts = great_circle(KPIT, KBOS, 24)
        short = build_corridor(pts[10:14])
        assert overlap_fraction(long_, short) > 0.9

    def test_a_wide_corridor_swallows_a_narrow_one(self):
        pts = great_circle(KPIT, KBOS, 24)
        wide = build_corridor(pts, width_nm=80)
        narrow = build_corridor(pts, width_nm=10)
        assert overlap_fraction(wide, narrow) == pytest.approx(1.0, abs=0.02)


class TestCriticAdapter:
    class FakeCorridor:
        def __init__(self, cid):
            self.id = cid

    def test_known_shapes_are_compared(self):
        shapes = {
            "a": build_corridor(great_circle(KPIT, KBOS, 24)),
            "b": build_corridor(great_circle(KPIT, KBOS, 24)),
        }
        fn = corridor_overlap_fn(shapes)
        assert fn(self.FakeCorridor("a"), self.FakeCorridor("b")) > 0.9

    def test_a_missing_shape_never_causes_a_prune(self):
        """Absent geometry must score zero overlap, not high overlap."""
        shapes = {"a": build_corridor(great_circle(KPIT, KBOS, 24))}
        fn = corridor_overlap_fn(shapes)
        assert fn(self.FakeCorridor("a"), self.FakeCorridor("missing")) == 0.0


class TestSimplify:
    def _dense_track(self, n=154):
        return great_circle(KPIT, KBOS, n)

    def test_a_dense_track_is_thinned(self):
        dense = self._dense_track()
        thinned = simplify_track(dense, tolerance_nm=2.0)
        assert len(thinned) < len(dense)

    def test_endpoints_survive(self):
        dense = self._dense_track()
        thinned = simplify_track(dense, tolerance_nm=2.0)
        assert thinned[0] == pytest.approx(dense[0], abs=0.01)
        assert thinned[-1] == pytest.approx(dense[-1], abs=0.01)

    def test_length_is_broadly_preserved(self):
        dense = self._dense_track()
        thinned = simplify_track(dense, tolerance_nm=2.0)
        assert path_length_nm(thinned) == pytest.approx(
            path_length_nm(dense), rel=0.02)

    def test_short_tracks_are_left_alone(self):
        assert simplify_track([KPIT, KBOS]) == [KPIT, KBOS]

    def test_the_corridor_barely_changes(self):
        """Thinning must not move the airspace the corridor occupies."""
        dense = self._dense_track()
        a = build_corridor(dense, simplify_nm=None)
        b = build_corridor(dense, simplify_nm=2.0)
        assert overlap_fraction(a, b) > 0.95


KSEA = (47.4502, -122.3088)
RJTT = (35.5533, 139.7811)


class TestAntimeridian:
    """A Seattle to Tokyo route runs -122, -130, ... 179, -179, ... 140.
    Averaging that raw puts the projection's centre near West Africa, and
    the corridor gets drawn eastward across three continents instead of
    northwest over the Aleutians."""

    @pytest.fixture
    def transpacific(self):
        return great_circle(KSEA, RJTT, 24)

    def test_the_route_is_detected_as_crossing(self, transpacific):
        from app.reasoning.geometry import crosses_antimeridian
        assert crosses_antimeridian(transpacific)
        assert not crosses_antimeridian(great_circle(KPIT, KBOS, 24))

    def test_longitudes_unwrap_continuously(self):
        from app.reasoning.geometry import unwrap_longitudes
        wrapped = [170.0, 175.0, 179.0, -178.0, -173.0]
        unwrapped = unwrap_longitudes(wrapped)
        steps = [abs(b - a) for a, b in zip(unwrapped, unwrapped[1:])]
        assert all(s < 180 for s in steps)
        assert unwrapped[-1] > 180

    def test_normalisation_returns_a_legal_longitude(self):
        from app.reasoning.geometry import normalize_longitude
        assert normalize_longitude(187.0) == pytest.approx(-173.0)
        assert normalize_longitude(-190.0) == pytest.approx(170.0)
        assert normalize_longitude(-122.0) == pytest.approx(-122.0)

    def test_the_midpoint_lands_in_the_pacific(self, transpacific):
        """Not off West Africa, which is what the raw average gives."""
        from app.reasoning.geometry import midpoint
        lat, lon = midpoint(transpacific)
        assert abs(lon) > 150, "the centre should be near the date line"
        assert 30 < lat < 60

    def test_the_geodesic_length_is_right(self, transpacific):
        """Seattle to Tokyo is about 4,100 nm, not 15,000."""
        assert path_length_nm(transpacific) == pytest.approx(4174, abs=50)

    def test_the_corridor_area_matches_its_dimensions(self, transpacific):
        """A 4,174 nm path buffered 25 nm each side is roughly 210,000 nm2.
        The broken projection gave 296,672."""
        corridor = build_corridor(transpacific)
        assert corridor.area_nm2() == pytest.approx(210_000, rel=0.05)

    def test_the_corridor_does_not_wrap_the_wrong_way(self, transpacific):
        """The clearest symptom: a band across Europe."""
        corridor = build_corridor(transpacific)
        assert not corridor.contains(51.5, -0.1), "London is not en route"
        assert not corridor.contains(40.7, -74.0), "New York is not en route"
        assert not corridor.contains(55.7, 37.6), "Moscow is not en route"

    def test_points_along_the_route_are_inside(self, transpacific):
        corridor = build_corridor(transpacific)
        for point in transpacific[2:-2]:
            assert corridor.contains(*point)

    def test_a_us_route_is_unaffected(self):
        """The fix must not move anything that already worked."""
        corridor = build_corridor(great_circle(KPIT, KBOS, 24))
        assert corridor.length_nm == pytest.approx(431, abs=3)
        assert corridor.contains(*great_circle(KPIT, KBOS, 24)[12])
