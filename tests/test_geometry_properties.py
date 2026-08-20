"""Properties and metamorphic relations for corridor geometry.

Every geometry test written before this file used a route between two
airports in the continental United States. That is not a small gap: it meant
the entire suite shared one assumption about the coordinate domain, and when
that assumption broke - a Seattle to Tokyo route crossing the antimeridian -
738 tests stayed green while the projection anchored itself off West Africa
and the corridor swept across three continents.

The gap was not in what the tests checked. It was in where they looked.

TWO TECHNIQUES, FOR TWO DIFFERENT PROBLEMS.

Property-based testing, with Hypothesis, generates coordinates across the
whole domain rather than the handful a person thinks to type. It answers
"does this hold everywhere" instead of "does this hold in Pennsylvania".

Metamorphic relations answer a harder question. For most of this module
there is no oracle: nobody knows what the corridor area for an arbitrary
route *should* be, so there is nothing to assert it equals. But we do know
what must stay true when an input is transformed. Sliding a route east by
forty degrees cannot change its length. Mirroring it across the equator
cannot change its area. Those relations need no oracle, and the antimeridian
bug violates the first one flagrantly.

WHAT THESE TESTS ALREADY FOUND. Southern hemisphere, equator-crossing and
polar routes all pass, which is worth knowing rather than assuming - the
antimeridian fix generalised. A zero-length route does not: it builds a
corridor with no area whose containment test rejects its own defining point.
That case has its own class at the end of this file.
"""

import math

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from pyproj import Geod

from app.reasoning.geometry import (
    build_corridor,
    great_circle,
    midpoint,
    normalize_longitude,
    overlap_fraction,
    path_length_nm,
    unwrap_longitudes,
)

#: Geodesic work through pyproj and shapely is not free, and a deadline that
#: fires on a slow example is noise rather than signal.
GEO = settings(max_examples=60, deadline=None,
               suppress_health_check=[HealthCheck.too_slow,
                                      HealthCheck.filter_too_much])

#: Latitudes beyond this are outside anywhere a scheduled flight operates,
#: and an azimuthal projection anchored near a pole distorts enough that a
#: tolerance meaningful elsewhere stops meaning anything.
FLYABLE_LAT = 78.0

latitudes = st.floats(min_value=-FLYABLE_LAT, max_value=FLYABLE_LAT,
                      allow_nan=False, allow_infinity=False)
longitudes = st.floats(min_value=-180.0, max_value=180.0,
                       allow_nan=False, allow_infinity=False)


#: Routes are built from a start, a bearing and a distance rather than from
#: two random points. Two random points are almost never a plausible flight
#: - most pairs on a sphere are thousands of miles apart - so filtering them
#: afterwards throws away nearly everything and Hypothesis rightly complains
#: that the surviving distribution no longer represents the domain.
_GEOD = Geod(ellps="WGS84")


@st.composite
def route(draw, min_nm: float = 100.0, max_nm: float = 6000.0):
    """A start point, a bearing and a distance. Valid by construction.

    Antipodal pairs are excluded by the distance bound rather than by
    filtering: they have infinitely many great circles between them and no
    aircraft flies one.
    """
    start_lat = draw(st.floats(min_value=-FLYABLE_LAT, max_value=FLYABLE_LAT,
                               allow_nan=False, allow_infinity=False))
    start_lon = draw(longitudes)
    bearing = draw(st.floats(min_value=0, max_value=360,
                             allow_nan=False, allow_infinity=False))
    distance_nm = draw(st.floats(min_value=min_nm, max_value=max_nm,
                                 allow_nan=False, allow_infinity=False))

    end_lon, end_lat, _ = _GEOD.fwd(start_lon, start_lat, bearing,
                                    distance_nm * 1852.0)
    # A bearing that runs over a pole produces a route no aircraft flies and
    # a projection nobody should trust. Nudged rather than discarded.
    assume(-FLYABLE_LAT <= end_lat <= FLYABLE_LAT)
    return (start_lat, start_lon), (end_lat, normalize_longitude(end_lon))


def shift_longitude(points, delta):
    """Slide a path east or west, wrapping at the antimeridian."""
    return [(lat, normalize_longitude(lon + delta)) for lat, lon in points]


def mirror_latitude(points):
    """Reflect a path across the equator."""
    return [(-lat, lon) for lat, lon in points]


# ------------------------------------------------------------ longitude


class TestLongitudeArithmetic:
    """Longitude is a circle, not a line. Every bug in this module so far
    has come from code that treated it as a line."""

    @given(st.floats(min_value=-1080, max_value=1080,
                     allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, deadline=None)
    def test_normalisation_always_lands_in_range(self, lon):
        result = normalize_longitude(lon)
        assert -180.0 <= result <= 180.0

    @given(st.floats(min_value=-180, max_value=180,
                     allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, deadline=None)
    def test_normalisation_is_idempotent(self, lon):
        once = normalize_longitude(lon)
        assert normalize_longitude(once) == pytest.approx(once, abs=1e-9)

    @given(st.lists(longitudes, min_size=2, max_size=40))
    @settings(max_examples=100, deadline=None)
    def test_unwrapping_never_leaves_a_discontinuity(self, lons):
        """The property the map renderer needs: a step near 360 degrees is
        drawn as a line back across the whole world."""
        unwrapped = unwrap_longitudes(lons)
        steps = [abs(b - a) for a, b in zip(unwrapped, unwrapped[1:])]
        assert all(step <= 180.0 + 1e-9 for step in steps)

    @given(st.lists(longitudes, min_size=1, max_size=40))
    @settings(max_examples=100, deadline=None)
    def test_unwrapping_preserves_position(self, lons):
        """Unwrapping may leave the legal range, but each value must still
        describe the same meridian.

        Compared as a position on a circle rather than as a number: +180 and
        -180 are the same place, and asserting numeric equality across that
        boundary fails on a difference of one ten-trillionth of a degree.
        """
        for original, unwrapped in zip(lons, unwrap_longitudes(lons)):
            separation = abs(unwrapped - original) % 360.0
            assert min(separation, 360.0 - separation) < 1e-6

    @given(st.lists(longitudes, min_size=2, max_size=20))
    @settings(max_examples=100, deadline=None)
    def test_unwrapping_is_idempotent(self, lons):
        once = unwrap_longitudes(lons)
        twice = unwrap_longitudes(once)
        assert once == pytest.approx(twice, abs=1e-9)


# ------------------------------------------------- metamorphic relations


class TestTranslationInvariance:
    """MR1. Sliding a route east or west changes nothing about its shape.

    This is the relation the antimeridian bug violated. The projection
    anchored on an average of raw longitudes, so a route spanning the date
    line anchored on the far side of the planet and every measurement that
    followed was wrong.
    """

    @given(route(), st.floats(min_value=-180, max_value=180,
                              allow_nan=False, allow_infinity=False))
    @GEO
    def test_length_survives_a_longitude_shift(self, pair, delta):
        a, b = pair
        original = great_circle(a, b, 16)
        shifted = shift_longitude(original, delta)
        assert path_length_nm(shifted) == pytest.approx(
            path_length_nm(original), rel=0.02)

    @given(route(max_nm=4000), st.sampled_from([-170, -90, -40, 40, 90, 170]))
    @GEO
    def test_area_survives_a_longitude_shift(self, pair, delta):
        a, b = pair
        original = build_corridor(great_circle(a, b, 16))
        shifted = build_corridor(shift_longitude(great_circle(a, b, 16), delta))
        assume(original.area_nm2() > 0)
        assert shifted.area_nm2() == pytest.approx(
            original.area_nm2(), rel=0.05)

    def test_the_specific_shift_that_broke(self):
        """A US route slid west until it straddles the date line. Before the
        fix this produced a corridor with 42% more area than the same shape
        in its original position."""
        pittsburgh_boston = great_circle((40.49, -80.23), (42.36, -71.01), 24)
        here = build_corridor(pittsburgh_boston)
        there = build_corridor(shift_longitude(pittsburgh_boston, -100))
        assert there.area_nm2() == pytest.approx(here.area_nm2(), rel=0.05)
        assert there.length_nm == pytest.approx(here.length_nm, rel=0.02)


class TestHemisphereSymmetry:
    """MR2. A route mirrored across the equator has the same shape.

    Every fixture in this project is in the northern hemisphere. This is the
    relation that would catch a sign error nobody has looked for.
    """

    @given(route(max_nm=4000))
    @GEO
    def test_length_survives_a_latitude_mirror(self, pair):
        a, b = pair
        original = great_circle(a, b, 16)
        assert path_length_nm(mirror_latitude(original)) == pytest.approx(
            path_length_nm(original), rel=0.02)

    @given(route(max_nm=4000))
    @GEO
    def test_area_survives_a_latitude_mirror(self, pair):
        a, b = pair
        original = build_corridor(great_circle(a, b, 16))
        assume(original.area_nm2() > 0)
        mirrored = build_corridor(mirror_latitude(great_circle(a, b, 16)))
        assert mirrored.area_nm2() == pytest.approx(
            original.area_nm2(), rel=0.05)

    @pytest.mark.parametrize("name,a,b", [
        ("Sydney to Auckland", (-33.95, 151.18), (-37.01, 174.79)),
        ("Sydney to Santiago", (-33.95, 151.18), (-33.39, -70.79)),
        ("Johannesburg to Perth", (-26.13, 28.24), (-31.94, 115.97)),
        ("Bogota to Santiago, crossing the equator",
         (4.70, -74.15), (-33.39, -70.79)),
        ("Singapore to Tokyo, crossing the equator",
         (1.36, 103.99), (35.55, 139.78)),
    ])
    def test_real_southern_and_equatorial_routes(self, name, a, b):
        """Named rather than generated, so a failure says which route."""
        points = great_circle(a, b, 24)
        corridor = build_corridor(points)
        expected = path_length_nm(points) * 50   # 25 nm each side, flat ends
        assert corridor.area_nm2() == pytest.approx(expected, rel=0.15), name
        assert corridor.contains(*points[len(points) // 2]), name


class TestReversalSymmetry:
    """MR3. A corridor is the same airspace whichever way you fly it."""

    @given(route(max_nm=4000))
    @GEO
    def test_length_survives_reversal(self, pair):
        a, b = pair
        forward = great_circle(a, b, 16)
        assert path_length_nm(forward[::-1]) == pytest.approx(
            path_length_nm(forward), rel=1e-6)

    @given(route(max_nm=4000))
    @GEO
    def test_area_survives_reversal(self, pair):
        a, b = pair
        forward = build_corridor(great_circle(a, b, 16))
        assume(forward.area_nm2() > 0)
        backward = build_corridor(great_circle(a, b, 16)[::-1])
        assert backward.area_nm2() == pytest.approx(
            forward.area_nm2(), rel=0.02)


class TestWidthMonotonicity:
    """MR4. A wider corridor covers more sky and contains the narrower one."""

    @given(route(max_nm=3000),
           st.floats(min_value=10, max_value=40),
           st.floats(min_value=5, max_value=30))
    @GEO
    def test_a_wider_corridor_has_more_area(self, pair, narrow, extra):
        a, b = pair
        points = great_circle(a, b, 16)
        thin = build_corridor(points, width_nm=narrow)
        thick = build_corridor(points, width_nm=narrow + extra)
        assume(thin.area_nm2() > 0)
        assert thick.area_nm2() > thin.area_nm2()

    @given(route(max_nm=3000))
    @GEO
    def test_a_wider_corridor_contains_the_narrower_one(self, pair):
        a, b = pair
        points = great_circle(a, b, 16)
        thin = build_corridor(points, width_nm=10)
        thick = build_corridor(points, width_nm=50)
        assume(thin.area_nm2() > 0)
        assert overlap_fraction(thin, thick) == pytest.approx(1.0, abs=0.02)


# ------------------------------------------------------------ invariants


class TestContainmentInvariants:
    """A corridor that does not contain its own centreline is not a
    corridor. This is the cheapest check in the file and it is the one that
    caught the antimeridian bug when everything else looked plausible."""

    @given(route(max_nm=5000))
    @GEO
    def test_every_interior_centreline_point_is_inside(self, pair):
        """Interior, not every: the two endpoints sit exactly on the
        boundary, and shapely's containment test excludes boundaries. See
        TestEndpointBoundary below."""
        a, b = pair
        points = great_circle(a, b, 16)
        corridor = build_corridor(points)
        assume(corridor.area_nm2() > 0)
        for point in points[1:-1]:
            assert corridor.contains(*point), (
                f"{point} is outside its own corridor")

    @given(route(min_nm=500, max_nm=5000))
    @GEO
    def test_a_point_on_the_far_side_of_the_planet_is_outside(self, pair):
        """The symptom of the antimeridian bug: a corridor stretching the
        long way round the world contained cities nowhere near the route."""
        a, b = pair
        corridor = build_corridor(great_circle(a, b, 16))
        assume(corridor.area_nm2() > 0)
        centre = midpoint(great_circle(a, b, 16))
        antipode = (-centre[0], normalize_longitude(centre[1] + 180))
        assert not corridor.contains(*antipode)


class TestAreaMatchesDimensions:
    """A path of length L buffered W nautical miles each side occupies about
    L x 2W, plus two end caps. An area far from that means the projection is
    distorting, which is what a wrongly anchored projection does."""

    @given(route(min_nm=200, max_nm=4000),
           st.floats(min_value=10, max_value=50))
    @GEO
    def test_area_is_length_times_twice_the_width(self, pair, width):
        """The buffer has flat ends rather than rounded ones, so there are
        no end caps and the area is exactly length by twice the width. That
        makes this a tight assertion rather than a loose one, and a tight
        assertion is what catches a projection drifting."""
        a, b = pair
        points = great_circle(a, b, 16)
        corridor = build_corridor(points, width_nm=width)
        expected = path_length_nm(points) * 2 * width
        assert corridor.area_nm2() == pytest.approx(expected, rel=0.10)


class TestOverlapInvariants:
    """Dominance pruning eliminates a corridor when a better-provenanced one
    covers 80% of the same airspace, so these are load-bearing."""

    @given(route(max_nm=4000))
    @GEO
    def test_a_corridor_completely_overlaps_itself(self, pair):
        a, b = pair
        corridor = build_corridor(great_circle(a, b, 16))
        assume(corridor.area_nm2() > 0)
        assert overlap_fraction(corridor, corridor) == pytest.approx(
            1.0, abs=1e-6)

    @given(route(min_nm=300, max_nm=900), route(min_nm=300, max_nm=900))
    @GEO
    def test_overlap_is_symmetric_at_comparable_scales(self, first, second):
        """The case a search actually produces.

        All four corridors in a search run between the same airports, so
        they are within a few percent of each other in length. Measured
        across forty random pairs at that scale the worst asymmetry is
        0.0005, so this assertion is tight on purpose: it is the regime the
        dominance rule operates in, and a 0.80 threshold deserves a
        symmetric input.
        """
        a = build_corridor(great_circle(*first, 16))
        b = build_corridor(great_circle(*second, 16))
        assume(a.area_nm2() > 0 and b.area_nm2() > 0)
        assert overlap_fraction(a, b) == pytest.approx(
            overlap_fraction(b, a), abs=0.005)

    @given(route(max_nm=4000), route(max_nm=4000))
    @GEO
    def test_overlap_is_roughly_symmetric_at_any_scale(self, first, second):
        """The general case, with a tolerance that reflects the method.

        Each corridor is projected into the other's local projection, and
        those differ, so a short corridor compared against a long one gives
        a slightly different answer depending on which is the reference.
        Measured worst case at a 25x length mismatch is 0.028; a search
        never produces that, but a caller might.
        """
        a = build_corridor(great_circle(*first, 16))
        b = build_corridor(great_circle(*second, 16))
        assume(a.area_nm2() > 0 and b.area_nm2() > 0)
        assert overlap_fraction(a, b) == pytest.approx(
            overlap_fraction(b, a), abs=0.05)

    def test_asymmetry_grows_with_the_scale_mismatch(self):
        """Documents where the approximation loses precision, so a future
        change that makes it worse is visible rather than absorbed."""
        from pyproj import Geod
        geod = Geod(ellps="WGS84")

        def corridor(lat, lon, bearing, nm):
            lon2, lat2, _ = geod.fwd(lon, lat, bearing, nm * 1852)
            return build_corridor(great_circle((lat, lon), (lat2, lon2), 16))

        same = corridor(40.0, -80.0, 90, 500)
        similar = corridor(40.2, -79.8, 80, 520)
        much_longer = corridor(40.2, -79.8, 80, 5000)

        comparable = abs(overlap_fraction(same, similar)
                         - overlap_fraction(similar, same))
        mismatched = abs(overlap_fraction(same, much_longer)
                         - overlap_fraction(much_longer, same))
        assert comparable < 0.01
        assert mismatched >= comparable

    @given(route(max_nm=4000), route(max_nm=4000))
    @GEO
    def test_overlap_is_a_fraction(self, first, second):
        a = build_corridor(great_circle(*first, 16))
        b = build_corridor(great_circle(*second, 16))
        assume(a.area_nm2() > 0 and b.area_nm2() > 0)
        assert 0.0 <= overlap_fraction(a, b) <= 1.0 + 1e-9

    def test_distant_corridors_do_not_overlap(self):
        atlantic = build_corridor(great_circle((40.6, -73.8), (51.5, -0.5), 24))
        pacific = build_corridor(great_circle((-33.9, 151.2), (-37.0, 174.8), 24))
        assert overlap_fraction(atlantic, pacific) == pytest.approx(0.0, abs=1e-6)


class TestGreatCircleIsTheFloor:
    """No path between two points is shorter than the geodesic. The critic
    rejects a corridor as physically implausible when it is, and that check
    is only as good as the geodesic it compares against."""

    @given(route(min_nm=200, max_nm=5000))
    @GEO
    def test_sampling_more_finely_converges_rather_than_shortens(self, pair):
        a, b = pair
        coarse = path_length_nm(great_circle(a, b, 8))
        fine = path_length_nm(great_circle(a, b, 64))
        assert fine >= coarse * 0.999
        assert fine == pytest.approx(coarse, rel=0.01)

    @given(route(min_nm=200, max_nm=5000))
    @GEO
    def test_a_detour_is_never_shorter_than_the_direct_path(self, pair):
        a, b = pair
        direct = path_length_nm(great_circle(a, b, 16))
        via = (max(-FLYABLE_LAT, min(FLYABLE_LAT, (a[0] + b[0]) / 2 + 10)),
               (a[1] + b[1]) / 2)
        detour = (path_length_nm(great_circle(a, via, 16))
                  + path_length_nm(great_circle(via, b, 16)))
        assert detour >= direct * 0.999


# ------------------------------------------------------------ degenerate


class TestDegenerateRoutes:
    """A search from an airport to itself is a plausible typo, and the
    geometry absorbs it badly: a zero-length path buffers to a polygon of no
    area whose containment test rejects its own defining point.

    The chosen fix was to refuse the search rather than to teach the buffer
    about degenerate paths, because there is no useful answer to give about
    a route with no length. These tests hold the geometry to what it
    actually does, so that the refusal upstream remains load-bearing rather
    than belt-and-braces - if the geometry is ever made to handle this case,
    the last test here fails and the guard can be revisited.

    The refusal itself is tested in test_api.py::TestSameAirportIsRejected.
    """

    IDENTICAL = ((40.0, -80.0), (40.0, -80.0))

    def test_a_zero_length_route_has_zero_length(self):
        points = great_circle(*self.IDENTICAL, 8)
        assert path_length_nm(points) == pytest.approx(0.0, abs=0.1)

    def test_the_geometry_still_cannot_represent_it(self):
        """Not a complaint: a corridor around a point is a circle, and a
        circle is not a corridor. The zero area is what makes the case
        detectable, and the API refuses it before anyone reaches here."""
        corridor = build_corridor(great_circle(*self.IDENTICAL, 8))
        assert corridor.area_nm2() == 0.0
        assert not corridor.contains(*self.IDENTICAL[0])

    def test_the_refusal_upstream_is_what_protects_this(self):
        """If the geometry ever learns to buffer a point into a circle,
        this fails and the upstream guard becomes optional rather than
        necessary. Worth being told."""
        corridor = build_corridor(great_circle(*self.IDENTICAL, 8))
        assert corridor.area_nm2() == 0.0, (
            "the degenerate case now produces a real polygon; the "
            "same-airport refusal in api.py can be reconsidered")


class TestEndpointBoundary:
    """The corridor buffer has flat ends, so an endpoint lies exactly on the
    boundary, and shapely's `contains` excludes boundaries while `covers`
    includes them.

    In practice a pilot report has to match an airport coordinate to the
    last floating-point digit to be affected, so nothing observed so far
    depends on it. It is recorded because a containment test that rejects
    the point defining the corridor is surprising, and surprising behaviour
    is worth knowing before it matters rather than after.
    """

    ROUTE = ((0.0, 0.0), (1.6748874816722792, 0.0))   # 100 nm due north

    def test_the_interior_is_contained(self):
        points = great_circle(*self.ROUTE, 16)
        corridor = build_corridor(points)
        assert all(corridor.contains(*p) for p in points[1:-1])

    def test_the_endpoints_are_on_the_boundary_not_inside_it(self):
        points = great_circle(*self.ROUTE, 16)
        corridor = build_corridor(points)
        assert not corridor.contains(*points[0])
        assert not corridor.contains(*points[-1])

    def test_a_point_just_inside_the_end_is_contained(self):
        """The exclusion is exact rather than approximate: a thousandth of a
        nautical mile along the route is inside."""
        points = great_circle(*self.ROUTE, 16)
        corridor = build_corridor(points)
        nudged = points[0][0] + 0.001 / 60.0
        assert corridor.contains(nudged, points[0][1])

    def test_the_area_confirms_the_ends_are_flat(self):
        """Rounded ends would add pi r squared. The area is exactly length
        by twice the width, so they are flat."""
        points = great_circle(*self.ROUTE, 16)
        corridor = build_corridor(points)
        assert corridor.area_nm2() == pytest.approx(
            path_length_nm(points) * 50, rel=0.01)


class TestKnownLimits:
    """Documented rather than asserted away. A test that pretends a limit
    does not exist is worse than no test."""

    def test_antipodal_routes_are_outside_the_projection_envelope(self):
        """Two points on opposite sides of the planet have infinitely many
        great circles between them, and an azimuthal projection anchored at
        their midpoint distorts by roughly a fifth at the edges. No
        scheduled flight is antipodal - the longest is around 9,000 nm - so
        this is a limit rather than a defect."""
        points = great_circle((0.0, 0.0), (0.0, 180.0), 24)
        corridor = build_corridor(points)
        expected = path_length_nm(points) * 50
        error = abs(corridor.area_nm2() - expected) / expected
        assert error > 0.10, (
            "antipodal distortion has improved; this limit can be revisited")

    def test_the_longest_real_route_is_within_tolerance(self):
        """Singapore to New York, about 8,300 nm, is the longest scheduled
        flight and it must be measured accurately."""
        points = great_circle((1.36, 103.99), (40.64, -73.78), 32)
        corridor = build_corridor(points)
        expected = path_length_nm(points) * 50
        assert corridor.area_nm2() == pytest.approx(expected, rel=0.15)
