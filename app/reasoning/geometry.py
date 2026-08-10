# install-to: app/reasoning
"""
Route geometry for corridor hypotheses.

Everything here is deterministic and offline. It answers the questions an
LLM cannot answer reliably: how long is this path, does it turn like a real
flight, which airspace does it occupy, and does a pilot report at these
coordinates and this altitude fall inside it.

PROJECTION. Buffering a path in degrees is wrong - a degree of longitude is
85 km at Pittsburgh and 0 at the pole - so each corridor builds a local
azimuthal equidistant projection centred on its own midpoint, does its work
in metres, and converts back. For a 500 nm route the residual distortion at
the endpoints is well under the corridor width.

CORRIDOR WIDTH. The default half-width is 25 nautical miles. A flight path
is a line but turbulence exposure is a volume, so the buffer decides which
pilot reports and which advisory polygons count as "along this route". 25 nm
is roughly the protected width of a jet airway plus a margin, and it matches
the scale at which a PIREP is meaningful - a single report is a point
observation of an air mass that is usually far larger.

The error is asymmetric and the default leans deliberately. Too narrow and a
report filed twenty miles off track by an aircraft in the same air mass is
missed. Too wide and conditions the flight never met are swept in, which for
a nervous passenger is the worse mistake: it manufactures turbulence that
was not there.

CONTAINMENT IS THREE-DIMENSIONAL. A G-AIRMET for turbulence between FL240
and FL390 does not apply to a flight cruising at FL410. Point-in-polygon
alone would say it does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pyproj import Geod, Transformer
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import transform as shapely_transform

NM_TO_M = 1852.0
M_TO_NM = 1.0 / NM_TO_M

#: Half-width of a corridor, each side of the path. See module docstring.
DEFAULT_WIDTH_NM = 25.0

#: Simplification tolerance for flown tracks. AeroAPI returns a position
#: roughly every 27 seconds - 154 points for a 69 minute flight - which is
#: far finer than a 25 nm corridor needs.
DEFAULT_SIMPLIFY_NM = 2.0

GEOD = Geod(ellps="WGS84")

LatLon = tuple[float, float]


# ------------------------------------------------------------------ basics


def great_circle(origin: LatLon, dest: LatLon, n_points: int = 24) -> list[LatLon]:
    """Points along the geodesic, endpoints included."""
    if n_points < 2:
        n_points = 2
    (lat1, lon1), (lat2, lon2) = origin, dest
    inter = GEOD.npts(lon1, lat1, lon2, lat2, n_points - 2)
    return [(lat1, lon1)] + [(lat, lon) for lon, lat in inter] + [(lat2, lon2)]


def path_length_nm(points: list[LatLon]) -> float:
    """Geodesic length of a polyline."""
    if len(points) < 2:
        return 0.0
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        _, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
        total += dist_m
    return round(total * M_TO_NM, 3)


def bearing(a: LatLon, b: LatLon) -> float:
    fwd, _, _ = GEOD.inv(a[1], a[0], b[1], b[0])
    return fwd % 360.0


#: Legs shorter than this are terminal manoeuvring, not enroute structure.
MIN_ENROUTE_LEG_NM = 25.0


def leg_length_nm(a: LatLon, b: LatLon) -> float:
    _, _, dist_m = GEOD.inv(a[1], a[0], b[1], b[0])
    return dist_m * M_TO_NM


def max_dogleg_deg(points: list[LatLon],
                   min_leg_nm: float = MIN_ENROUTE_LEG_NM) -> float:
    """Sharpest turn between consecutive *enroute* legs.

    A transport aircraft does not turn 120 degrees at cruise, so a large
    enroute dogleg means the path is mis-parsed or is not a flight path.

    Turns involving a short leg are excluded, and that exclusion is load
    bearing. Departure and arrival fixes sit close to their airports, so the
    first and last legs are often under 20 nm and turn sharply by design.
    Measured naively, a perfectly ordinary KPIT filed route shows a 74
    degree turn at EWC purely because EWC is 19 nm due north of the field.
    Scoring that as implausible would eliminate real routes.
    """
    if len(points) < 3:
        return 0.0
    worst = 0.0
    for a, b, c in zip(points, points[1:], points[2:]):
        if a == b or b == c:
            continue
        if leg_length_nm(a, b) < min_leg_nm or leg_length_nm(b, c) < min_leg_nm:
            continue
        turn = abs((bearing(b, c) - bearing(a, b) + 180.0) % 360.0 - 180.0)
        worst = max(worst, turn)
    return round(worst, 2)


def midpoint(points: list[LatLon]) -> LatLon:
    """Centre of the bounding extent, used to anchor the local projection."""
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    return ((min(lats) + max(lats)) / 2.0, (min(lons) + max(lons)) / 2.0)


# ------------------------------------------------------------------ corridor


@dataclass
class CorridorShape:
    """A flight path widened into the airspace it occupies.

    The polygon lives in a local metric projection; latitude/longitude only
    appear at the boundary of this class.
    """
    points: list[LatLon]
    width_nm: float = DEFAULT_WIDTH_NM
    altitude_min_ft: int | None = None
    altitude_max_ft: int | None = None
    polygon: Polygon = field(repr=False, default=None)
    _to_local: Transformer = field(repr=False, default=None)
    _to_wgs84: Transformer = field(repr=False, default=None)

    @property
    def length_nm(self) -> float:
        return path_length_nm(self.points)

    @property
    def max_dogleg(self) -> float:
        return max_dogleg_deg(self.points)

    def to_local(self, lat: float, lon: float) -> tuple[float, float]:
        return self._to_local.transform(lon, lat)

    def contains(self, lat: float, lon: float,
                 altitude_ft: int | None = None) -> bool:
        """Is this observation inside the corridor?

        Containment is three-dimensional when an altitude band is set on the
        corridor and an altitude is supplied. A report at FL410 is not inside
        a corridor that tops out at FL390, however well the coordinates line
        up.
        """
        x, y = self.to_local(lat, lon)
        if not self.polygon.contains(Point(x, y)):
            return False
        if altitude_ft is None:
            return True
        if self.altitude_min_ft is not None and altitude_ft < self.altitude_min_ft:
            return False
        if self.altitude_max_ft is not None and altitude_ft > self.altitude_max_ft:
            return False
        return True

    def boundary_latlon(self) -> list[LatLon]:
        """Corridor outline in latitude/longitude, for display."""
        ring = self.polygon.exterior.coords
        return [(lat, lon) for lon, lat in
                (self._to_wgs84.transform(x, y) for x, y in ring)]

    def area_nm2(self) -> float:
        return round(self.polygon.area * M_TO_NM * M_TO_NM, 2)


def _local_transformers(centre: LatLon) -> tuple[Transformer, Transformer]:
    """Azimuthal equidistant projection anchored on the corridor midpoint."""
    lat, lon = centre
    crs = (f"+proj=aeqd +lat_0={lat} +lon_0={lon} "
           f"+x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs")
    return (Transformer.from_crs("EPSG:4326", crs, always_xy=True),
            Transformer.from_crs(crs, "EPSG:4326", always_xy=True))


def build_corridor(points: list[LatLon],
                   width_nm: float = DEFAULT_WIDTH_NM,
                   altitude_min_ft: int | None = None,
                   altitude_max_ft: int | None = None,
                   simplify_nm: float | None = DEFAULT_SIMPLIFY_NM
                   ) -> CorridorShape:
    """Widen a path into a corridor polygon.

    `simplify_nm` thins dense flown tracks before buffering. It is a
    tolerance, not a sample rate: points are dropped only where doing so
    moves the path by less than the tolerance.
    """
    if len(points) < 2:
        raise ValueError("a corridor needs at least two points")

    to_local, to_wgs84 = _local_transformers(midpoint(points))
    local = [to_local.transform(lon, lat) for lat, lon in points]
    line = LineString(local)

    if simplify_nm and simplify_nm > 0:
        line = line.simplify(simplify_nm * NM_TO_M, preserve_topology=False)

    polygon = line.buffer(width_nm * NM_TO_M, cap_style=2, join_style=2)

    return CorridorShape(
        points=list(points),
        width_nm=width_nm,
        altitude_min_ft=altitude_min_ft,
        altitude_max_ft=altitude_max_ft,
        polygon=polygon,
        _to_local=to_local,
        _to_wgs84=to_wgs84,
    )


def simplify_track(points: list[LatLon],
                   tolerance_nm: float = DEFAULT_SIMPLIFY_NM) -> list[LatLon]:
    """Thin a dense flown track, keeping its shape."""
    if len(points) < 3:
        return list(points)
    to_local, to_wgs84 = _local_transformers(midpoint(points))
    local = LineString([to_local.transform(lon, lat) for lat, lon in points])
    thinned = local.simplify(tolerance_nm * NM_TO_M, preserve_topology=False)
    return [(lat, lon) for lon, lat in
            (to_wgs84.transform(x, y) for x, y in thinned.coords)]


# ------------------------------------------------------------------ overlap


def overlap_fraction(a: CorridorShape, b: CorridorShape) -> float:
    """How much airspace two corridors share, 0 to 1.

    Measured against the smaller corridor, so a short route lying entirely
    inside a longer one reports 1.0. This is what the critic's dominance
    rule needs: it asks whether a corridor is redundant, not whether two
    corridors are the same size.

    Both are compared in the first corridor's projection.
    """
    if a.polygon is None or b.polygon is None:
        return 0.0

    def reproject(x, y):
        lon, lat = b._to_wgs84.transform(x, y)
        return a._to_local.transform(lon, lat)

    b_in_a = shapely_transform(reproject, b.polygon)
    if not a.polygon.is_valid or not b_in_a.is_valid:
        return 0.0

    inter = a.polygon.intersection(b_in_a).area
    smaller = min(a.polygon.area, b_in_a.area)
    if smaller <= 0:
        return 0.0
    return round(min(1.0, inter / smaller), 4)


def corridor_overlap_fn(shapes: dict[str, CorridorShape]):
    """Adapter for the critic's `overlap_fn` slot.

    The critic works with Corridor objects and knows nothing about geometry;
    this closure maps their ids onto the shapes built here. Corridors with no
    shape return 0, so a missing geometry can never cause a prune.
    """
    def overlap(c_a, c_b) -> float:
        sa, sb = shapes.get(c_a.id), shapes.get(c_b.id)
        if sa is None or sb is None:
            return 0.0
        return overlap_fraction(sa, sb)
    return overlap
