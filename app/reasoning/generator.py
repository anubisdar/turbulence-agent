# install-to: app/reasoning
"""
Corridor hypothesis generator - the thought generator in the ToT split.

Given a trip, produces the candidate corridors the critic scores and the
controller prunes. Four sources at depth 1, chosen because they fail in
different ways rather than because there are four of them:

    flown track       where the aircraft actually went last time
    filed route       where this flight said it would go
    alternate routing the other corridor traffic on this pair commonly files
    great circle      the geometric floor, always available, always weakest

The first three cost API calls. The fourth costs nothing and exists so the
search can never return empty - a corridor derived from geometry is a poor
hypothesis but it is an honest one, and the critic already scores it last.

WHY THE ORDER OF CALLS MATTERS. The filed-route call returns named fixes
*with* coordinates, which is the only way coordinates enter this system.
Alternate routings arrive as bare strings. So the filed route is fetched
first, its fixes are cached, and the alternate routing is then resolved
against a cache that the previous call just warmed. On a repeated route the
alternate resolves for free.

EVIDENCE IS NOT ATTACHED HERE. Corridors come back with empty Evidence:
coverage and agreement are None, the reading is UNRESOLVED. Gathering pilot
reports and advisories along a corridor is a separate step against the
Aviation Weather Center. Until that exists, the critic scores on provenance
and geometry, which is the behaviour it was written for - absent evidence
scores neutral, never zero, and never prunes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

from app.reasoning.controller import Budget
from app.reasoning.critic import (
    Corridor,
    Evidence,
    Geometry,
    Provenance,
    Severity,
)
from app.reasoning.evidence import GatherResult, gather_evidence
from app.reasoning.geometry import (
    DEFAULT_WIDTH_NM,
    CorridorShape,
    build_corridor,
    corridor_overlap_fn,
    great_circle,
    max_dogleg_deg,
    path_length_nm,
    simplify_track,
)
from app.sources.aeroapi import AeroAPIClient, AeroAPIError, FlightSegment
from app.sources.fixes import resolve_route, upsert_fixes

#: A flown track includes taxi, climb and descent. Turbulence exposure that
#: matters to a passenger is overwhelmingly at cruise, and a band derived
#: from every position runs from the ground upward - which would match
#: low-level advisories that have nothing to do with the cruise segment.
#: Positions within this much of the maximum altitude are treated as cruise.
CRUISE_BAND_FT = 4000

#: Below this, a track has no meaningful cruise segment to band.
MIN_CRUISE_ALTITUDE_FT = 10000

#: Depth beyond which this generator has nothing further to offer. Depth 3
#: was reserved for data-gap strategies, which need turbulence evidence to
#: choose between - so it stays unused until the weather layer lands, rather
#: than inventing branches that differ only cosmetically.
MAX_USEFUL_DEPTH = 2

LatLon = tuple[float, float]


@dataclass
class CorridorGenerator:
    """Callable matching the controller's Generator protocol.

    Holds the geometry it builds so the critic's dominance rule can compare
    corridors by shared airspace: see `overlap_fn`.
    """
    client: AeroAPIClient
    conn: object                      # sqlite3.Connection with route_fixes
    origin: str
    dest: str
    origin_latlon: LatLon | None = None
    dest_latlon: LatLon | None = None
    width_nm: float = DEFAULT_WIDTH_NM
    #: Time of day the passenger is flying, as "HH:MM" UTC. A 07:00 and a
    #: 19:00 departure on the same route fly different air, so the reference
    #: flight is chosen by time of day rather than simply by recency.
    target_time: str | None = None

    #: Turbulence sources. Both optional: without them the search still runs
    #: and every corridor reports its reading as unresolved, which is the
    #: honest answer rather than a silent assumption of calm.
    fetch_pireps: object | None = None
    gairmet_client: object | None = None
    when: object | None = None

    shapes: dict[str, CorridorShape] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    _flight: FlightSegment | None = field(default=None, repr=False)
    _looked_for_flight: bool = field(default=False, repr=False)
    _segments: list = field(default_factory=list, repr=False)
    _pair_altitude_band: tuple[int, int] | None = field(default=None, repr=False)
    #: Evidence keyed by corridor id, kept so the interface can show which
    #: reports and forecasts produced a reading.
    evidence: dict[str, GatherResult] = field(default_factory=dict)

    # ------------------------------------------------------------ helpers

    @property
    def overlap_fn(self):
        """Adapter for the critic's dominance check."""
        return corridor_overlap_fn(self.shapes)

    def _note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def _endpoints(self, budget: Budget | None = None
                   ) -> tuple[LatLon, LatLon] | None:
        """Airport coordinates, from the constructor, the cache, or a lookup.

        The cache normally learns airports from filed routes, since AeroAPI
        returns the origin and destination as route fixes. That fails on a
        pair where no usable route comes back - a wrong reference flight, an
        unparseable route string, an airport nobody files through - and it
        took the great-circle corridor down with it. The geometric source
        needs no external data to compute, so it should not be the source
        that fails first.

        The direct lookup costs one call per unknown airport, once, and the
        result is cached permanently.
        """
        if self.origin_latlon and self.dest_latlon:
            return self.origin_latlon, self.dest_latlon

        found, missing = _lookup(self.conn, [self.origin, self.dest])
        if not missing:
            return found[self.origin], found[self.dest]

        if budget is None:
            return None

        for code in list(missing):
            if not budget.spend():
                self._note(
                    f"Tool budget exhausted before {code} could be located, "
                    f"so no corridor could be built."
                )
                return None
            try:
                airport = self.client.airport(code)
            except AeroAPIError as e:
                self._note(f"Could not look up {code}: {e}")
                return None
            if airport is None:
                self._note(
                    f"{code} is not an airport AeroAPI recognises, so no "
                    f"corridor could be built for this trip."
                )
                return None
            upsert_fixes(self.conn, [airport.as_cache_row()],
                         source="AeroAPI /airports")
            self._note(f"Located {code} directly and cached it.")

        found, still_missing = _lookup(self.conn, [self.origin, self.dest])
        if still_missing:
            return None
        return found[self.origin], found[self.dest]

    def _register(self, cid: str, points: list[LatLon],
                  altitude_min: int | None = None,
                  altitude_max: int | None = None) -> CorridorShape | None:
        try:
            shape = build_corridor(points, width_nm=self.width_nm,
                                   altitude_min_ft=altitude_min,
                                   altitude_max_ft=altitude_max)
        except ValueError:
            return None
        self.shapes[cid] = shape
        return shape

    def _corridor(self, cid: str, provenance: Provenance,
                  points: list[LatLon], gc_nm: float,
                  altitude_min: int | None = None,
                  altitude_max: int | None = None,
                  depth: int = 1, parent_id: str | None = None,
                  label: str = "") -> Corridor | None:
        shape = self._register(cid, points, altitude_min, altitude_max)
        if shape is None:
            return None
        return Corridor(
            id=cid,
            provenance=provenance,
            geometry=Geometry(
                length_nm=path_length_nm(points),
                great_circle_nm=gc_nm,
                max_dogleg_deg=max_dogleg_deg(points),
                endpoints_match_airports=True,
                altitude_profile_valid=True,
            ),
            evidence=Evidence(reading=Severity.UNRESOLVED),
            depth=depth,
            parent_id=parent_id,
            label=label,
        )

    def _get_flight(self, budget: Budget) -> FlightSegment | None:
        """The most recently departed flight on this pair. One call, reused."""
        if self._looked_for_flight:
            return self._flight
        self._looked_for_flight = True
        if not budget.spend():
            self._note("Tool budget exhausted before a flight could be found.")
            return None
        try:
            self._segments = self.client.flights_between(self.origin, self.dest)
        except AeroAPIError as e:
            self._note(f"Could not list flights on this pair: {e}")
            return None
        flown = [s for s in self._segments if s.has_flown]
        self._flight = _pick_reference(flown, self.target_time)
        if self._flight and self.target_time:
            self._note(
                f"Reference flight {self._flight.ident} chosen for departing "
                f"nearest {self.target_time} UTC, not for being the most "
                f"recent. Morning and evening departures fly different air."
            )
        if self._flight is None:
            if not self._segments:
                # Distinct from "nothing has departed lately". A pair with no
                # nonstop service has no route to describe at all, and the
                # great circle that follows is a geometric line rather than a
                # path anyone flies.
                self._note(
                    f"No nonstop flights operate between {self.origin} and "
                    f"{self.dest}. Every itinerary on this pair connects "
                    f"through another airport, so there is no single flown "
                    f"route to examine. Any corridor shown is the geometric "
                    f"shortest path, not a route an aircraft takes."
                )
            else:
                self._note(
                    f"No nonstop flight on {self.origin}-{self.dest} has "
                    f"departed within the available window, so no flown track "
                    f"or filed route is available. Corridors from those "
                    f"sources are absent, which is not the same as their "
                    f"being smooth."
                )
        return self._flight

    # ------------------------------------------------------------ depth 1

    def _great_circle_corridor(self, gc_nm: float,
                               points: list[LatLon]) -> Corridor | None:
        return self._corridor("gc", Provenance.GREAT_CIRCLE, points, gc_nm,
                              label="great circle")

    def _flown_track_corridor(self, budget: Budget,
                              gc_nm: float) -> Corridor | None:
        flight = self._get_flight(budget)
        if flight is None:
            return None
        if not budget.spend():
            self._note("Tool budget exhausted before the flown track was fetched.")
            return None
        try:
            positions = self.client.track(flight.fa_flight_id)
        except AeroAPIError as e:
            self._note(f"Could not fetch the flown track: {e}")
            return None
        if len(positions) < 2:
            self._note("The flown track returned too few positions to use.")
            return None

        points = simplify_track([(p.latitude, p.longitude) for p in positions])
        self._note(
            f"Flown track from {flight.ident} departing {flight.actual_off}: "
            f"{len(positions)} positions thinned to {len(points)}."
        )

        band = cruise_band(
            [p.altitude_ft for p in positions if p.altitude_ft is not None])
        if band:
            self._note(
                f"Cruise band from the flown track: "
                f"FL{band[0] // 100:03d} to FL{band[1] // 100:03d}. Taxi, "
                f"climb and descent positions are excluded - laterally the "
                f"corridor covers the whole route, but vertically it covers "
                f"the cruise segment."
            )
        else:
            self._note(
                "The flown track reached no identifiable cruise altitude, so "
                "the corridor carries no altitude band."
            )

        return self._corridor(
            "track", Provenance.ACTUAL_TRACK, points, gc_nm,
            altitude_min=band[0] if band else None,
            altitude_max=band[1] if band else None,
            label=f"flown track {flight.ident}",
        )

    def _fetch_filed_route(self, budget: Budget) -> list | None:
        """Fetch and cache the filed route. Run first: this is the only call
        that brings coordinates into the system, including the airports."""
        flight = self._get_flight(budget)
        if flight is None:
            return None
        if not budget.spend():
            self._note("Tool budget exhausted before the filed route was fetched.")
            return None
        try:
            fixes = self.client.route_fixes(flight.fa_flight_id)
        except AeroAPIError as e:
            self._note(f"Could not fetch the filed route: {e}")
            return None
        if len(fixes) < 2:
            self._note("The filed route returned too few fixes to use.")
            return None
        stored = upsert_fixes(self.conn, [f.as_cache_row() for f in fixes])
        self._note(f"Cached {stored} route fix(es) from {flight.ident}.")
        return fixes

    def _filed_corridor(self, fixes: list, gc_nm: float) -> Corridor | None:
        flight = self._flight
        points = [(f.latitude, f.longitude) for f in fixes]
        return self._corridor(
            "filed", Provenance.FILED_ROUTE, points, gc_nm,
            altitude_min=flight.filed_altitude_ft if flight else None,
            altitude_max=flight.filed_altitude_ft if flight else None,
            label=f"filed route {flight.ident}" if flight else "filed route",
        )

    def _alternate_routing_corridor(self, budget: Budget,
                                    gc_nm: float) -> Corridor | None:
        if not budget.spend():
            self._note("Tool budget exhausted before alternate routings were fetched.")
            return None
        try:
            routings = self.client.alternate_routings(self.origin, self.dest)
        except AeroAPIError as e:
            self._note(f"Could not fetch alternate routings: {e}")
            return None
        if not routings:
            return None

        filed = (self._flight.route if self._flight else None) or ""
        # The most-filed routing is usually the one this flight filed. Take
        # the most popular one that differs, so the branch is a genuine
        # alternative rather than a duplicate the critic would prune anyway.
        candidates = [r for r in routings if r.route.strip() != filed.strip()]
        if not candidates:
            self._note("Every filed routing on this pair matches this flight's.")
            return None

        # Record the altitude spread filed across this pair. A single flight
        # files one level; the pair shows the range real traffic uses, which
        # is what depth 2 branches on.
        lows = [r.filed_altitude_min_ft for r in routings if r.filed_altitude_min_ft]
        highs = [r.filed_altitude_max_ft for r in routings if r.filed_altitude_max_ft]
        if lows and highs:
            self._pair_altitude_band = (min(lows), max(highs))

        for routing in candidates:
            resolution = resolve_route(self.conn, routing.route)

            # Cold cache: fetch the fixes from a flight that filed this exact
            # routing. One call, and it permanently covers those waypoints.
            if not resolution.resolved and resolution.missing:
                donor = next((s for s in self._segments
                              if (s.route or "").strip() == routing.route.strip()),
                             None)
                if donor and budget.spend():
                    try:
                        fixes = self.client.route_fixes(donor.fa_flight_id)
                        stored = upsert_fixes(
                            self.conn, [f.as_cache_row() for f in fixes])
                        self._note(
                            f"Cached {stored} fix(es) from {donor.ident}, which "
                            f"filed the alternate routing.")
                        resolution = resolve_route(self.conn, routing.route)
                    except AeroAPIError as e:
                        self._note(f"Could not resolve the alternate routing: {e}")

            if resolution.resolved:
                for n in resolution.notes():
                    self._note(n)
                points = [(lat, lon) for _, lat, lon in resolution.points]

                # A filed route string names enroute fixes, not the airports:
                # `TYROO PSB J49 HNK PONCT JFUND2` starts 35 nm from the
                # field. Building a corridor straight from that gives a path
                # shorter than the great circle, which the critic correctly
                # rejects as impossible. Anchor it at both airports.
                endpoints = self._endpoints()
                if endpoints:
                    origin_ll, dest_ll = endpoints
                    if points[0] != origin_ll:
                        points.insert(0, origin_ll)
                    if points[-1] != dest_ll:
                        points.append(dest_ll)
                self._note(
                    f"Alternate routing filed by {routing.count} flight(s): "
                    f"{routing.route}"
                )
                return self._corridor(
                    "alternate", Provenance.PUBLISHED_AIRWAY, points, gc_nm,
                    altitude_min=routing.filed_altitude_min_ft,
                    altitude_max=routing.filed_altitude_max_ft,
                    label=f"alternate routing ({routing.count} filings)",
                )
            for n in resolution.notes():
                self._note(n)

        self._note(
            "No alternate routing could be resolved: the fix cache does not "
            "yet cover their waypoints. The gap is reported rather than "
            "bridged with a straight line."
        )
        return None

    # ------------------------------------------------------------ depth 2

    def _gather_for(self, corridor: Corridor, budget: Budget) -> Corridor:
        """Attach turbulence evidence to one corridor.

        Called on survivors rather than on every candidate. Gathering for a
        corridor that is about to be pruned by dominance would spend a call
        on an answer nobody reads, and depth-1 pruning is decided by
        provenance and geometry regardless.
        """
        shape = self.shapes.get(corridor.id)
        if shape is None:
            return corridor
        if corridor.id in self.evidence:
            return corridor

        if not (self.fetch_pireps or self.gairmet_client):
            return corridor

        if not budget.spend():
            self._note(
                f"Tool budget exhausted before turbulence evidence could be "
                f"gathered for {corridor.id}. Its reading stays unresolved, "
                f"which is not the same as smooth."
            )
            return corridor

        result = gather_evidence(
            shape,
            fetch_pireps=self.fetch_pireps,
            gairmet_client=self.gairmet_client,
            when=self.when,
        )
        self.evidence[corridor.id] = result
        for n in result.notes:
            self._note(f"{corridor.id}: {n}")

        return replace(corridor, evidence=result.evidence)

    def gather_for_survivors(self, survivors: Sequence[Corridor],
                             budget: Budget) -> list[Corridor]:
        """Attach evidence to the corridors that made it through the beam."""
        return [self._gather_for(c, budget) for c in survivors]

    def _altitude_branches(self, parent: Corridor) -> list[Corridor]:
        """Branch a surviving corridor on the altitudes actually filed.

        Real filed altitudes, not invented profiles. A corridor with no
        altitude information, or with a single filed level, produces no
        branch - there is nothing to choose between.
        """
        shape = self.shapes.get(parent.id)
        if shape is None:
            return []
        lo, hi = shape.altitude_min_ft, shape.altitude_max_ft
        source = "this corridor"
        if (lo is None or hi is None or hi - lo < 2000) and self._pair_altitude_band:
            # One flight files one level, so a single corridor offers nothing
            # to choose between. The spread real traffic files on this pair is
            # a genuine choice: it is the range the planned flight is likely
            # to be assigned, not an invented profile.
            lo, hi = self._pair_altitude_band
            source = "altitudes filed across this pair"
        if lo is None or hi is None or hi - lo < 2000:
            return []
        self._note(
            f"Altitude branches for {parent.id} taken from {source}: "
            f"FL{lo // 100:03d} to FL{hi // 100:03d}."
        )

        out: list[Corridor] = []
        for suffix, band, label in (
            ("low", (lo, lo + 2000), f"low band FL{lo // 100:03d}"),
            ("high", (hi - 2000, hi), f"high band FL{hi // 100:03d}"),
        ):
            cid = f"{parent.id}/{suffix}"
            child = self._corridor(
                cid, parent.provenance, list(shape.points),
                parent.geometry.great_circle_nm,
                altitude_min=band[0], altitude_max=band[1],
                depth=parent.depth + 1, parent_id=parent.id, label=label,
            )
            if child:
                out.append(child)
        return out

    # ------------------------------------------------------------ protocol

    def __call__(self, parent: Corridor | None, depth: int,
                 budget: Budget) -> Sequence[Corridor]:
        if depth > MAX_USEFUL_DEPTH:
            return []

        if depth == 1:
            # The filed route is fetched first for two reasons: it caches the
            # route fixes the alternate routing needs, and it caches the
            # origin and destination airports, which is how the great-circle
            # baseline becomes computable. Every corridor is scored against
            # that baseline, so it has to exist before any corridor is built.
            filed_fixes = self._fetch_filed_route(budget)

            endpoints = self._endpoints(budget)
            if endpoints is None:
                self._note(
                    f"Airport coordinates for {self.origin}/{self.dest} could "
                    f"not be established, so no corridor could be built or "
                    f"scored against a great-circle baseline. No route was "
                    f"established, so no turbulence conclusion follows from "
                    f"the absence of one."
                )
                return []

            gc_points = great_circle(endpoints[0], endpoints[1], 24)
            gc_nm = path_length_nm(gc_points)

            out: list[Corridor] = []
            if filed_fixes:
                c = self._filed_corridor(filed_fixes, gc_nm)
                if c:
                    out.append(c)
            for maker in (
                lambda: self._flown_track_corridor(budget, gc_nm),
                lambda: self._alternate_routing_corridor(budget, gc_nm),
                lambda: self._great_circle_corridor(gc_nm, gc_points),
            ):
                c = maker()
                if c:
                    out.append(c)

            if not out:
                self._note(
                    "No corridor hypothesis could be generated for this trip. "
                    "No route was established, so no turbulence conclusion "
                    "follows from the absence of one."
                )
            return out

        if parent is None:
            return []
        return self._altitude_branches(parent)


def _minutes(stamp: str | None) -> int | None:
    """Minutes past midnight UTC from an ISO timestamp."""
    if not stamp or "T" not in stamp:
        return None
    try:
        hh, mm = stamp.split("T", 1)[1][:5].split(":")
        return int(hh) * 60 + int(mm)
    except (ValueError, IndexError):
        return None


def _pick_reference(flown: list, target_time: str | None):
    """The flight whose corridor best represents the trip being planned.

    Without a target time, the most recent departure. With one, the flight
    departing nearest that time of day - wrapping across midnight, so 23:50
    is twenty minutes from 00:10 rather than twenty-three hours.
    """
    if not flown:
        return None
    if not target_time:
        return max(flown, key=lambda s: s.actual_off or "")

    want = _minutes(f"T{target_time}") if ":" in target_time else None
    if want is None:
        return max(flown, key=lambda s: s.actual_off or "")

    def distance(seg):
        got = _minutes(seg.actual_off) or _minutes(seg.scheduled_out)
        if got is None:
            return 10_000
        gap = abs(got - want)
        return min(gap, 1440 - gap)

    return min(flown, key=lambda s: (distance(s), -(_minutes(s.actual_off) or 0)))


def cruise_band(altitudes: list[int]) -> tuple[int, int] | None:
    """The cruise portion of a flown track's altitude profile.

    A track runs from the ground up, so min and max across all positions
    gives a band starting below zero. Only positions near the top of climb
    describe where the aircraft actually spent its cruise.
    """
    usable = [a for a in altitudes if a is not None and a >= MIN_CRUISE_ALTITUDE_FT]
    if not usable:
        return None
    top = max(usable)
    cruise = [a for a in usable if a >= top - CRUISE_BAND_FT]
    return (min(cruise), top)


def _lookup(conn, names):
    from app.sources.fixes import lookup
    return lookup(conn, names)
