# install-to: app/sources
"""
AeroAPI client, scoped to the four endpoints the corridor generator needs.

Every endpoint here was verified against a live Personal-tier key before
being written. Notes on what that tier actually does:

  - `/history/*` is gated to Standard and Premium. It returns 401 with the
    title "Invalid API key", so a tier rejection is indistinguishable from a
    bad key unless you look at the body. We do not use those endpoints;
    `/flights/{ident}?start=` reaches back far enough on Personal.
  - Bursts are throttled. A dozen calls in a few seconds draws 429s that
    clear when spaced, so calls are spaced by default and a 429 is retried
    once after a longer pause.
  - `/airports/{a}/flights/to/{b}` returns *itineraries*, not flights. Each
    entry is {"segments": [...]} so one-stop connections can be expressed.
    The flight objects are one level down.

UNITS. AeroAPI reports `route_distance` as a bare integer on a flight object
and as a string like "557 sm" on a routing. For a Pittsburgh to Boston pair
whose geodesic is 431 nm, the flight object reported 495 - which matches the
geodesic in *statute* miles, not nautical, despite a filed route necessarily
being longer than the geodesic. The unit is not reliably documented and the
two endpoints disagree with each other.

So this module never uses AeroAPI's distance for geometry. Path length is
computed from resolved fix coordinates with pyproj. The reported distance is
carried through as a cross-check and flagged when it disagrees, because an
unlabelled number in a geometry calculation is how a corridor ends up
quietly wrong.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

BASE_URL = "https://aeroapi.flightaware.com/aeroapi"
DEFAULT_SPACING_SECONDS = 1.5
RATE_LIMIT_BACKOFF_SECONDS = 5.0

SM_TO_NM = 0.868976


class AeroAPIError(RuntimeError):
    pass


class TierRestricted(AeroAPIError):
    """A 401 that is really a tier restriction rather than a bad key."""


class RateLimited(AeroAPIError):
    pass


#: (path, params) -> (status, parsed_json_or_None, raw_body)
Transport = Callable[[str, dict], tuple[int, dict | None, str]]


# ------------------------------------------------------------------ shapes


@dataclass(frozen=True)
class FlightSegment:
    ident: str
    fa_flight_id: str
    aircraft_type: str | None
    status: str | None
    actual_off: str | None
    scheduled_out: str | None
    route: str | None
    filed_altitude_ft: int | None
    reported_distance: float | None      # cross-check only, unit unreliable
    origin: str | None = None
    destination: str | None = None

    @property
    def has_flown(self) -> bool:
        return bool(self.actual_off)


@dataclass(frozen=True)
class Airport:
    """Enough of an airport to anchor a corridor."""
    code: str
    latitude: float
    longitude: float
    name: str | None = None

    def as_cache_row(self) -> dict:
        return {"name": self.code, "latitude": self.latitude,
                "longitude": self.longitude, "type": "Airport"}


@dataclass(frozen=True)
class RouteFix:
    name: str
    latitude: float
    longitude: float
    fix_type: str | None = None
    distance_from_origin: float | None = None

    def as_cache_row(self) -> dict:
        return {"name": self.name, "latitude": self.latitude,
                "longitude": self.longitude, "type": self.fix_type}


@dataclass(frozen=True)
class TrackPosition:
    latitude: float
    longitude: float
    altitude_ft: int | None
    timestamp: str | None
    groundspeed: int | None = None
    update_type: str | None = None


@dataclass(frozen=True)
class AlternateRouting:
    route: str
    count: int
    filed_altitude_min_ft: int | None
    filed_altitude_max_ft: int | None
    reported_distance_nm: float | None   # normalised from "NNN sm"
    last_departure_time: str | None = None


def parse_distance_to_nm(value) -> float | None:
    """Normalise AeroAPI's distance field to nautical miles.

    Handles the string form ("557 sm", "480 nm") from the routings endpoint.
    A bare number is returned unchanged and should be treated as unverified -
    see the module docstring.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.match(r"^\s*([\d.]+)\s*(sm|nm|mi|km)?\s*$", str(value), re.I)
    if not m:
        return None
    n = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit in ("sm", "mi"):
        return round(n * SM_TO_NM, 2)
    if unit == "km":
        return round(n * 0.539957, 2)
    return n


# ------------------------------------------------------------------ client


@dataclass
class AeroAPIClient:
    #: Optional timing sink. Set by the service so a search can report where
    #: its seconds went; None everywhere else so tests and scripts are
    #: unaffected.
    """Thin wrapper. Counts its own calls so a caller can budget them.

    `transport` is injectable so tests never touch the network.
    """
    api_key: str = ""
    base_url: str = BASE_URL
    spacing_seconds: float = DEFAULT_SPACING_SECONDS
    transport: Transport | None = None
    sleep: Callable[[float], None] = time.sleep
    calls_made: int = 0
    timings: object | None = None
    call_log: list[str] = field(default_factory=list)

    # -------------------------------------------------------------- request

    def _http(self, path: str, params: dict) -> tuple[int, dict | None, str]:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={
            "x-apikey": self.api_key,
            "Accept": "application/json; charset=UTF-8",
        })
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
                return resp.status, json.loads(raw), raw
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            return e.code, None, raw

    def _timed(self, fn):
        """Run a call, recording its wall time against the AeroAPI bucket.

        Wall clock rather than CPU: what an external call costs a search is
        the waiting, and that is the number worth putting on a chart.
        """
        if self.timings is None:
            return fn()
        with self.timings.track("aeroapi"):
            return fn()

    def request(self, path: str, params: dict | None = None,
                _retried: bool = False) -> dict:
        params = params or {}
        transport = self.transport or self._http

        self.calls_made += 1
        self.call_log.append(path)
        # Timed here rather than around the whole method: retries and the
        # inter-call spacing sleep are part of what a search waits for, and
        # both happen below this line.
        status, body, raw = self._timed(lambda: transport(path, params))

        if status == 200 and body is not None:
            if self.spacing_seconds:
                self._timed(lambda: self.sleep(self.spacing_seconds))
            return body

        if status == 429:
            if _retried:
                raise RateLimited(f"rate limited twice on {path}")
            self._timed(lambda: self.sleep(RATE_LIMIT_BACKOFF_SECONDS))
            return self.request(path, params, _retried=True)

        if status == 401:
            # AeroAPI uses 401 for tier restrictions as well as bad keys.
            if "tier" in raw.lower() or "standard" in raw.lower():
                raise TierRestricted(
                    f"{path} is not available on this tier: {raw[:200]}")
            raise AeroAPIError(f"401 on {path} - key rejected: {raw[:200]}")

        raise AeroAPIError(f"HTTP {status} on {path}: {raw[:200]}")

    # -------------------------------------------------------------- endpoints

    def flights_between(self, origin: str, dest: str,
                        max_pages: int = 1,
                        nonstop_only: bool = True) -> list[FlightSegment]:
        """Flights on an airport pair, flattened out of their itineraries.

        NONSTOP ONLY, BY DEFAULT. This endpoint returns itineraries, and an
        itinerary can be a connection: a San Diego to Tokyo query comes back
        as KSAN to KLAX followed by KLAX to RJTT. Flattening every segment
        into one pool and picking by departure time means the reference
        flight can be a regional feeder leg on a different aircraft between
        two different airports, which is exactly what happened - a Dash 8
        turboprop was returned as the reference for Seattle to Tokyo.

        Filtering to segments that actually fly the requested pair means a
        pair with no nonstop service returns nothing, which is the honest
        answer. The caller reports that rather than describing someone
        else's flight.
        """
        body = self.request(f"/airports/{origin}/flights/to/{dest}",
                            {"max_pages": max_pages})
        origin, dest = origin.upper(), dest.upper()
        out: list[FlightSegment] = []
        for itinerary in body.get("flights") or []:
            for seg in itinerary.get("segments") or []:
                if not seg.get("fa_flight_id"):
                    continue
                if nonstop_only:
                    seg_origin = (seg.get("origin") or {}).get("code", "")
                    seg_dest = (seg.get("destination") or {}).get("code", "")
                    if (seg_origin or "").upper() != origin:
                        continue
                    if (seg_dest or "").upper() != dest:
                        continue
                out.append(FlightSegment(
                    ident=seg.get("ident") or "",
                    fa_flight_id=seg["fa_flight_id"],
                    aircraft_type=seg.get("aircraft_type"),
                    status=seg.get("status"),
                    actual_off=seg.get("actual_off"),
                    scheduled_out=seg.get("scheduled_out"),
                    route=seg.get("route"),
                    filed_altitude_ft=(seg.get("filed_altitude") or 0) * 100
                                      if seg.get("filed_altitude") else None,
                    reported_distance=parse_distance_to_nm(
                        seg.get("route_distance")),
                    origin=(seg.get("origin") or {}).get("code"),
                    destination=(seg.get("destination") or {}).get("code"),
                ))
        return out

    def most_recently_flown(self, origin: str, dest: str
                            ) -> FlightSegment | None:
        """The latest departed flight on the pair.

        A flown track only exists for a flight that has actually departed;
        scheduled flights have no positions.
        """
        flown = [s for s in self.flights_between(origin, dest) if s.has_flown]
        if not flown:
            return None
        flown.sort(key=lambda s: s.actual_off or "", reverse=True)
        return flown[0]

    def route_fixes(self, fa_flight_id: str) -> list[RouteFix]:
        """Filed route with coordinates. Also warms the fix cache."""
        body = self.request(f"/flights/{fa_flight_id}/route")
        out: list[RouteFix] = []
        for f in body.get("fixes") or []:
            lat, lon = f.get("latitude"), f.get("longitude")
            if lat is None or lon is None:
                continue
            out.append(RouteFix(
                name=(f.get("name") or "").strip().upper(),
                latitude=float(lat), longitude=float(lon),
                fix_type=f.get("type"),
                distance_from_origin=f.get("distance_from_origin"),
            ))
        return out

    def track(self, fa_flight_id: str,
              include_estimated: bool = True) -> list[TrackPosition]:
        """Flown positions. Roughly one every 27 seconds on a live ADS-B leg."""
        body = self.request(f"/flights/{fa_flight_id}/track",
                            {"include_estimated_positions":
                             "true" if include_estimated else "false"})
        out: list[TrackPosition] = []
        for p in body.get("positions") or []:
            lat, lon = p.get("latitude"), p.get("longitude")
            if lat is None or lon is None:
                continue
            alt = p.get("altitude")
            out.append(TrackPosition(
                latitude=float(lat), longitude=float(lon),
                # AeroAPI reports altitude in hundreds of feet.
                altitude_ft=int(alt) * 100 if alt is not None else None,
                timestamp=p.get("timestamp"),
                groundspeed=p.get("groundspeed"),
                update_type=p.get("update_type"),
            ))
        return out

    def airport(self, code: str) -> Airport | None:
        """Look up an airport's position directly.

        The fix cache normally learns airports from filed routes, since
        AeroAPI returns the origin and destination as route fixes. That
        breaks on a pair nobody has filed a usable route for, which leaves
        even the great-circle corridor unbuildable. This is the fallback
        that makes the geometric source genuinely unconditional.
        """
        code = (code or "").strip().upper()
        if not code:
            return None
        body = self.request(f"/airports/{code}")
        lat, lon = body.get("latitude"), body.get("longitude")
        if lat is None or lon is None:
            return None
        return Airport(code=code, latitude=float(lat), longitude=float(lon),
                       name=body.get("name"))

    def alternate_routings(self, origin: str, dest: str
                           ) -> list[AlternateRouting]:
        """Routings actually filed on this pair, most-used first.

        No coordinates - these are route strings. They resolve through the
        fix cache, which the route_fixes call populates.
        """
        body = self.request(f"/airports/{origin}/routes/{dest}")
        out: list[AlternateRouting] = []
        for r in body.get("routes") or []:
            if not r.get("route"):
                continue
            out.append(AlternateRouting(
                route=r["route"],
                count=int(r.get("count") or 0),
                filed_altitude_min_ft=(r.get("filed_altitude_min") or 0) * 100
                                      if r.get("filed_altitude_min") else None,
                filed_altitude_max_ft=(r.get("filed_altitude_max") or 0) * 100
                                      if r.get("filed_altitude_max") else None,
                reported_distance_nm=parse_distance_to_nm(
                    r.get("route_distance")),
                last_departure_time=r.get("last_departure_time"),
            ))
        out.sort(key=lambda r: -r.count)
        return out
