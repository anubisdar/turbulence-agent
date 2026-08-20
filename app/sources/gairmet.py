# install-to: app/sources
"""
G-AIRMET turbulence forecasts from the Aviation Weather Center.

A G-AIRMET is a forecast polygon: a shape on a map, an altitude band, and a
validity window, saying where bumpy air is expected. It is the forecast half
of the turbulence picture. PIREPs, handled in `awc.py`, are the observed
half, and the two are deliberately kept apart all the way to the critic so a
disagreement between them surfaces instead of being averaged away.

Three things about this endpoint were found by probing it rather than by
reading about it, and each one would have caused a quiet bug:

  THE `type` PARAMETER IS IGNORED. Requesting `type=turb-hi` returns the
  same payload as requesting nothing: every hazard the bulletin carries,
  including icing, IFR and freezing level. Filtering must happen here, on
  the `hazard` field. Trusting the server would mean testing flight
  corridors against freezing-level advisories.

  `product` IS NOT THE HAZARD. Turbulence features come back with
  `"product": "TANGO"`, but so do several other hazards - Tango is the
  bulletin, not the subject. Filtering on it would sweep in icing and IFR.
  `hazard` is the field that identifies turbulence.

  THERE IS NO GeoJSON. `geom` and `geometryType` are both the literal string
  "AREA". The actual shape is in `coords`, a list of {lat, lon} dicts whose
  values are strings. The ring is closed, so it converts cleanly, but it has
  to be parsed rather than handed to shapely directly.

The older textual `/airmet` endpoint returns 204 No Content. CONUS textual
AIRMETs were retired in January 2025; the typed G-AIRMET products replaced
them.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Callable, Iterable, Sequence

BASE_URL = "https://aviationweather.gov/api/data"
GAIRMET_PATH = "/gairmet"

#: Hazard values that mean turbulence. High and low altitude are separate
#: products; both matter, since a corridor can cross either band.
TURBULENCE_HAZARDS: frozenset[str] = frozenset({"TURB-HI", "TURB-LO"})

#: G-AIRMETs are issued in three-hour forecast steps.
FORECAST_STEP_HOURS = 3

#: A forecast this far from the time of interest is not describing the same
#: air. Beyond it, the advisory is reported as unavailable rather than
#: stretched to fit.
MAX_FORECAST_DISTANCE = timedelta(hours=FORECAST_STEP_HOURS)


class GairmetFetchError(RuntimeError):
    pass


class TurbulenceSeverity(StrEnum):
    """Mirrors the enum in `awc.py` so PIREPs and forecasts speak the same
    language. NONE means a forecast of smooth air, which is not the same as
    no forecast at all."""
    NONE = "none"
    LIGHT = "light"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"


_SEVERITY_CODES: dict[str, TurbulenceSeverity] = {
    "NEG": TurbulenceSeverity.NONE, "NIL": TurbulenceSeverity.NONE,
    "NONE": TurbulenceSeverity.NONE, "SMTH": TurbulenceSeverity.NONE,
    "LGT": TurbulenceSeverity.LIGHT, "LT": TurbulenceSeverity.LIGHT,
    "LIGHT": TurbulenceSeverity.LIGHT,
    "MOD": TurbulenceSeverity.MODERATE, "MDT": TurbulenceSeverity.MODERATE,
    "MODERATE": TurbulenceSeverity.MODERATE,
    "SEV": TurbulenceSeverity.SEVERE, "SEVERE": TurbulenceSeverity.SEVERE,
    "EXTM": TurbulenceSeverity.EXTREME, "EXTRM": TurbulenceSeverity.EXTREME,
    "EXTREME": TurbulenceSeverity.EXTREME,
}

_SEVERITY_RANK: dict[TurbulenceSeverity, int] = {
    TurbulenceSeverity.NONE: 0, TurbulenceSeverity.LIGHT: 1,
    TurbulenceSeverity.MODERATE: 2, TurbulenceSeverity.SEVERE: 3,
    TurbulenceSeverity.EXTREME: 4,
}


def parse_severity(code: str | None) -> TurbulenceSeverity | None:
    """Map an AWC severity code. Unknown codes return None, never a guess.

    A code we cannot read is an unknown severity, not a mild one.
    """
    if not code:
        return None
    token = str(code).strip().upper()
    if token in _SEVERITY_CODES:
        return _SEVERITY_CODES[token]
    # Compound forms like "LGT-MOD" or "MOD OCNL SEV" take the worst part,
    # which is the conservative reading and the one a passenger cares about.
    worst: TurbulenceSeverity | None = None
    for part in token.replace("/", " ").replace("-", " ").split():
        found = _SEVERITY_CODES.get(part)
        if found and (worst is None
                      or _SEVERITY_RANK[found] > _SEVERITY_RANK[worst]):
            worst = found
    return worst


# ------------------------------------------------------------------ model


@dataclass(frozen=True)
class TurbulenceAdvisory:
    """One forecast polygon with its altitude band and validity window."""
    hazard: str
    severity: TurbulenceSeverity | None
    base_ft: int | None
    top_ft: int | None
    valid_time: datetime | None
    expire_time: datetime | None
    issue_time: datetime | None
    forecast_hour: int | None
    ring: list[tuple[float, float]] = field(default_factory=list)
    tag: str | None = None
    status: str | None = None
    source: str = "AWC G-AIRMET"

    @property
    def usable(self) -> bool:
        """A polygon needs a shape and an altitude band to be testable.

        Without the band, a forecast for FL300-FL400 would match a flight at
        FL410, which is a different piece of sky.
        """
        return (len(self.ring) >= 4
                and self.base_ft is not None
                and self.top_ft is not None)

    def covers_altitude(self, altitude_ft: int) -> bool:
        if self.base_ft is None or self.top_ft is None:
            return False
        return self.base_ft <= altitude_ft <= self.top_ft

    def overlaps_band(self, base_ft: int | None, top_ft: int | None) -> bool:
        """Does this advisory's altitude band intersect a corridor's?"""
        if self.base_ft is None or self.top_ft is None:
            return False
        if base_ft is None or top_ft is None:
            return True          # unbanded corridor: fall back to 2D
        return not (top_ft < self.base_ft or base_ft > self.top_ft)

    def valid_at(self, when: datetime,
                 tolerance: timedelta = MAX_FORECAST_DISTANCE) -> bool:
        """Is this forecast describing the air at a given time?

        Inside its own validity window is a clear yes. Otherwise the
        forecast step nearest the time of interest is accepted up to
        `tolerance`, beyond which it is describing different air.
        """
        if self.valid_time is None:
            return False
        if self.expire_time and self.valid_time <= when <= self.expire_time:
            return True
        return abs(when - self.valid_time) <= tolerance


# ------------------------------------------------------------------ parsing


def _to_int_ft(value) -> int | None:
    """Flight level string to feet. `'400'` is FL400, which is 40,000 ft.

    `'SFC'` means the surface. Anything unrecognised returns None rather
    than a default, so a missing band is visible instead of invented.
    """
    if value is None:
        return None
    token = str(value).strip().upper()
    if not token:
        return None
    if token in ("SFC", "SURFACE", "GND", "GROUND"):
        return 0
    if token.startswith("FL"):
        token = token[2:]
    try:
        return int(float(token)) * 100
    except ValueError:
        return None


def _to_datetime(value) -> datetime | None:
    """AWC mixes ISO strings and unix seconds in the same payload."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, ValueError, OverflowError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_ring(coords) -> list[tuple[float, float]]:
    """`coords` is a list of {lat, lon} dicts with string values.

    Returns (lat, lon) pairs, the ordering used throughout this project.
    The ring arrives closed; the closing point is dropped so the ring is
    described once, and the geometry layer closes it as its own convention.
    """
    if not isinstance(coords, (list, tuple)):
        return []
    ring: list[tuple[float, float]] = []
    for point in coords:
        if isinstance(point, dict):
            lat, lon = point.get("lat"), point.get("lon")
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            lon, lat = point[0], point[1]      # GeoJSON ordering, if it appears
        else:
            continue
        try:
            ring.append((float(lat), float(lon)))
        except (TypeError, ValueError):
            continue
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring.pop()
    return ring


def parse_advisory(feature) -> TurbulenceAdvisory | None:
    """One payload feature into an advisory, or None if it is not usable."""
    if not isinstance(feature, dict):
        return None
    props = feature.get("properties") if isinstance(
        feature.get("properties"), dict) else feature

    hazard = str(props.get("hazard") or "").strip().upper()
    if not hazard:
        return None

    return TurbulenceAdvisory(
        hazard=hazard,
        severity=parse_severity(props.get("severity")),
        base_ft=_to_int_ft(props.get("base")),
        top_ft=_to_int_ft(props.get("top")),
        valid_time=_to_datetime(props.get("validTime")),
        expire_time=_to_datetime(props.get("expireTime")),
        issue_time=_to_datetime(props.get("issueTime")),
        forecast_hour=(int(props["forecastHour"])
                       if str(props.get("forecastHour", "")).strip().isdigit()
                       else None),
        ring=parse_ring(props.get("coords")),
        tag=props.get("tag"),
        status=props.get("status"),
    )


def turbulence_only(features: Iterable) -> list[TurbulenceAdvisory]:
    """Keep the turbulence advisories and discard the rest.

    This filter is not optional. The endpoint ignores its own `type`
    parameter and returns every hazard in the bulletin, so without it a
    corridor would be scored against icing and freezing-level forecasts.
    """
    out: list[TurbulenceAdvisory] = []
    for feature in features or []:
        advisory = parse_advisory(feature)
        if advisory and advisory.hazard in TURBULENCE_HAZARDS:
            out.append(advisory)
    return out


# ------------------------------------------------------------------ client


#: (path, params) -> (status, parsed_or_None, raw). Injectable for tests.
Transport = Callable[[str, dict], tuple[int, object, str]]


def _http(path: str, params: dict) -> tuple[int, object, str]:
    url = f"{BASE_URL}{path}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "turbulence-agent/0.1",
    })
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read().decode("utf-8")
            if resp.status == 204 or not raw.strip():
                return resp.status, [], raw
            return resp.status, json.loads(raw), raw
    except urllib.error.HTTPError as e:
        return e.code, None, e.read().decode("utf-8", "replace")[:400]
    except json.JSONDecodeError:
        return 200, None, "response was not JSON"


@dataclass
class GairmetClient:
    """Fetches turbulence advisories. Synchronous on purpose.

    The reasoning layer is synchronous and heavily tested, so the event loop
    stops at the source boundary rather than being threaded through the
    graph.
    """
    transport: Transport | None = None
    calls_made: int = 0
    #: Optional timing sink, set by the service. See AeroAPIClient.
    timings: object | None = None

    def fetch(self) -> list[TurbulenceAdvisory]:
        """Current turbulence G-AIRMETs.

        No bounding box: the endpoint returns the full CONUS bulletin
        regardless, so filtering by geography happens against the corridor
        rather than in the request.
        """
        transport = self.transport or _http
        self.calls_made += 1
        if self.timings is not None:
            with self.timings.track("awc"):
                status, body, raw = transport(GAIRMET_PATH, {"format": "json"})
        else:
            status, body, raw = transport(GAIRMET_PATH, {"format": "json"})

        if status == 204:
            return []
        if status != 200 or body is None:
            raise GairmetFetchError(
                f"G-AIRMET fetch failed: HTTP {status}: {str(raw)[:200]}")

        if isinstance(body, dict):
            features = body.get("features") or body.get("data") or []
        elif isinstance(body, list):
            features = body
        else:
            raise GairmetFetchError(
                f"unexpected G-AIRMET payload type: {type(body).__name__}")

        return turbulence_only(features)


def select_for_time(advisories: Sequence[TurbulenceAdvisory],
                    when: datetime,
                    tolerance: timedelta = MAX_FORECAST_DISTANCE
                    ) -> list[TurbulenceAdvisory]:
    """Advisories describing the air at a given time.

    Forecasts step every three hours, so a corridor at 18:00Z must not be
    scored against a forecast valid at 06:00Z. Anything outside the
    tolerance is dropped, and the caller reports the gap rather than
    stretching a stale forecast over it.
    """
    return [a for a in advisories if a.usable and a.valid_at(when, tolerance)]
