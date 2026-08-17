# install-to: app/reasoning
"""
Turbulence evidence for a corridor.

Takes a corridor shape and produces the `Evidence` the critic scores: what
pilots reported inside it, what the forecast says about it, how much of it
anyone has looked at lately, and whether those two sources agree.

THE TWO SOURCES STAY APART. A pilot report is one aircraft at one moment.
A forecast polygon covers hours and a wide band of sky. They answer
different questions and they can disagree, so each keeps its own reading and
its own count. The critic computes agreement from the pair rather than being
handed a blended number.

WORST WINS, WITH THE COUNT KEPT. Four reports along one corridor, three
light and one severe, resolve to severe with a count of four. The count is
what lets a reader tell one outlier from four in agreement. Averaging would
bury the severe report, and a lone severe report along the route is exactly
what a nervous passenger wants surfaced.

ABSENCE IS NEVER A READING. A corridor nobody has flown recently has no
reports, and that produces UNRESOLVED, never SMOOTH. Coverage falls, which
lowers the score, but it can never eliminate a branch - see the critic.

The async boundary stops here. PIREP fetching is async; the reasoning layer
is synchronous and heavily tested. Rather than thread an event loop through
the graph, this module runs the coroutine and hands back plain data.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Protocol, Sequence

from app.reasoning.critic import Evidence, Severity
from app.reasoning.geometry import (
    CorridorShape,
    crosses_antimeridian,
    intersects_ring,
    normalize_longitude,
    unwrap_longitudes,
)
from app.sources.gairmet import (
    GairmetClient,
    GairmetFetchError,
    TurbulenceAdvisory,
)
from app.sources.gairmet import TurbulenceSeverity as ForecastSeverity

#: How far back to look for pilot reports. Beyond this a report is
#: describing air that has moved on.
DEFAULT_LOOKBACK_HOURS = 3

#: Reports older than this contribute nothing to coverage. Mirrors the decay
#: already used in the critic's coverage scoring.
COVERAGE_HORIZON_MINUTES = 180.0

#: The corridor is divided into this many segments to measure coverage. A
#: single report should not make a 500 nm corridor look fully observed.
COVERAGE_SEGMENTS = 10

#: Source-layer severity strings mapped onto the critic's enum. Both source
#: modules use the same vocabulary, so one table serves both.
_TO_CRITIC: dict[str, Severity] = {
    "none": Severity.SMOOTH,
    "light": Severity.LIGHT,
    "moderate": Severity.MODERATE,
    "severe": Severity.SEVERE,
    "extreme": Severity.EXTREME,
}

_RANK: dict[Severity, int] = {
    Severity.SMOOTH: 0, Severity.LIGHT: 1, Severity.MODERATE: 2,
    Severity.SEVERE: 3, Severity.EXTREME: 4,
}


def to_critic_severity(value) -> Severity:
    """Source severity to the critic's enum.

    An unreadable or absent value becomes UNRESOLVED, never SMOOTH. The
    difference is the whole point: one means calm air, the other means
    nobody knows.
    """
    if value is None:
        return Severity.UNRESOLVED
    return _TO_CRITIC.get(str(value).strip().lower(), Severity.UNRESOLVED)


def worst(*readings: Severity) -> Severity:
    """The most severe of several readings, ignoring the unresolved ones."""
    known = [r for r in readings if r is not Severity.UNRESOLVED]
    if not known:
        return Severity.UNRESOLVED
    return max(known, key=lambda r: _RANK.get(r, 0))


# ------------------------------------------------------------------ inputs


class Report(Protocol):
    """The shape this module needs from a pilot report.

    Structural rather than a concrete import, so `awc.PilotReport` satisfies
    it without this module depending on that class.
    """
    latitude: float
    longitude: float
    altitude_ft: int | None
    turbulence_severity: object
    observation_time: datetime


#: Fetches pilot reports for a bounding box. Injectable so tests never touch
#: the network and never start an event loop.
PirepFetcher = Callable[[tuple[float, float, float, float], int],
                        Sequence[object]]


@dataclass
class GatherResult:
    """Evidence plus the notes explaining how it was reached."""
    evidence: Evidence
    notes: list[str]
    reports_inside: int = 0
    reports_considered: int = 0
    advisories_inside: int = 0
    advisories_considered: int = 0
    #: One plain sentence for a reader who will not read the notes. Set
    #: whenever the reading is unresolved, because "unresolved" on its own
    #: tells a nervous passenger nothing about why.
    summary: str | None = None


# ------------------------------------------------------------------ helpers


def bounding_box(shape: CorridorShape,
                 pad_deg: float = 1.0) -> tuple[float, float, float, float]:
    """A box around the corridor as (min_lat, min_lon, max_lat, max_lon).

    Latitude first, which is what AWC's bbox parameter wants and what
    `awc.BBox` documents. Getting this backwards is not loud: a Pittsburgh
    longitude of -80 is a legal latitude, so the request succeeds and
    quietly searches the South Atlantic. It returns zero reports, which
    looks exactly like a calm afternoon.

    Padded so the fetch does not clip reports just outside the corridor.
    Precise containment is tested afterwards, in three dimensions.
    """
    lats = [p[0] for p in shape.points]
    lons = [p[1] for p in shape.points]

    if crosses_antimeridian(shape.points):
        # A box from -122 to +139 spans most of the planet rather than the
        # narrow band the route occupies. Unwrapping gives the true extent,
        # but the result may exceed 180, and AWC would reject that. A route
        # crossing the date line is outside the CONUS products anyway, so
        # the box is clamped and the caller gets an honest empty result
        # rather than a request for half the world.
        unwrapped = unwrap_longitudes(lons)
        lo = normalize_longitude(min(unwrapped) - pad_deg)
        hi = normalize_longitude(max(unwrapped) + pad_deg)
        return (min(lats) - pad_deg, min(lo, hi),
                max(lats) + pad_deg, max(lo, hi))

    return (min(lats) - pad_deg, min(lons) - pad_deg,
            max(lats) + pad_deg, max(lons) + pad_deg)


def _age_minutes(when: datetime | None, now: datetime) -> float | None:
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (now - when).total_seconds() / 60.0)


def coverage_fraction(shape: CorridorShape,
                      points: Sequence[tuple[float, float]],
                      segments: int = COVERAGE_SEGMENTS) -> float:
    """Fraction of the corridor's length with an observation near it.

    The corridor is split into equal segments along its path and a segment
    counts as covered when a report falls nearer to it than to any other.
    One report on a 500 nm route yields 0.1, not 1.0, which is the honest
    reading of what has actually been looked at.
    """
    if not points or len(shape.points) < 2:
        return 0.0
    path = shape.points
    if len(path) < segments:
        # Interpolate so short paths still divide into comparable segments.
        path = [path[int(i * (len(path) - 1) / (segments - 1))]
                for i in range(segments)]
    step = max(1, len(path) // segments)
    anchors = path[::step][:segments]

    covered = set()
    for lat, lon in points:
        best, best_d = None, None
        for i, (alat, alon) in enumerate(anchors):
            d = (alat - lat) ** 2 + (alon - lon) ** 2
            if best_d is None or d < best_d:
                best, best_d = i, d
        if best is not None:
            covered.add(best)
    return round(len(covered) / len(anchors), 4)


# ------------------------------------------------------------------ gather


def gather_observed(shape: CorridorShape, reports: Iterable,
                    now: datetime) -> tuple[Severity, int, float | None,
                                            float, list[str], int, int]:
    """Pilot reports falling inside the corridor, in three dimensions."""
    considered = 0
    inside_points: list[tuple[float, float]] = []
    ages: list[float] = []
    readings: list[Severity] = []
    worst_reading = Severity.UNRESOLVED
    worst_where: str | None = None
    notes: list[str] = []

    # Why a report was rejected matters to a reader. "Somewhere else
    # entirely" and "directly overhead but at 3,000 feet" are different
    # facts, and only one of them is worth worrying about.
    elsewhere = 0                       # not near the route
    wrong_altitude: list[int] = []       # along the route, wrong height

    for r in reports or []:
        considered += 1
        lat = getattr(r, "latitude", None)
        lon = getattr(r, "longitude", None)
        if lat is None or lon is None:
            continue
        alt = getattr(r, "altitude_ft", None)
        if not shape.contains(lat, lon, altitude_ft=alt):
            if shape.contains(lat, lon):
                if alt is not None:
                    wrong_altitude.append(alt)
            else:
                elsewhere += 1
            continue

        inside_points.append((lat, lon))
        age = _age_minutes(getattr(r, "observation_time", None), now)
        if age is not None:
            ages.append(age)

        reading = to_critic_severity(
            getattr(getattr(r, "turbulence_severity", None), "value",
                    getattr(r, "turbulence_severity", None)))
        if reading is Severity.UNRESOLVED:
            continue
        readings.append(reading)
        if worst(worst_reading, reading) is reading and reading is not worst_reading:
            worst_reading = reading
            worst_where = f"{lat:.2f}, {lon:.2f}"
        elif worst_reading is Severity.UNRESOLVED:
            worst_reading = reading
            worst_where = f"{lat:.2f}, {lon:.2f}"

    count = len(readings)
    cov = coverage_fraction(shape, inside_points)
    mean_age = round(sum(ages) / len(ages), 1) if ages else None

    if not considered:
        # Fetching nothing at all must still be said out loud. Silence here
        # would be the one place absence passes without comment, which is
        # the failure this project exists to avoid.
        notes.append(
            "No pilot reports were filed anywhere near this route in the "
            "lookback window. Nobody has flown it and said anything, which "
            "is an absence of observation, not a report of smooth air."
        )
    elif not inside_points:
        if wrong_altitude:
            highest = max(wrong_altitude)
            notes.append(
                f"{len(wrong_altitude)} pilot report(s) were filed along "
                f"this route but all of them below "
                f"FL{highest // 100:03d}, well under the cruise altitude. "
                f"Turbulence at 3,000 feet says nothing about the ride at "
                f"cruise, so none of them counts as evidence here."
            )
        if elsewhere:
            notes.append(
                f"{elsewhere} other pilot report(s) were in the surrounding "
                f"airspace but not along this corridor."
            )
        notes.append(
            "No pilot report describes the air this flight will actually be "
            "in. That is an absence of observation, not a report of smooth "
            "air."
        )
    elif count:
        spread = {r.value for r in readings}
        if len(spread) > 1:
            notes.append(
                f"{count} pilot report(s) inside the corridor disagree "
                f"({', '.join(sorted(spread))}). The worst is used, and the "
                f"count is kept so one outlier is distinguishable from "
                f"several in agreement."
            )
        else:
            notes.append(
                f"{count} pilot report(s) inside the corridor, all "
                f"{worst_reading.value}."
            )

    return (worst_reading, count, mean_age, cov, notes,
            len(inside_points), considered)


def gather_forecast(shape: CorridorShape,
                    advisories: Sequence[TurbulenceAdvisory],
                    ) -> tuple[Severity, int, list[str], int, int]:
    """Forecast polygons overlapping the corridor in three dimensions."""
    considered = len(advisories or [])
    matched: list[TurbulenceAdvisory] = []
    notes: list[str] = []

    for a in advisories or []:
        if not a.usable:
            continue
        if not a.overlaps_band(shape.altitude_min_ft, shape.altitude_max_ft):
            continue
        # Polygon intersection, not vertex containment. A G-AIRMET is
        # usually far larger than a 25 nm corridor, so an advisory that
        # completely covers the route has every vertex outside it.
        if intersects_ring(shape, a.ring):
            matched.append(a)

    reading = worst(*[to_critic_severity(
        getattr(a.severity, "value", a.severity)) for a in matched]) \
        if matched else Severity.UNRESOLVED

    if considered and not matched:
        notes.append(
            f"{considered} turbulence forecast(s) were active but none cover "
            f"this corridor at its altitudes. No forecast is not a forecast "
            f"of smooth air."
        )
    elif matched:
        bands = ", ".join(
            f"FL{(a.base_ft or 0)//100:03d}-FL{(a.top_ft or 0)//100:03d}"
            for a in matched[:3])
        notes.append(
            f"{len(matched)} turbulence forecast(s) overlap this corridor "
            f"({bands}), worst {reading.value}."
        )

    return reading, len(matched), notes, len(matched), considered


#: A route shorter than this rarely climbs high enough, or stays at cruise
#: long enough, for anyone to file a report about it.
SHORT_ROUTE_NM = 600

#: What each level means in the cabin, paraphrased from the FAA turbulence
#: reporting criteria - AIM Table 7-1-11 and Advisory Circular 120-88A,
#: "Preventing Injuries Caused by Turbulence". These describe what occupants
#: experience, which is the part a passenger can use.
#:
#: Nothing here promises a comfortable flight. "Light" is the level most
#: likely to be misread as reassurance, so it describes the sensation and
#: stops. Whether a given flight is comfortable is not something an advisory
#: or a stranger's pilot report can tell you.
SENSATION: dict[Severity, str] = {
    Severity.SMOOTH:
        "Reported as smooth, meaning no appreciable turbulence.",
    Severity.LIGHT:
        "Light turbulence causes slight, erratic changes in altitude or "
        "attitude. Occupants may feel a slight strain against seat belts. "
        "Objects stay put and cabin service can continue.",
    Severity.MODERATE:
        "Moderate turbulence causes changes in altitude or attitude, though "
        "the aircraft stays in positive control throughout. Occupants feel "
        "definite strain against seat belts, unsecured objects are "
        "dislodged, and walking is difficult.",
    Severity.SEVERE:
        "Severe turbulence causes large, abrupt changes in altitude or "
        "attitude, and the aircraft may be momentarily out of control. "
        "Occupants are forced violently against seat belts and unsecured "
        "objects are tossed about. Cabin service and walking are "
        "impossible.",
    Severity.EXTREME:
        "Extreme turbulence means the aircraft is violently tossed about "
        "and is practically impossible to control. It may cause structural "
        "damage. This is rare, and crews route around it wherever it is "
        "forecast.",
}

#: Where the FAA descriptions come from. Cited rather than paraphrased
#: anonymously, since the whole point of this project is that claims carry
#: their provenance.
SENSATION_SOURCE = ("FAA turbulence reporting criteria, AIM Table 7-1-11 "
                    "and Advisory Circular 120-88A")


def explain_absence(shape: CorridorShape, reports_considered: int,
                    reports_inside: int, advisories_considered: int,
                    advisories_inside: int) -> str:
    """One sentence saying why there is no reading.

    "Unresolved" is accurate and useless on its own. A passenger wants to
    know whether nobody looked, or people looked and found nothing, and
    those are different situations with the same label.
    """
    length = shape.length_nm
    short = length < SHORT_ROUTE_NM

    if reports_considered == 0 and advisories_considered == 0:
        return ("Neither pilot reports nor forecasts were available, so "
                "nothing is known about the air on this route.")

    if reports_inside == 0 and advisories_inside == 0:
        if short:
            return (f"This is a short hop of about {length:.0f} nautical "
                    f"miles. No turbulence forecast covers it and no pilot "
                    f"has reported conditions at cruise along it, which is "
                    f"common on routes this length. Nothing is known either "
                    f"way.")
        return (f"No turbulence forecast covers this route and no pilot has "
                f"reported conditions at cruise along it in the last few "
                f"hours. Nothing is known either way.")

    if reports_inside == 0:
        return ("A forecast covers this route but no pilot has reported "
                "actual conditions along it, so the forecast is the only "
                "thing to go on.")

    return ("Pilots have reported along this route but no forecast covers "
            "it, so the reports are the only thing to go on.")


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    """`1 pilot report` rather than `1 pilot report(s)`. The severe message
    is the one most likely to be read closely, so it should read cleanly."""
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def explain_reading(evidence: Evidence, shape: CorridorShape) -> str:
    """Plain language for a resolved reading: what it means, where it came
    from, and how much it is worth.

    The last part matters most. Moderate from one pilot report half an hour
    ago and moderate from a forecast covering six hours and half a continent
    are the same word describing very different claims, and a passenger
    cannot tell them apart unless the sentence says so.
    """
    reading = evidence.reading
    if reading is Severity.UNRESOLVED:
        return ""

    parts = [SENSATION.get(reading, "")]

    obs, fc = evidence.observed_reading, evidence.forecast_reading
    obs_n, fc_n = evidence.observed_count, evidence.forecast_count

    if evidence.sources_disagree:
        parts.append(
            f"This is the worse of two sources that disagree: "
            f"{_plural(obs_n, 'pilot report')} said {obs.value} while the forecast "
            f"said {fc.value}. The worse one is used because an average "
            f"would match neither.")
    elif obs is not Severity.UNRESOLVED and fc is not Severity.UNRESOLVED:
        parts.append(
            f"Both sources agree: {_plural(obs_n, 'pilot report')} and a forecast "
            f"both say {reading.value}.")
    elif obs is not Severity.UNRESOLVED:
        age = evidence.mean_age_minutes
        when = f", the most recent about {age:.0f} minutes old" if age else ""
        parts.append(
            f"This comes from {_plural(obs_n, 'pilot report')} inside the corridor"
            f"{when}. No forecast covers this route, so pilots flying it are "
            f"the only source.")
    else:
        parts.append(
            f"This comes from a turbulence forecast covering the route. A "
            f"forecast is a broad shape over several hours, so it says what "
            f"is expected rather than what anyone has felt. No pilot has "
            f"reported actual conditions along this corridor.")

    coverage = evidence.coverage_fraction
    if coverage is not None and obs is not Severity.UNRESOLVED:
        if coverage < 0.34:
            parts.append(
                f"Pilot reports cover only about {coverage:.0%} of the "
                f"route, so most of it is unobserved.")
        elif coverage < 0.67:
            parts.append(
                f"Pilot reports cover roughly {coverage:.0%} of the route.")

    if obs_n == 1 and reading in (Severity.SEVERE, Severity.EXTREME):
        parts.append(
            "This rests on a single report. One aircraft encountering "
            "severe air does not mean every aircraft will, but it is worth "
            "knowing about.")

    return " ".join(p for p in parts if p)


def gather_evidence(shape: CorridorShape,
                    fetch_pireps: PirepFetcher | None = None,
                    gairmet_client: GairmetClient | None = None,
                    when: datetime | None = None,
                    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
                    ) -> GatherResult:
    """Evidence for one corridor, from both sources, held apart.

    Either source failing leaves its half unresolved and says so. A gap in
    one source never becomes a reading in the other.
    """
    now = when or datetime.now(timezone.utc)
    notes: list[str] = []

    # ---- observed
    reports: Sequence[object] = []
    if fetch_pireps is not None:
        try:
            reports = fetch_pireps(bounding_box(shape), lookback_hours) or []
        except Exception as e:  # noqa: BLE001 - one source failing is not fatal
            notes.append(
                f"Pilot reports could not be fetched ({type(e).__name__}). "
                f"The observed side is unknown, not clear."
            )
    (obs_reading, obs_count, mean_age, cov,
     obs_notes, inside, considered) = gather_observed(shape, reports, now)
    notes.extend(obs_notes)

    # ---- forecast
    advisories: Sequence[TurbulenceAdvisory] = []
    if gairmet_client is not None:
        try:
            advisories = [a for a in gairmet_client.fetch()
                          if a.valid_at(now)]
        except GairmetFetchError as e:
            notes.append(
                f"Turbulence forecasts could not be fetched ({e}). The "
                f"forecast side is unknown, not clear."
            )
    (fc_reading, fc_count, fc_notes,
     fc_inside, fc_considered) = gather_forecast(shape, advisories)
    notes.extend(fc_notes)

    # ---- combine, without blending
    combined = worst(obs_reading, fc_reading)

    if obs_reading is not Severity.UNRESOLVED and \
            fc_reading is not Severity.UNRESOLVED and \
            obs_reading is not fc_reading:
        notes.append(
            f"Pilots reported {obs_reading.value} where the forecast says "
            f"{fc_reading.value}. Both are reported and the worse is used; "
            f"averaging them would produce a number neither source supports."
        )

    summary = None
    if combined is Severity.UNRESOLVED:
        summary = explain_absence(shape, considered, inside,
                                  fc_considered, fc_inside)
        notes.append(f"{summary} Unresolved is not smooth.")

    evidence = Evidence(
        coverage_fraction=cov if fetch_pireps is not None else None,
        mean_age_minutes=mean_age,
        agreement=None,                 # the critic computes this
        reading=combined,
        observation_count=obs_count,
        observed_reading=obs_reading,
        observed_count=obs_count,
        forecast_reading=fc_reading,
        forecast_count=fc_count,
    )

    if summary is None:
        summary = explain_reading(evidence, shape)

    return GatherResult(
        evidence=evidence, notes=notes, summary=summary,
        reports_inside=inside, reports_considered=considered,
        advisories_inside=fc_inside, advisories_considered=fc_considered,
    )


# ------------------------------------------------------------------ async


def sync_pirep_fetcher(async_fetch) -> PirepFetcher:
    """Wrap the async PIREP fetcher for the synchronous reasoning layer.

    The event loop stops here on purpose. Making the graph async would mean
    touching a controller and critic that are already tested, for no gain
    beyond concurrency this agent does not need.
    """
    def fetch(bbox, hours):
        coro = async_fetch(bbox=bbox, hours_back=hours)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # Already inside a loop: run the coroutine on its own thread rather
        # than nesting, which asyncio forbids.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return fetch
