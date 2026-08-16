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
from app.reasoning.geometry import CorridorShape, intersects_ring
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


# ------------------------------------------------------------------ helpers


def bounding_box(shape: CorridorShape,
                 pad_deg: float = 1.0) -> tuple[float, float, float, float]:
    """A lat/lon box around the corridor, padded so the fetch does not clip
    reports just outside it. Precise containment is tested afterwards."""
    lats = [p[0] for p in shape.points]
    lons = [p[1] for p in shape.points]
    return (min(lons) - pad_deg, min(lats) - pad_deg,
            max(lons) + pad_deg, max(lats) + pad_deg)


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

    for r in reports or []:
        considered += 1
        lat = getattr(r, "latitude", None)
        lon = getattr(r, "longitude", None)
        if lat is None or lon is None:
            continue
        alt = getattr(r, "altitude_ft", None)
        if not shape.contains(lat, lon, altitude_ft=alt):
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

    if considered and not inside_points:
        notes.append(
            f"{considered} pilot report(s) were fetched near this route but "
            f"none fell inside the corridor at its altitudes. That is an "
            f"absence of observation, not a report of smooth air."
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

    if combined is Severity.UNRESOLVED:
        notes.append(
            "No turbulence evidence was found for this corridor from either "
            "source. The reading stays unresolved. Unresolved is not smooth."
        )

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

    return GatherResult(
        evidence=evidence, notes=notes,
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
