"""Aviation Weather Center (aviationweather.gov) - pilot reports.

`fetch_pireps` pulls PIREPs and AIREPs for a bounding box over the last N hours and
returns them as validated `PilotReport` models.

Three things this module deliberately does *not* do:

* **It never invents a severity.** AWC codes turbulence intensity in `tbInt1`/`tbInt2`.
  When those are absent - no `/TB` group, or a `/TB` group with a type but no
  intensity - `turbulence_severity` is `None`, never `LIGHT`. A pilot explicitly
  reporting a smooth ride (`TB NEG`) is different data, and gets
  `TurbulenceSeverity.NONE`. Downstream code must keep those two apart
  (CLAUDE.md rule 3).
* **It does not read free text.** Intensity comes from AWC's coded fields only.
  `AGC UA .../RM LIGHT TURB` has no coded intensity, so it parses to `None` even
  though a human can see "LIGHT" in the remark. Guessing from remarks is exactly the
  assumed-value failure rule 3 forbids; `raw_text` is preserved so a later stage can
  surface the remark to the user without it becoming a number.
* **It does not cache.** Callers layer a TTL cache on top with an explicit
  `expires_at` (CLAUDE.md rule 4).

Every report carries `source` and `fetched_at` so provenance survives into the
output (rule 5).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Final

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.config import Settings, get_settings

__all__ = [
    "AwcFetchError",
    "BBox",
    "PilotReport",
    "TurbulenceSeverity",
    "fetch_pirep_geojson",
    "fetch_pireps",
    "parse_pirep_feature",
    "parse_turbulence_severity",
]

logger = logging.getLogger(__name__)

#: ``(min_lat, min_lon, max_lat, max_lon)`` - the order AWC's ``bbox`` param wants.
BBox = tuple[float, float, float, float]

PIREP_PATH: Final = "pirep"
SOURCE_NAME: Final = "AWC"


class AwcFetchError(RuntimeError):
    """The AWC request failed, or returned something that is not a GeoJSON object.

    Raised rather than swallowed: the agent's tool wrapper turns this into a typed
    Failure so the graph can widen its search instead of silently seeing no reports
    (CLAUDE.md - a failed fetch must never look like "no turbulence out there").
    """


class TurbulenceSeverity(StrEnum):
    """Reported turbulence intensity.

    ``NONE`` means a pilot explicitly reported smooth air. It does *not* mean
    "unknown" - that is represented by `None` on the field itself.
    """

    NONE = "NONE"
    LIGHT = "LIGHT"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
    EXTREME = "EXTREME"

    @property
    def rank(self) -> int:
        """Ordinal for comparisons; higher is worse."""
        return _SEVERITY_RANK[self]


_SEVERITY_RANK: Final[dict[TurbulenceSeverity, int]] = {
    TurbulenceSeverity.NONE: 0,
    TurbulenceSeverity.LIGHT: 1,
    TurbulenceSeverity.MODERATE: 2,
    TurbulenceSeverity.SEVERE: 3,
    TurbulenceSeverity.EXTREME: 4,
}

# AWC's coded intensities, plus the spelled-out variants that occasionally appear.
_SEVERITY_CODES: Final[dict[str, TurbulenceSeverity]] = {
    "NEG": TurbulenceSeverity.NONE,
    "NEGATIVE": TurbulenceSeverity.NONE,
    "NIL": TurbulenceSeverity.NONE,
    "NONE": TurbulenceSeverity.NONE,
    "SMTH": TurbulenceSeverity.NONE,
    "SMOOTH": TurbulenceSeverity.NONE,
    "LGT": TurbulenceSeverity.LIGHT,
    "LT": TurbulenceSeverity.LIGHT,
    "LIGHT": TurbulenceSeverity.LIGHT,
    "MOD": TurbulenceSeverity.MODERATE,
    "MDT": TurbulenceSeverity.MODERATE,
    "MODERATE": TurbulenceSeverity.MODERATE,
    "SEV": TurbulenceSeverity.SEVERE,
    "SEVERE": TurbulenceSeverity.SEVERE,
    "EXTM": TurbulenceSeverity.EXTREME,
    "EXTRM": TurbulenceSeverity.EXTREME,
    "EXTREME": TurbulenceSeverity.EXTREME,
}


class PilotReport(BaseModel):
    """One PIREP or AIREP, as reported. Immutable once built."""

    model_config = ConfigDict(frozen=True)

    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    #: Reported flight level in feet MSL. `None` when the report carries no level
    #: (e.g. "during climb"); never defaulted to 0.
    altitude_ft: int | None = Field(default=None, ge=0)
    #: `None` means AWC coded no intensity - not "smooth". See module docstring.
    turbulence_severity: TurbulenceSeverity | None = None
    #: ICAO type designator for a PIREP (``B738``), or the callsign for an AIREP.
    aircraft_type: str | None = None
    observation_time: datetime
    raw_text: str
    #: Provenance (CLAUDE.md rule 5): when we pulled it, and from where.
    fetched_at: datetime
    source: str = SOURCE_NAME

    @field_validator("observation_time", "fetched_at")
    @classmethod
    def _must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def parse_turbulence_severity(code: str | None) -> TurbulenceSeverity | None:
    """Map an AWC intensity code to the enum, or `None` if nothing is coded.

    Handles ranges (``LGT-MOD``) by returning the worse end: for a flier deciding
    between itineraries, a report that *might* be moderate is a moderate report.
    Unrecognised tokens are dropped rather than guessed at.
    """
    if not code:
        return None

    worst: TurbulenceSeverity | None = None
    for token in str(code).upper().replace("/", "-").split("-"):
        severity = _SEVERITY_CODES.get(token.strip())
        if severity is None:
            continue
        if worst is None or severity.rank > worst.rank:
            worst = severity
    return worst


def _worst(*severities: TurbulenceSeverity | None) -> TurbulenceSeverity | None:
    """The most severe of the given values, ignoring `None`s."""
    known = [s for s in severities if s is not None]
    return max(known, key=lambda s: s.rank) if known else None


def _parse_timestamp(value: Any) -> datetime | None:
    """AWC returns ISO-8601 with a `Z`; some endpoints return epoch seconds."""
    if isinstance(value, bool) or value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            return datetime.fromtimestamp(int(text), tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None

    # AWC timestamps are UTC; a bare one is still UTC, not local.
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _flight_level_to_feet(value: Any) -> int | None:
    """`fltlvl` is hundreds of feet: 280 -> 28000. Absent stays absent."""
    if value is None or isinstance(value, bool):
        return None
    try:
        flight_level = int(value)
    except (TypeError, ValueError):
        return None
    return flight_level * 100 if flight_level >= 0 else None


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def parse_pirep_feature(feature: Any, fetched_at: datetime) -> PilotReport | None:
    """Build a `PilotReport` from one GeoJSON feature, or `None` if unusable.

    A feature is unusable without a point location or an observation time - those
    two are what makes a report placeable in space and time. Everything else is
    allowed to be missing.
    """
    if not isinstance(feature, dict):
        return None

    properties = feature.get("properties")
    geometry = feature.get("geometry")
    if not isinstance(properties, dict) or not isinstance(geometry, dict):
        return None

    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) < 2:
        return None
    try:
        longitude, latitude = float(coordinates[0]), float(coordinates[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(longitude) and math.isfinite(latitude)):
        return None

    observation_time = _parse_timestamp(properties.get("obsTime"))
    if observation_time is None:
        return None

    raw_text = properties.get("rawOb")
    severity = _worst(
        parse_turbulence_severity(properties.get("tbInt1")),
        parse_turbulence_severity(properties.get("tbInt2")),
    )

    try:
        return PilotReport(
            latitude=latitude,
            longitude=longitude,
            altitude_ft=_flight_level_to_feet(properties.get("fltlvl")),
            turbulence_severity=severity,
            aircraft_type=_optional_text(properties.get("acType")),
            observation_time=observation_time,
            raw_text=raw_text if isinstance(raw_text, str) else "",
            fetched_at=fetched_at,
        )
    except ValidationError:
        logger.warning("AWC pirep: dropping feature that failed validation: %r", feature)
        return None


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def _validate_bbox(bbox: BBox) -> BBox:
    try:
        min_lat, min_lon, max_lat, max_lon = (float(value) for value in bbox)
    except (TypeError, ValueError):
        raise ValueError(
            "bbox must be four numbers (min_lat, min_lon, max_lat, max_lon), "
            f"got {bbox!r}"
        ) from None

    if not all(map(math.isfinite, (min_lat, min_lon, max_lat, max_lon))):
        raise ValueError(f"bbox values must be finite, got {bbox!r}")
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise ValueError(f"bbox latitudes must be within [-90, 90], got {bbox!r}")
    if not (-180.0 <= min_lon <= 180.0 and -180.0 <= max_lon <= 180.0):
        raise ValueError(f"bbox longitudes must be within [-180, 180], got {bbox!r}")
    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError(
            "bbox must be ordered (min_lat, min_lon, max_lat, max_lon) with "
            f"min < max on both axes, got {bbox!r}"
        )

    return (min_lat, min_lon, max_lat, max_lon)


def _validate_hours_back(hours_back: float) -> float:
    try:
        value = float(hours_back)
    except (TypeError, ValueError):
        raise ValueError(f"hours_back must be a number, got {hours_back!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"hours_back must be a positive, finite number, got {value!r}")
    return hours_back


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------


async def fetch_pirep_geojson(
    bbox: BBox,
    hours_back: float,
    *,
    settings: Settings,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """The one function in this module that touches the network.

    Kept separate and module-level so tests can monkeypatch it wholesale
    (CLAUDE.md: every external call goes through a patchable function).
    """
    url = f"{settings.awc_base_url.rstrip('/')}/{PIREP_PATH}"
    params = {
        "format": "geojson",
        "age": str(hours_back),
        "bbox": ",".join(str(value) for value in bbox),
    }
    headers = {
        "User-Agent": settings.awc_user_agent,
        "Accept": "application/geo+json, application/json",
    }

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=settings.awc_timeout_seconds)

    try:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise AwcFetchError(
            f"AWC pirep request failed with HTTP {exc.response.status_code}"
        ) from exc
    except httpx.HTTPError as exc:
        raise AwcFetchError(f"AWC pirep request failed: {exc}") from exc
    except ValueError as exc:  # includes json.JSONDecodeError
        raise AwcFetchError("AWC pirep response was not valid JSON") from exc
    finally:
        if owns_client:
            await client.aclose()

    if not isinstance(payload, dict):
        raise AwcFetchError(
            f"AWC pirep response was {type(payload).__name__}, expected a GeoJSON object"
        )
    return payload


async def fetch_pireps(
    bbox: BBox,
    hours_back: float = 6.0,
    *,
    client: httpx.AsyncClient | None = None,
    settings: Settings | None = None,
) -> list[PilotReport]:
    """Fetch PIREPs/AIREPs in `bbox` from the last `hours_back` hours.

    Args:
        bbox: `(min_lat, min_lon, max_lat, max_lon)` in degrees. AWC filters
            server-side, so this is a real narrowing, not a client-side trim.
        hours_back: How far back to look, in hours. Fractional values are fine.
        client: An `httpx.AsyncClient` to reuse. One is created and closed per
            call when omitted.
        settings: Overrides the process settings; the `User-Agent` AWC requires
            comes from here.

    Returns:
        Reports sorted newest observation first. Features that cannot be placed in
        space and time are skipped and logged, not raised over.

    Raises:
        ValueError: `bbox` or `hours_back` is malformed.
        AwcFetchError: the request failed or the body was not GeoJSON.
    """
    bbox = _validate_bbox(bbox)
    hours_back = _validate_hours_back(hours_back)
    settings = settings if settings is not None else get_settings()

    fetched_at = datetime.now(timezone.utc)
    payload = await fetch_pirep_geojson(
        bbox, hours_back, settings=settings, client=client
    )

    features = payload.get("features")
    if features is None:
        return []
    if not isinstance(features, list):
        raise AwcFetchError(
            f"AWC pirep response had {type(features).__name__} features, expected a list"
        )

    reports: list[PilotReport] = []
    for feature in features:
        report = parse_pirep_feature(feature, fetched_at=fetched_at)
        if report is None:
            continue
        reports.append(report)

    skipped = len(features) - len(reports)
    if skipped:
        logger.warning(
            "AWC pirep: skipped %d of %d unusable feature(s)", skipped, len(features)
        )

    reports.sort(key=lambda report: report.observation_time, reverse=True)
    return reports
