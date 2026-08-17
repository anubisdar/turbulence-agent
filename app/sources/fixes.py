# install-to: app/sources
"""
Cache of named route fixes, accumulated from AeroAPI.

Filed routes arrive as strings of identifiers - `EWC WOMBT TOSTR PONCT
JFUND2` - with no coordinates. But `/flights/{id}/route` returns those same
identifiers *with* latitude and longitude, 19 of them in a single call. So
every route lookup we make for one purpose populates a fix database for
another, at no additional API cost and covering exactly the fixes our routes
actually use.

Fix coordinates are immutable reference data. They belong in long-term
memory with no TTL, alongside the NTSB index, and unlike turbulence
advisories they never go stale.

WHAT A ROUTE STRING CONTAINS. Not every token is a point:

    EWC          navaid, 3 letters              -> a point
    WOMBT        enroute fix, 5 letters         -> a point
    J49 / Q82    airway, letter plus digits     -> a path, not a point
    JFUND2       SID or STAR, name plus digit   -> a terminal procedure
    KPIT         airport, 4 letters             -> a point

Airways and procedures are dropped, and the drop is recorded rather than
hidden. Two approximations follow from that, both deliberate:

  - `PSB J49 HNK` becomes a straight leg from PSB to HNK. Jet airways run
    essentially direct between navaids, so the error is small at corridor
    width.
  - Departure and arrival procedures are dropped entirely. They are
    terminal-area manoeuvring within a few dozen miles of airports that are
    already corridor endpoints, and they carry no cruise-altitude exposure.

A token that looks like a point but is not in the cache is reported as
missing. It is never skipped silently and never interpolated, because a
corridor with a hole in the middle is not a shorter corridor - it is an
unknown one.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

FIXES_DDL = """
CREATE TABLE IF NOT EXISTS route_fixes (
    name            TEXT PRIMARY KEY,
    latitude        REAL NOT NULL,
    longitude       REAL NOT NULL,
    fix_type        TEXT,
    seen_count      INTEGER NOT NULL DEFAULT 1,
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    source          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fixes_type ON route_fixes(fix_type);
"""


class TokenKind(str, Enum):
    POINT = "point"          # navaid, fix, or airport - has coordinates
    OCEANIC = "oceanic"      # 5700N/15000W - coordinates in the name itself
    AIRWAY = "airway"        # J49, Q82 - a path between points
    PROCEDURE = "procedure"  # JFUND2, OOSHN5 - SID or STAR
    UNKNOWN = "unknown"      # does not match any known shape


_AIRWAY = re.compile(r"^[A-Z]\d{1,3}$")
_PROCEDURE = re.compile(r"^[A-Z]{3,5}\d$")
_AIRPORT = re.compile(r"^[A-Z]{4}$")
_FIX = re.compile(r"^[A-Z]{2,5}$")

#: Oceanic waypoints carry their own position. Over water there is nothing to
#: name a fix after, so routes use the coordinates directly:
#:
#:     5700N/15000W   57 degrees north, 150 degrees west
#:     3600N/15000E   36 north, 150 east
#:     5230N/04000W   52 degrees 30 minutes north, 40 west
#:
#: Degrees and minutes are packed without a separator: the last two digits of
#: each group are minutes. Latitude is always four digits, longitude five,
#: which is what distinguishes them.
#:
#: These were classified as UNKNOWN and discarded, which is why every
#: transpacific routing failed to resolve. They need no cache lookup at all -
#: the position is in the token.
_OCEANIC = re.compile(
    r"^(\d{2})(\d{2})([NS])/?(\d{3})(\d{2})([EW])$")


def classify(token: str) -> TokenKind:
    """What kind of thing is this route token?"""
    t = token.strip().upper()
    if not t:
        return TokenKind.UNKNOWN
    if _AIRWAY.match(t):
        return TokenKind.AIRWAY
    if _PROCEDURE.match(t):
        return TokenKind.PROCEDURE
    if _OCEANIC.match(t):
        return TokenKind.OCEANIC
    if _AIRPORT.match(t) or _FIX.match(t):
        return TokenKind.POINT
    return TokenKind.UNKNOWN


def parse_oceanic(token: str) -> tuple[float, float] | None:
    """`5700N/15000W` to (57.0, -150.0).

    Returns (lat, lon), the ordering used throughout this project. Minutes
    are the last two digits of each group, so `5230N` is 52.5 degrees.
    """
    match = _OCEANIC.match((token or "").strip().upper())
    if not match:
        return None
    lat_deg, lat_min, ns, lon_deg, lon_min, ew = match.groups()
    lat = int(lat_deg) + int(lat_min) / 60.0
    lon = int(lon_deg) + int(lon_min) / 60.0
    if ns == "S":
        lat = -lat
    if ew == "W":
        lon = -lon
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return (round(lat, 4), round(lon, 4))


def tokenize(route: str) -> list[str]:
    """Split a filed route into tokens.

    Procedures sometimes carry a transition, `JFUND2.PONCT`, so dots are
    separators too. Coordinate-style tokens (`4030N08015W`) are left intact
    for the classifier to reject.
    """
    if not route:
        return []
    parts = re.split(r"[\s.]+", route.strip().upper())
    return [p for p in parts if p]


# ------------------------------------------------------------------ storage


def _normalize_type(raw) -> str | None:
    """Fold AeroAPI's inconsistent casing without mangling acronyms.

    The same kind of thing comes back as both "Waypoint" and "WAYPOINT",
    which splits the cache statistics in two. But "VOR" and "VOR-DME
    (NAVAID)" are acronyms and must stay as they are, so only long,
    fully-alphabetic, fully-uppercase words are folded.
    """
    if not raw:
        return None
    out = []
    for word in str(raw).split():
        if word.isalpha() and word.isupper() and len(word) > 4:
            out.append(word.capitalize())
        else:
            out.append(word)
    return " ".join(out)


def init_fixes(conn: sqlite3.Connection) -> None:
    conn.executescript(FIXES_DDL)
    conn.commit()


def upsert_fixes(conn: sqlite3.Connection, fixes: list[dict],
                 source: str = "AeroAPI /flights/{id}/route") -> int:
    """Store fixes from an AeroAPI route response.

    Expects the shape AeroAPI returns: name, latitude, longitude, type.
    Entries without coordinates are skipped - a named fix with no position
    is not a fix we can use.
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    stored = 0
    for f in fixes or []:
        name = (f.get("name") or "").strip().upper()
        lat, lon = f.get("latitude"), f.get("longitude")
        if not name or lat is None or lon is None:
            continue
        fix_type = _normalize_type(f.get("type"))
        conn.execute(
            """INSERT INTO route_fixes
                   (name, latitude, longitude, fix_type, seen_count,
                    first_seen, last_seen, source)
               VALUES (?,?,?,?,1,?,?,?)
               ON CONFLICT(name) DO UPDATE SET
                   latitude   = excluded.latitude,
                   longitude  = excluded.longitude,
                   fix_type   = COALESCE(excluded.fix_type, route_fixes.fix_type),
                   seen_count = route_fixes.seen_count + 1,
                   last_seen  = excluded.last_seen""",
            (name, float(lat), float(lon), fix_type, now, now, source),
        )
        stored += 1
    conn.commit()
    return stored


def lookup(conn: sqlite3.Connection, names: list[str]
           ) -> tuple[dict[str, tuple[float, float]], list[str]]:
    """Resolve names to coordinates. Returns (found, missing).

    Missing names are returned, never dropped. The caller decides what an
    incomplete route means; this layer does not decide for it.
    """
    if not names:
        return {}, []
    wanted = [n.strip().upper() for n in names if n and n.strip()]
    placeholders = ",".join("?" for _ in wanted)
    rows = conn.execute(
        f"SELECT name, latitude, longitude FROM route_fixes "
        f"WHERE name IN ({placeholders})", wanted
    ).fetchall()
    found = {r[0]: (r[1], r[2]) for r in rows}
    missing = [n for n in wanted if n not in found]
    return found, missing


def cache_stats(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        "SELECT COUNT(*), MIN(first_seen), MAX(last_seen) FROM route_fixes"
    ).fetchone()
    by_type = dict(conn.execute(
        "SELECT COALESCE(fix_type,'unknown'), COUNT(*) "
        "FROM route_fixes GROUP BY 1"
    ).fetchall())
    return {"total": row[0], "first_seen": row[1], "last_seen": row[2],
            "by_type": by_type}


# ------------------------------------------------------------------ resolution


@dataclass
class RouteResolution:
    """What a filed route string became, and what it cost to get there."""
    route: str
    points: list[tuple[str, float, float]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    #: Oceanic waypoints resolved from their own names, needing no cache.
    oceanic_points: list[str] = field(default_factory=list)
    airways_dropped: list[str] = field(default_factory=list)
    procedures_dropped: list[str] = field(default_factory=list)
    unknown_tokens: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        """Usable only if every point-like token was found and at least two
        points remain. A route with a hole is not a shorter route."""
        return not self.missing and len(self.points) >= 2

    @property
    def coverage(self) -> float:
        """Fraction of point-like tokens that resolved."""
        total = len(self.points) + len(self.missing)
        return 1.0 if total == 0 else len(self.points) / total

    def notes(self) -> list[str]:
        out: list[str] = []
        if self.missing:
            out.append(
                f"{len(self.missing)} route point(s) are not in the fix cache "
                f"({', '.join(self.missing[:6])}). This corridor cannot be "
                f"built; the gap is not treated as a shortcut."
            )
        if self.airways_dropped:
            out.append(
                f"Airway segment(s) {', '.join(sorted(set(self.airways_dropped)))} "
                f"approximated as straight legs between their endpoints."
            )
        if self.procedures_dropped:
            out.append(
                f"Terminal procedure(s) "
                f"{', '.join(sorted(set(self.procedures_dropped)))} dropped; "
                f"they manoeuvre near airports already used as endpoints."
            )
        if self.oceanic_points:
            out.append(
                f"{len(self.oceanic_points)} oceanic waypoint(s) resolved "
                f"from their own coordinates, which need no lookup: "
                f"{', '.join(self.oceanic_points[:4])}."
            )
        if self.unknown_tokens:
            out.append(
                f"Unrecognised route token(s): "
                f"{', '.join(sorted(set(self.unknown_tokens)))}."
            )
        return out


def resolve_route(conn: sqlite3.Connection, route: str) -> RouteResolution:
    """Turn a filed route string into an ordered list of coordinates."""
    res = RouteResolution(route=route or "")
    tokens = tokenize(route)
    if not tokens:
        return res

    # Sequence, not set: a route is ordered, and an oceanic point sits
    # between named fixes rather than after them.
    ordered: list[tuple[str, tuple[float, float] | None]] = []
    point_names: list[str] = []
    for t in tokens:
        kind = classify(t)
        if kind is TokenKind.AIRWAY:
            res.airways_dropped.append(t)
        elif kind is TokenKind.PROCEDURE:
            res.procedures_dropped.append(t)
        elif kind is TokenKind.OCEANIC:
            # The position is in the token, so no cache lookup is needed and
            # none can fail.
            position = parse_oceanic(t)
            if position:
                ordered.append((t, position))
                res.oceanic_points.append(t)
            else:
                res.unknown_tokens.append(t)
        elif kind is TokenKind.POINT:
            ordered.append((t, None))
            point_names.append(t)
        else:
            res.unknown_tokens.append(t)

    found, missing = lookup(conn, point_names)
    res.missing = missing
    seen: set[str] = set()
    for name, position in ordered:
        if position is not None:
            res.points.append((name, position[0], position[1]))
            continue
        if name in found and name not in seen:
            seen.add(name)
            lat, lon = found[name]
            res.points.append((name, lat, lon))
    return res
