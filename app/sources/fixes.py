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
    AIRWAY = "airway"        # J49, Q82 - a path between points
    PROCEDURE = "procedure"  # JFUND2, OOSHN5 - SID or STAR
    UNKNOWN = "unknown"      # does not match any known shape


_AIRWAY = re.compile(r"^[A-Z]\d{1,3}$")
_PROCEDURE = re.compile(r"^[A-Z]{3,5}\d$")
_AIRPORT = re.compile(r"^[A-Z]{4}$")
_FIX = re.compile(r"^[A-Z]{2,5}$")


def classify(token: str) -> TokenKind:
    """What kind of thing is this route token?"""
    t = token.strip().upper()
    if not t:
        return TokenKind.UNKNOWN
    if _AIRWAY.match(t):
        return TokenKind.AIRWAY
    if _PROCEDURE.match(t):
        return TokenKind.PROCEDURE
    if _AIRPORT.match(t) or _FIX.match(t):
        return TokenKind.POINT
    return TokenKind.UNKNOWN


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
            (name, float(lat), float(lon), f.get("type"), now, now, source),
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

    point_names: list[str] = []
    for t in tokens:
        kind = classify(t)
        if kind is TokenKind.AIRWAY:
            res.airways_dropped.append(t)
        elif kind is TokenKind.PROCEDURE:
            res.procedures_dropped.append(t)
        elif kind is TokenKind.POINT:
            point_names.append(t)
        else:
            res.unknown_tokens.append(t)

    found, missing = lookup(conn, point_names)
    res.missing = missing
    # Order matters: the route is a sequence, not a set.
    seen: set[str] = set()
    for name in point_names:
        if name in found and name not in seen:
            seen.add(name)
            lat, lon = found[name]
            res.points.append((name, lat, lon))
    return res
