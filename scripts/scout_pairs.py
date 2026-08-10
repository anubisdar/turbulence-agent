#!/usr/bin/env python3
# install-to: scripts
"""
Scout airport pairs for a workable demo route.

A corridor search needs a flight that has actually departed - a flown track
only exists for a flight that flew - and the pair endpoint returns a mix of
departed and scheduled. This checks that cheaply: one call per pair instead
of the five a full search costs, and it reports the departure times so a
sensible value can be picked for the time-of-day control.

Usage:
    export AEROAPI_KEY=...
    python3 scripts/scout_pairs.py                       # a default shortlist
    python3 scripts/scout_pairs.py KJFK-KLAX KORD-KSFO
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.sources.aeroapi import AeroAPIClient, AeroAPIError  # noqa: E402

SHORTLIST = [
    ("KJFK", "KLAX"), ("KLAX", "KJFK"),
    ("KORD", "KSFO"), ("KATL", "KLAX"),
    ("KEWR", "KSFO"), ("KBOS", "KSFO"),
    ("KIAD", "KLAX"), ("KPIT", "KBOS"),
]


def local_hhmm(stamp: str | None) -> str:
    return stamp[11:16] if stamp and len(stamp) >= 16 else "  ?  "


def main():
    key = os.environ.get("AEROAPI_KEY")
    if not key:
        sys.exit("AEROAPI_KEY is not set")

    args = [a for a in sys.argv[1:] if "-" in a]
    pairs = [tuple(a.upper().split("-", 1)) for a in args] or SHORTLIST

    client = AeroAPIClient(api_key=key)
    now = datetime.now(timezone.utc).strftime("%H:%M")
    print(f"Scouting {len(pairs)} pair(s). Current time {now} UTC.\n")
    print(f"  {'PAIR':<12} {'DEPARTED':>8} {'SCHED':>6}  TYPES / DEPARTURE TIMES (UTC)")
    print("  " + "-" * 74)

    workable = []
    for origin, dest in pairs:
        try:
            segments = client.flights_between(origin, dest)
        except AeroAPIError as e:
            print(f"  {origin}-{dest:<7} {'error':>8}  {e}")
            continue

        flown = [s for s in segments if s.has_flown]
        flown.sort(key=lambda s: s.actual_off or "")
        sched = len(segments) - len(flown)

        types = sorted({s.aircraft_type for s in flown if s.aircraft_type})
        times = " ".join(local_hhmm(s.actual_off) for s in flown[:6])
        print(f"  {origin}-{dest:<7} {len(flown):>8} {sched:>6}  "
              f"{','.join(types[:4]) or '-'}")
        if times:
            print(f"  {'':<12} {'':>8} {'':>6}  {times}")
        if flown:
            workable.append((origin, dest, len(flown), flown))

    print(f"\n  {client.calls_made} call(s) used.\n")

    if not workable:
        print("  No pair had a departed flight in the returned window. Try a")
        print("  different hour, or a pair with more daily frequency.")
        return

    workable.sort(key=lambda w: -w[2])
    origin, dest, count, flown = workable[0]
    mid = flown[len(flown) // 2]
    print(f"  Best bet: {origin} to {dest} - {count} departed flight(s).")
    print(f"  A good time value is {local_hhmm(mid.actual_off)} UTC, which "
          f"selects {mid.ident} ({mid.aircraft_type}).")
    print(f"  Its filed route: {mid.route or 'not recorded'}")


if __name__ == "__main__":
    main()
