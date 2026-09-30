# install-to: scripts
"""Show the raw evidence behind a turbulence reading for one corridor.

Answers a narrower, more useful question than "is the code buggy": for
*this* route, right now, what did AWC actually hand back, and why did it
combine to whatever severity you saw in the app?

Uses the exact same functions the live search calls
(`app.reasoning.evidence.gather_evidence`, `app.sources.awc.fetch_pireps`,
`app.sources.gairmet.GairmetClient`) against a real corridor - not a
synthetic report, not a mock. This has to run somewhere with real network
access to aviationweather.gov; the sandbox this was written in does not
have that, which is why this is a script for you to run rather than
something already run and pasted back.

Usage:
    python scripts/inspect_evidence.py KPIT KBOS
    python scripts/inspect_evidence.py KPIT KBOS --width-nm 40 --alt 35000

Picks the great-circle line between the two airports as the corridor -
the same fallback shape the app itself uses when no filed route or flown
track is available - so this works for any pair without needing a live
flight to already exist between them.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, ".")

from app.reasoning.evidence import gather_evidence, sync_pirep_fetcher
from app.reasoning.geometry import build_corridor, great_circle
from app.sources.aeroapi import AeroAPIClient
from app.sources.awc import fetch_pireps
from app.sources.gairmet import GairmetClient


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("origin", help="ICAO code, e.g. KPIT")
    p.add_argument("dest", help="ICAO code, e.g. KBOS")
    p.add_argument("--width-nm", type=float, default=25.0)
    p.add_argument("--alt", type=int, default=35000,
                   help="cruise altitude in feet, for both the corridor "
                        "band and the vertical check against advisories")
    args = p.parse_args()

    api_key = os.environ.get("AEROAPI_KEY")
    if not api_key:
        sys.exit("AEROAPI_KEY is not set in this shell - needed to look up "
                 "the two airports' coordinates. (Same variable the server "
                 "itself uses.)")
    aero = AeroAPIClient(api_key=api_key)

    origin = aero.airport(args.origin)
    dest = aero.airport(args.dest)
    if origin is None or dest is None:
        sys.exit(f"AeroAPI did not recognise one of these codes: "
                 f"{args.origin!r}, {args.dest!r}")

    points = great_circle((origin.latitude, origin.longitude),
                          (dest.latitude, dest.longitude))
    shape = build_corridor(points, width_nm=args.width_nm,
                           altitude_min_ft=args.alt - 2000,
                           altitude_max_ft=args.alt + 2000)

    now = datetime.now(timezone.utc)
    result = gather_evidence(
        shape, fetch_pireps=sync_pirep_fetcher(fetch_pireps),
        gairmet_client=GairmetClient(), when=now)
    ev = result.evidence

    print(f"{args.origin.upper()} -> {args.dest.upper()}  "
         f"(width {args.width_nm} nm, FL{(args.alt - 2000)//100:03d}-"
         f"FL{(args.alt + 2000)//100:03d})")
    print(f"as of {now.isoformat()}")
    print()
    print(f"COMBINED READING: {ev.reading.value}")
    print()
    print(f"  observed (PIREPs): {ev.observed_reading.value}  "
         f"count={ev.observed_count}  "
         f"considered={result.reports_considered}  "
         f"inside_corridor={result.reports_inside}")
    print(f"  forecast (G-AIRMET): {ev.forecast_reading.value}  "
         f"count={ev.forecast_count}  "
         f"considered={result.advisories_considered}  "
         f"inside_laterally={result.advisories_inside}  "
         f"matched_altitude={len(result.matched_advisories)}")
    print()
    print("Why:")
    for n in result.notes:
        print(f"  - {n}")
    print()
    print("Summary the app would show:")
    print(f"  {result.summary}")

    if result.raw_advisories:
        print()
        print(f"Raw G-AIRMET advisories considered "
             f"({len(result.raw_advisories)}):")
        for a in result.raw_advisories:
            print(f"  - {a.severity.value}  FL{(a.base_ft or 0)//100:03d}-"
                 f"FL{(a.top_ft or 0)//100:03d}  valid {a.valid_time}")


if __name__ == "__main__":
    main()
