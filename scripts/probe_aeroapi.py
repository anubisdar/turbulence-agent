#!/usr/bin/env python3
# install-to: scripts
"""
Probe AeroAPI to find out what the Personal tier actually gives us.

Four questions decide the corridor generator's design, and none of them are
answerable from the documentation:

  Q1  Does /flights/{id}/route return coordinates, or only waypoint names?
      Names would mean an FAA navaid/fix database is a hidden dependency for
      the filed-route corridor.
  Q2  How dense is /flights/{id}/track? Positions every minute or every ten?
      That sets the resolution of the flown-track corridor.
  Q3  Does /airports/{a}/routes/{b} work on Personal, and in what format?
      That is the published-airway corridor source.
  Q4  How far back does /flights/{ident}?start= actually reach on this key?
      The /history/* endpoints are tier-gated; this is the alternative.

Costs real money. Personal tier includes about $5/month of usage, so the
probe is deliberately small and counts every call it makes. Calls are spaced
because Personal throttles bursts.

Usage:
    export AEROAPI_KEY=...
    python3 scripts/probe_aeroapi.py                 # KPIT -> KBOS
    python3 scripts/probe_aeroapi.py KJFK KLAX
    python3 scripts/probe_aeroapi.py --dry-run       # show the plan only
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

BASE = "https://aeroapi.flightaware.com/aeroapi"
OUT_DIR = Path("data/aeroapi_probe")
SPACING_SECONDS = 2.0

CALLS = 0
KEY = os.environ.get("AEROAPI_KEY", "")


def get(path: str, params: dict | None = None, note: str = "") -> dict | None:
    """One authenticated GET. Counts against the monthly allowance."""
    global CALLS
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    CALLS += 1
    print(f"\n[call {CALLS}] GET {path}")
    if params:
        print(f"          params: {params}")
    if note:
        print(f"          why: {note}")

    req = urllib.request.Request(url, headers={
        "x-apikey": KEY,
        "Accept": "application/json; charset=UTF-8",
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        print(f"          HTTP {resp.status}")
        return body
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        print(f"          HTTP {e.code} {e.reason}")
        print(f"          body: {detail}")
        if e.code == 401:
            print("          NOTE: AeroAPI returns 401 for tier-gating as well as")
            print("                for a bad key. If other calls work, this is a")
            print("                tier restriction, not an auth problem.")
        if e.code == 429:
            print("          NOTE: rate limited. Personal throttles bursts.")
        return None
    finally:
        time.sleep(SPACING_SECONDS)


def save(name: str, obj) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{name}.json").write_text(json.dumps(obj, indent=2))
    print(f"          saved -> {OUT_DIR / name}.json")


def looks_like_coords(obj) -> bool:
    """Does this structure carry latitude/longitude anywhere shallow?"""
    def walk(o, depth=0):
        if depth > 4:
            return False
        if isinstance(o, dict):
            keys = {k.lower() for k in o}
            if {"latitude", "longitude"} <= keys or {"lat", "lon"} <= keys:
                return True
            return any(walk(v, depth + 1) for v in o.values())
        if isinstance(o, list):
            return any(walk(v, depth + 1) for v in o[:3])
        return False
    return walk(obj)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    origin = args[0] if args else "KPIT"
    dest = args[1] if len(args) > 1 else "KBOS"

    print(f"AeroAPI probe: {origin} -> {dest}")
    print("Plan (about 6 calls, roughly a few cents):")
    print("  1. /account/usage                  confirm the key and see spend")
    print(f"  2. /airports/{origin}/routes/{dest}      Q3 published airway routings")
    print(f"  3. /airports/{origin}/flights/to/{dest}  find a real recent flight")
    print("  4. /flights/{ident}?start=...       Q4 how far back the window reaches")
    print("  5. /flights/{id}/route              Q1 coordinates or waypoint names")
    print("  6. /flights/{id}/track              Q2 position density")
    if dry:
        print("\nDry run - no calls made.")
        return
    if not KEY:
        sys.exit("AEROAPI_KEY is not set")

    # ---- key check and current spend
    usage = get("/account/usage", note="confirm the key works and see spend")
    if usage:
        save("account_usage", usage)
        print(f"          {json.dumps(usage)[:300]}")

    # ---- Q3: published airway routings
    routes = get(f"/airports/{origin}/routes/{dest}",
                 note="Q3: is this the published-airway corridor source?")
    if routes:
        save("airport_routes", routes)
        items = routes.get("routes") or []
        print(f"          {len(items)} routing(s) returned")
        if items:
            print(f"          first: {json.dumps(items[0])[:300]}")
        print(f"          carries coordinates: {looks_like_coords(routes)}")

    # ---- find a real flight on the pair
    #
    # This endpoint returns *itineraries*, not flights: each entry is
    # {"segments": [flight, ...]} so that one-stop connections can be
    # expressed. The flight objects live one level down.
    pair = get(f"/airports/{origin}/flights/to/{dest}",
               {"max_pages": 1},
               note="find a real recent flight to inspect")
    ident = fa_id = None
    if pair:
        save("airport_pair_flights", pair)
        itineraries = pair.get("flights") or []
        segments = [seg for it in itineraries for seg in (it.get("segments") or [])]
        print(f"          {len(itineraries)} itinerary(s), {len(segments)} segment(s)")

        statuses = {}
        for seg in segments:
            statuses[seg.get("status")] = statuses.get(seg.get("status"), 0) + 1
        print(f"          statuses: {statuses}")

        # A flown track only exists for a flight that has actually departed.
        # Prefer the most recently departed, fall back to anything with an id.
        flown = [s for s in segments if s.get("actual_off")]
        flown.sort(key=lambda s: s.get("actual_off") or "", reverse=True)
        pick = None
        if flown:
            pick = flown[0]
            print(f"          {len(flown)} segment(s) have departed")
        else:
            print("          none have departed yet in this window")
            with_id = [s for s in segments if s.get("fa_flight_id")]
            if with_id:
                pick = with_id[0]
                print("          falling back to a scheduled flight - the track")
                print("          call will likely return nothing")

        if pick:
            ident = pick.get("ident")
            fa_id = pick.get("fa_flight_id")
            print(f"          picked {ident}  ({pick.get('aircraft_type')})  "
                  f"{pick.get('status')}")
            print(f"          actual_off    : {pick.get('actual_off')}")
            print(f"          route string  : {pick.get('route')!r}")
            print(f"          route_distance: {pick.get('route_distance')} nm")
            print(f"          filed_altitude: {pick.get('filed_altitude')}")

    if not fa_id:
        print("\nNo usable flight on that pair. Try a busier pair, e.g.")
        print("    python3 scripts/probe_aeroapi.py KJFK KLAX")
        print(f"\nTOTAL CALLS: {CALLS}")
        return

    # ---- Q4: how far back does the non-history window reach
    start = (datetime.now(timezone.utc) - timedelta(days=9)).strftime("%Y-%m-%d")
    hist = get(f"/flights/{ident}", {"start": start, "max_pages": 1},
               note="Q4: how far back does start= reach on this tier?")
    if hist:
        save("flights_ident_window", hist)
        fl = hist.get("flights") or []
        dates = sorted({(f.get("scheduled_out") or "")[:10] for f in fl if f.get("scheduled_out")})
        print(f"          {len(fl)} flight(s), dates {dates[:1]} .. {dates[-1:]}")
        print(f"          requested start={start}")
        print(f"          oldest returned: {dates[0] if dates else 'none'}")

    # ---- Q1: filed route - coordinates or names?
    route = get(f"/flights/{fa_id}/route",
                note="Q1: does the filed route carry coordinates?")
    if route:
        save("flight_route", route)
        fixes = route.get("fixes") or []
        print(f"          {len(fixes)} fix(es)")
        if fixes:
            print(f"          first fix: {json.dumps(fixes[0])}")
        has = looks_like_coords(route)
        print(f"\n  === Q1 ANSWER: coordinates present: {has} ===")
        if has:
            print("      Filed-route corridors can be built directly.")
        else:
            print("      Waypoint names only. Building a filed-route corridor")
            print("      would require an FAA navaid/fix database (NASR).")

    # ---- Q2: track density
    track = get(f"/flights/{fa_id}/track",
                {"include_estimated_positions": "true"},
                note="Q2: how dense are the flown-track positions?")
    if track:
        save("flight_track", track)
        pos = track.get("positions") or []
        print(f"          {len(pos)} position(s)")
        if pos:
            print(f"          first: {json.dumps(pos[0])}")
            stamps = [p["timestamp"] for p in pos if p.get("timestamp")]
            if len(stamps) > 1:
                t0 = datetime.fromisoformat(stamps[0].replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(stamps[-1].replace("Z", "+00:00"))
                span = (t1 - t0).total_seconds() / 60.0
                print(f"          span {span:.0f} min, "
                      f"mean gap {span/max(len(stamps)-1,1)*60:.0f}s")
            alts = [p.get("altitude") for p in pos if p.get("altitude")]
            if alts:
                print(f"          altitude {min(alts)}..{max(alts)} (hundreds of ft)")
            srcs = {}
            for p in pos:
                srcs[p.get("update_type")] = srcs.get(p.get("update_type"), 0) + 1
            print(f"          update types: {srcs}")
        print(f"\n  === Q2 ANSWER: {len(pos)} positions for the flown corridor ===")

    print(f"\nTOTAL CALLS: {CALLS}")
    print(f"Saved under {OUT_DIR}/")
    print("\nRe-run /account/usage later to see what this cost.")


if __name__ == "__main__":
    main()
