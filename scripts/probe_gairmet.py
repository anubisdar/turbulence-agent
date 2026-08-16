#!/usr/bin/env python3
# install-to: scripts
"""
Probe the Aviation Weather Center G-AIRMET endpoint.

Same probe-first pattern that NTSB and AeroAPI both needed: three APIs into
this project, three payloads that did not match what the documentation
implied. Free and unauthenticated, so this costs nothing but a request.

Four questions decide how the weather layer is built:

  Q1  Which product names actually return turbulence? AIRMET "Tango" is not
      turbulence-specific, and CONUS textual AIRMETs were retired in
      January 2025. The typed products should be TURB-HI and TURB-LO, but
      the parameter spelling matters.
  Q2  Do the features carry an altitude band? Containment has to be three
      dimensional - a forecast for FL240 to FL390 says nothing about a
      flight at FL410 - and without top and base the polygon is unusable.
  Q3  What is the geometry? A Polygon ring is what shapely wants. A
      MultiPolygon or a bare coordinate list needs different handling.
  Q4  What validity window does each feature carry? G-AIRMETs are issued in
      three-hour steps, so a corridor at 18:00Z should not be tested against
      a forecast valid at 06:00Z.

Usage:
    python3 scripts/probe_gairmet.py
    python3 scripts/probe_gairmet.py --save
"""

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://aviationweather.gov/api/data"
OUT_DIR = Path("data/awc_probe")

# Bounding box loose enough to catch something somewhere over CONUS.
CONUS_BBOX = "-125,24,-66,50"


def get(path: str, params: dict, note: str = "") -> tuple[int, object, str]:
    url = f"{BASE}{path}?" + urllib.parse.urlencode(params)
    print(f"\nGET {path}")
    print(f"    params: {params}")
    if note:
        print(f"    why: {note}")
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "turbulence-agent-research/0.1 (capstone project)",
    })
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read().decode("utf-8")
            print(f"    HTTP {resp.status}  {len(raw):,} bytes")
            try:
                return resp.status, json.loads(raw), raw
            except json.JSONDecodeError:
                print(f"    not JSON. First 300 chars:\n{raw[:300]}")
                return resp.status, None, raw
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        print(f"    HTTP {e.code} {e.reason}")
        print(f"    body: {body}")
        return e.code, None, body
    except Exception as e:  # noqa: BLE001
        print(f"    failed: {type(e).__name__}: {e}")
        return 0, None, ""


def features_of(body) -> list:
    if isinstance(body, dict) and body.get("type") == "FeatureCollection":
        return body.get("features") or []
    if isinstance(body, list):
        return body
    return []


def describe(body, label: str) -> list:
    feats = features_of(body)
    print(f"    {label}: {len(feats)} feature(s)")
    if not feats:
        return []

    first = feats[0]
    if isinstance(first, dict) and "properties" in first:
        props = first.get("properties") or {}
        geom = first.get("geometry") or {}
    else:
        props, geom = first, {}

    print(f"    property keys: {sorted(props)[:18]}")
    if geom:
        print(f"    geometry type: {geom.get('type')}")
    return feats


def main():
    save = "--save" in sys.argv
    print("AWC G-AIRMET probe")
    print(f"now: {datetime.now(timezone.utc):%Y-%m-%dT%H:%M}Z")

    results = {}

    # ---- Q1: which product spelling returns turbulence
    attempts = [
        ("/gairmet", {"format": "json", "type": "turb-hi"},
         "Q1: typed high-altitude turbulence"),
        ("/gairmet", {"format": "json", "type": "turb-lo"},
         "Q1: typed low-altitude turbulence"),
        ("/gairmet", {"format": "json"},
         "Q1: everything, to see what types exist"),
        ("/airmet", {"format": "json"},
         "Q1: the older AIRMET endpoint, for comparison"),
    ]
    for path, params, note in attempts:
        status, body, _ = get(path, params, note)
        if status == 200 and body is not None:
            key = f"{path.strip('/')}_{params.get('type', 'all')}"
            feats = describe(body, "returned")
            results[key] = feats
            if save and feats:
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                (OUT_DIR / f"{key}.json").write_text(json.dumps(body, indent=2))
                print(f"    saved -> {OUT_DIR / key}.json")

    if not any(results.values()):
        print("\nNothing came back from any endpoint. Either the API moved or")
        print("there are no active G-AIRMETs right now. Both are worth knowing;")
        print("try again in a few hours before assuming the endpoint changed.")
        return

    # ---- pick the richest response for the remaining questions
    key = max(results, key=lambda k: len(results[k]))
    feats = results[key]
    print(f"\n{'=' * 68}")
    print(f"ANALYSING: {key}  ({len(feats)} features)")
    print("=" * 68)

    # ---- what hazard types are present
    def prop(f, *names):
        p = f.get("properties", f) if isinstance(f, dict) else {}
        for n in names:
            if p.get(n) is not None:
                return p[n]
        return None

    hazards = Counter(str(prop(f, "hazard", "product", "type")) for f in feats)
    print(f"\nhazard values: {dict(hazards)}")

    turb = [f for f in feats
            if "turb" in str(prop(f, "hazard", "product", "type")).lower()]
    print(f"turbulence features: {len(turb)}")
    sample = (turb or feats)[0]
    props = sample.get("properties", sample)

    # ---- Q2: altitude band
    print(f"\n=== Q2: ALTITUDE BAND ===")
    alt_keys = [k for k in props
                if any(t in k.lower()
                       for t in ("alt", "top", "base", "level", "fl"))]
    print(f"altitude-ish keys: {alt_keys}")
    for k in alt_keys:
        print(f"    {k:<16} {props[k]!r}")
    has_band = len(alt_keys) >= 2
    print(f"\n  ANSWER: usable altitude band present: {has_band}")
    if not has_band:
        print("  Without a top and a base the polygon cannot be tested in 3D,")
        print("  and a forecast for FL240-FL390 would wrongly match FL410.")

    # ---- Q3: geometry
    print(f"\n=== Q3: GEOMETRY ===")
    geom = sample.get("geometry")
    if geom:
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        print(f"    type: {gtype}")
        if coords:
            ring = coords[0] if gtype == "Polygon" else coords[0][0]
            print(f"    ring points: {len(ring)}")
            print(f"    first three: {ring[:3]}")
            print(f"    closed ring: {ring[0] == ring[-1]}")
        print(f"\n  ANSWER: {gtype} - "
              f"{'shapely takes this directly' if gtype in ('Polygon', 'MultiPolygon') else 'needs conversion'}")
    else:
        geom_keys = [k for k in props if "coord" in k.lower() or "geom" in k.lower()]
        print(f"    no geometry object. candidate keys: {geom_keys}")
        for k in geom_keys:
            print(f"    {k}: {str(props[k])[:160]}")

    # ---- Q4: validity window
    print(f"\n=== Q4: VALIDITY WINDOW ===")
    time_keys = [k for k in props
                 if any(t in k.lower()
                        for t in ("time", "valid", "issue", "expire", "from", "to"))]
    for k in time_keys:
        print(f"    {k:<18} {props[k]!r}")
    print(f"\n  ANSWER: {'window present' if len(time_keys) >= 2 else 'INCOMPLETE'}")
    print("  A corridor at 18:00Z must not be scored against a forecast")
    print("  valid at 06:00Z, so both ends of the window are needed.")

    # ---- full sample
    print(f"\n=== ONE FULL FEATURE (truncated) ===")
    print(json.dumps(sample, indent=2)[:2500])

    if save:
        print(f"\nSaved responses under {OUT_DIR}/")


if __name__ == "__main__":
    main()
