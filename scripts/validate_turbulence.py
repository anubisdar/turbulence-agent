#!/usr/bin/env python3
# install-to: scripts
"""
Validate the turbulence layer end to end against a running instance.

Written after a live IAD to LAX search returned MODERATE but did not render
the two source boxes under the reading. That could be a stale page, a
missing payload field, or a rendering bug, and guessing between them wastes
more time than checking.

Every check states what it expects and why it matters. A failure here is a
real defect, not a style note.

Usage:
    ./scripts/serve.sh            # in another terminal
    python3 scripts/validate_turbulence.py
    python3 scripts/validate_turbulence.py --host http://blueadept:8000
    python3 scripts/validate_turbulence.py --origin KPIT --dest KBOS
    python3 scripts/validate_turbulence.py --fixtures    # no API spend
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PASS, FAIL, WARN, INFO = "PASS", "FAIL", "WARN", "INFO"
_RESULTS: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> bool:
    _RESULTS.append((status, name, detail))
    colour = {"PASS": "\033[32m", "FAIL": "\033[31m",
              "WARN": "\033[33m", "INFO": "\033[36m"}[status]
    print(f"  {colour}{status:<4}\033[0m {name}")
    if detail:
        for line in str(detail).splitlines():
            print(f"         {line}")
    return status == PASS


def section(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m")


def post(host: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{host}{path}", method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def get(host: str, path: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(f"{host}{path}", timeout=30) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


# ------------------------------------------------------------------ checks


def check_reachable(host: str) -> dict | None:
    section("Service")
    try:
        status, raw = get(host, "/api/health")
    except Exception as e:  # noqa: BLE001
        record(FAIL, "service reachable",
               f"{type(e).__name__}: {e}\nStart it with ./scripts/serve.sh")
        return None
    if status != 200:
        record(FAIL, "service reachable", f"HTTP {status}")
        return None
    health = json.loads(raw)
    record(PASS, "service reachable", host)
    record(PASS if health.get("database_present") else WARN,
           "database present", health.get("database"))
    record(PASS if health.get("aeroapi_key_configured") else WARN,
           "AeroAPI key configured",
           "" if health.get("aeroapi_key_configured")
           else "live searches will be refused")
    return health


def check_payload_shape(data: dict) -> None:
    """The fields the verdict strip reads. A rename here empties the UI."""
    section("Response shape")

    wx = (data.get("outcome") or {}).get("turbulence")
    if wx is None:
        record(FAIL, "outcome.turbulence present",
               "The page reads this to draw the two source boxes. Missing "
               "means the running service predates the change - restart it, "
               "or app/web/service.py was not installed.")
        return
    record(PASS, "outcome.turbulence present")

    for side in ("observed", "forecast"):
        block = wx.get(side)
        if not isinstance(block, dict):
            record(FAIL, f"turbulence.{side} present")
            continue
        missing = [k for k in ("reading", "count") if k not in block]
        record(FAIL if missing else PASS, f"turbulence.{side} shape",
               f"missing {missing}" if missing else
               f"{block.get('reading')} from {block.get('count')} source(s)")

    record(PASS if "disagree" in wx else FAIL, "turbulence.disagree present",
           "a flag, not an average" if "disagree" in wx else "")

    kept = [c for c in data.get("corridors", []) if c.get("kept")]
    if kept:
        want = ("reading", "observed_reading", "forecast_reading",
                "sources_disagree")
        missing = sorted({k for c in kept for k in want if k not in c})
        record(FAIL if missing else PASS, "surviving corridors carry readings",
               f"missing {missing}" if missing else
               f"{len(kept)} corridor(s) carry their own readings")


def check_sources_held_apart(data: dict) -> None:
    """The design decision this layer exists for: observed and forecast are
    separate opinions, and the combined reading is the worse of them."""
    section("Sources held apart")

    wx = (data.get("outcome") or {}).get("turbulence") or {}
    obs = (wx.get("observed") or {}).get("reading")
    fc = (wx.get("forecast") or {}).get("reading")
    combined = wx.get("reading") or data["outcome"].get("reading")

    record(INFO, "readings",
           f"pilots {obs} · forecast {fc} · combined {combined}")

    order = {"smooth": 0, "light": 1, "moderate": 2, "severe": 3, "extreme": 4}
    known = [r for r in (obs, fc) if r in order]

    if len(known) == 2:
        expected = max(known, key=lambda r: order[r])
        record(PASS if combined == expected else FAIL,
               "combined reading is the worse of the two",
               f"expected {expected}, got {combined}")
        should_split = obs != fc
        record(PASS if wx.get("disagree") == should_split else FAIL,
               "disagreement flag matches the readings",
               f"disagree={wx.get('disagree')}, readings "
               f"{'differ' if should_split else 'match'}")
    elif len(known) == 1:
        record(PASS if combined == known[0] else FAIL,
               "one silent source does not soften the other",
               f"only {known[0]} spoke, combined is {combined}")
        record(PASS if not wx.get("disagree") else FAIL,
               "a silent source is a gap, not a disagreement")
    else:
        record(INFO, "neither source spoke",
               "nothing to compare on this route right now")


def check_absence_is_not_smooth(data: dict) -> None:
    """The rule the whole project turns on."""
    section("Absence is not smooth")

    wx = (data.get("outcome") or {}).get("turbulence") or {}
    obs = (wx.get("observed") or {}).get("reading")
    fc = (wx.get("forecast") or {}).get("reading")
    combined = wx.get("reading") or data["outcome"].get("reading")

    if obs == "unresolved" and fc == "unresolved":
        record(PASS if combined == "unresolved" else FAIL,
               "no evidence yields unresolved, never smooth",
               f"combined reading is {combined}")
    else:
        record(PASS, "at least one source spoke", f"pilots {obs}, forecast {fc}")

    notes = " ".join(data.get("notes", []) + data.get("generator_notes", []))
    if obs == "unresolved":
        record(PASS if "not a report of smooth air" in notes
               or "unknown, not clear" in notes else WARN,
               "silence on the observed side is explained",
               "" if "smooth air" in notes else
               "no note explains why there were no pilot reports")

    covers = [c.get("components", {}).get("coverage", 0)
              for c in data.get("corridors", []) if c.get("kept")]
    if covers and all(c == 0 for c in covers):
        pruned_for_coverage = [
            c for c in data.get("corridors", [])
            if not c.get("kept") and "coverage" in (c.get("reason") or "").lower()]
        record(PASS if not pruned_for_coverage else FAIL,
               "zero coverage did not prune anything",
               "every corridor scored zero on coverage and none was "
               "eliminated for it")


def check_evidence_is_survivor_only(data: dict) -> None:
    """Gathering for a corridor about to be pruned spends a call on an
    answer nobody reads."""
    section("Evidence gathered for survivors only")

    with_reading = {c["id"] for c in data.get("corridors", [])
                    if c.get("reading") and c["reading"] != "unresolved"}
    kept = {c["id"] for c in data.get("corridors", []) if c.get("kept")}
    stray = with_reading - kept

    record(PASS if not stray else FAIL,
           "only surviving corridors carry evidence",
           f"pruned corridors with readings: {sorted(stray)}" if stray
           else f"{len(with_reading)} of {len(kept)} survivors have a reading")

    used = data["outcome"].get("calls_used", 0)
    cap = data["request"].get("max_tool_calls", 0)
    record(PASS if used <= cap else FAIL, "call budget respected",
           f"{used} of {cap} calls")


def check_altitude_branches(data: dict) -> None:
    """Same ground path, different air. A report at FL340 is not evidence
    about FL315."""
    section("Altitude branches")

    children = [c for c in data.get("corridors", [])
                if "/" in c.get("id", "") and c.get("kept")]
    if len(children) < 2:
        record(INFO, "fewer than two altitude branches survived",
               "nothing to compare on this search")
        return

    bands = {(c.get("altitude_min_ft"), c.get("altitude_max_ft"))
             for c in children}
    record(PASS if len(bands) > 1 else FAIL,
           "branches cover different altitude bands",
           " · ".join(f"{c['id']} {c.get('altitude_min_ft')}-"
                      f"{c.get('altitude_max_ft')}" for c in children))

    readings = {c.get("reading") for c in children}
    if len(readings) == 1:
        record(INFO, "branches share a reading",
               f"all {readings.pop()} - the same evidence covered both bands, "
               f"which is possible when a forecast spans them")
    else:
        record(PASS, "branches reached different readings",
               " · ".join(f"{c['id']} {c.get('reading')}" for c in children))


def check_toggle(host: str, base_body: dict) -> None:
    """Switching turbulence off must leave the reading unresolved rather
    than implying calm air."""
    section("Turbulence toggle")

    body = {**base_body, "include_turbulence": False}
    try:
        off = post(host, "/api/search/corridors", body)
    except Exception as e:  # noqa: BLE001
        record(FAIL, "search with turbulence off", f"{type(e).__name__}: {e}")
        return

    reading = off["outcome"].get("reading")
    record(PASS if reading == "unresolved" else FAIL,
           "off yields unresolved", f"got {reading}")
    record(PASS if reading != "smooth" else FAIL,
           "off never yields smooth")

    notes = " ".join(off.get("notes", []))
    record(PASS if "switched off" in notes else WARN,
           "the reason is stated in the notes",
           "" if "switched off" in notes else "no note explains the absence")

    record(PASS if off["request"].get("include_turbulence") is False else FAIL,
           "the flag is echoed back")


def check_narration(data: dict) -> None:
    section("Agent Processing View")

    beats = data.get("narration") or []
    if not beats:
        record(FAIL, "narration present")
        return
    record(PASS, "narration present", f"{len(beats)} beats")

    concepts = {b.get("concept") for b in beats}
    record(PASS if "Evidence" in concepts else WARN,
           "evidence gathering is narrated",
           "" if "Evidence" in concepts else
           "no Evidence beat - sources may not have been reached")

    text = " ".join(b.get("text", "") for b in beats)
    record(PASS if "Coverage is never allowed to prune" in text
           or "coverage" in text.lower() else WARN,
           "the coverage guardrail is stated")

    wx = (data.get("outcome") or {}).get("turbulence") or {}
    if wx.get("disagree"):
        flagged = [b for b in beats
                   if b.get("kind") == "caution" and "disagree" in b.get("text", "")]
        record(PASS if flagged else FAIL,
               "a disagreement is flagged, not stated quietly")


def check_page(host: str) -> None:
    """The page is served separately from the API, so it can be stale even
    when the service is current."""
    section("Web page")

    status, html = get(host, "/")
    if status != 200:
        record(WARN, "page served", f"HTTP {status} - API only")
        return
    record(PASS, "page served", f"{len(html):,} bytes")

    # Each entry accepts any of its alternatives, so a rename in the page
    # does not fail a check that is really asking whether the feature is
    # present at all.
    checks = [
        (('id="turbulence"',), "turbulence toggle present"),
        (("include_turbulence",), "toggle is sent with the request"),
        (('class="sources',), "two-source strip present"),
        (("split-note",), "disagreement note present"),
        (("o.turbulence", "outcome.turbulence", "wx.summary"),
         "page reads the turbulence summary"),
    ]
    stale = False
    for needles, name in checks:
        ok = any(n in html for n in needles)
        record(PASS if ok else FAIL, name,
               "" if ok else
               f"none of {list(needles)} found in the served page")
        stale = stale or not ok
    if stale:
        record(WARN, "page looks stale",
               "run ./scripts/install_static.sh, then hard refresh the "
               "browser with Ctrl+Shift+R")

    if "cartocdn.com/voyager/" in html and "rastertiles" not in html:
        record(FAIL, "light basemap tile path",
               "CARTO serves voyager from /rastertiles/voyager/. Fix with:\n"
               "sed -i 's|cartocdn.com/voyager/|cartocdn.com/rastertiles/"
               "voyager/|' app/web/static/index.html")
    else:
        record(PASS, "light basemap tile path")

    # the wording defect spotted on screen
    m = re.search(r"Derived from observations along the selected corridor",
                  html)
    record(WARN if m else PASS, "verdict wording",
           "the page says 'Derived from observations' even when only a "
           "forecast spoke. Observations and forecasts are different things."
           if m else "")


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--origin", default="KIAD")
    ap.add_argument("--dest", default="KLAX")
    ap.add_argument("--time", default=None, help="departure HH:MM UTC")
    ap.add_argument("--fixtures", action="store_true",
                    help="replay saved payloads, no API spend")
    ap.add_argument("--save", action="store_true",
                    help="write the response to data/validate_turbulence.json")
    args = ap.parse_args()

    host = args.host.rstrip("/")
    print(f"Validating {host}  ·  {args.origin} to {args.dest}"
          f"{'  (fixtures)' if args.fixtures else ''}")

    if check_reachable(host) is None:
        return 2

    body = {
        "origin": args.origin, "dest": args.dest,
        "include_turbulence": True, "include_reputation": False,
        "use_fixtures": args.fixtures,
    }
    if args.time:
        body["departure_time"] = args.time

    section("Search")
    try:
        data = post(host, "/api/search/corridors", body)
    except urllib.error.HTTPError as e:
        record(FAIL, "search completed",
               f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
        return 2
    except Exception as e:  # noqa: BLE001
        record(FAIL, "search completed", f"{type(e).__name__}: {e}")
        return 2

    out = data["outcome"]
    record(PASS, "search completed",
           f"{out.get('nodes_generated')} nodes · {out.get('calls_used')} calls "
           f"· {out.get('elapsed_seconds')}s · winner {out.get('winner')}")
    record(INFO, "reading", out.get("reading"))

    if args.save:
        path = Path("data/validate_turbulence.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        record(INFO, "response saved", str(path))

    check_payload_shape(data)
    check_sources_held_apart(data)
    check_absence_is_not_smooth(data)
    check_evidence_is_survivor_only(data)
    check_altitude_branches(data)
    check_narration(data)
    check_toggle(host, body)
    check_page(host)

    # ---- summary
    counts = {s: sum(1 for st, _, _ in _RESULTS if st == s)
              for s in (PASS, FAIL, WARN, INFO)}
    print(f"\n\033[1mSummary\033[0m")
    print(f"  {counts[PASS]} passed · {counts[FAIL]} failed · "
          f"{counts[WARN]} warnings")

    if counts[FAIL]:
        print("\n  Failures:")
        for status, name, detail in _RESULTS:
            if status == FAIL:
                print(f"    - {name}")
                if detail:
                    print(f"      {detail.splitlines()[0]}")
    if counts[WARN]:
        print("\n  Warnings:")
        for status, name, detail in _RESULTS:
            if status == WARN:
                print(f"    - {name}")

    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
