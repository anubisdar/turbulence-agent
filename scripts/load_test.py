#!/usr/bin/env python3
# install-to: scripts
"""
Run twenty real routes against a deployed instance and report what happened.

Not a load test in the throughput sense - the agent is a single worker doing
one metered search at a time, and hammering it concurrently would measure
FlightAware's rate limiter rather than anything about this system. What this
measures is breadth: twenty different routes, chosen to exercise different
parts of the geometry and different data availability, run one at a time
with a pause between them.

ROUTE SELECTION. Every pair is at least 250 nautical miles apart. Below
that the flown track, the filed route and the great circle are nearly
identical, dominance prunes almost everything, and the search stops being
interesting. The set deliberately includes:

  - long transcontinental routes, where the projection has to work hard
  - short-but-not-tiny hops, where corridors converge
  - routes over water, where pilot reports are sparse
  - routes over the Rockies, where turbulence forecasts are common
  - a transpacific pair, to keep the antimeridian path exercised

Costs real API calls: roughly 8 per route, so about 160 for a full run.
Pace it or the upstream rate limiter will start refusing, which the report
will show as degraded searches rather than as failures.

Usage:
    python3 scripts/load_test.py --host https://turbulence.adeptsecurity.net \\
        --user demo --password ...
    python3 scripts/load_test.py --routes 5          # a shorter pass
    python3 scripts/load_test.py --explain           # include the model
"""

import argparse
import base64
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

#: (origin, destination, why this pair is here)
ROUTES = [
    ("KSEA", "KBOS", "transcontinental, northern"),
    ("KLAX", "KJFK", "transcontinental, the busiest pair in the country"),
    ("KDEN", "KATL", "over the plains, out of high terrain"),
    ("KORD", "KDFW", "midwest to Texas, heavy traffic"),
    ("KSFO", "KORD", "over the Rockies, forecast turbulence common"),
    ("KIAH", "KLAS", "southwest desert crossing"),
    ("KMSP", "KPHX", "long diagonal, sparse middle"),
    ("KMIA", "KEWR", "coastal, mostly over water"),
    ("KSLC", "KMCO", "west to Florida, crosses several air masses"),
    ("KPDX", "KDEN", "northwest into the Front Range"),
    ("KCLT", "KDEN", "southeast to the Rockies"),
    ("KBNA", "KLAX", "mid-south transcontinental"),
    ("KDTW", "KMCO", "midwest to Florida, well-flown"),
    ("KPHL", "KORD", "short-haul, corridors converge"),
    ("KSAN", "KSFO", "west coast, short but distinct routings"),
    ("KAUS", "KATL", "gulf coast"),
    ("KBOS", "KMIA", "full east coast run"),
    ("KLAS", "KSEA", "desert to the Pacific northwest"),
    ("KANC", "KSEA", "Alaska to the lower 48, sparse reporting"),
    ("KSEA", "RJTT", "transpacific, crosses the antimeridian"),
    ("KDCA", "KSJC", "transcontinental, mid-atlantic to the bay"),
    ("KBWI", "KOAK", "transcontinental, low-cost trunk route"),
    ("KOKC", "KMDW", "southern plains into the midwest"),
    ("KSTL", "KSAT", "mississippi valley to south Texas"),
    ("KOAK", "KMCI", "over the Sierra and the Rockies"),
    ("KABQ", "KMDW", "high desert to the Great Lakes"),
    ("KMKE", "KTPA", "Great Lakes to the gulf coast"),
    ("KTPA", "KBWI", "coastal, much of it over water"),
    ("KMCI", "KFLL", "plains to south Florida"),
    ("KBOI", "KOAK", "high desert into northern California"),
    ("KRDU", "KMDW", "piedmont to the midwest"),
    ("KMSY", "KBWI", "gulf coast to the mid-atlantic"),
    ("KCLE", "KTPA", "Great Lakes to Florida, well flown"),
    ("KLGA", "KBUF", "short-haul, corridors converge"),
    ("KBUR", "KOAK", "California coast, short but distinct routings"),
    ("KHOU", "KTPA", "across the gulf"),
    ("KBDL", "KFLL", "full east coast run"),
    ("KELP", "KHOU", "desert to the gulf"),
    ("PHNL", "KSMF", "mid-Pacific, no pilot reports for most of it"),
    ("KSJC", "RJAA", "transpacific, crosses the antimeridian"),
]


def request(host, path, body, auth, timeout=180):
    url = f"{host}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if data else "GET",
        headers={"Content-Type": "application/json"})
    if auth:
        req.add_header("Authorization", "Basic " + base64.b64encode(
            auth.encode()).decode())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--user")
    ap.add_argument("--password")
    ap.add_argument("--routes", type=int, default=len(ROUTES),
                    help="how many of the twenty to run")
    ap.add_argument("--cap", type=int, default=16)
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--pause", type=float, default=8.0,
                    help="seconds between routes; below about 5 the upstream "
                         "rate limiter starts refusing")
    args = ap.parse_args()

    host = args.host.rstrip("/")
    auth = (f"{args.user}:{args.password}"
            if args.user and args.password else None)
    routes = ROUTES[:max(1, min(args.routes, len(ROUTES)))]

    print(f"Running {len(routes)} routes against {host}")
    print(f"  cap {args.cap} calls · explainer "
          f"{'on' if args.explain else 'off'} · {args.pause}s between\n")

    header = (f"  {'route':<12} {'nm':>6} {'calls':>6} {'secs':>7} {'cut':>4} "
              f"{'deg':>4}  {'reading':<11} {'winner':<14} explainer")
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = []
    started_all = time.time()

    for i, (origin, dest, note) in enumerate(routes, 1):
        body = {
            "origin": origin, "dest": dest,
            "max_tool_calls": args.cap,
            "include_turbulence": True,
            "include_reputation": False,
            "include_explanation": args.explain,
        }
        try:
            data = request(host, "/api/search/corridors", body, auth)
        except urllib.error.HTTPError as e:
            print(f"  {origin}-{dest:<7} HTTP {e.code}: "
                  f"{e.read().decode('utf-8', 'replace')[:70]}")
            results.append({"route": f"{origin}-{dest}", "error": e.code})
            time.sleep(args.pause)
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  {origin}-{dest:<7} {type(e).__name__}: {e}")
            results.append({"route": f"{origin}-{dest}", "error": str(e)})
            time.sleep(args.pause)
            continue

        out = data.get("outcome") or {}
        wx = out.get("turbulence") or {}
        ex = data.get("explanation") or {}
        corridors = data.get("corridors") or []
        winner = next((c for c in corridors
                       if c.get("id") == out.get("winner")), {})

        row = {
            "route": f"{origin}-{dest}", "note": note,
            "length_nm": winner.get("length_nm"),
            "calls": out.get("calls_used", 0),
            "elapsed": out.get("elapsed_seconds", 0),
            "truncated": bool(out.get("truncated")),
            "degraded": bool(out.get("degraded")),
            "degraded_reason": (out.get("degraded_reasons") or [None])[0],
            "reading": out.get("reading"),
            "observed": (wx.get("observed") or {}).get("reading"),
            "forecast": (wx.get("forecast") or {}).get("reading"),
            "winner": out.get("winner"),
            "nodes": out.get("nodes_generated", 0),
            "explainer": ("-" if not args.explain
                          else "ok" if ex.get("source") == "model"
                          else "rejected"),
            "reject": (ex.get("rejected") or [None])[0],
        }
        results.append(row)

        print(f"  {row['route']:<12} "
              f"{(row['length_nm'] or 0):>6.0f} {row['calls']:>6} "
              f"{row['elapsed']:>7.1f} "
              f"{'YES' if row['truncated'] else '-':>4} "
              f"{'YES' if row['degraded'] else '-':>4}  "
              f"{str(row['reading']):<11} {str(row['winner']):<14} "
              f"{row['explainer']}")

        if i < len(routes):
            time.sleep(args.pause)

    # ------------------------------------------------------------- report
    ok = [r for r in results if "error" not in r]
    print(f"\n\033[1mOutcome\033[0m  ({len(ok)} of {len(results)} completed, "
          f"{time.time() - started_all:.0f}s total)")

    if not ok:
        print("  Nothing completed. Check the host and credentials.")
        return 2

    readings = {}
    for r in ok:
        readings[r["reading"]] = readings.get(r["reading"], 0) + 1
    resolved = sum(n for k, n in readings.items() if k != "unresolved")
    print(f"  resolved       {resolved} of {len(ok)} "
          f"({resolved / len(ok):.0%})")
    for reading, count in sorted(readings.items(), key=lambda x: -x[1]):
        print(f"    {str(reading):<12} {count}")

    print(f"\n\033[1mSources\033[0m")
    pairs = {}
    for r in ok:
        obs = r["observed"] != "unresolved"
        fc = r["forecast"] != "unresolved"
        key = ("both" if obs and fc else "forecast only" if fc
               else "pilots only" if obs else "neither")
        pairs[key] = pairs.get(key, 0) + 1
    for key, count in sorted(pairs.items(), key=lambda x: -x[1]):
        print(f"  {key:<15} {count}")

    print(f"\n\033[1mCost and time\033[0m")
    calls = [r["calls"] for r in ok]
    times = [r["elapsed"] for r in ok if r["elapsed"]]
    print(f"  calls          {sum(calls)} total, "
          f"{statistics.mean(calls):.1f} mean, {max(calls)} worst")
    if times:
        print(f"  seconds        {min(times):.1f} to {max(times):.1f}, "
              f"median {statistics.median(times):.1f}")

    truncated = [r for r in ok if r["truncated"]]
    degraded = [r for r in ok if r["degraded"]]
    if truncated:
        print(f"\n\033[33m  {len(truncated)} truncated\033[0m: "
              f"{', '.join(r['route'] for r in truncated)}")
        print("    A budget stopped these, so they explored less of the tree.")
    if degraded:
        print(f"\n\033[33m  {len(degraded)} degraded\033[0m: "
              f"{', '.join(r['route'] for r in degraded)}")
        reasons = {}
        for r in degraded:
            key = str(r["degraded_reason"])[:60]
            reasons[key] = reasons.get(key, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {count}x  {reason}")

    if args.explain:
        accepted = sum(1 for r in ok if r["explainer"] == "ok")
        print(f"\n\033[1mExplainer\033[0m")
        print(f"  {accepted} of {len(ok)} accepted "
              f"({accepted / len(ok):.0%})")
        for r in ok:
            if r["explainer"] == "rejected":
                print(f"    {r['route']:<12} {str(r['reject'])[:64]}")

    thin = [r for r in ok if r["nodes"] < 4]
    if thin:
        print(f"\n\033[1mWorth a look\033[0m")
        for r in thin:
            print(f"  {r['route']:<12} only {r['nodes']} node(s) "
                  f"� {r['note']}")

    print(f"\n  Status page: {host}/status")
    return 1 if len(ok) < len(results) else 0


if __name__ == "__main__":
    sys.exit(main())
