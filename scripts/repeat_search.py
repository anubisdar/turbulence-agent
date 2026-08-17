#!/usr/bin/env python3
# install-to: scripts
"""
Repeat one search and watch what it costs.

The claim in every write-up is that cost falls with use: route fixes arrive
with coordinates, they are cached permanently, and a second search of the
same pair is cheaper than the first. That has been observed once, informally,
going from five calls to four. This measures it.

What a healthy result looks like: call count drops after the first run and
then holds flat. What a problem looks like: it never settles, which means
something is being fetched that should have been cached, and every search
pays for it forever.

The reading is also compared across runs. Corridor selection is
deterministic, so the same route within a few minutes should reach the same
verdict. A reading that flickers between runs would mean something
non-deterministic reached the scoring path, which is the property this whole
project rests on.

Costs real API calls. Six runs of a domestic route is roughly 25 calls, so
run it deliberately rather than casually.

Usage:
    ./scripts/serve.sh                    # in another terminal
    python3 scripts/repeat_search.py
    python3 scripts/repeat_search.py --origin KJFK --dest KLAX --runs 8
    python3 scripts/repeat_search.py --fixtures        # free, no API spend
    python3 scripts/repeat_search.py --explain         # include the model
"""

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request


def post(host: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{host}/api/search/corridors", method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--origin", default="KPIT")
    ap.add_argument("--dest", default="KBOS")
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--cap", type=int, default=16)
    ap.add_argument("--fixtures", action="store_true",
                    help="replay saved payloads, no API spend")
    ap.add_argument("--explain", action="store_true",
                    help="include the model-written explanation")
    ap.add_argument("--pause", type=float, default=1.0,
                    help="seconds between runs")
    args = ap.parse_args()

    host = args.host.rstrip("/")
    body = {
        "origin": args.origin.upper(), "dest": args.dest.upper(),
        "max_tool_calls": args.cap, "include_turbulence": True,
        "include_reputation": False,
        "include_explanation": args.explain,
        "use_fixtures": args.fixtures,
    }

    print(f"Repeating {body['origin']} to {body['dest']} "
          f"{args.runs} times{'  (fixtures)' if args.fixtures else ''}")
    print(f"  cap {args.cap} calls · explainer "
          f"{'on' if args.explain else 'off'}\n")

    header = (f"  {'run':>3} {'calls':>6} {'fixes':>12} {'nodes':>6} "
              f"{'secs':>7} {'cut':>4}  {'reading':<11} {'winner':<12} "
              f"explainer")
    print(header)
    print("  " + "-" * (len(header) - 2))

    runs = []
    for i in range(1, args.runs + 1):
        try:
            started = time.time()
            data = post(host, body)
            wall = time.time() - started
        except urllib.error.HTTPError as e:
            print(f"  {i:>3}  HTTP {e.code}: "
                  f"{e.read().decode('utf-8', 'replace')[:120]}")
            return 2
        except Exception as e:  # noqa: BLE001
            print(f"  {i:>3}  {type(e).__name__}: {e}")
            return 2

        out = data["outcome"]
        cache = data.get("fix_cache", {})
        ex = data.get("explanation") or {}
        explainer = ("-" if not args.explain
                     else ("accepted" if ex.get("source") == "model"
                           else "rejected"))

        runs.append({
            "calls": out.get("calls_used", 0),
            "fixes_before": cache.get("before", 0),
            "fixes_after": cache.get("after", 0),
            "nodes": out.get("nodes_generated", 0),
            "elapsed": out.get("elapsed_seconds", wall),
            "reading": out.get("reading"),
            "winner": out.get("winner"),
            "truncated": bool(out.get("truncated")),
            "stop": out.get("stop"),
            "explainer": explainer,
            "rejected": ex.get("rejected") or [],
        })
        r = runs[-1]
        print(f"  {i:>3} {r['calls']:>6} "
              f"{r['fixes_before']:>5} -> {r['fixes_after']:<4} "
              f"{r['nodes']:>6} {r['elapsed']:>7.2f} "
              f"{'YES' if r['truncated'] else '-':>4}  "
              f"{str(r['reading']):<11} {str(r['winner']):<12} {explainer}")
        if i < args.runs:
            time.sleep(args.pause)

    # ------------------------------------------------------------ analysis
    #
    # The verdict depends on whether there was a saving available to make.
    # A route whose cache is already warm has nothing left to learn, so a
    # flat cost is the correct outcome rather than a failure to converge.
    # Judging cost without looking at cache growth first reports a healthy
    # warm route as broken.
    growth = [r["fixes_after"] - r["fixes_before"] for r in runs]
    already_warm = all(g == 0 for g in growth)

    print("\n\033[1mCost\033[0m")
    calls = [r["calls"] for r in runs]
    first, rest = calls[0], calls[1:]
    print(f"  first run      {first} calls")
    if rest:
        print(f"  later runs     {min(rest)}-{max(rest)} calls "
              f"(mean {statistics.mean(rest):.1f})")

        if max(rest) < first:
            print(f"  \033[32mconverged\033[0m: repeat searches cost "
                  f"{first - max(rest)} fewer call(s) than the first")
        elif already_warm and len(set(calls)) == 1:
            print(f"  \033[32mstable\033[0m: every run costs {calls[0]}. "
                  f"The cache was already warm before this test, so there "
                  f"was no saving left to make.")
            print(f"  To measure the saving itself, run against a pair this "
                  f"agent has not searched before.")
        elif len(set(rest)) == 1:
            print(f"  \033[33mflat\033[0m: every run costs {rest[0]} while "
                  f"the cache is still growing, so the calls being made are "
                  f"not the ones the cache covers")
        else:
            print(f"  \033[31mnot settling\033[0m: cost still varies after "
                  f"the first run, so something is being refetched that "
                  f"should be cached")

        # A warm cache and a steady cost still leaves the question of what
        # those calls are. Per-pair work should be cached; per-route-string
        # work is not.
        if already_warm and calls[0] > 4:
            print(f"  Note: {calls[0]} calls on a warm cache is more than the "
                  f"four a domestic search needs. Check the call log for "
                  f"fetches repeated per alternate routing rather than per "
                  f"pair.")

    print("\n\033[1mFix cache\033[0m")
    print(f"  {runs[0]['fixes_before']} at the start, "
          f"{runs[-1]['fixes_after']} at the end")
    print(f"  added per run: {growth}")
    if already_warm:
        print("  \033[32mwarm before this test started\033[0m: every "
              "waypoint was already known, so no lookups were needed")
    elif all(g == 0 for g in growth[1:]):
        print("  \033[32mwarmed on the first run\033[0m: nothing new was "
              "needed afterwards")
    else:
        print("  \033[33mstill learning\033[0m: later runs are still caching "
              "fixes, which is expected on a route with many alternates")

    cut = [r for r in runs if r["truncated"]]
    if cut:
        stops = {r["stop"] for r in cut}
        print(f"\n\033[33m{len(cut)} of {len(runs)} runs were truncated\033[0m "
              f"({', '.join(sorted(map(str, stops)))})")
        print("  A truncated search explores less of the tree, so it can "
              "reach a different corridor from the same query. That is not "
              "the scoring changing, it is the search being cut short.")

    print("\n\033[1mDeterminism\033[0m")
    readings = {r["reading"] for r in runs}
    winners = {r["winner"] for r in runs}
    if len(readings) == 1:
        print(f"  \033[32mstable\033[0m: every run read "
              f"{readings.pop()}")
    elif cut:
        print(f"  readings {sorted(map(str, readings))}, and "
              f"{len(cut)} run(s) were truncated. Check whether the varying "
              f"runs are the truncated ones before suspecting the scoring.")
    else:
        print(f"  \033[31mvaried\033[0m: readings {sorted(map(str, readings))}")
        print("  Turbulence data is live, so a change over minutes can be "
              "real. A change within seconds would not be.")
    if len(winners) == 1:
        print(f"  winner stable: {winners.pop()}")
    else:
        print(f"  \033[33mwinner varied\033[0m: {sorted(map(str, winners))}")

    if args.explain:
        print("\n\033[1mExplainer\033[0m")
        accepted = sum(1 for r in runs if r["explainer"] == "accepted")
        rate = accepted / len(runs)
        print(f"  {accepted} of {len(runs)} accepted ({rate:.0%})")
        reasons: dict[str, int] = {}
        for r in runs:
            for reason in r["rejected"]:
                key = str(reason).split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
        for reason, n in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {n}x  {reason}")
        if rate < 0.5:
            print("  \033[33mrejecting more than half\033[0m: worth reading "
                  "the discarded text before deciding whether the checks are "
                  "too tight or the model is genuinely reaching")

    print("\n\033[1mLatency\033[0m")
    times = [r["elapsed"] for r in runs]
    print(f"  {min(times):.2f}s to {max(times):.2f}s, "
          f"median {statistics.median(times):.2f}s")
    if len(times) > 1 and times[0] > 3 * statistics.median(times[1:]):
        print(f"  The first run took {times[0]:.1f}s against a later median "
              f"of {statistics.median(times[1:]):.1f}s. That is the sentence "
              f"transformer loading, which happens once per process, not per "
              f"search.")
    if not args.fixtures:
        print("  Nearly all of the rest is waiting on external APIs rather "
              "than on the search itself.")

    print(f"\n  {sum(calls)} API calls spent across {len(runs)} runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
