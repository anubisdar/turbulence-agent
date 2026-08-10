#!/usr/bin/env python3
# install-to: scripts
"""
Run the corridor hypothesis search against the live AeroAPI.

Everything below has been exercised against canned payloads; this is the
first time it meets the real thing. Costs about five API calls, fewer once
the fix cache is warm.

The fix cache lives in the same SQLite file as the NTSB index. Both are
long-term reference data with unbounded validity - fix coordinates and
closed accident reports do not go stale - which is the same distinction
drawn in Checkpoint 2.1 between reference material and turbulence advisories
carrying an explicit TTL.

Usage:
    export AEROAPI_KEY=...
    python3 scripts/run_corridor_search.py                 # KPIT -> KBOS
    python3 scripts/run_corridor_search.py KJFK KLAX
    python3 scripts/run_corridor_search.py --graph         # via LangGraph
    python3 scripts/run_corridor_search.py --dry-run       # plan only
"""

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.reasoning.controller import Budget, search  # noqa: E402
from app.reasoning.generator import CorridorGenerator  # noqa: E402
from app.reasoning.geometry import overlap_fraction  # noqa: E402
from app.reasoning.graph import search_graph  # noqa: E402
from app.sources.aeroapi import AeroAPIClient  # noqa: E402
from app.sources.fixes import cache_stats, init_fixes  # noqa: E402
from app.retrieval.schema import connect  # noqa: E402

DEFAULT_DB = "data/retrieval.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("origin", nargs="?", default="KPIT")
    ap.add_argument("dest", nargs="?", default="KBOS")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--beam-width", type=int, default=2)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max-calls", type=int, default=8)
    ap.add_argument("--confidence", type=float, default=0.85)
    ap.add_argument("--width-nm", type=float, default=25.0)
    ap.add_argument("--graph", action="store_true",
                    help="run through the LangGraph controller")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"Corridor search: {args.origin} -> {args.dest}")
    print(f"  beam width {args.beam_width}, depth limit {args.depth}, "
          f"call cap {args.max_calls}, corridor half-width {args.width_nm} nm")
    print(f"  controller: {'LangGraph StateGraph' if args.graph else 'plain loop'}")

    if args.dry_run:
        print("\nDry run - no calls made.")
        return

    key = os.environ.get("AEROAPI_KEY")
    if not key:
        sys.exit("AEROAPI_KEY is not set")

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    init_fixes(conn)

    before = cache_stats(conn)
    print(f"  fix cache before: {before['total']} fix(es)")

    client = AeroAPIClient(api_key=key)
    gen = CorridorGenerator(client=client, conn=conn,
                            origin=args.origin.upper(),
                            dest=args.dest.upper(),
                            width_nm=args.width_nm)

    runner = search_graph if args.graph else search
    result = runner(gen,
                    beam_width=args.beam_width,
                    depth_limit=args.depth,
                    confidence_threshold=args.confidence,
                    budget=Budget(max_tool_calls=args.max_calls),
                    overlap_fn=gen.overlap_fn)

    print("\n" + "=" * 72)
    print("SEARCH TRACE")
    print("=" * 72)
    for line in result.trace():
        print(line)

    print("\n" + "=" * 72)
    print("CORRIDORS BUILT")
    print("=" * 72)
    print(f"  {'id':<18} {'length':>9} {'dogleg':>8} {'area':>12}  altitude")
    for cid, shape in gen.shapes.items():
        alt = "-"
        if shape.altitude_min_ft or shape.altitude_max_ft:
            alt = (f"FL{(shape.altitude_min_ft or 0)//100:03d}"
                   f"-FL{(shape.altitude_max_ft or 0)//100:03d}")
        print(f"  {cid:<18} {shape.length_nm:>7.1f}nm {shape.max_dogleg:>7.1f}° "
              f"{shape.area_nm2():>10,.0f}nm²  {alt}")

    depth1 = [cid for cid in gen.shapes
              if "/" not in cid]
    if len(depth1) > 1:
        print("\n" + "=" * 72)
        print("AIRSPACE OVERLAP  (dominance prunes at 0.80)")
        print("=" * 72)
        for a, b in combinations(depth1, 2):
            frac = overlap_fraction(gen.shapes[a], gen.shapes[b])
            flag = "  <- dominance range" if frac >= 0.80 else ""
            print(f"  {a:<12} vs {b:<12} {frac:>6.3f}{flag}")

    print("\n" + "=" * 72)
    print("OUTCOME")
    print("=" * 72)
    print(f"  stop reason   : {result.stop.value}")
    print(f"  truncated     : {result.truncated}")
    print(f"  depth reached : {result.depth_reached}")
    print(f"  nodes explored: {result.nodes_generated}")
    print(f"  api calls     : {result.calls_used}  "
          f"(client counted {client.calls_made})")
    print(f"  elapsed       : {result.elapsed}s")
    print(f"  winner        : {result.winner.id if result.winner else 'none'}")
    print(f"  reading       : {result.reading.value}")
    if result.survivors:
        print(f"  survivors     : {', '.join(c.id for c in result.survivors)}")

    print("\n  Turbulence evidence is not attached yet, so the reading is")
    print("  unresolved by construction. That is the Aviation Weather Center")
    print("  step, not a gap in the search.")

    if gen.notes:
        print("\n" + "=" * 72)
        print("GENERATOR NOTES")
        print("=" * 72)
        for n in gen.notes:
            print(f"  - {n}")

    if result.notes:
        print("\n" + "=" * 72)
        print("SEARCH NOTES")
        print("=" * 72)
        for n in result.notes:
            print(f"  - {n}")

    after = cache_stats(conn)
    print("\n" + "=" * 72)
    print(f"FIX CACHE: {before['total']} -> {after['total']} "
          f"(+{after['total'] - before['total']})")
    print("=" * 72)
    for kind, n in sorted(after["by_type"].items(), key=lambda x: -x[1]):
        print(f"  {n:>4}  {kind}")
    print("\n  Cached fixes are permanent. A repeat of this route will make")
    print("  fewer calls, and neighbouring routes that share waypoints will")
    print("  resolve without any donor fetch at all.")

    print(f"\n  Call paths: {', '.join(client.call_log)}")
    conn.close()


if __name__ == "__main__":
    main()
