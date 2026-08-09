#!/usr/bin/env python3
"""
Measure aircraft type resolution across the full cached Part 121 corpus.

Offline. Reads data/ntsb_cache/ only - does not touch NTSB.

Usage:
    python3 scripts/coverage_types.py

Reports:
  - resolution rate by confidence, per distinct string and weighted by case
  - every UNRESOLVED string, so gaps are visible rather than averaged away
  - the canonical type buckets the corpus collapses into
  - MAX vs NG separation, since that is the claim the design rests on
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.aircraft_types import Confidence, resolve  # noqa: E402

CACHE = Path("data/ntsb_cache")
AIRLINE_PARTS = {"121"}


def load_part121() -> list:
    if not CACHE.exists():
        sys.exit(f"No cache at {CACHE}. Run scope_ntsb.py first.")
    seen, cases = set(), []
    for f in sorted(CACHE.glob("*.json")):
        for case in json.loads(f.read_text()):
            mkey = case.get("cm_mkey")
            if mkey in seen:
                continue
            parts = {
                (v.get("regulationFlightConductedUnder") or "").strip()
                for v in (case.get("cm_vehicles") or [])
            }
            if parts & AIRLINE_PARTS:
                seen.add(mkey)
                cases.append(case)
    return cases


def main():
    cases = load_part121()
    print(f"Part 121 cases in cache: {len(cases):,}\n")

    by_conf_strings = Counter()
    by_conf_cases = Counter()
    unresolved = Counter()
    family_only = Counter()
    buckets = Counter()
    seen_strings = set()
    unresolved_cases = 0

    for case in cases:
        case_confs = []
        for v in (case.get("cm_vehicles") or []):
            make = (v.get("make") or "").strip()
            model = (v.get("model") or "").strip()
            if not (make or model):
                continue
            r = resolve(make, model)
            case_confs.append(r.confidence)

            pair = (make.upper(), model.upper())
            if pair not in seen_strings:
                seen_strings.add(pair)
                by_conf_strings[r.confidence.value] += 1
                if r.confidence is Confidence.UNRESOLVED:
                    unresolved[pair] += 1
                elif r.confidence is Confidence.FAMILY_ONLY:
                    family_only[pair] += 1
            if r.usable:
                buckets[r.key] += 1

        if case_confs:
            best = min(case_confs, key=lambda c: list(Confidence).index(c))
            by_conf_cases[best.value] += 1
            if all(c is Confidence.UNRESOLVED for c in case_confs):
                unresolved_cases += 1

    total_strings = sum(by_conf_strings.values())
    print("--- Resolution by distinct make/model string ---")
    for k in ("exact", "derived", "family_only", "unresolved"):
        n = by_conf_strings.get(k, 0)
        print(f"  {k:<14} {n:>5,}  {n/total_strings*100:5.1f}%")
    print(f"  {'TOTAL':<14} {total_strings:>5,}")

    total_cases = sum(by_conf_cases.values())
    print("\n--- Resolution weighted by case (best vehicle in each) ---")
    for k in ("exact", "derived", "family_only", "unresolved"):
        n = by_conf_cases.get(k, 0)
        print(f"  {k:<14} {n:>5,}  {n/total_cases*100:5.1f}%")
    usable = total_cases - by_conf_cases.get("unresolved", 0)
    print(f"\n  RETRIEVABLE BY TYPE: {usable:,}/{total_cases:,} "
          f"({usable/total_cases*100:.1f}%)")
    print(f"  cases with no usable type at all: {unresolved_cases:,}")

    print("\n--- UNRESOLVED strings (every one; fix or accept explicitly) ---")
    if unresolved:
        for (mk, md), n in unresolved.most_common():
            print(f"  {mk:<32} | {md}")
    else:
        print("  none")

    print("\n--- FAMILY_ONLY strings (variant absent in source data) ---")
    for (mk, md), n in list(family_only.most_common())[:40]:
        print(f"  {mk:<32} | {md}")
    if len(family_only) > 40:
        print(f"  ... and {len(family_only)-40} more")

    print(f"\n--- Canonical buckets: {len(buckets)} distinct types ---")
    for key, n in buckets.most_common(40):
        print(f"  {n:>5,}  {key}")

    print("\n--- The claim under test: MAX never shares a bucket with NG ---")
    max_keys = {k: n for k, n in buckets.items() if "MAX" in k}
    ng_keys = {k: n for k, n in buckets.items()
               if k.startswith("737-") and k not in max_keys}
    print(f"  MAX buckets: {max_keys or 'none in corpus'}")
    print(f"  737 NG/Classic buckets: {ng_keys}")
    overlap = set(max_keys) & set(ng_keys)
    print(f"  overlap: {overlap or 'none'}")
    assert not overlap, "MAX and NG collided - the filter is unsafe"


if __name__ == "__main__":
    main()
