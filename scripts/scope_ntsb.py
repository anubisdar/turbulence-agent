#!/usr/bin/env python3
"""
Step 2: measure the corpus before committing to a scope.

Pulls NTSB aviation cases over a date range, caching each chunk to disk so
NTSB is hit exactly once per window. Auto-splits any window that returns the
500-record cap. Then reports the numbers that decide ingest scope:

  - how many cases are Part 121 (scheduled air carrier) vs everything else
  - what make/model strings actually look like for Part 121
  - whether narratives are present often enough to be worth embedding
  - roughly how many chunks we'd end up embedding

Usage:
    python3 scope_ntsb.py 2000-01-01 2026-08-05
    python3 scope_ntsb.py 2000-01-01 2026-08-05 --report-only

Cache lives in data/ntsb_cache/. Safe to re-run; it will not re-fetch.
Stdlib only.
"""

import json
import io
import sys
import time
import zipfile
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

FILE_EXPORT_URL = "https://data.ntsb.gov/carol-main-public/api/Query/FileExport"
RESULT_SET_SIZE = 500
SLEEP_SECONDS = 1.5

HEADERS = {
    "Accept": "*/*",
    "Content-Type": "application/json",
    "Origin": "https://data.ntsb.gov",
    "User-Agent": "turbulence-agent-research/0.1 (capstone project; contact matthew.darlage@gmail.com)",
}

CACHE = Path("data/ntsb_cache")

# FAR Parts that mean "the kind of flight my user is actually on"
AIRLINE_PARTS = {"121"}
# Part 135 is commuter/on-demand - adjacent, reported separately so you can decide
COMMUTER_PARTS = {"135"}


# ---------------------------------------------------------------- fetching

def _date_rule(value: str, operator: str) -> dict:
    return {
        "RuleType": "Simple",
        "Values": [value],
        "Columns": ["Event.EventDate"],
        "Operator": operator,
        "overrideColumn": "",
        "selectedOption": {
            "FieldName": "EventDate", "DisplayText": "Event date",
            "Columns": ["Event.EventDate"], "Selectable": True,
            "InputType": "Date", "RuleType": 0, "Options": None,
            "TargetCollection": "cases", "UnderDevelopment": True,
        },
    }


def build_payload(start: str, end: str) -> dict:
    return {
        "QueryGroups": [{
            "QueryRules": [
                _date_rule(start, "is on or after"),
                _date_rule(end, "is on or before"),
                {
                    "RuleType": "Simple", "Values": ["Aviation"],
                    "Columns": ["Event.Mode"], "Operator": "is", "overrideColumn": "",
                    "selectedOption": {
                        "FieldName": "Mode", "DisplayText": "Investigation mode",
                        "Columns": ["Event.Mode"], "Selectable": True,
                        "InputType": "Dropdown", "RuleType": 0, "Options": None,
                        "TargetCollection": "cases", "UnderDevelopment": True,
                    },
                },
            ],
            "AndOr": "and", "inLastSearch": False, "editedSinceLastSearch": False,
        }],
        "AndOr": "and",
        "TargetCollection": "cases",
        "ExportFormat": "data",
        "SessionId": 227230,
        "ResultSetSize": RESULT_SET_SIZE,
        "SortDescending": True,
    }


def fetch_window(start: date, end: date) -> list:
    """Fetch one window, using cache. Recursively splits if the cap is hit."""
    key = CACHE / f"{start.isoformat()}_{end.isoformat()}.json"
    if key.exists():
        records = json.loads(key.read_text())
        print(f"  cached  {start} .. {end}  ({len(records):>4} records)")
        return records

    body = json.dumps(build_payload(start.isoformat(), end.isoformat())).encode()
    req = urllib.request.Request(FILE_EXPORT_URL, data=body, headers=HEADERS, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code} on {start} .. {end}: {e.reason}")
        raise
    time.sleep(SLEEP_SECONDS)

    if raw[:2] != b"PK":
        raise RuntimeError(f"Non-ZIP response for {start}..{end}: {raw[:300]!r}")

    zf = zipfile.ZipFile(io.BytesIO(raw))
    names = [n for n in zf.namelist() if n.lower().endswith(".json")]
    records = json.loads(zf.read(names[0]).decode("utf-8"))

    if len(records) >= RESULT_SET_SIZE and (end - start).days > 1:
        mid = start + (end - start) / 2
        print(f"  CAP HIT {start} .. {end} -> splitting")
        left = fetch_window(start, mid)
        right = fetch_window(mid + timedelta(days=1), end)
        return left + right

    CACHE.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps(records))
    print(f"  fetched {start} .. {end}  ({len(records):>4} records)")
    return records


def quarters(start: date, end: date):
    cur = start
    while cur <= end:
        q_end = min(cur + timedelta(days=91), end)
        yield cur, q_end
        cur = q_end + timedelta(days=1)


def load_all(start: date, end: date, report_only: bool) -> list:
    seen, out = set(), []
    for a, b in quarters(start, end):
        if report_only:
            key = CACHE / f"{a.isoformat()}_{b.isoformat()}.json"
            if not key.exists():
                continue
            recs = json.loads(key.read_text())
        else:
            recs = fetch_window(a, b)
        for r in recs:
            mkey = r.get("cm_mkey")
            if mkey not in seen:
                seen.add(mkey)
                out.append(r)
    return out


# ---------------------------------------------------------------- reporting

def parts_of(case) -> set:
    return {
        (v.get("regulationFlightConductedUnder") or "").strip()
        for v in (case.get("cm_vehicles") or [])
    } - {""}


def types_of(case):
    for v in (case.get("cm_vehicles") or []):
        make = (v.get("make") or "").strip()
        model = (v.get("model") or "").strip()
        if make or model:
            yield make, model


def textlen(case, field) -> int:
    return len(case.get(field) or "")


def report(cases: list):
    print("\n" + "=" * 70)
    print(f"TOTAL CASES: {len(cases):,}")
    print("=" * 70)

    part_counter = Counter()
    for c in cases:
        for p in parts_of(c) or {"(none)"}:
            part_counter[p] += 1
    print("\n--- FAR Part distribution (cases, may double-count multi-vehicle) ---")
    for p, n in part_counter.most_common(20):
        print(f"  {p:<10} {n:>7,}   {n/len(cases)*100:5.1f}%")

    airline = [c for c in cases if parts_of(c) & AIRLINE_PARTS]
    commuter = [c for c in cases if parts_of(c) & COMMUTER_PARTS]
    print(f"\n  Part 121 (scheduled air carrier): {len(airline):,}")
    print(f"  Part 135 (commuter / on-demand):  {len(commuter):,}")

    if not airline:
        print("\n!!! No Part 121 cases found. Check the field name before going further.")
        return

    print("\n--- Part 121 cases by year ---")
    by_year = Counter((c.get("cm_eventDate") or "????")[:4] for c in airline)
    for y in sorted(by_year):
        print(f"  {y}  {'#' * min(by_year[y], 60)} {by_year[y]}")

    print("\n--- Part 121 event type ---")
    for k, n in Counter(c.get("cm_eventType") or "?" for c in airline).most_common():
        print(f"  {k:<8} {n:>6,}")

    print("\n--- Part 121 MAKE strings (raw, as entered) ---")
    makes = Counter()
    for c in airline:
        for mk, _ in types_of(c):
            makes[mk.upper()] += 1
    for mk, n in makes.most_common(40):
        print(f"  {n:>5,}  {mk}")

    print("\n--- Part 121 MAKE + MODEL pairs (this is your normalization problem) ---")
    pairs = Counter()
    for c in airline:
        for mk, md in types_of(c):
            pairs[(mk.upper(), md.upper())] += 1
    for (mk, md), n in pairs.most_common(80):
        print(f"  {n:>5,}  {mk:<28} | {md}")
    print(f"\n  distinct make strings:  {len(makes):,}")
    print(f"  distinct make+model:    {len(pairs):,}")

    print("\n--- Boeing/Airbus model strings only (variant collapse preview) ---")
    for target in ("BOEING", "AIRBUS"):
        models = Counter()
        for c in airline:
            for mk, md in types_of(c):
                if target in mk.upper():
                    models[md.upper()] += 1
        print(f"\n  {target}: {len(models)} distinct model strings")
        for md, n in models.most_common(50):
            print(f"    {n:>5,}  {md}")

    print("\n--- Narrative availability (Part 121) ---")
    fields = ["cm_probableCause", "analysisNarrative", "factualNarrative", "prelimNarrative"]
    for f in fields:
        present = [textlen(c, f) for c in airline if textlen(c, f) > 0]
        if present:
            present.sort()
            print(f"  {f:<20} present in {len(present):>5,}/{len(airline):,} "
                  f"({len(present)/len(airline)*100:4.1f}%)  "
                  f"median {present[len(present)//2]:>6,}  max {present[-1]:>7,} chars")
        else:
            print(f"  {f:<20} present in     0/{len(airline):,}")

    print("\n--- Narrative availability by era (analysisNarrative, Part 121) ---")
    era = defaultdict(lambda: [0, 0])
    for c in airline:
        y = (c.get("cm_eventDate") or "????")[:4]
        bucket = f"{y[:3]}0s" if y[:4].isdigit() else "?"
        era[bucket][0] += 1
        if textlen(c, "analysisNarrative") > 0:
            era[bucket][1] += 1
    for k in sorted(era):
        tot, have = era[k]
        print(f"  {k}  {have:>5,}/{tot:<5,}  {have/tot*100:5.1f}%")

    print("\n--- Estimated embedding load (Part 121, ~1200 char chunks) ---")
    chunks = 0
    for c in airline:
        for f in ("cm_probableCause", "analysisNarrative", "factualNarrative"):
            n = textlen(c, f)
            if n:
                chunks += max(1, -(-n // 1200))
    print(f"  approx chunks: {chunks:,}")
    print(f"  at ~40 chunks/sec on CPU: ~{chunks/40/60:.1f} minutes to embed")

    print("\n--- Same estimate if you included Part 135 too ---")
    chunks135 = 0
    for c in commuter:
        for f in ("cm_probableCause", "analysisNarrative", "factualNarrative"):
            n = textlen(c, f)
            if n:
                chunks135 += max(1, -(-n // 1200))
    print(f"  additional chunks: {chunks135:,}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    report_only = "--report-only" in sys.argv
    if len(args) != 2:
        print(__doc__)
        sys.exit(1)
    start, end = date.fromisoformat(args[0]), date.fromisoformat(args[1])

    print(f"Range {start} .. {end}   (cache: {CACHE}/)")
    cases = load_all(start, end, report_only)
    if not cases:
        print("No cases loaded.")
        sys.exit(2)
    report(cases)


if __name__ == "__main__":
    main()
