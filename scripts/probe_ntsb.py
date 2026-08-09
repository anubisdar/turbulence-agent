#!/usr/bin/env python3
"""
Step 1 probe: what does a CAROL FileExport record actually contain?

Not ingest code. Pulls one narrow date slice, saves the raw response,
and reports on field structure so we can design chunking and the
metadata filter against reality instead of assumption.

Usage:
    python3 probe_ntsb.py 2024-01-01 2024-01-31

Stdlib only. Nothing to install.
"""

import json
import io
import sys
import zipfile
import urllib.request
import urllib.error
from datetime import date
from pathlib import Path

FILE_EXPORT_URL = "https://data.ntsb.gov/carol-main-public/api/Query/FileExport"

HEADERS = {
    "Accept": "*/*",
    "Content-Type": "application/json",
    "Origin": "https://data.ntsb.gov",
    "User-Agent": "turbulence-agent-research/0.1 (capstone project; contact matthew.darlage@gmail.com)",
}

OUT_DIR = Path("data/ntsb_probe")


def _date_rule(value: str, operator: str) -> dict:
    return {
        "RuleType": "Simple",
        "Values": [value],
        "Columns": ["Event.EventDate"],
        "Operator": operator,
        "overrideColumn": "",
        "selectedOption": {
            "FieldName": "EventDate",
            "DisplayText": "Event date",
            "Columns": ["Event.EventDate"],
            "Selectable": True,
            "InputType": "Date",
            "RuleType": 0,
            "Options": None,
            "TargetCollection": "cases",
            "UnderDevelopment": True,
        },
    }


def build_payload(start: str, end: str, result_set_size: int = 500) -> dict:
    return {
        "QueryGroups": [
            {
                "QueryRules": [
                    _date_rule(start, "is on or after"),
                    _date_rule(end, "is on or before"),
                    {
                        "RuleType": "Simple",
                        "Values": ["Aviation"],
                        "Columns": ["Event.Mode"],
                        "Operator": "is",
                        "overrideColumn": "",
                        "selectedOption": {
                            "FieldName": "Mode",
                            "DisplayText": "Investigation mode",
                            "Columns": ["Event.Mode"],
                            "Selectable": True,
                            "InputType": "Dropdown",
                            "RuleType": 0,
                            "Options": None,
                            "TargetCollection": "cases",
                            "UnderDevelopment": True,
                        },
                    },
                ],
                "AndOr": "and",
                "inLastSearch": False,
                "editedSinceLastSearch": False,
            }
        ],
        "AndOr": "and",
        "TargetCollection": "cases",
        "ExportFormat": "data",
        "SessionId": 227230,
        "ResultSetSize": result_set_size,
        "SortDescending": True,
    }


def fetch(payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(FILE_EXPORT_URL, data=body, headers=HEADERS, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        print(f"HTTP {resp.status}  content-type={resp.headers.get('Content-Type')}")
        return resp.read()


def walk_keys(obj, prefix="", out=None, depth=0):
    """Collect dotted key paths so we can see nested structure, not just top level."""
    if out is None:
        out = {}
    if depth > 4:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                out.setdefault(path, type(v).__name__)
                walk_keys(v, path, out, depth + 1)
            else:
                sample = "" if v is None else str(v)
                out.setdefault(path, f"{type(v).__name__} (len {len(sample)})")
    elif isinstance(obj, list) and obj:
        walk_keys(obj[0], f"{prefix}[]", out, depth + 1)
    return out


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    start, end = sys.argv[1], sys.argv[2]
    date.fromisoformat(start)
    date.fromisoformat(end)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Requesting Aviation cases {start} .. {end}\n")
    try:
        raw = fetch(build_payload(start, end))
    except urllib.error.HTTPError as e:
        print(f"HTTP ERROR {e.code}: {e.reason}")
        print(e.read()[:2000].decode("utf-8", "replace"))
        sys.exit(2)

    raw_path = OUT_DIR / f"raw_{start}_{end}.bin"
    raw_path.write_bytes(raw)
    print(f"Saved {len(raw):,} bytes -> {raw_path}\n")

    if raw[:2] != b"PK":
        print("NOT a ZIP. First 1000 bytes:\n")
        print(raw[:1000].decode("utf-8", "replace"))
        sys.exit(3)

    zf = zipfile.ZipFile(io.BytesIO(raw))
    print("ZIP members:")
    for n in zf.namelist():
        print(f"  {n}  ({zf.getinfo(n).file_size:,} bytes)")
    print()

    json_names = [n for n in zf.namelist() if n.lower().endswith(".json")]
    if not json_names:
        print("No .json inside the ZIP. Stop here and tell me what the members are.")
        sys.exit(4)

    data = json.loads(zf.read(json_names[0]).decode("utf-8"))
    (OUT_DIR / "extracted.json").write_bytes(json.dumps(data, indent=2)[:5_000_000].encode())

    # Find the list of cases wherever it lives
    if isinstance(data, list):
        records = data
        print(f"Top level: list of {len(records):,}")
    elif isinstance(data, dict):
        print(f"Top level: dict with keys {list(data.keys())}")
        records = None
        for k, v in data.items():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                records = v
                print(f"Treating '{k}' as the record list ({len(v):,} records)")
                break
        if records is None:
            print("Could not find a record list. Dumping top level:")
            print(json.dumps(data, indent=2)[:3000])
            sys.exit(5)
    else:
        print(f"Unexpected top-level type: {type(data)}")
        sys.exit(6)

    if len(records) >= 500:
        print("\n*** Hit the 500 cap. Real ingest must chunk by date. ***")

    print("\n=== FIELD PATHS (first record) ===")
    keys = walk_keys(records[0])
    for path in sorted(keys):
        print(f"  {path:<60} {keys[path]}")

    # The two questions that decide the retrieval design
    print("\n=== ANSWERING THE TWO DESIGN QUESTIONS ===")
    flat = {k.lower(): k for k in keys}

    narrative_hits = [v for k, v in flat.items() if any(
        t in k for t in ("narrative", "analysis", "probablecause", "cause", "factual", "remarks", "summary")
    )]
    print("\nQ1 narrative-ish fields:")
    print("  " + ("\n  ".join(narrative_hits) if narrative_hits else "NONE FOUND"))

    aircraft_hits = [v for k, v in flat.items() if any(
        t in k for t in ("make", "model", "series", "aircraft", "registration", "amateur", "engine")
    )]
    print("\nQ2 aircraft-type fields:")
    print("  " + ("\n  ".join(aircraft_hits) if aircraft_hits else "NONE FOUND"))

    # Longest string field across a sample - that's the embeddable text
    print("\n=== LONGEST TEXT FIELDS (sampled over 50 records) ===")
    longest = {}
    def scan(o, prefix=""):
        if isinstance(o, dict):
            for k, v in o.items():
                p = f"{prefix}.{k}" if prefix else k
                if isinstance(v, str):
                    longest[p] = max(longest.get(p, 0), len(v))
                elif isinstance(v, (dict, list)):
                    scan(v, p)
        elif isinstance(o, list):
            for item in o[:3]:
                scan(item, f"{prefix}[]")
    for r in records[:50]:
        scan(r)
    for p, n in sorted(longest.items(), key=lambda x: -x[1])[:15]:
        print(f"  {n:>8,}  {p}")

    print("\n=== ONE FULL RECORD (truncated to 6000 chars) ===")
    print(json.dumps(records[0], indent=2)[:6000])


if __name__ == "__main__":
    main()
