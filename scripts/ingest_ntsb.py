#!/usr/bin/env python3
"""
Build the retrieval database from the cached CAROL export.

Offline. Reads data/ntsb_cache/, writes data/retrieval.db. Does not embed -
that is a separate pass, so a schema change never means re-downloading and a
re-embed never means re-parsing.

Idempotent: re-running replaces each case in place.

Usage:
    python3 scripts/ingest_ntsb.py
    python3 scripts/ingest_ntsb.py --db data/retrieval.db --rebuild
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.aircraft_types import resolve  # noqa: E402
from app.retrieval.chunking import (  # noqa: E402
    SECTION_PRIORITY,
    chunk_case,
    has_narrative,
)
from app.retrieval.schema import connect, init_db  # noqa: E402

CACHE = Path("data/ntsb_cache")
DEFAULT_DB = Path("data/retrieval.db")
AIRLINE_PARTS = {"121"}

# NTSB does not publish a stable per-case deep link in the export payload.
# Leaving this None rather than constructing a URL that may not resolve -
# provenance that points nowhere is worse than provenance that is absent.
CASE_URL_TEMPLATE: str | None = None


def load_part121() -> list[tuple[str, dict]]:
    if not CACHE.exists():
        sys.exit(f"No cache at {CACHE}. Run scope_ntsb.py first.")
    seen: set = set()
    out: list[tuple[str, dict]] = []
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
                out.append((f.stem, case))
    return out


def _year(iso: str | None) -> int | None:
    if not iso or len(iso) < 4 or not iso[:4].isdigit():
        return None
    return int(iso[:4])


def insert_case(conn, case: dict, window: str, now: str) -> list:
    mkey = case["cm_mkey"]
    conn.execute("DELETE FROM cases WHERE mkey = ?", (mkey,))
    conn.execute("DELETE FROM case_aircraft WHERE mkey = ?", (mkey,))
    conn.execute("DELETE FROM case_findings WHERE mkey = ?", (mkey,))
    conn.execute("DELETE FROM chunks WHERE mkey = ?", (mkey,))

    conn.execute(
        """INSERT INTO cases (mkey, ntsb_num, event_date, event_year, event_type,
               report_type, completion_status, highest_injury, fatal_count,
               city, state, country, latitude, longitude,
               source, source_class, source_url, ingested_at, export_window)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            mkey,
            case.get("cm_ntsbNum") or "",
            case.get("cm_eventDate"),
            _year(case.get("cm_eventDate")),
            case.get("cm_eventType"),
            case.get("cm_mostRecentReportType"),
            case.get("cm_completionStatus"),
            case.get("cm_highestInjury"),
            case.get("cm_fatalInjuryCount"),
            case.get("cm_city"),
            case.get("cm_state"),
            case.get("cm_country"),
            case.get("cm_Latitude"),
            case.get("cm_Longitude"),
            "NTSB CAROL",
            "formal",
            CASE_URL_TEMPLATE.format(mkey=mkey) if CASE_URL_TEMPLATE else None,
            now,
            window,
        ),
    )

    resolved = []
    for v in (case.get("cm_vehicles") or []):
        make = (v.get("make") or "").strip()
        model = (v.get("model") or "").strip()
        r = resolve(make, model)
        resolved.append(r)
        conn.execute(
            """INSERT INTO case_aircraft (mkey, vehicle_num, far_part,
                   raw_make, raw_model, manufacturer, family, variant,
                   generation, type_confidence, operator_name, registration,
                   damage_level)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                mkey,
                v.get("cm_vehicleNum"),
                (v.get("regulationFlightConductedUnder") or "").strip() or None,
                make or None,
                model or None,
                r.manufacturer,
                r.family,
                r.variant,
                r.generation,
                r.confidence.value,
                (v.get("operatorName") or "").strip() or None,
                (v.get("registrationNumber") or "").strip() or None,
                v.get("DamageLevel"),
            ),
        )
        for f in (v.get("cm_findings") or []):
            conn.execute(
                """INSERT INTO case_findings (mkey, finding_code, finding_text,
                       in_probable_cause) VALUES (?,?,?,?)""",
                (mkey, f.get("cm_findingCode"), f.get("cm_findingText"),
                 1 if f.get("cm_inPc") else 0),
            )

    chunks = chunk_case(case, resolved)
    for c in chunks:
        conn.execute(
            """INSERT INTO chunks (mkey, section, section_priority, ordinal,
                   ordinal_of, text, context_header, char_count)
               VALUES (?,?,?,?,?,?,?,?)""",
            (mkey, c.section.value, SECTION_PRIORITY[c.section], c.ordinal,
             c.meta.get("of", 1), c.text, c.context_header, c.char_count),
        )
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--rebuild", action="store_true",
                    help="delete the database first")
    args = ap.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.rebuild and db_path.exists():
        db_path.unlink()
        print(f"removed {db_path}")

    cases = load_part121()
    print(f"Part 121 cases in cache: {len(cases):,}")

    conn = connect(db_path)
    init_db(conn)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    section_counts = Counter()
    embedded = 0
    no_narrative = []
    by_decade = defaultdict(lambda: [0, 0])

    for i, (window, case) in enumerate(cases, 1):
        chunks = insert_case(conn, case, window, now)
        for c in chunks:
            section_counts[c.section.value] += 1
            if c.embedded:
                embedded += 1

        yr = _year(case.get("cm_eventDate"))
        decade = f"{yr // 10 * 10}s" if yr else "unknown"
        by_decade[decade][0] += 1
        if has_narrative(case):
            by_decade[decade][1] += 1
        else:
            no_narrative.append((case.get("cm_ntsbNum"), yr))

        if i % 250 == 0:
            conn.commit()
            print(f"  {i:,}/{len(cases):,}")
    conn.commit()

    print(f"\nwrote {db_path}  ({db_path.stat().st_size/1024/1024:.1f} MB)")

    print("\n--- Rows ---")
    for t in ("cases", "case_aircraft", "case_findings", "chunks"):
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<16} {n:>7,}")

    print("\n--- Chunks by section ---")
    total = sum(section_counts.values())
    for s, n in section_counts.most_common():
        print(f"  {s:<16} {n:>7,}  {n/total*100:5.1f}%")
    print(f"\n  to embed: {embedded:,}  ({embedded/total*100:.0f}% of stored)")
    print(f"  stored, not embedded: {total-embedded:,}")

    print("\n--- Narrative coverage by decade ---")
    print("  (cases with no narrative are invisible to semantic search)")
    for d in sorted(by_decade):
        tot, have = by_decade[d]
        print(f"  {d:<8} {have:>4,}/{tot:<5,} {have/tot*100:5.1f}% have narrative")
    print(f"\n  total with no narrative: {len(no_narrative):,}")

    print("\n--- Type buckets with narrative-bearing cases ---")
    rows = conn.execute("""
        SELECT a.variant AS v, COUNT(DISTINCT a.mkey) AS cases,
               COUNT(DISTINCT CASE WHEN ch.id IS NOT NULL THEN a.mkey END) AS with_text
        FROM case_aircraft a
        LEFT JOIN chunks ch ON ch.mkey = a.mkey
        WHERE a.variant IS NOT NULL AND a.far_part = '121'
        GROUP BY a.variant ORDER BY cases DESC LIMIT 20
    """).fetchall()
    for r in rows:
        gap = r["cases"] - r["with_text"]
        flag = f"   <-- {gap} with no narrative" if gap else ""
        print(f"  {r['cases']:>4} cases  {r['with_text']:>4} retrievable  "
              f"{r['v']}{flag}")

    conn.close()


if __name__ == "__main__":
    main()
