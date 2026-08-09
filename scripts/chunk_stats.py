#!/usr/bin/env python3
"""
Measure what the chunker actually produces across the cached Part 121 corpus.

Offline. Reads data/ntsb_cache/ only. Writes nothing.

The scope report estimated ~9,858 chunks using a flat 1,200-char division.
This runs the real structural chunker and reports the true shape, so we know
what we are committing to embed - and whether the factual narratives are
dominating the index.

Usage:
    python3 scripts/chunk_stats.py
    python3 scripts/chunk_stats.py --sample 3     # print a few real chunks
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.aircraft_types import resolve  # noqa: E402
from app.retrieval.chunking import Section, chunk_case  # noqa: E402

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
    sample_n = 0
    if "--sample" in sys.argv:
        sample_n = int(sys.argv[sys.argv.index("--sample") + 1])

    cases = load_part121()
    print(f"Part 121 cases: {len(cases):,}\n")

    by_section = Counter()
    chars_by_section = Counter()
    per_case_counts = []
    sizes = []
    cases_with_no_chunks = 0
    worst = []
    samples = []

    for case in cases:
        aircraft = [
            resolve((v.get("make") or ""), (v.get("model") or ""))
            for v in (case.get("cm_vehicles") or [])
        ]
        chunks = chunk_case(case, aircraft)
        if not chunks:
            cases_with_no_chunks += 1
            continue
        per_case_counts.append(len(chunks))
        worst.append((len(chunks), case.get("cm_ntsbNum")))
        for c in chunks:
            by_section[c.section.value] += 1
            chars_by_section[c.section.value] += c.char_count
            sizes.append(c.char_count)
        if len(samples) < sample_n:
            samples.append(chunks)

    total = sum(by_section.values())
    print("--- Chunks by section ---")
    for s in (Section.PROBABLE_CAUSE, Section.ANALYSIS,
              Section.FACTUAL, Section.PRELIMINARY):
        n = by_section.get(s.value, 0)
        share = n / total * 100 if total else 0
        chars = chars_by_section.get(s.value, 0)
        print(f"  {s.value:<16} {n:>7,}  {share:5.1f}%   {chars:>11,} chars")
    print(f"  {'TOTAL':<16} {total:>7,}")
    print(f"\n  scope-report estimate was ~9,858")

    print("\n--- Chunk size distribution ---")
    sizes.sort()
    print(f"  min {sizes[0]:,}   p50 {sizes[len(sizes)//2]:,}   "
          f"p90 {sizes[int(len(sizes)*0.9)]:,}   max {sizes[-1]:,}")
    print(f"  mean {statistics.mean(sizes):,.0f}")
    tiny = sum(1 for s in sizes if s < 100)
    print(f"  chunks under 100 chars: {tiny:,} ({tiny/len(sizes)*100:.1f}%)")

    print("\n--- Chunks per case ---")
    per_case_counts.sort()
    print(f"  min {per_case_counts[0]}   p50 {per_case_counts[len(per_case_counts)//2]}   "
          f"p90 {per_case_counts[int(len(per_case_counts)*0.9)]}   "
          f"max {per_case_counts[-1]}")
    print(f"  cases producing no chunks at all: {cases_with_no_chunks:,} "
          f"({cases_with_no_chunks/len(cases)*100:.1f}%)")

    print("\n--- Heaviest cases (single-case domination risk) ---")
    for n, num in sorted(worst, reverse=True)[:10]:
        print(f"  {n:>4} chunks   {num}")

    print("\n--- Index cost ---")
    print(f"  embedding at ~40 chunks/sec: ~{total/40/60:.1f} min")
    print(f"  at 384 dims float32: ~{total*384*4/1024/1024:.1f} MB of vectors")
    print(f"  text volume: {sum(chars_by_section.values())/1024/1024:.1f} MB")

    print("\n--- What the index looks like without factual narratives ---")
    lean = total - by_section.get("factual", 0)
    print(f"  chunks: {lean:,} ({lean/total*100:.0f}% of full index)")
    print(f"  embedding time: ~{lean/40/60:.1f} min")

    for chunks in samples:
        print("\n" + "=" * 70)
        print(f"SAMPLE CASE {chunks[0].ntsb_num} - {len(chunks)} chunks")
        print(f"header: {chunks[0].context_header}")
        for c in chunks[:3]:
            print(f"\n  [{c.section.value} {c.ordinal+1}/{c.meta['of']}] "
                  f"{c.char_count} chars")
            print("  " + c.text[:400].replace("\n", "\n  "))


if __name__ == "__main__":
    main()
