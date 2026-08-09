#!/usr/bin/env bash
#
# run_demo.sh - install the latest drop, then exercise the retrieval tool
# against the real index and print the evidence the write-up needs.
#
# Usage:
#   ./scripts/run_demo.sh                 # install from /home/darlage, then demo
#   ./scripts/run_demo.sh --demo-only     # skip the install step
#
set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
SRC="${SRC:-/home/darlage}"
DB="${DB:-data/retrieval.db}"

cd "$PROJ"

if [[ "${1:-}" != "--demo-only" ]]; then
  if [[ ! -x scripts/install_drop.sh ]]; then
    echo "scripts/install_drop.sh not found or not executable" >&2
    exit 1
  fi
  ./scripts/install_drop.sh "$SRC"
fi

if [[ ! -f "$DB" ]]; then
  echo "No index at $DB - run ingest_ntsb.py then embed_chunks.py first" >&2
  exit 1
fi

printf '\n\033[1m=== Retrieval demonstration ===\033[0m\n'

python3 - "$DB" <<'PYEOF'
import sys
from app.retrieval.schema import connect
from app.retrieval.embedding import SentenceTransformerEncoder
from app.retrieval.search import search_aircraft_reputation

DB = sys.argv[1]
QUERY = "flight control system malfunction during climb"

conn = connect(DB, load_vec=True)
enc = SentenceTransformerEncoder()
print(f"model: {enc.name}  dim={enc.dim}\n")


def show(type_str, query=QUERY, k=6, **kw):
    out = search_aircraft_reputation(conn, enc, type_str, query, k=k, **kw)
    rt = out.resolved_type
    print("=" * 74)
    print(f"QUERY TYPE : {type_str!r}")
    print(f"RESOLVED   : variant={rt.variant}  family={rt.family} "
          f"generation={rt.generation}  confidence={rt.confidence.value}")
    c = out.coverage
    print(f"CORPUS     : exact-variant tier : {c.cases_variant} case(s), "
          f"{c.cases_variant_with_text} with narrative, "
          f"{c.cases_variant_without_text} without")
    print(f"             family-only tier   : {c.cases_family} case(s), "
          f"{c.cases_family_with_text} with narrative, "
          f"{c.cases_family_without_text} without")
    if c.oldest_event_year:
        print(f"             span {c.oldest_event_year}-{c.newest_event_year}")
    for n in out.notes:
        print(f"  NOTE  {n}")
    if not out.hits:
        print("  (no results)")
    for h in out.hits:
        flag = " [PRELIMINARY]" if h.provisional else ""
        print(f"\n  [{h.tier:<7}] {h.ntsb_num}  {h.event_year}  "
              f"score={h.score}  {h.section}{flag}")
        print(f"            type={h.variant or h.family} "
              f"(raw {h.raw_model!r}, {h.type_confidence})  op={h.operator}")
        print(f"            {h.text[:180].strip()}")
    print()
    return out


# 1. The demonstration example: MAX 8 vs the NG it is constantly confused with
max8 = show("737 MAX 8")
ng800 = show("737-800")

print("=" * 74)
print("SEPARATION CHECK")
a = {h.mkey for h in max8.hits}
b = {h.mkey for h in ng800.hits}
print(f"  MAX 8 cases returned  : {sorted(a)}")
print(f"  737-800 cases returned: {sorted(b)}")
print(f"  overlap               : {sorted(a & b) or 'none'}")
gens_a = {h.generation for h in max8.hits}
gens_b = {h.generation for h in ng800.hits}
print(f"  generations in MAX 8 result  : {gens_a - {None} or '-'}")
print(f"  generations in 737-800 result: {gens_b - {None} or '-'}")
assert not (a & b), "MAX and NG result sets overlap"
print("  OK - disjoint\n")

# 2. The same fleet reached by its raw NTSB string
print("=" * 74)
print("CUSTOMER-CODE ROUND TRIP")
raw = search_aircraft_reputation(conn, enc, "737-8H4", QUERY, k=6,
                                 include_family_tier=False)
canon = search_aircraft_reputation(conn, enc, "737-800", QUERY, k=6,
                                   include_family_tier=False)
print(f"  '737-8H4' -> {raw.resolved_type.variant}")
print(f"  '737-800' -> {canon.resolved_type.variant}")
print(f"  same cases: "
      f"{ {h.mkey for h in raw.hits} == {h.mkey for h in canon.hits} }\n")

# 3. A type where the corpus is thin - absence must not read as safety
show("767-300", k=4)

# 4. An unresolvable type - nothing is guessed
show("Spaceship One", k=4)

conn.close()
print("=" * 74)
print("done")
PYEOF
