#!/usr/bin/env bash
# reconcile_chunks.sh - explain the gap between "chunks to embed" and
# "vectors in index" in the NTSB CAROL retrieval database.
#
# Usage:  ./reconcile_chunks.sh /path/to/carol.db
#
# Read-only. Opens the database with an immutable URI so it cannot write,
# lock, or journal anything.

set -uo pipefail

DB="${1:-}"
if [[ -z "$DB" ]]; then
  echo "usage: $0 /path/to/carol.db" >&2
  exit 2
fi
if [[ ! -r "$DB" ]]; then
  echo "error: cannot read $DB" >&2
  exit 2
fi
command -v sqlite3 >/dev/null || { echo "error: sqlite3 not found" >&2; exit 2; }

# Absolute path, URI-encoded for the immutable open.
ABS=$(readlink -f "$DB")
URI="file:${ABS}?immutable=1"

# sqlite-vec tables need the extension loaded to be queried. Find it if present.
VEC_EXT=""
for c in "$(python3 -c 'import sqlite_vec,sys; sys.stdout.write(sqlite_vec.loadable_path())' 2>/dev/null)" \
         ./vec0.so /usr/local/lib/vec0.so; do
  [[ -n "$c" && -e "$c" ]] && { VEC_EXT="$c"; break; }
done
LOAD=""
[[ -n "$VEC_EXT" ]] && LOAD=".load ${VEC_EXT%.so}"

q() { sqlite3 "$URI" "$LOAD" "$@" 2>/dev/null; }

rule() { printf '%s\n' "------------------------------------------------------------"; }

echo
echo "database: $ABS"
echo "size:     $(du -h "$ABS" | cut -f1)"
[[ -n "$VEC_EXT" ]] && echo "sqlite-vec: $VEC_EXT" || echo "sqlite-vec: not found (vector table counts may fail)"

rule
echo "TABLES"
rule
q ".mode list" "SELECT type||'  '||name FROM sqlite_master
                WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'
                ORDER BY type, name;"

# ---- discover the chunks table --------------------------------------------
CHUNKS=$(q "SELECT name FROM sqlite_master WHERE type='table'
            AND name IN ('chunks','chunk') LIMIT 1;")
[[ -z "$CHUNKS" ]] && CHUNKS=$(q "SELECT name FROM sqlite_master WHERE type='table'
                                  AND name LIKE '%chunk%' AND name NOT LIKE '%vec%'
                                  ORDER BY LENGTH(name) LIMIT 1;")
if [[ -z "$CHUNKS" ]]; then
  echo; echo "could not find a chunks table - inspect the list above and edit CHUNKS in this script." >&2
  exit 1
fi

rule
echo "SCHEMA: $CHUNKS"
rule
q ".mode list" "SELECT name||'  ('||type||')' FROM pragma_table_info('$CHUNKS');"

COLS=$(q "SELECT group_concat(lower(name),' ') FROM pragma_table_info('$CHUNKS');")
has() { [[ " $COLS " == *" $1 "* ]]; }

# marker column: something that says "this chunk has a vector"
MARK=""
for c in embedded is_embedded has_embedding embedded_at vec_rowid embedding_id; do
  has "$c" && { MARK="$c"; break; }
done

# timestamp column, for separating runs
TS=""
for c in embedded_at embedded_ts created_at inserted_at updated_at; do
  has "$c" && { TS="$c"; break; }
done

# section column, to reproduce the "to embed" figure
SEC=""
for c in section section_type kind doc_section; do
  has "$c" && { SEC="$c"; break; }
done

# ---- discover the vector table ---------------------------------------------
VEC=$(q "SELECT name FROM sqlite_master WHERE type='table'
         AND (sql LIKE '%vec0%' OR name LIKE '%vec%')
         AND name NOT LIKE '%_info' AND name NOT LIKE '%_chunks%rowid%'
         ORDER BY LENGTH(name) LIMIT 1;")

echo
echo "detected -> chunks:'$CHUNKS'  marker:'${MARK:-none}'  timestamp:'${TS:-none}'  section:'${SEC:-none}'  vectors:'${VEC:-none}'"

# ---- section breakdown (reproduces the Exhibit A block) --------------------
if [[ -n "$SEC" ]]; then
  rule
  echo "CHUNKS BY SECTION"
  rule
  q ".headers on" ".mode column" "
    SELECT $SEC AS section,
           COUNT(*) AS n,
           ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM $CHUNKS),1) AS pct
    FROM $CHUNKS GROUP BY 1 ORDER BY n DESC;"
fi

# ---- the reconciliation ----------------------------------------------------
rule
echo "RECONCILIATION"
rule

TOTAL=$(q "SELECT COUNT(*) FROM $CHUNKS;")
printf '%-34s %s\n' "chunks total" "${TOTAL:-?}"

if [[ -n "$SEC" ]]; then
  ELIG=$(q "SELECT COUNT(*) FROM $CHUNKS WHERE lower($SEC) <> 'factual';")
  printf '%-34s %s\n' "eligible to embed (non-factual)" "${ELIG:-?}"
fi

if [[ -n "$MARK" ]]; then
  MARKED=$(q "SELECT COUNT(*) FROM $CHUNKS WHERE $MARK IS NOT NULL AND $MARK <> 0 AND $MARK <> '';")
  printf '%-34s %s\n' "marked embedded" "${MARKED:-?}"
fi

if [[ -n "$VEC" ]]; then
  VECN=$(q "SELECT COUNT(*) FROM $VEC;")
  printf '%-34s %s\n' "vectors in index" "${VECN:-? (load sqlite-vec)}"
fi

# ---- run separation: the resumable-run hypothesis --------------------------
if [[ -n "$MARK" && -n "$TS" ]]; then
  rule
  echo "EMBEDDED CHUNKS BY RUN (grouped by minute)"
  rule
  q ".headers on" ".mode column" "
    SELECT substr($TS,1,16) AS run_minute, COUNT(*) AS n
    FROM $CHUNKS
    WHERE $MARK IS NOT NULL AND $MARK <> 0 AND $MARK <> ''
    GROUP BY 1 ORDER BY 1;"
  echo
  echo "Two clusters with the smaller one holding the difference = earlier run,"
  echo "skipped as already-done. One cluster = the gap came from somewhere else."
fi

# ---- orphan checks: the bad case -------------------------------------------
if [[ -n "$MARK" && -n "$VEC" ]]; then
  rule
  echo "ORPHAN CHECK (both should be 0)"
  rule
  PK=$(q "SELECT name FROM pragma_table_info('$CHUNKS') WHERE pk=1 LIMIT 1;")
  PK=${PK:-rowid}

  A=$(q "SELECT COUNT(*) FROM $CHUNKS c
         WHERE c.$MARK IS NOT NULL AND c.$MARK <> 0 AND c.$MARK <> ''
           AND NOT EXISTS (SELECT 1 FROM $VEC v WHERE v.rowid = c.$PK);")
  B=$(q "SELECT COUNT(*) FROM $VEC v
         WHERE NOT EXISTS (SELECT 1 FROM $CHUNKS c WHERE c.$PK = v.rowid);")

  printf '%-34s %s\n' "marked embedded, no vector" "${A:-?}"
  printf '%-34s %s\n' "vector with no parent chunk" "${B:-?}"
  echo
  if [[ "${A:-0}" == "0" && "${B:-0}" == "0" ]]; then
    echo "Clean. Index and markers agree; the gap is a bookkeeping artifact of the"
    echo "embed pass, not a missing-vector problem."
  else
    echo "NOT clean. The index is over- or under-reporting - this is the case"
    echo "worth writing up, and worth fixing before the numbers go in the document."
  fi
fi

rule
echo "done"
echo
