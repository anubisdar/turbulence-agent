#!/usr/bin/env bash
#
# restore.sh - return the project to a previous checkpoint.
#
# Safety model, in order of how much it would hurt to lose something:
#   1. Uncommitted work is stashed with a named label before anything moves.
#   2. The current database is set aside, never overwritten in place.
#   3. Archives are integrity-checked before extraction.
#   4. Restores land on a new branch, not a detached HEAD.
#
# Usage:
#   ./scripts/restore.sh                            # list what is available
#   ./scripts/restore.sh checkpoint-3-complete      # restore code + database
#   ./scripts/restore.sh <tag> --code-only          # leave the database alone
#   ./scripts/restore.sh <tag> --db-only            # leave the code alone
#   ./scripts/restore.sh <tag> --dry-run            # show the plan, change nothing
#
set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
ARCHIVE_DIR="$PROJ/.checkpoints"
DB="data/retrieval.db"
STAMP="$(date +%Y%m%d-%H%M%S)"

cd "$PROJ"

say()   { printf '  %s\n' "$*"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }
warn()  { printf '  \033[33m%s\033[0m\n' "$*"; }

[[ -d .git ]] || { echo "no .git here - run from the repo root" >&2; exit 1; }

TAG=""
DO_CODE=1
DO_DB=1
DRY=0
DETACH=0

for arg in "$@"; do
  case "$arg" in
    --code-only) DO_DB=0 ;;
    --db-only)   DO_CODE=0 ;;
    --dry-run)   DRY=1 ;;
    --detach)    DETACH=1 ;;
    --*)         echo "unknown option: $arg" >&2; exit 1 ;;
    *)           TAG="$arg" ;;
  esac
done

run() { if (( DRY )); then say "would: $*"; else "$@"; fi; }

# ---------------------------------------------------------------- listing

list_checkpoints() {
  head_ "Available checkpoints"
  local tags
  tags=$(git tag -l --sort=-creatordate)
  if [[ -z "$tags" ]]; then
    say "no tags found - run checkpoint.sh first"
    return
  fi
  printf '  %-34s %-12s %s\n' "TAG" "DATE" "DATABASE ARCHIVE"
  while read -r t; do
    [[ -z "$t" ]] && continue
    local when archive
    when=$(git log -1 --format=%ad --date=short "$t" 2>/dev/null || echo "?")
    archive="$ARCHIVE_DIR/retrieval-db-$t.tar.gz"
    if [[ -f "$archive" ]]; then
      printf '  %-34s %-12s %s\n' "$t" "$when" "yes ($(du -h "$archive" | cut -f1))"
    else
      printf '  %-34s %-12s %s\n' "$t" "$when" "-"
    fi
  done <<< "$tags"

  head_ "Current position"
  local branch
  branch=$(git rev-parse --abbrev-ref HEAD)
  if [[ "$branch" == "HEAD" ]]; then
    warn "detached HEAD at $(git rev-parse --short HEAD)"
    say "return to development with: git checkout master"
  else
    say "on branch '$branch' at $(git rev-parse --short HEAD)"
  fi
  if [[ -f "$DB" ]]; then
    say "database present: $(du -h "$DB" | cut -f1)"
  else
    warn "no $DB - the index is not built"
  fi
}

if [[ -z "$TAG" ]]; then
  list_checkpoints
  head_ "To restore"
  say "./scripts/restore.sh <tag>"
  exit 0
fi

git rev-parse "$TAG" >/dev/null 2>&1 || {
  echo "no such tag: $TAG" >&2
  echo "run without arguments to see what is available" >&2
  exit 1
}

head_ "Restoring $TAG"
(( DRY )) && warn "dry run - nothing will change"

# ---------------------------------------------------------------- protect work

if (( DO_CODE )); then
  if ! git diff --quiet || ! git diff --cached --quiet || \
     [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
    head_ "Uncommitted work found"
    git status --short | head -20 | sed 's/^/    /'
    LABEL="pre-restore-$STAMP"
    say ""
    say "stashing as '$LABEL' - nothing is discarded"
    run git stash push -u -m "$LABEL"
    say "recover it later with: git stash list  /  git stash pop"
  else
    say "working tree clean"
  fi
fi

# ---------------------------------------------------------------- code

if (( DO_CODE )); then
  head_ "Code"
  if (( DETACH )); then
    run git checkout -q "$TAG"
    warn "detached HEAD - commits here belong to no branch"
  else
    BRANCH="restore/$TAG-$STAMP"
    run git checkout -q -b "$BRANCH" "$TAG"
    say "checked out $TAG on new branch '$BRANCH'"
    say "this keeps you off a detached HEAD, so work here is not lost"
  fi
else
  say ""
  say "code: skipped (--db-only)"
fi

# ---------------------------------------------------------------- database

if (( DO_DB )); then
  head_ "Database"
  ARCHIVE="$ARCHIVE_DIR/retrieval-db-$TAG.tar.gz"

  if [[ ! -f "$ARCHIVE" ]]; then
    warn "no archive for $TAG at ${ARCHIVE#$PROJ/}"
    say "the index can be rebuilt from the cache instead:"
    say "    python3 scripts/ingest_ntsb.py --rebuild"
    say "    python3 scripts/embed_chunks.py"
  elif ! tar tzf "$ARCHIVE" >/dev/null 2>&1; then
    warn "archive is corrupt: ${ARCHIVE#$PROJ/}"
    say "rebuild from the cache instead - do not extract this file"
    exit 1
  else
    say "archive verified: $(du -h "$ARCHIVE" | cut -f1)"
    if [[ -f "$DB" ]]; then
      SET_ASIDE="$ARCHIVE_DIR/superseded-$STAMP.db"
      run mkdir -p "$ARCHIVE_DIR"
      run mv "$DB" "$SET_ASIDE"
      say "current database set aside at ${SET_ASIDE#$PROJ/}"
    fi
    run mkdir -p data
    run tar xzf "$ARCHIVE" -C "$PROJ"
    if (( ! DRY )) && [[ -f "$DB" ]]; then
      say "restored $DB ($(du -h "$DB" | cut -f1))"
    fi
  fi
else
  say ""
  say "database: skipped (--code-only)"
fi

# ---------------------------------------------------------------- verify

if (( ! DRY )); then
  head_ "Verify"
  if [[ -f "$DB" ]] && command -v python3 >/dev/null; then
    python3 - "$DB" <<'PYEOF' 2>/dev/null || say "could not read the database"
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
for t in ("cases", "case_aircraft", "chunks"):
    try:
        n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<16} {n:>7,}")
    except Exception:
        print(f"  {t:<16} missing")
try:
    n = c.execute("SELECT COUNT(*) FROM chunks WHERE embedded_at IS NOT NULL").fetchone()[0]
    print(f"  {'embedded':<16} {n:>7,}")
except Exception:
    pass
c.close()
PYEOF
  fi
  say ""
  say "run the suite to confirm the code is coherent:"
  say "    pytest tests/"
fi

head_ "Done"
if (( DO_CODE )) && (( ! DETACH )) && (( ! DRY )); then
  say "you are on $(git rev-parse --abbrev-ref HEAD)"
  say "return to the tip of development with: git checkout master"
fi
if (( ! DRY )); then
  STASHES=$(git stash list 2>/dev/null | wc -l)
  (( STASHES > 0 )) && say "$STASHES stash(es) held - see: git stash list"
fi
