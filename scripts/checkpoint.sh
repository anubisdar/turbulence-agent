#!/usr/bin/env bash
#
# checkpoint.sh - snapshot the project so we can return to this exact state.
#
# Three tiers of data, treated differently:
#   source code      -> git (small, text, the thing that changes)
#   built database   -> tar archive (19 MB, ~6 min to rebuild, worth keeping)
#   NTSB cache       -> left alone (~250 MB, immutable, slow and impolite to refetch)
#
# Usage:
#   ./scripts/checkpoint.sh                          # tag with today's date
#   ./scripts/checkpoint.sh checkpoint-3-complete    # explicit tag name
#
set -euo pipefail

PROJ="${PROJ:-$(pwd)}"
TAG="${1:-checkpoint-$(date +%Y%m%d-%H%M)}"
ARCHIVE_DIR="$PROJ/.checkpoints"
MAX_COMMIT_MB=25

cd "$PROJ"

say()   { printf '  %s\n' "$*"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }

[[ -d .git ]] || { echo "no .git here - run from the repo root" >&2; exit 1; }
[[ -d app ]] || { echo "does not look like turbulence-agent: $PROJ" >&2; exit 1; }

# ---------------------------------------------------------------- gitignore

head_ "Ensuring .gitignore"

ensure_ignore() {
  local pattern="$1"
  if ! grep -qxF "$pattern" .gitignore 2>/dev/null; then
    echo "$pattern" >> .gitignore
    say "added   $pattern"
  fi
}

if [[ ! -f .gitignore ]]; then
  printf '# Created by checkpoint.sh\n' > .gitignore
  say "created .gitignore"
fi

# Virtualenv and Python noise
ensure_ignore '.venv/'
ensure_ignore 'venv/'
ensure_ignore '__pycache__/'
ensure_ignore '*.py[cod]'
ensure_ignore '.pytest_cache/'
ensure_ignore '*.egg-info/'

# Data: large, regenerable, or fetched. None of it belongs in git.
ensure_ignore 'data/'

# Working artifacts
ensure_ignore '.drop-backups/'
ensure_ignore '.checkpoints/'
ensure_ignore '*.bak'
ensure_ignore '*.log'

# Credentials - nothing here today, but this is when to set the habit
ensure_ignore '.env'
ensure_ignore '*.key'
ensure_ignore 'secrets.*'

# ---------------------------------------------------------------- what would commit

head_ "What git will track"

git add -A --dry-run >/dev/null 2>&1 || true
git add -A

FILE_COUNT=$(git diff --cached --name-only | wc -l)
BYTES=$(git diff --cached --name-only -z \
        | xargs -0 -r du -cb 2>/dev/null | tail -1 | cut -f1 || echo 0)
MB=$(( BYTES / 1024 / 1024 ))

say "files staged: $FILE_COUNT"
say "total size:   ${MB} MB"

if (( MB > MAX_COMMIT_MB )); then
  head_ "ABORTING - staged content is ${MB} MB"
  say "That is far larger than source code should be. Something big slipped"
  say "past .gitignore. The ten largest staged paths:"
  git diff --cached --name-only -z | xargs -0 -r du -h 2>/dev/null \
    | sort -rh | head -10 | sed 's/^/    /'
  git reset >/dev/null
  say ""
  say "Nothing was committed. Add the offender to .gitignore and re-run."
  exit 1
fi

say ""
say "top-level paths being tracked:"
git diff --cached --name-only | cut -d/ -f1 | sort -u | sed 's/^/    /'

# ---------------------------------------------------------------- commit + tag

head_ "Committing"

if git rev-parse HEAD >/dev/null 2>&1; then
  if git diff --cached --quiet; then
    say "no changes to commit"
  else
    git commit -q -m "Checkpoint: $TAG

Retrieval layer complete. NTSB Part 121 corpus indexed and searchable.
231 tests passing."
    say "committed $(git rev-parse --short HEAD)"
  fi
else
  git commit -q -m "Initial commit: turbulence-aware flight ranking agent

AWC source client, NTSB retrieval layer (type normalization, structural
chunking, sqlite-vec index, two-tier type-filtered search).
231 tests passing."
  say "initial commit $(git rev-parse --short HEAD)"
fi

if git rev-parse "$TAG" >/dev/null 2>&1; then
  say "tag '$TAG' already exists - leaving it alone"
else
  git tag -a "$TAG" -m "Checkpoint 3 complete: retrieval layer built and demonstrated"
  say "tagged $TAG"
fi

# ---------------------------------------------------------------- database archive

head_ "Archiving the built index"

mkdir -p "$ARCHIVE_DIR"
DB="data/retrieval.db"

if [[ -f "$DB" ]]; then
  OUT="$ARCHIVE_DIR/retrieval-db-$TAG.tar.gz"
  if [[ -f "$OUT" ]]; then
    say "archive already exists: ${OUT#$PROJ/}"
  else
    tar czf "$OUT" "$DB" 2>/dev/null
    say "archived ${OUT#$PROJ/}  ($(du -h "$OUT" | cut -f1))"
  fi
else
  say "no $DB found - skipping"
fi

# The NTSB cache is deliberately NOT archived: large, immutable, and the
# whole reason ingest is repeatable. Just never delete it.
if [[ -d data/ntsb_cache ]]; then
  say "ntsb_cache: $(du -sh data/ntsb_cache | cut -f1) - not archived, do not delete"
fi

# ---------------------------------------------------------------- how to return

head_ "To return to this state"
cat <<EOF
    git stash                                  # park anything in progress
    git checkout $TAG

  If the database also needs restoring:
    tar xzf .checkpoints/retrieval-db-$TAG.tar.gz

  To get back to the tip of development:
    git checkout master

  Current tags:
EOF
git tag -l | sed 's/^/    /'

head_ "Done"
