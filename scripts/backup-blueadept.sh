#!/usr/bin/env bash
# backup-blueadept.sh - back up the turbulence-agent dev box.
#
# Captures the working tree (including uncommitted work), consistent copies of
# every SQLite database, and the environment files. Deliberately excludes
# anything reproducible from source: .venv, caches, downloaded model weights,
# GRIB2 files.
#
# Usage:
#   ./backup-blueadept.sh                    # write archive to $DEST
#   ./backup-blueadept.sh --dest /mnt/usb    # somewhere else
#   ./backup-blueadept.sh --encrypt          # gpg symmetric, prompts for passphrase
#   ./backup-blueadept.sh --stdout > b.tgz   # stream, nothing lands on disk
#   ./backup-blueadept.sh --keep 7           # prune to the newest 7 in $DEST
#   ./backup-blueadept.sh --dry-run

set -euo pipefail

# ---------------------------------------------------------------- configuration
PROJECT="${TURBULENCE_PROJECT:-/root/projects/turbulence-agent}"
DEST="${TURBULENCE_BACKUP_DEST:-/root/backups}"
KEEP=0
ENCRYPT=0
STDOUT=0
DRYRUN=0

# Paths inside $PROJECT that are reproducible and should never be archived.
EXCLUDES=(
  '.venv'
  'cache'
  'data/ntsb_cache'       # 348M of raw NTSB API responses; retrieval.db is the product
  '.checkpoints'          # 226M of immutable db snapshots; archive once, separately
  '.hypothesis'
  '__pycache__'
  '.pytest_cache'
  '.mypy_cache'
  '*.pyc'
  '*.db.bak'
  '*.grib2'
  '*.grib2.idx'
  'node_modules'
)
# ---------------------------------------------------------------- plumbing
say() { printf '%s\n' "$*" >&2; }          # all logging to stderr, so --stdout is clean
die() { say "error: $*"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)     DEST="$2"; shift 2 ;;
    --project)  PROJECT="$2"; shift 2 ;;
    --keep)     KEEP="$2"; shift 2 ;;
    --encrypt)  ENCRYPT=1; shift ;;
    --stdout)   STDOUT=1; shift ;;
    --dry-run)  DRYRUN=1; shift ;;
    -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
    *)          die "unknown option: $1" ;;
  esac
done

[[ -d "$PROJECT" ]] || die "project not found: $PROJECT"
command -v tar >/dev/null || die "tar not found"
if [[ $ENCRYPT -eq 1 ]]; then command -v gpg >/dev/null || die "gpg not found"; fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
NAME="turbulence-blueadept-${TS}"
WORK="$(mktemp -d)"
STAGE="${WORK}/${NAME}"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$STAGE"/{tree,db,git}

say "Backup ${NAME}"
say "  project ${PROJECT}"

# ---------------------------------------------------------------- 1. git state
# The tarball carries the working tree, but recording HEAD, the branch, and
# anything dirty makes it obvious later what this snapshot actually was.
if [[ -d "$PROJECT/.git" ]]; then
  git -C "$PROJECT" rev-parse HEAD                  > "$STAGE/git/HEAD.txt"      2>/dev/null || true
  git -C "$PROJECT" rev-parse --abbrev-ref HEAD     > "$STAGE/git/branch.txt"    2>/dev/null || true
  git -C "$PROJECT" status --porcelain              > "$STAGE/git/dirty.txt"     2>/dev/null || true
  git -C "$PROJECT" log --oneline -20               > "$STAGE/git/recent.txt"    2>/dev/null || true
  git -C "$PROJECT" remote -v                       > "$STAGE/git/remotes.txt"   2>/dev/null || true
  # Uncommitted changes to tracked files, as a patch you can replay.
  git -C "$PROJECT" diff HEAD                       > "$STAGE/git/uncommitted.patch" 2>/dev/null || true
  DIRTY="$(wc -l < "$STAGE/git/dirty.txt" | tr -d ' ')"
  say "  git     $(cat "$STAGE/git/branch.txt" 2>/dev/null || echo '?') @ $(cut -c1-8 "$STAGE/git/HEAD.txt" 2>/dev/null || echo '?'), ${DIRTY} dirty path(s)"
else
  say "  git     not a repository"
fi

# ---------------------------------------------------------------- 2. databases
# cp on a live SQLite file can capture a torn page or miss the WAL. The backup
# API takes a consistent snapshot while the writer keeps working.
snapshot_db() {
  local src="$1" out="$2"
  if command -v sqlite3 >/dev/null; then
    sqlite3 "$src" ".backup '$out'"
  else
    python3 - "$src" "$out" <<'PY'
import sqlite3, sys
src, dst = sys.argv[1], sys.argv[2]
s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
d = sqlite3.connect(dst)
with d:
    s.backup(d)
d.close(); s.close()
PY
  fi
}

DBCOUNT=0
while IFS= read -r -d '' db; do
  base="$(basename "$db")"
  if snapshot_db "$db" "$STAGE/db/$base" 2>/dev/null; then
    DBCOUNT=$((DBCOUNT+1))
    say "  db      ${base} ($(du -h "$STAGE/db/$base" | cut -f1))"
  else
    say "  db      ${base} SNAPSHOT FAILED - not a database, or locked"
    rm -f "$STAGE/db/$base"
  fi
done < <(find "$PROJECT/data" -maxdepth 2 -name '*.db' -type f -print0 2>/dev/null)
[[ $DBCOUNT -eq 0 ]] && say "  db      none found under ${PROJECT}/data"

# ---------------------------------------------------------------- 3. tree
# The live *.db files are excluded here because the consistent copies above
# supersede them; keeping both would double the size and confuse a restore.
TAR_EXCLUDES=()
for e in "${EXCLUDES[@]}"; do TAR_EXCLUDES+=( --exclude="$e" ); done
TAR_EXCLUDES+=( --exclude='data/*.db' --exclude='data/*.db-wal' --exclude='data/*.db-shm' )

tar -C "$PROJECT" -cf - "${TAR_EXCLUDES[@]}" . | tar -C "$STAGE/tree" -xf -
say "  tree    $(du -sh "$STAGE/tree" | cut -f1) after excludes"

# ---------------------------------------------------------------- 4. environment
python3 --version                       > "$STAGE/python-version.txt" 2>&1 || true
if [[ -x "$PROJECT/.venv/bin/pip" ]]; then
  "$PROJECT/.venv/bin/pip" freeze        > "$STAGE/pip-freeze.txt" 2>/dev/null || true
  say "  venv    frozen to pip-freeze.txt ($(wc -l < "$STAGE/pip-freeze.txt" | tr -d ' ') packages)"
fi

# ---------------------------------------------------------------- 5. manifest
SECRETS=0
[[ -f "$STAGE/tree/.env" ]] && SECRETS=1

cat > "$STAGE/MANIFEST.txt" <<EOF
turbulence-agent dev box backup
created    ${TS}
host       $(hostname)
project    ${PROJECT}
databases  ${DBCOUNT} snapshot(s) via SQLite backup API
secrets    $([[ $SECRETS -eq 1 ]] && echo "YES - tree/.env is present in this archive" || echo "no .env found")

contents
  tree/       working tree, excluding: ${EXCLUDES[*]}
  db/         consistent SQLite snapshots (restore these, not tree/data/*.db)
  git/        HEAD, branch, dirty paths, and uncommitted.patch
  pip-freeze.txt

restore
  1. mkdir -p /root/projects/turbulence-agent && tar xzf <archive> --strip-components=2 -C /root/projects/turbulence-agent
     (adjust --strip-components for your archive layout; contents sit under ${NAME}/)
  2. cp db/*.db /root/projects/turbulence-agent/data/
  3. python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
     pip-freeze.txt has exact pinned versions if requirements.txt has drifted
  4. Verify: TURBULENCE_DB=/tmp/t.db .venv/bin/pytest -q

note
  .venv, cache/, and downloaded model weights are NOT here by design - they are
  reproducible. bge-small-en-v1.5 re-downloads on first run.
EOF

# ---------------------------------------------------------------- 6. emit
if [[ $DRYRUN -eq 1 ]]; then
  say ""
  say "dry run - would archive:"
  du -sh "$STAGE"/* >&2
  exit 0
fi

if [[ $STDOUT -eq 1 ]]; then
  say "  writing to stdout"
  tar -C "$WORK" -czf - "$NAME"
  exit 0
fi

mkdir -p "$DEST"
ARCHIVE="${DEST}/${NAME}.tar.gz"
tar -C "$WORK" -czf "$ARCHIVE" "$NAME"
chmod 600 "$ARCHIVE"

if [[ $ENCRYPT -eq 1 ]]; then
  gpg --symmetric --cipher-algo AES256 --output "${ARCHIVE}.gpg" "$ARCHIVE"
  shred -u "$ARCHIVE" 2>/dev/null || rm -f "$ARCHIVE"
  ARCHIVE="${ARCHIVE}.gpg"
  chmod 600 "$ARCHIVE"
fi

sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"

say ""
say "  archive ${ARCHIVE} ($(du -h "$ARCHIVE" | cut -f1))"
say "  sha256  $(cut -d' ' -f1 "${ARCHIVE}.sha256")"
if [[ $SECRETS -eq 1 && $ENCRYPT -eq 0 ]]; then
  say ""
  say "  WARNING: this archive contains .env in cleartext. It is mode 600, but"
  say "           anywhere you copy it inherits whatever permissions land there."
  say "           Use --encrypt if it is leaving this box."
fi

# ---------------------------------------------------------------- 7. retention
if [[ "$KEEP" -gt 0 ]]; then
  mapfile -t OLD < <(ls -1t "${DEST}"/turbulence-blueadept-*.tar.gz* 2>/dev/null | grep -v '\.sha256$' | tail -n +$((KEEP+1)))
  for f in "${OLD[@]:-}"; do
    [[ -n "$f" ]] || continue
    rm -f "$f" "${f}.sha256"
    say "  pruned  $(basename "$f")"
  done
fi
