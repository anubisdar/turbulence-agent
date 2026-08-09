#!/usr/bin/env bash
#
# install_drop.sh - put files dropped from a chat into the project tree and
# rewrite the flat imports they arrive with.
#
# Files written in a sandbox sit side by side, so tests import `from chunking
# import ...`. In this project they live under app/retrieval/, so every test
# needs `from app.retrieval.chunking import ...`. This does that rewrite for
# any module that actually exists in app/retrieval/, rather than a fixed list,
# so it keeps working as the package grows.
#
# Usage:
#   ./scripts/install_drop.sh                  # from ~/darlage, into $PWD
#   ./scripts/install_drop.sh /some/other/dir
#   DRY_RUN=1 ./scripts/install_drop.sh        # show what it would do
#
set -euo pipefail

SRC="${1:-/home/darlage}"
PROJ="${PROJ:-$(pwd)}"
DRY_RUN="${DRY_RUN:-0}"

RETRIEVAL_DIR="$PROJ/app/retrieval"
TESTS_DIR="$PROJ/tests"
SCRIPTS_DIR="$PROJ/scripts"
BACKUP_DIR="$PROJ/.drop-backups/$(date +%Y%m%d-%H%M%S)"

say()  { printf '  %s\n' "$*"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }
run()  { if [[ "$DRY_RUN" == "1" ]]; then say "would: $*"; else "$@"; fi; }

# ---------------------------------------------------------------- sanity

[[ -d "$SRC" ]] || { echo "source dir not found: $SRC" >&2; exit 1; }
[[ -d "$PROJ/app" ]] || {
  echo "does not look like the project root: $PROJ" >&2
  echo "cd to the repo root, or set PROJ=/path/to/turbulence-agent" >&2
  exit 1
}

mkdir -p "$RETRIEVAL_DIR" "$TESTS_DIR" "$SCRIPTS_DIR"
[[ -f "$RETRIEVAL_DIR/__init__.py" ]] || run touch "$RETRIEVAL_DIR/__init__.py"

# ---------------------------------------------------------------- routing
#
# test_*.py                     -> tests/
# verb-prefixed operational files -> scripts/
# everything else                 -> app/retrieval/
#
# Scripts are things you run; modules are things you import. The rule is the
# leading verb: probe_, scope_, ingest_, embed_, build_, export_, coverage_.
# `embedding.py` is a module, `embed_chunks.py` is a script - the underscore
# after the verb is what separates them.

route_for() {
  local base="$1"
  case "$base" in
    test_*.py)
      echo "$TESTS_DIR" ;;
    probe_*.py|scope_*.py|coverage_*.py|ingest_*.py|embed_*.py|build_*.py|\
    export_*.py|search_*.py|eval_*.py|*_stats.py)
      echo "$SCRIPTS_DIR" ;;
    *.py)
      echo "$RETRIEVAL_DIR" ;;
    *.sh)
      echo "$SCRIPTS_DIR" ;;
    *)
      echo "" ;;
  esac
}

head_ "Installing from $SRC"

moved=()
shopt -s nullglob
for f in "$SRC"/*.py "$SRC"/*.sh; do
  base="$(basename "$f")"
  # never overwrite the running script from under itself
  [[ "$base" == "install_drop.sh" && "$f" -ef "${BASH_SOURCE[0]}" ]] && continue
  dest_dir="$(route_for "$base")"
  [[ -z "$dest_dir" ]] && { say "skip     $base"; continue; }

  dest="$dest_dir/$base"
  if [[ -f "$dest" ]]; then
    mkdir -p "$BACKUP_DIR"
    run cp "$dest" "$BACKUP_DIR/$base"
    say "replace  $base -> ${dest#$PROJ/}   (backup kept)"
  else
    say "new      $base -> ${dest#$PROJ/}"
  fi
  run cp "$f" "$dest"
  [[ "$base" == *.sh ]] && run chmod +x "$dest"
  moved+=("$dest")
done
shopt -u nullglob

[[ ${#moved[@]} -eq 0 ]] && { echo "nothing to install"; exit 0; }

# ---------------------------------------------------------------- imports

head_ "Rewriting imports"

# Every module that lives in app/retrieval/ is a candidate for rewriting.
mapfile -t MODULES < <(
  find "$RETRIEVAL_DIR" -maxdepth 1 -name '*.py' ! -name '__init__.py' \
    -exec basename {} .py \;
)
say "package modules: ${MODULES[*]:-none}"

fixed_any=0
for dest in "${moved[@]}"; do
  # only python under tests/ and scripts/ carries flat imports
  [[ "$dest" == *.py ]] || continue
  case "$dest" in
    "$TESTS_DIR"/*|"$SCRIPTS_DIR"/*) ;;
    *) continue ;;
  esac
  # in a dry run the file was never copied, so there is nothing to inspect
  [[ -f "$dest" ]] || { say "would scan ${dest#$PROJ/} for flat imports"; continue; }

  for mod in "${MODULES[@]}"; do
    # only rewrite bare `from mod import` / `import mod`, never an already
    # qualified `from app.retrieval.mod import`
    if grep -qE "^[[:space:]]*(from[[:space:]]+$mod[[:space:]]+import|import[[:space:]]+$mod\b)" "$dest"; then
      if [[ "$DRY_RUN" == "1" ]]; then
        say "would fix ${dest#$PROJ/}: $mod -> app.retrieval.$mod"
      else
        sed -i -E \
          -e "s/^([[:space:]]*)from[[:space:]]+$mod[[:space:]]+import/\1from app.retrieval.$mod import/" \
          -e "s/^([[:space:]]*)import[[:space:]]+$mod\b/\1import app.retrieval.$mod as $mod/" \
          "$dest"
        say "fixed    ${dest#$PROJ/}: $mod -> app.retrieval.$mod"
      fi
      fixed_any=1
    fi
  done
done
[[ "$fixed_any" == "0" ]] && say "no flat imports found"

# ---------------------------------------------------------------- verify

if [[ "$DRY_RUN" == "1" ]]; then
  head_ "Dry run - nothing written"
  exit 0
fi

head_ "Remaining flat imports (should be none)"
if grep -rnE "^[[:space:]]*from[[:space:]]+($(IFS='|'; echo "${MODULES[*]}"))[[:space:]]+import" \
     "$TESTS_DIR" "$SCRIPTS_DIR" 2>/dev/null; then
  echo "  ^ still flat - fix by hand" >&2
else
  say "clean"
fi

head_ "Optional dependencies"
for mod in sqlite_vec sentence_transformers; do
  if python3 -c "import $mod" 2>/dev/null; then
    say "ok       $mod"
  else
    say "MISSING  $mod   (pip install ${mod//_/-})"
  fi
done

head_ "Running tests"
cd "$PROJ"
python3 -m pytest tests/ -q

head_ "Done"
say "backups: ${BACKUP_DIR#$PROJ/}"
