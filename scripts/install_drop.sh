#!/usr/bin/env bash
#
# install_drop.sh - put files dropped from a chat into the project tree,
# rewrite the flat imports they arrive with, and verify they actually load.
#
# Routing is decided in this order, most reliable first:
#
#   1. A file already in the tree with the same name -> same directory.
#      Replacements never move. This alone fixes most misrouting.
#   2. An explicit marker in the file:  # install-to: app/reasoning
#   3. Filename pattern (tests, scripts, known module names).
#   4. Fallback, with a loud warning. Silent fallback is what put search.py
#      and graph.py in the wrong package.
#
# After installing, every placed module is import-checked. A file in the
# wrong package fails here with a clear message rather than surfacing later
# as a pytest collection error.
#
# Usage:
#   ./scripts/install_drop.sh                  # from /home/darlage, into $PWD
#   ./scripts/install_drop.sh /some/other/dir
#   DRY_RUN=1 ./scripts/install_drop.sh        # show the plan, change nothing
#
set -euo pipefail

SRC="${1:-/home/darlage}"
PROJ="${PROJ:-$(pwd)}"
DRY_RUN="${DRY_RUN:-0}"

APP_DIR="$PROJ/app"
TESTS_DIR="$PROJ/tests"
SCRIPTS_DIR="$PROJ/scripts"
RETRIEVAL_DIR="$APP_DIR/retrieval"
REASONING_DIR="$APP_DIR/reasoning"
SOURCES_DIR="$APP_DIR/sources"
BACKUP_DIR="$PROJ/.drop-backups/$(date +%Y%m%d-%H%M%S)"

say()   { printf '  %s\n' "$*"; }
warn()  { printf '  \033[33m%s\033[0m\n' "$*"; }
err()   { printf '  \033[31m%s\033[0m\n' "$*"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }
run()   { if [[ "$DRY_RUN" == "1" ]]; then say "would: $*"; else "$@"; fi; }

[[ -d "$SRC" ]]     || { echo "source dir not found: $SRC" >&2; exit 1; }
[[ -d "$APP_DIR" ]] || {
  echo "does not look like the project root: $PROJ" >&2
  echo "cd to the repo root, or set PROJ=/path/to/turbulence-agent" >&2
  exit 1
}

mkdir -p "$RETRIEVAL_DIR" "$REASONING_DIR" "$SOURCES_DIR" "$TESTS_DIR" "$SCRIPTS_DIR"
for d in "$RETRIEVAL_DIR" "$REASONING_DIR" "$SOURCES_DIR"; do
  [[ -f "$d/__init__.py" ]] || run touch "$d/__init__.py"
done

# ---------------------------------------------------------------- routing

# route_for runs in a command substitution, so it cannot set variables in
# the caller. A "GUESS:" prefix on the returned path carries that signal out.
route_for() {
  local file="$1" base existing marker
  base="$(basename "$file")"

  # 1. Replacement: this name already exists somewhere in the tree.
  existing=$(find "$APP_DIR" "$TESTS_DIR" "$SCRIPTS_DIR" -name "$base" \
               -not -path '*/.venv/*' -print -quit 2>/dev/null || true)
  if [[ -n "$existing" ]]; then
    dirname "$existing"
    return
  fi

  # 2. Explicit marker in the first 20 lines: # install-to: app/reasoning
  marker=$(head -20 "$file" 2>/dev/null \
           | sed -n 's/^[[:space:]]*#[[:space:]]*install-to:[[:space:]]*//p' \
           | head -1 | tr -d '[:space:]')
  if [[ -n "$marker" ]]; then
    echo "$PROJ/$marker"
    return
  fi

  # 3. Filename pattern.
  case "$base" in
    test_*.py)
      echo "$TESTS_DIR" ; return ;;
    probe_*.py|scope_*.py|coverage_*.py|ingest_*.py|embed_*.py|build_*.py|export_*.py|eval_*.py|patch_*.py|run_*.py|*_stats.py|*.sh)
      echo "$SCRIPTS_DIR" ; return ;;
    critic.py|controller.py|graph.py|generator.py|state.py|corridor*.py|beam*.py|tot_*.py)
      echo "$REASONING_DIR" ; return ;;
    awc.py|aeroapi.py|gtg.py|noaa*.py|pirep*.py|airmet*.py|flightaware*.py)
      echo "$SOURCES_DIR" ; return ;;
    aircraft_types.py|chunking.py|schema.py|embedding.py|search.py|retriev*.py)
      echo "$RETRIEVAL_DIR" ; return ;;
  esac

  # 4. Fallback. Flag it so the caller can warn.
  echo "GUESS:$RETRIEVAL_DIR"
}

head_ "Installing from $SRC"

declare -a MOVED=()
declare -a GUESSED=()
shopt -s nullglob
for f in "$SRC"/*.py "$SRC"/*.sh; do
  base="$(basename "$f")"
  [[ "$base" == "install_drop.sh" && "$f" -ef "${BASH_SOURCE[0]}" ]] && continue

  dest_dir="$(route_for "$f")"
  guessed=0
  if [[ "$dest_dir" == GUESS:* ]]; then
    guessed=1
    dest_dir="${dest_dir#GUESS:}"
  fi
  [[ -z "$dest_dir" ]] && { say "skip     $base"; continue; }
  mkdir -p "$dest_dir"
  dest="$dest_dir/$base"

  if [[ -f "$dest" ]]; then
    mkdir -p "$BACKUP_DIR"
    run cp "$dest" "$BACKUP_DIR/$base"
    say "replace  $base -> ${dest#$PROJ/}"
  elif (( guessed )); then
    warn "GUESSED  $base -> ${dest#$PROJ/}  (no rule matched)"
    GUESSED+=("$base")
  else
    say "new      $base -> ${dest#$PROJ/}"
  fi

  run cp "$f" "$dest"
  [[ "$base" == *.sh ]] && run chmod +x "$dest"
  MOVED+=("$dest")
done
shopt -u nullglob

[[ ${#MOVED[@]} -eq 0 ]] && { echo "nothing to install"; exit 0; }

if (( ${#GUESSED[@]} )); then
  warn ""
  warn "${#GUESSED[@]} file(s) had no routing rule and were placed by fallback."
  warn "If that is wrong, move them and add a rule, or put this near the top"
  warn "of the file so it routes itself next time:"
  warn "    # install-to: app/reasoning"
fi

# ---------------------------------------------------------------- imports

head_ "Rewriting imports"

declare -A MODULE_PKG=()
while IFS= read -r f; do
  b="$(basename "$f" .py)"
  MODULE_PKG["$b"]="app.$(basename "$(dirname "$f")")"
done < <(find "$APP_DIR" -mindepth 2 -maxdepth 2 -name '*.py' ! -name '__init__.py')
MODULES=("${!MODULE_PKG[@]}")
say "package modules: ${MODULES[*]:-none}"

fixed_any=0
for dest in "${MOVED[@]}"; do
  [[ "$dest" == *.py ]] || continue
  case "$dest" in
    "$TESTS_DIR"/*|"$SCRIPTS_DIR"/*|"$APP_DIR"/*) ;;
    *) continue ;;
  esac
  if [[ ! -f "$dest" ]]; then
    say "would scan ${dest#$PROJ/}"
    continue
  fi

  self_mod="$(basename "$dest" .py)"
  for mod in "${MODULES[@]}"; do
    [[ "$mod" == "$self_mod" ]] && continue
    grep -qE "^[[:space:]]*(from[[:space:]]+$mod[[:space:]]+import|import[[:space:]]+$mod\b)" \
      "$dest" || continue
    pkg="${MODULE_PKG[$mod]}"
    if [[ "$DRY_RUN" == "1" ]]; then
      say "would fix ${dest#$PROJ/}: $mod -> $pkg.$mod"
    else
      sed -i -E \
        -e "s/^([[:space:]]*)from[[:space:]]+$mod[[:space:]]+import/\1from $pkg.$mod import/" \
        -e "s/^([[:space:]]*)import[[:space:]]+$mod\b/\1import $pkg.$mod as $mod/" \
        "$dest"
      say "fixed    ${dest#$PROJ/}: $mod -> $pkg.$mod"
    fi
    fixed_any=1
  done
done
(( fixed_any )) || say "no flat imports found"

if [[ "$DRY_RUN" == "1" ]]; then
  head_ "Dry run - nothing written"
  exit 0
fi

# ---------------------------------------------------------------- verify

head_ "Undefined names"

# An import succeeding proves a module loads, not that every line in it
# runs. A name used where it is not bound sits quiet until that branch
# executes, which in one case meant reaching production and surfacing as a
# failed safety lookup on every search. pyflakes finds it in under a second.
if python3 -c "import pyflakes" 2>/dev/null; then
  undefined=$(python3 -m pyflakes "$PROJ/app" 2>/dev/null | grep "undefined name" || true)
  if [[ -n "$undefined" ]]; then
    err "undefined names found:"
    printf '  %s\n' "$undefined"
    err "not installing. Fix these first."
    exit 1
  fi
  say "none"
else
  warn "pyflakes is not installed, so undefined names will not be caught."
  warn "  pip install pyflakes"
fi

head_ "Import check"

cd "$PROJ"
import_failures=0
for dest in "${MOVED[@]}"; do
  [[ "$dest" == "$APP_DIR"/* && "$dest" == *.py ]] || continue
  rel="${dest#$PROJ/}"
  dotted="${rel%.py}"
  dotted="${dotted//\//.}"
  if python3 -c "import $dotted" 2>/dev/null; then
    say "ok       $dotted"
  else
    err "FAILED   $dotted"
    python3 -c "import $dotted" 2>&1 | tail -3 | sed 's/^/      /'
    import_failures=$((import_failures + 1))
  fi
done
if (( import_failures )); then
  err "$import_failures module(s) failed to import - likely misrouted"
fi

head_ "Optional dependencies"
for mod in sqlite_vec sentence_transformers langgraph; do
  if python3 -c "import $mod" 2>/dev/null; then
    say "ok       $mod"
  else
    warn "MISSING  $mod   (pip install ${mod//_/-})"
  fi
done

head_ "Running tests"
python3 -m pytest tests/ -q

head_ "Done"
say "backups: ${BACKUP_DIR#$PROJ/}"
