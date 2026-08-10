#!/usr/bin/env bash
# install-to: scripts
#
# loc.sh - count lines of code in this project, broken down usefully.
#
# Counts three things separately, because a single total hides the shape of
# the work: production code under app/, tests under tests/, and operational
# scripts. Blank lines and comment-only lines are reported apart from
# statements so the numbers mean something.
#
# Uses git ls-files, so .gitignore already excludes .venv, data/, and
# caches. Untracked files are not counted - commit first if that matters.
#
# Usage:
#   ./scripts/loc.sh
#   ./scripts/loc.sh --files      # per-file detail, largest first
#
set -euo pipefail

cd "${PROJ:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

DETAIL=0
[[ "${1:-}" == "--files" ]] && DETAIL=1

head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# Emit "statements blanks comments total" for a set of files.
count() {
  local files=("$@")
  [[ ${#files[@]} -eq 0 ]] && { echo "0 0 0 0"; return; }
  awk '
    { total++ }
    /^[[:space:]]*$/            { blank++;   next }
    /^[[:space:]]*#/            { comment++; next }
    { code++ }
    END { printf "%d %d %d %d\n", code+0, blank+0, comment+0, total+0 }
  ' "${files[@]}"
}

row() {
  local label="$1"; shift
  read -r code blank comment total <<< "$(count "$@")"
  printf '  %-22s %7s %8s %9s %8s %7s\n' \
    "$label" "$code" "$blank" "$comment" "$total" "${#@}"
}

mapfile -t APP_FILES    < <(git ls-files 'app/*.py' | grep -v '__init__.py' || true)
mapfile -t TEST_FILES   < <(git ls-files 'tests/*.py' || true)
mapfile -t SCRIPT_FILES < <(git ls-files 'scripts/*' || true)

head_ "Lines of code"
printf '  %-22s %7s %8s %9s %8s %7s\n' \
  "" "code" "blank" "comment" "total" "files"
printf '  %s\n' "-------------------------------------------------------------------------"

row "app/ (production)"  "${APP_FILES[@]}"
row "tests/"             "${TEST_FILES[@]}"
row "scripts/"           "${SCRIPT_FILES[@]}"
printf '  %s\n' "-------------------------------------------------------------------------"
row "TOTAL" "${APP_FILES[@]}" "${TEST_FILES[@]}" "${SCRIPT_FILES[@]}"

head_ "By package"
for pkg in $(git ls-files 'app/*/*.py' | cut -d/ -f2 | sort -u); do
  mapfile -t files < <(git ls-files "app/$pkg/*.py" | grep -v '__init__.py' || true)
  row "app/$pkg" "${files[@]}"
done

head_ "Test coverage by weight"
read -r app_code _ _ _ <<< "$(count "${APP_FILES[@]}")"
read -r test_code _ _ _ <<< "$(count "${TEST_FILES[@]}")"
if (( app_code > 0 )); then
  printf '  %s lines of test per line of production code\n' \
    "$(awk -v t="$test_code" -v a="$app_code" 'BEGIN{printf "%.2f", t/a}')"
fi
printf '  %s test(s) collected\n' \
  "$(python3 -m pytest tests/ --collect-only -q 2>/dev/null | tail -1 | grep -oE '^[0-9]+' || echo '?')"

if (( DETAIL )); then
  head_ "Largest files"
  git ls-files '*.py' '*.sh' \
    | xargs wc -l 2>/dev/null \
    | sort -rn | sed '1d' | head -25 | sed 's/^/  /'
fi

head_ "Note"
echo "  Counts tracked files only. Docstrings count as code here - awk cannot"
echo "  see block strings - so app/ is inflated by the module documentation."
echo "  For a strict split, use: cloc --vcs=git ."
