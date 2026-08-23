#!/usr/bin/env bash
# install-to: scripts
#
# install_static.sh - place the web pages, which install_drop.sh does not
# route.
#
# The installer handles .py and .sh. A page is neither, and it belongs in a
# directory the API serves from rather than anywhere a pattern would guess.
#
# Every .html in the drop is installed. An earlier version looked for
# index.html specifically, so a drop containing only status.html failed with
# "no index.html" - which reads as a missing file rather than as a file the
# script had no rule for.
#
# Usage:
#   ./scripts/install_static.sh              # from /home/darlage
#   ./scripts/install_static.sh /some/dir
#
set -euo pipefail

SRC="${1:-/home/darlage}"
PROJ="${PROJ:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
DEST="$PROJ/app/web/static"

shopt -s nullglob
pages=("$SRC"/*.html)
shopt -u nullglob

if [[ ${#pages[@]} -eq 0 ]]; then
  echo "no .html files in $SRC" >&2
  echo >&2
  echo "This installs pages into app/web/static. Python and shell files go" >&2
  echo "through install_drop.sh instead." >&2
  exit 1
fi

mkdir -p "$DEST"
backup=""
installed=0

for page in "${pages[@]}"; do
  name="$(basename "$page")"
  if [[ -f "$DEST/$name" ]]; then
    if cmp -s "$page" "$DEST/$name"; then
      echo "  unchanged  $name"
      continue
    fi
    # One backup directory per run, created only when something is actually
    # replaced.
    if [[ -z "$backup" ]]; then
      backup="$PROJ/.drop-backups/$(date +%Y%m%d-%H%M%S)"
      mkdir -p "$backup"
    fi
    cp "$DEST/$name" "$backup/"
    echo "  replaced   $name"
  else
    echo "  new        $name"
  fi
  cp "$page" "$DEST/$name"
  installed=$((installed + 1))
done

echo
if [[ "$installed" -eq 0 ]]; then
  echo "Nothing changed."
  exit 0
fi

echo "$installed page(s) installed to app/web/static"
[[ -n "$backup" ]] && echo "backups: ${backup#$PROJ/}"

# Named rather than counted, because a page that never gets served is a
# page nobody notices is missing.
echo
echo "served at:"
for page in "${pages[@]}"; do
  name="$(basename "$page")"
  case "$name" in
    index.html)  echo "  /            $name" ;;
    status.html) echo "  /status      $name" ;;
    *)           echo "  (no route)   $name  -- add one in app/web/api.py" ;;
  esac
done

echo
echo "start the server:  ./scripts/serve.sh"
