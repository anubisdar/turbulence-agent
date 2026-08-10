#!/usr/bin/env bash
# install-to: scripts
#
# install_static.sh - place index.html, which install_drop.sh does not route.
#
# The installer handles .py and .sh. The web page is neither, and it belongs
# in a directory the API serves from rather than anywhere a pattern would
# guess. One job, run after the usual drop.
#
# Usage:
#   ./scripts/install_static.sh              # from /home/darlage
#   ./scripts/install_static.sh /some/dir
#
set -euo pipefail

SRC="${1:-/home/darlage}"
PROJ="${PROJ:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
DEST="$PROJ/app/web/static"

[[ -f "$SRC/index.html" ]] || { echo "no index.html in $SRC" >&2; exit 1; }

mkdir -p "$DEST"
if [[ -f "$DEST/index.html" ]]; then
  backup="$PROJ/.drop-backups/$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$backup"
  cp "$DEST/index.html" "$backup/"
  echo "  replaced (backup in ${backup#$PROJ/})"
fi
cp "$SRC/index.html" "$DEST/index.html"
echo "  installed -> app/web/static/index.html"
echo
echo "start the server:  ./scripts/serve.sh"
echo "then open:         http://<this-host>:8000/"
