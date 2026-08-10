#!/usr/bin/env bash
# install-to: scripts
#
# serve.sh - run the web API.
#
# Fixture mode needs no key; live mode needs AEROAPI_KEY. The health
# endpoint reports which is available.
#
# Usage:
#   ./scripts/serve.sh                 # port 8000, all interfaces
#   PORT=8080 ./scripts/serve.sh
#   ./scripts/serve.sh --reload        # auto-restart while editing
#
set -euo pipefail

cd "${PROJ:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

python3 -c "import fastapi, uvicorn" 2>/dev/null || {
  echo "missing dependencies. install with:" >&2
  echo "    pip install fastapi 'uvicorn[standard]'" >&2
  exit 1
}

if [[ -z "${AEROAPI_KEY:-}" ]]; then
  echo "note: AEROAPI_KEY is not set - live searches will be refused."
  echo "      fixture mode still works: pass use_fixtures true."
fi

echo "docs at http://${HOST}:${PORT}/docs"
exec python3 -m uvicorn app.web.api:app --host "$HOST" --port "$PORT" "$@"
