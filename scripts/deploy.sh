#!/usr/bin/env bash
# install-to: scripts
#
# deploy.sh - push the agent to an EC2 instance.
#
# Sends the application, the scripts, and the built SQLite index. Does NOT
# send the NTSB cache: it is ~250 MB, it exists only to rebuild the index,
# and the index itself is 19 MB. Nor the virtualenv, which bootstrap.sh
# builds on the far side.
#
# Usage:
#   ./scripts/deploy.sh 3.14.159.26
#   ./scripts/deploy.sh ec2-user@myhost --dry-run
#   KEY=~/.ssh/demo.pem ./scripts/deploy.sh 3.14.159.26
#
set -euo pipefail

PROJ="${PROJ:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
REMOTE_DIR="/opt/turbulence-agent"
APP_USER="turbulence"

HOST="${1:-}"
shift || true
DRY=""
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY="--dry-run"
done

[[ -n "$HOST" ]] || { echo "usage: $0 <host> [--dry-run]" >&2; exit 1; }
[[ "$HOST" == *@* ]] || HOST="ubuntu@$HOST"

SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
[[ -n "${KEY:-}" ]] && SSH_OPTS+=(-i "$KEY")

cd "$PROJ"
say()   { printf '  %s\n' "$*"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }

head_ "Checking locally"
[[ -d app/web ]] || { echo "no app/web here - wrong directory?" >&2; exit 1; }
if [[ -f data/retrieval.db ]]; then
  say "index: $(du -h data/retrieval.db | cut -f1)"
else
  say "no data/retrieval.db - the safety record lookup will not work"
fi
if ! python3 -m pytest tests/ -q >/dev/null 2>&1; then
  echo "  tests are failing. Fix before deploying." >&2
  exit 1
fi
say "tests pass"

head_ "Syncing to $HOST"
rsync -az --info=stats1 $DRY \
  -e "ssh ${SSH_OPTS[*]}" \
  --exclude '.venv/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '.git/' --exclude '.pytest_cache/' \
  --exclude 'data/ntsb_cache/' --exclude '.drop-backups/' \
  --exclude '.checkpoints/' \
  app scripts tests pyproject.toml \
  "$HOST:/tmp/turbulence-sync/"

if [[ -f data/retrieval.db ]]; then
  head_ "Sending the index"
  rsync -az --info=progress2 $DRY -e "ssh ${SSH_OPTS[*]}" \
    data/retrieval.db "$HOST:/tmp/turbulence-sync-db"
fi

[[ -n "$DRY" ]] && { head_ "Dry run - nothing installed"; exit 0; }

head_ "Installing on the remote"
ssh "${SSH_OPTS[@]}" "$HOST" "sudo bash -s" <<REMOTE
set -euo pipefail
rsync -a --delete /tmp/turbulence-sync/ $REMOTE_DIR/ \
  --exclude 'data/' --exclude '.venv/' --exclude '.cache/'
if [[ -f /tmp/turbulence-sync-db ]]; then
  mkdir -p $REMOTE_DIR/data
  mv /tmp/turbulence-sync-db $REMOTE_DIR/data/retrieval.db
fi
chown -R $APP_USER:$APP_USER $REMOTE_DIR
rm -rf /tmp/turbulence-sync
systemctl restart turbulence-agent
sleep 2
systemctl is-active --quiet turbulence-agent \
  && echo "  service running" \
  || { echo "  service failed to start:"; journalctl -u turbulence-agent -n 20 --no-pager; exit 1; }
REMOTE

head_ "Health check"
ssh "${SSH_OPTS[@]}" "$HOST" \
  "curl -fsS http://127.0.0.1:8000/api/health" | python3 -m json.tool | sed 's/^/  /'

head_ "Done"
say "logs: ssh $HOST 'journalctl -u turbulence-agent -f'"
