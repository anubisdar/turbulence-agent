#!/usr/bin/env bash
# install-to: scripts
#
# refresh_edge.sh - pull the edge events in now, instead of waiting for
# the timer.
#
# The status page reads the database live; there is no server-side cache.
# So the lag between something happening at the edge and the page showing
# it is the ingest timer, which fires every five minutes. This runs the
# same ingest immediately and tells you what changed.
#
# WHAT IT DOES NOT DO. It changes no configuration. The timer keeps its
# cadence, the service keeps its window, and nothing is enabled or
# disabled. Running this twice in a row is harmless: the ingest
# deduplicates on insert, so the second run reads the same window and
# writes nothing.
#
# WHY MANUAL RATHER THAN A TIGHTER TIMER. A one-minute cadence is a
# change to the running system that has to be right at three in the
# morning as well as now. This is a change to nothing, that you run when
# you are looking at the page. If it turns out you run it constantly,
# that is the evidence for tightening the timer - and you will know what
# the ingest costs by then, which you do not yet.
#
# Usage:
#   sudo ./refresh_edge.sh              # read the last 30 minutes
#   sudo ./refresh_edge.sh --minutes 240
#   sudo ./refresh_edge.sh --dry-run    # show what would be read
#   sudo ./refresh_edge.sh --no-probe-filter
#
set -uo pipefail

APP_DIR="${TURBULENCE_APP:-/opt/turbulence-agent}"
APP_USER="${TURBULENCE_USER:-turbulence}"
MINUTES=30
DRY=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --minutes)          MINUTES="$2"; shift ;;
    --dry-run)          DRY=1 ;;
    --no-probe-filter)  EXTRA+=(--no-probe-filter) ;;
    -h|--help)          sed -n '/^# Usage:/,/^$/p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

[[ "$MINUTES" =~ ^[0-9]+$ ]] \
  || { echo "--minutes wants a number" >&2; exit 2; }

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
say()  { printf '  %s\n' "$*"; }
dim()  { printf '  \033[2m%s\033[0m\n' "$*"; }
warn() { printf '  \033[33m%s\033[0m\n' "$*"; }
die()  { printf '  \033[31mSTOP  %s\033[0m\n' "$*" >&2; exit 1; }

PY="$APP_DIR/.venv/bin/python"
DB="$APP_DIR/data/retrieval.db"
INGEST="$APP_DIR/scripts/ingest_edge_events.py"

# ------------------------------------------------------------- preflight
[[ $EUID -eq 0 ]] || die "run with sudo (journald and the database need it)"
[[ -x "$PY" ]]     || die "no interpreter at $PY"
[[ -f "$INGEST" ]] || die "no ingest script at $INGEST"
[[ -r "$DB" ]]     || die "cannot read $DB"
id "$APP_USER" >/dev/null 2>&1 \
  || die "no such user: $APP_USER"

#: Every query runs as the account that owns the database. sqlite3 writes
#: -wal and -journal beside the file, and a sidecar owned by root leaves
#: the application unable to write with no error that says so.
q() {
  sudo -u "$APP_USER" sqlite3 -cmd ".timeout 5000" "$DB" "$1" 2>/dev/null
}

HAS_PROBE=$(q "SELECT COUNT(*) FROM pragma_table_info('edge_events')
               WHERE name='probe';")

bold "Before"
BEFORE=$(q "SELECT COUNT(*) FROM edge_events;")
LATEST=$(q "SELECT COALESCE(MAX(occurred_at),'never') FROM edge_events;")
say "$BEFORE row(s), most recent $LATEST"
if [[ "${HAS_PROBE:-0}" != "1" ]]; then
  warn "no probe column - this database predates the self-check marking,"
  warn "so the counts below include the operator's own health check"
fi

# ------------------------------------------------------------- the run
bold "Reading the last $MINUTES minutes"
ARGS=(--lookback-minutes "$MINUTES")
[[ "${#EXTRA[@]}" -gt 0 ]] && ARGS+=("${EXTRA[@]}")
[[ "$DRY" -eq 1 ]] && ARGS+=(--dry-run)

# Run as the app user, from the app directory, under the venv - the same
# three things the systemd unit does. Running it as root under system
# python is how the mail report spent weeks printing "?" for every
# address, and it would fail the same way here.
if ! (cd "$APP_DIR" && sudo -u "$APP_USER" "$PY" "$INGEST" "${ARGS[@]}"); then
  die "the ingest failed - nothing was written"
fi

if [[ "$DRY" -eq 1 ]]; then
  bold "Dry run"
  dim "nothing was written. Re-run without --dry-run to ingest."
  exit 0
fi

# ------------------------------------------------------------- after
bold "After"
AFTER=$(q "SELECT COUNT(*) FROM edge_events;")
say "$((AFTER - BEFORE)) new row(s), $AFTER total"

WINDOW="datetime('now','-${MINUTES} minutes')"
if [[ "${HAS_PROBE:-0}" == "1" ]]; then
  ROWS=$(sudo -u "$APP_USER" sqlite3 -header -column "$DB" "
    SELECT kind,
           CASE probe WHEN 1 THEN 'self-check' ELSE 'outside' END AS whose,
           COALESCE(detail,'?') AS detail,
           COUNT(*)             AS n
      FROM edge_events
     WHERE occurred_at >= $WINDOW
     GROUP BY kind, probe, detail
     ORDER BY n DESC LIMIT 12;" 2>/dev/null)
else
  ROWS=$(sudo -u "$APP_USER" sqlite3 -header -column "$DB" "
    SELECT kind, COALESCE(detail,'?') AS detail, COUNT(*) AS n
      FROM edge_events
     WHERE occurred_at >= $WINDOW
     GROUP BY kind, detail ORDER BY n DESC LIMIT 12;" 2>/dev/null)
fi
if [[ -n "$ROWS" ]]; then
  sed 's/^/    /' <<<"$ROWS"
else
  dim "nothing in the window"
fi

# ------------------------------------------------------------- what next
bold "The page"
dim "/api/status reads the database on every request, so there is nothing"
dim "to flush. Reload the status page and it will show the rows above."
dim ""
dim "If it does not, the application is running older code than the"
dim "database has columns for:  sudo systemctl restart turbulence-agent"

NEXT=$(systemctl list-timers ingest-edge-events.timer --no-pager 2>/dev/null \
       | awk 'NR==2 {print $1, $2, $3}')
[[ -n "$NEXT" ]] && dim "" && dim "the timer would have done this at $NEXT"
