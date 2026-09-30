#!/usr/bin/env bash
# install-to: scripts
#
# weekly_check.sh - the regression and upstream-health check, run weekly.
#
# Every commit already runs the pytest suite (deterministic, network-blocked
# by tests/conftest.py, costs nothing). What that suite cannot catch is the
# thing that has actually broken this project four times: an upstream API
# quietly changing its response shape while the fixtures keep encoding the
# old belief (see tests/test_upstream_contracts.py's own docstring). Catching
# that needs a real call to the real API, which is why it is not in the
# per-commit suite and has to happen on its own schedule instead.
#
# CORE (default, safe to run unattended - free or a few cents):
#   1. Full pytest suite
#   2. Refresh the G-AIRMET probe (free) and re-run the upstream contract
#      tests against it
#   3. scripts/validate_turbulence.py --fixtures (end-to-end, no API spend)
#   4. pip-audit, if installed (dependency vulnerabilities)
#
# OPTIONAL, cost real money against metered APIs - opt in explicitly:
#   --aeroapi-probe   refresh the AeroAPI probe (~6 calls, a few cents),
#                     then re-run the contract tests against it too
#   --live-validate   validate_turbulence.py against a deployed --host,
#                     using live AeroAPI + AWC data instead of fixtures
#   --load-test       20 real routes against a deployed --host
#                     (scripts/load_test.py)
#   --redteam         adversarial run against the live model
#                     (scripts/redteam_explainer.py --yes) - the most
#                     expensive check here, LLM calls not API metering
#   --full            all of the above
#
# EMAIL. On by default - the report is sent via scripts/send_weekly_report.py
# once every check has run, to matthew.darlage@gmail.com unless --to or
# WEEKLY_REPORT_TO says otherwise. Needs WEEKLY_REPORT_SMTP_HOST,
# _SMTP_USER and _SMTP_PASS set (see turbulence-agent.env.example); if
# they are not, the email step prints why and exits nonzero, but that
# never rewrites this run's own PASS/FAIL - the report is still written
# to disk and this script's exit code still reflects the checks alone,
# not whether the email went out. --no-email skips the step entirely.
#
# Usage:
#   ./scripts/weekly_check.sh                                   # core only
#   ./scripts/weekly_check.sh --full --host https://turbulence.adeptsecurity.net
#   ./scripts/weekly_check.sh --aeroapi-probe --live-validate --host <url>
#   ./scripts/weekly_check.sh --no-email                        # local run
#
# Needs AEROAPI_KEY for anything that touches AeroAPI (--aeroapi-probe,
# --live-validate, --load-test) and ANTHROPIC_API_KEY for --redteam. Missing
# a key for a check you asked for fails that check rather than skipping it
# quietly - same principle the app itself follows.
#
# Needs TURBULENCE_OPERATOR_TOKEN too, whenever --host has a Turnstile
# challenge in front of it (the public deployment does) - both the fixtures
# check and --live-validate are automated clients and cannot solve it, same
# as load_test.py. validate_turbulence.py picks the token up from this
# variable itself; nothing further to wire here. Note the name is
# TURBULENCE_OPERATOR_TOKEN, not TURNSTILE_OPERATOR_TOKEN - an easy
# mismatch turbulence-agent.env.example itself used to have.
#
# Runs from blueadept against the deployed instance, same as check_edge.sh -
# not from /opt/turbulence-agent, which is the EC2 deployment target
# deploy.sh pushes to, not where this (or check_edge.sh) actually runs from.
#
# Cron, Sundays at 06:40 (source the env file first so the SMTP and API
# keys are in the shell that runs this - cron does not read them itself):
#   40 6 * * 0 cd /root/projects/turbulence-agent && set -a && \
#       . turbulence-agent.env && set +a && ./scripts/weekly_check.sh \
#       --host https://turbulence.adeptsecurity.net >> reports/weekly_cron.log 2>&1
#
set -uo pipefail

PROJ="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$PROJ"

HOST="${TURBULENCE_HOST:-https://turbulence.adeptsecurity.net}"
DO_AEROAPI_PROBE=0
DO_LIVE_VALIDATE=0
DO_LOAD_TEST=0
DO_REDTEAM=0
DO_EMAIL=1
MAIL_TO="${WEEKLY_REPORT_TO:-matthew.darlage@gmail.com}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --aeroapi-probe) DO_AEROAPI_PROBE=1; shift ;;
    --live-validate) DO_LIVE_VALIDATE=1; shift ;;
    --load-test) DO_LOAD_TEST=1; shift ;;
    --redteam) DO_REDTEAM=1; shift ;;
    --no-email) DO_EMAIL=0; shift ;;
    --to) MAIL_TO="$2"; shift 2 ;;
    --full)
      DO_AEROAPI_PROBE=1; DO_LIVE_VALIDATE=1; DO_LOAD_TEST=1; DO_REDTEAM=1
      shift ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
REPORT_DIR="reports/weekly"
mkdir -p "$REPORT_DIR"
REPORT="$REPORT_DIR/$STAMP.md"
LOG_DIR="$REPORT_DIR/$STAMP-logs"
mkdir -p "$LOG_DIR"

# Marks "before this run" so a probe script's own exit code can be checked
# against whether it actually wrote anything. Both probe scripts swallow
# HTTP and network errors internally and still exit 0 - that is correct for
# a script a person reads, and wrong for one this wrapper trusts blindly.
# An exit-0 probe that wrote no fresh file is exactly the "API is failing"
# case this exists to catch, so freshness is checked, not just the exit code.
SENTINEL="$LOG_DIR/sentinel"
touch "$SENTINEL"
fresh_json() { find "$1" -name '*.json' -newer "$SENTINEL" 2>/dev/null | grep -q .; }

# name / command / cost-tier / skip-reason, in run order
declare -a NAMES=() CMDS=() TIERS=() SKIP_MSGS=() STATUSES=()

add_check() {
  NAMES+=("$1"); CMDS+=("$2"); TIERS+=("$3"); SKIP_MSGS+=("${4:-}")
}

# Deliberately hermetic, regardless of what the calling shell has sourced.
# The documented cron line sources turbulence-agent.env before this script
# runs, which puts real TURNSTILE_SITE_KEY/SECRET_KEY and TURBULENCE_PUBLIC
# into this same shell - and app/web/api.py and turnstile.py read those at
# request time (PUBLIC at import time). Left in place, the pytest suite's
# in-process TestClient starts requiring a challenge no test sends,
# cascading into dozens of unrelated-looking 403s. Caught this by actually
# running the documented cron recipe end to end rather than just the bare
# script - it's exactly the gap a --dry-run of the command alone would
# have missed.
PYTEST_ENV_ISOLATION="env -u TURNSTILE_SITE_KEY -u TURNSTILE_SECRET_KEY \
  -u TURBULENCE_PUBLIC -u TURBULENCE_SESSION_SECRET -u TURBULENCE_OPERATOR_TOKEN"

add_check "pytest suite" \
  "$PYTEST_ENV_ISOLATION python -m pytest -q --ignore=tests/test_geometry_properties.py" \
  "free"

add_check "refresh G-AIRMET probe" \
  "python scripts/probe_gairmet.py --save && fresh_json data/awc_probe" \
  "free"

add_check "upstream contract tests" \
  "python -m pytest -q tests/test_upstream_contracts.py" \
  "free"

# FixtureTransport reads data/aeroapi_probe relative to wherever the FastAPI
# process's own cwd is - on a deployed --host that is the EC2 instance
# (/opt/turbulence-agent), never this box. `-d data/aeroapi_probe` here was
# checking blueadept's own filesystem, which has no bearing on whether the
# thing this check actually calls (the deployed --host) can serve fixtures -
# it can pass when the remote has none, or skip when the remote does. The
# app already answers this correctly of itself: /api/health reports
# `fixtures_available` from its own cwd, so ask the host instead of guessing
# from a directory on a different machine.
HEALTH="$(curl -sk --max-time 15 "$HOST/api/health" 2>/dev/null)"
if [[ "$HEALTH" == *'"fixtures_available":true'* ]]; then
  add_check "end-to-end against $HOST (fixtures, no spend)" \
    "python scripts/validate_turbulence.py --host $HOST --fixtures" \
    "free"
elif [[ -z "$HEALTH" ]]; then
  add_check "end-to-end (fixtures, no spend)" "" "missing" \
    "$HOST/api/health did not respond — is the host reachable and running?"
else
  add_check "end-to-end (fixtures, no spend)" "" "missing" \
    "$HOST reports no captured AeroAPI payloads (fixtures_available: false) — SSH into that host and run ./scripts/probe_aeroapi.py there once to seed them (a few cents, one-time); running it here on blueadept would seed the wrong machine"
fi

# pip-audit scans whatever Python environment it happens to be running
# under - not necessarily the one actually serving traffic. On a host
# where the weekly check runs from its own isolated venv (AWS, after the
# 2026-09-30 outage taught us not to install test tooling into the app's
# own .venv), a bare `pip-audit` on $PATH would silently audit that
# isolated venv's packages instead of production's, giving a clean report
# that never looked at what's actually deployed. AUDIT_VENV names the venv
# to audit explicitly when it differs from whichever one is on $PATH; set
# it in the cron line on hosts that split the two (AWS:
# AUDIT_VENV=/opt/turbulence-agent/.venv). Left unset - the common case,
# including blueadept, which has never had more than one venv - this
# behaves exactly as it always has.
if command -v pip-audit >/dev/null 2>&1; then
  if [[ -n "${AUDIT_VENV:-}" ]]; then
    # pip-audit's -r takes a real file, not stdin ("-r -" is rejected
    # outright) - verified directly rather than assumed, since guessing
    # wrong here would mean shipping this untested onto the AWS box.
    # Process substitution gives it a real path to open.
    add_check "dependency audit (pip-audit)" \
      "pip-audit -r <(\"$AUDIT_VENV/bin/pip\" freeze)" \
      "free"
  else
    add_check "dependency audit (pip-audit)" "pip-audit" "free"
  fi
else
  add_check "dependency audit (pip-audit)" "" "missing" \
    "pip-audit is not installed — pip install pip-audit --break-system-packages"
fi

if [[ "$DO_AEROAPI_PROBE" == 1 ]]; then
  add_check "refresh AeroAPI probe" \
    "python scripts/probe_aeroapi.py && fresh_json data/aeroapi_probe" \
    "\$ (a few cents, ~6 calls)"
  add_check "upstream contract tests, re-run against fresh AeroAPI probe" \
    "python -m pytest -q tests/test_upstream_contracts.py" \
    "free"
fi

if [[ "$DO_LIVE_VALIDATE" == 1 ]]; then
  add_check "end-to-end against $HOST (live data)" \
    "python scripts/validate_turbulence.py --host $HOST" \
    "\$ (one metered search)"
fi

if [[ "$DO_LOAD_TEST" == 1 ]]; then
  add_check "breadth check against $HOST (20 routes)" \
    "python scripts/load_test.py --host $HOST" \
    "\$\$ (20 metered searches)"
fi

if [[ "$DO_REDTEAM" == 1 ]]; then
  add_check "explainer red team (live model)" \
    "python scripts/redteam_explainer.py --yes --out $LOG_DIR/redteam.jsonl" \
    "\$\$ (LLM calls, not API-metered)"
fi

say()  { printf '  %s\n' "$*"; }
head_() { printf '\n\033[1m%s\033[0m\n' "$*"; }

# Detail blocks are written here during the loop and folded into the real
# report afterward, so the summary table can sit at the top of the file -
# the one thing worth reading first, whether that's a skim on disk or the
# first screen of an email.
DETAIL="$LOG_DIR/detail.md"
: > "$DETAIL"

OVERALL=0

for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"; cmd="${CMDS[$i]}"; tier="${TIERS[$i]}"
  msg="${SKIP_MSGS[$i]}"
  head_ "$name  [$tier]"

  if [[ "$tier" == "missing" ]]; then
    say "SKIPPED — $msg"
    STATUSES+=("SKIPPED")
    {
      echo "## $name — SKIPPED"
      echo
      echo "$msg"
      echo
    } >> "$DETAIL"
    continue
  fi

  logfile="$LOG_DIR/$i.log"
  if eval "$cmd" > "$logfile" 2>&1; then
    STATUSES+=("PASS")
    say "PASS"
  else
    STATUSES+=("FAIL")
    say "FAIL — see $logfile"
    OVERALL=1
  fi

  # Head and tail, not just tail: a long pytest failure list buries its own
  # "N failed, M passed" summary line past any fixed tail window, and the
  # summary is the one line a skim actually needs.
  {
    echo "## $name — ${STATUSES[$i]}"
    echo
    echo '```'
    if [[ "$(wc -l < "$logfile")" -gt 60 ]]; then
      head -n 30 "$logfile"
      echo "... [$(wc -l < "$logfile") lines total, truncated — full log was at $logfile] ..."
      tail -n 30 "$logfile"
    else
      cat "$logfile"
    fi
    echo '```'
    echo
  } >> "$DETAIL"
done

head_ "Summary"
STATUS_WORD="PASS"
[[ "$OVERALL" != 0 ]] && STATUS_WORD="FAIL"

{
  echo "# Weekly check — $STAMP — $STATUS_WORD"
  echo
  echo "Host: \`$HOST\`"
  echo
  echo "## Summary"
  echo
  echo "| Check | Cost | Result |"
  echo "|---|---|---|"
} > "$REPORT"
for i in "${!NAMES[@]}"; do
  printf '  %-55s %s\n' "${NAMES[$i]}" "${STATUSES[$i]}"
  echo "| ${NAMES[$i]} | ${TIERS[$i]} | ${STATUSES[$i]} |" >> "$REPORT"
done
echo >> "$REPORT"
cat "$DETAIL" >> "$REPORT"

echo
echo "Full report: $REPORT"
echo "Logs kept at: $LOG_DIR"

if [[ "$DO_EMAIL" == 1 ]]; then
  head_ "Emailing report to $MAIL_TO"
  if python scripts/send_weekly_report.py "$REPORT" --status "$STATUS_WORD" \
      --to "$MAIL_TO"; then
    say "sent"
  else
    say "not sent (see above) — the report is still at $REPORT"
  fi
fi

exit "$OVERALL"
