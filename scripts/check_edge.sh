#!/usr/bin/env bash
# install-to: scripts
#
# check_edge.sh - daily verification that the edge controls still work.
#
# Runs from blueadept against the deployed instance. Every control at the
# edge has broken silently at least once: the geo filter was present and
# dead for days because directive ordering put basic auth first, the WAF
# audit path wrote credentials to disk, and a missing Python dependency
# killed pilot reports while the output stayed honest. None of those
# announced themselves.
#
# WHY THE PROBES COST NOTHING. Every attack vector targets "/" and is
# refused by basic auth before reaching the application, so no AeroAPI or
# model call is spent. A health check that costs nine cents a run is a
# health check nobody runs.
#
# The probes carry a distinctive user agent so they can be excluded from
# the status page rather than inflating its detection counts.
#
# Usage:
#   ./scripts/check_edge.sh                    # all checks
#   ./scripts/check_edge.sh --no-remote        # HTTP only, no SSH
#   ./scripts/check_edge.sh --geo              # include foreign-origin test
#   ./scripts/check_edge.sh --quiet            # only report problems
#
# Cron, weekdays at 07:15:
#   15 7 * * 1-5 /root/projects/turbulence-agent/scripts/check_edge.sh \
#       --quiet >> /var/log/check_edge.log 2>&1

set -uo pipefail

HOST="${TURBULENCE_HOST:-https://turbulence.adeptsecurity.net}"
SSH_HOST="${TURBULENCE_SSH:-ubuntu@3.17.33.86}"
SSH_KEY="${TURBULENCE_KEY:-/root/.ssh/turbulence-agent.pem}"

#: Marks a request as ours, so the status page can exclude it.
PROBE_UA="turbulence-edge-check/1.0 (daily health probe)"

#: Below this many days the GeoLite2 databases are considered current. The
#: weekly cron gives four days of slack before anything is really stale.
GEOIP_MAX_AGE_DAYS=14
#: Certificate renewal happens automatically; this is when to worry.
CERT_MIN_DAYS=14

REMOTE=1
GEO=0
QUIET=0
for arg in "$@"; do
  case "$arg" in
    --no-remote) REMOTE=0 ;;
    --geo)       GEO=1 ;;
    --quiet)     QUIET=1 ;;
    -h|--help)   sed -n '3,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

PASS=0; FAIL=0; WARN=0
declare -a PROBLEMS=()

ok()   { PASS=$((PASS+1)); [[ "$QUIET" -eq 1 ]] || printf '  ok    %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); PROBLEMS+=("$1"); printf '  FAIL  %s\n' "$1"; }
warn() { WARN=$((WARN+1)); PROBLEMS+=("warn: $1"); printf '  warn  %s\n' "$1"; }
head_() { [[ "$QUIET" -eq 1 ]] || printf '\n%s\n' "$1"; }

probe() {  # probe <path-and-query> -> status code, or 000 if unreachable
  # curl prints the code even when the request fails, so `|| echo` would
  # append rather than replace. Take the last three characters.
  local out
  out=$(curl -sk -o /dev/null -w '%{http_code}' -A "$PROBE_UA" \
             --max-time 20 "${HOST}${1}" 2>/dev/null)
  echo "${out: -3}"
}

remote() { ssh -i "$SSH_KEY" -o ConnectTimeout=10 -o BatchMode=yes \
                "$SSH_HOST" "$@" 2>/dev/null; }

[[ "$QUIET" -eq 1 ]] || echo "Edge check against $HOST at $(date -u '+%F %T UTC')"

# ---------------------------------------------------------------- reachable
head_ "Reachability"
STATUS=$(probe "/")
case "$STATUS" in
  401|200) ok "site responds ($STATUS)" ;;
  000)     bad "site unreachable - TLS, DNS or the service is down"
           echo
           echo "Nothing below this would mean anything, so the rest is skipped."
           exit 1 ;;
  403)     bad "site returns 403 from here; the geo filter may be blocking a US address" ;;
  *)       warn "unexpected status $STATUS" ;;
esac

# The status page has its own rate-limit zone, so it must survive a burst
# that the search endpoint would not.
BURST_OK=1
for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
  [[ "$(probe "/api/status")" == "429" ]] && BURST_OK=0
done
[[ "$BURST_OK" -eq 1 ]] \
  && ok "status page survives twelve requests (its own zone)" \
  || bad "status page hit the search rate limit; the zone split is not working"

# ------------------------------------------------------------------ the WAF
head_ "Web application firewall"

# Four families. The scores these produce are known from the deployment:
# traversal 40, XSS 20, SQLi 5, scanner 5.
declare -A VECTORS=(
  ["path traversal"]="/?file=../../../../etc/passwd"
  ["SQL injection"]="/?id=1%27%20OR%20%271%27=%271"
  ["cross-site scripting"]="/?q=%3Cscript%3Ealert(1)%3C/script%3E"
)
BEFORE_TS=$(date -u '+%Y-%m-%d %H:%M:%S')
sleep 1

for name in "${!VECTORS[@]}"; do
  code=$(probe "${VECTORS[$name]}")
  # With the engine blocking, a refusal is the pass. A probe that reaches
  # the application means the rule set stopped matching it, which is the
  # failure worth hearing about.
  case "$code" in
    403)     ok "$name refused at the edge (403)" ;;
    401|200) bad "$name reached the application ($code) - the rule no longer matches" ;;
    *)       bad "$name probe returned $code" ;;
  esac
done

# Scanner detection keys on the user agent, so it needs its own request.
SCANNER_CODE=$(curl -sk -o /dev/null -w '%{http_code}' -A "Nikto/2.1.6" \
                    --max-time 20 "$HOST/" 2>/dev/null)
SCANNER_CODE="${SCANNER_CODE: -3}"
[[ "$SCANNER_CODE" == "403" ]] \
  && ok "scanner user agent refused (403)" \
  || bad "scanner user agent reached the application ($SCANNER_CODE)"

if [[ "$REMOTE" -eq 1 ]]; then
  sleep 3
  DETECTED=$(remote "sudo journalctl -u caddy --since '${BEFORE_TS}' -o cat --no-pager \
                     | grep -c 'http.handlers.waf'" || echo 0)
  DETECTED=${DETECTED:-0}
  if [[ "$DETECTED" -ge 4 ]]; then
    ok "firewall logged $DETECTED rule matches for those probes"
  elif [[ "$DETECTED" -gt 0 ]]; then
    warn "only $DETECTED rule matches logged; expected at least four"
  else
    bad "firewall logged nothing - Coraza may not be loaded"
  fi

  # Comments stripped first: a note above the directive mentions
  # "SecRuleEngine On", and matching it reported blocking as live while the
  # engine was still only detecting.
  ENGINE=$(remote "grep -v '^[[:space:]]*#' /etc/caddy/Caddyfile \
                   | grep -o 'SecRuleEngine [A-Za-z]*' | head -1")
  # Blocking since 2026-08-27, after five days of detection logs showed no
  # rule matching a legitimate search. A drop back to DetectionOnly would
  # be a silent loss of protection, so it is a failure rather than a note.
  case "$ENGINE" in
    *On)            ok "engine is On, blocking as intended" ;;
    *DetectionOnly) bad "engine has dropped back to DetectionOnly - nothing is being refused" ;;
    *)              bad "could not read SecRuleEngine from the Caddyfile" ;;
  esac
fi

# -------------------------------------------------------------- geo filter
head_ "Geographic filter"

if [[ "$REMOTE" -eq 1 ]]; then
  for db in GeoLite2-Country GeoLite2-City GeoLite2-ASN; do
    AGE=$(remote "find /usr/share/GeoIP/${db}.mmdb -mtime +${GEOIP_MAX_AGE_DAYS} 2>/dev/null | wc -l")
    EXISTS=$(remote "test -f /usr/share/GeoIP/${db}.mmdb && echo 1 || echo 0")
    if [[ "$EXISTS" != "1" ]]; then
      [[ "$db" == "GeoLite2-Country" ]] \
        && bad "$db is missing - the geo filter cannot work" \
        || warn "$db is missing - the status page loses region or network"
    elif [[ "${AGE:-0}" -gt 0 ]]; then
      warn "$db is older than ${GEOIP_MAX_AGE_DAYS} days; check the weekly cron"
    else
      ok "$db present and current"
    fi
  done

  # Resolution is verified against the database rather than by sending
  # traffic, because egress from another country is the hard part.
  RESOLVED=$(remote "cd /opt/turbulence-agent && sudo -u turbulence .venv/bin/python -c \"
import sys; sys.path.insert(0,'.')
from app.runs import resolve_origin
bad = 0
# One address the free database is known to carry, and one it is known
# not to. Asserting a specific country for several addresses turned out
# to test the addresses rather than the lookup: two anycast resolvers
# return nothing at all, and a third had changed country since the check
# was written. What is worth asserting is narrower - that the database
# opens, answers where it can, and returns nothing rather than a wrong
# answer where it cannot.
if resolve_origin('8.8.8.8').country != 'US':
    bad += 1                      # a well-known address must resolve
if resolve_origin('127.0.0.1').country is not None:
    bad += 1                      # loopback has no country to give
print(bad)
\"" || echo "x")
  case "$RESOLVED" in
    0) ok "known addresses resolve to the expected countries" ;;
    x) bad "could not run the resolution check on the instance" ;;
    *) bad "$RESOLVED of three known addresses resolved to the wrong country" ;;
  esac
fi

# A real foreign-origin request. Tor is the only free egress that lands in
# another country reliably; without it this check is skipped rather than
# faked, because a filter you cannot test is not a filter you can trust.
if [[ "$GEO" -eq 1 ]]; then
  if command -v torsocks >/dev/null 2>&1; then
    TOR_CODE=$(torsocks curl -sk -o /dev/null -w '%{http_code}' -A "$PROBE_UA" \
                        --max-time 45 "$HOST/" 2>/dev/null)
    TOR_CODE="${TOR_CODE: -3}"
    case "$TOR_CODE" in
      403) ok "foreign-origin request blocked (403) - filter is live end to end" ;;
      401) warn "foreign-origin request reached auth; the exit node may be in the US" ;;
      000) warn "could not reach the site over Tor" ;;
      *)   bad "foreign-origin request returned $TOR_CODE - expected 403" ;;
    esac
  else
    warn "torsocks not installed, so foreign-origin egress cannot be tested"
    [[ "$QUIET" -eq 1 ]] || echo "        apt-get install tor torsocks, then re-run with --geo"
  fi
fi

# ------------------------------------------------------------ the app gate
head_ "Application controls"

CONFIG=$(curl -sk --max-time 15 -A "$PROBE_UA" "$HOST/api/config" 2>/dev/null)
if [[ "$CONFIG" == *'"turnstile_enabled":true'* ]]; then
  ok "the challenge is configured"
elif [[ -z "$CONFIG" ]]; then
  ok "/api/config is behind auth, as it should be"
else
  bad "the challenge is NOT enabled - an open endpoint would be unprotected"
fi

# The middleware sets the policy on every response, but basic auth sits in
# front of the application: an unauthenticated probe gets Caddy's 401 and
# the application never runs. So there is no route that carries the policy
# without credentials, and the check needs them or must skip.
#
# Set TURBULENCE_AUTH=user:password to enable this check.
if [[ -n "${TURBULENCE_AUTH:-}" ]]; then
  CSP=$(curl -sk -D- -o /dev/null --max-time 15 -A "$PROBE_UA" \
             -u "$TURBULENCE_AUTH" "$HOST/api/health" 2>/dev/null \
        | grep -i 'content-security-policy' || true)
else
  CSP=""
fi

if [[ -z "${TURBULENCE_AUTH:-}" ]]; then
  warn "no credentials set, so the content security policy was not checked"
  [[ "$QUIET" -eq 1 ]] || echo "        export TURBULENCE_AUTH=user:password to enable it"
elif [[ -n "$CSP" ]]; then
  ok "content security policy present"
  # The map tiles now carry an API key as a query parameter, so the
  # policy still needs the tile host. Losing it would break the map
  # silently, with the page otherwise working.
  [[ "$CSP" == *"basemaps.cartocdn.com"* ]] \
    && ok "policy still permits the basemap tiles" \
    || warn "policy no longer permits the basemap tiles; the map will not draw"
else
  bad "no content security policy on an authenticated response"
fi

DAYS=$(echo | openssl s_client -connect "${HOST#https://}:443" \
        -servername "${HOST#https://}" 2>/dev/null | openssl x509 -noout -enddate 2>/dev/null \
        | cut -d= -f2)
if [[ -n "$DAYS" ]]; then
  LEFT=$(( ( $(date -d "$DAYS" +%s) - $(date +%s) ) / 86400 ))
  [[ "$LEFT" -gt "$CERT_MIN_DAYS" ]] \
    && ok "certificate valid for $LEFT more days" \
    || warn "certificate expires in $LEFT days"
fi

# ------------------------------------------------------------------ summary
echo
if [[ "$FAIL" -gt 0 ]]; then
  echo "$FAIL failed, $WARN warning(s), $PASS passed"
  printf '  %s\n' "${PROBLEMS[@]}"
  exit 1
elif [[ "$WARN" -gt 0 ]]; then
  echo "$WARN warning(s), $PASS passed"
  printf '  %s\n' "${PROBLEMS[@]}"
  exit 0
else
  [[ "$QUIET" -eq 1 ]] || echo "all $PASS checks passed"
  exit 0
fi
