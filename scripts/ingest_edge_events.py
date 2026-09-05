#!/usr/bin/env python3
# install-to: scripts
"""
Move edge events out of the logs and into the database.

Run by a systemd timer every five minutes. Reads further back than that on
purpose - clock skew and a slow writer both mean the previous run's
boundary is not trustworthy - and relies on a deduplication key to make
re-reading harmless. A run that fails leaves nothing half-written; the next
one picks up the same window.

Three sources, none of which the application itself ever sees:

  firewall detections   journald, logger http.handlers.waf
  challenge refusals    journald, this application's own log
  edge blocks           the Caddy access log, status 403 and 429

Addresses are read to resolve a country and network, then discarded. The
deduplication key is a hash and cannot be reversed into one.

Usage:
    python3 scripts/ingest_edge_events.py
    python3 scripts/ingest_edge_events.py --lookback-minutes 1440
    python3 scripts/ingest_edge_events.py --dry-run
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.edge_events import (  # noqa: E402
    CHALLENGE_REFUSAL,
    EDGE_BLOCK,
    WAF,
    dedup_key,
    init_edge_events,
    record_events,
)
from app.runs import resolve_origin  # noqa: E402

#: Six times the timer interval. Overlap is cheap; a gap is not.
DEFAULT_LOOKBACK_MINUTES = 30

CADDY_LOG = os.environ.get("TURBULENCE_CADDY_LOG",
                           "/var/log/caddy/access.log")
DB_PATH = os.environ.get("TURBULENCE_DB",
                         "/opt/turbulence-agent/data/retrieval.db")

_ATTACK_NAMES = {
    "attack-reputation-scanner": "scanner detection",
    "attack-lfi": "path traversal",
    "attack-rfi": "remote file inclusion",
    "attack-rce": "remote command execution",
    "attack-sqli": "SQL injection",
    "attack-xss": "cross-site scripting",
    "attack-injection-php": "PHP injection",
    "attack-injection-generic": "code injection",
    "attack-protocol": "protocol violation",
    "attack-disclosure": "information disclosure",
    "attack-reputation-ip": "known bad address",
    "attack-generic": "generic attack",
}

_ANOMALY_RULE = "949110"
_SCORE = re.compile(r"Total Score:\s*(\d+)")
_UNIQUE_ID = re.compile(r'\[unique_id "([^"]+)"')
_URI = re.compile(r'\[uri "([^"]*)"')
_CLIENT = re.compile(r'\[client "([^"]+)"')
_TAG = re.compile(r'\[tag "([^"]+)"')
_REFUSAL = re.compile(r"challenge outcome=(no_token|rejected)")

#: check_edge.sh sets this user agent on every probe it fires, and its
#: header says the probes carry it "so they can be excluded from the
#: status page rather than inflating its detection counts". Nothing read
#: it. Four vectors an hour, each tripping several rule families, put the
#: operator's own traffic at 625 of 673 firewall detections - a panel
#: that was almost entirely the health check watching itself.
#:
#: Kept as a pattern rather than an equality test because the version
#: string moves. Override with TURBULENCE_PROBE_UA if the agent changes.
PROBE_UA = re.compile(
    os.environ.get("TURBULENCE_PROBE_UA", r"turbulence-edge-check"),
    re.IGNORECASE)

#: The scanner-detection vector deliberately sends a scanner's own agent,
#: so it cannot be recognised by the pattern above. It is identified by
#: coming from an address that presented the probe agent in the same
#: window, which is what `probe_addresses` collects.
_UA_HEADER = "User-Agent"


def _origin(cache: dict, ip: str) -> tuple[str, str]:
    """Country and network for an address, resolved once per address."""
    if ip not in cache:
        o = resolve_origin(ip)
        cache[ip] = (o.country_name or o.country or "unknown",
                     o.asn_name or "unknown")
    return cache[ip]


def _user_agent(entry: dict) -> str:
    """The agent on a Caddy access log entry, which stores headers as
    lists."""
    headers = ((entry.get("request") or {}).get("headers") or {})
    value = headers.get(_UA_HEADER) or headers.get(_UA_HEADER.lower()) or ""
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value)


def probe_addresses(path: str, since: datetime) -> set[str]:
    """Addresses that presented the probe agent inside the window.

    The firewall's own log lines do not carry a user agent - Coraza
    reports the rule, the URI and the client, and nothing else - so a WAF
    detection cannot be recognised as a probe on its own. The access log
    does carry one, and both records name the same address, so the agent
    is read here and applied there.

    Kept to the window rather than remembered, so a health check that is
    moved to another host stops being excluded within the hour instead of
    silently forever. If the log cannot be read the set is empty, every
    event is treated as real traffic, and the counts are wrong in the
    direction that is easier to notice.
    """
    found: set[str] = set()
    if not os.path.exists(path):
        return found
    cutoff = since.timestamp()
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                when = entry.get("ts")
                if not isinstance(when, (int, float)) or when < cutoff:
                    continue
                if not PROBE_UA.search(_user_agent(entry)):
                    continue
                ip = ((entry.get("request") or {}).get("remote_ip")
                      or (entry.get("request") or {}).get("client_ip")
                      or "").strip()
                if ip:
                    found.add(ip)
    except OSError as e:
        print(f"  could not read {path} for probe agents: {type(e).__name__}",
              file=sys.stderr)
    return found


def _label_path(uri: str) -> str:
    """A path that says something.

    Stripping the query makes every injection attempt look like "/", since
    the payload lives in the query and the path is bare. Keep a decoded
    query only when the path alone is uninformative.
    """
    path, _, query = uri.partition("?")
    path = path or "/"
    if path == "/" and query:
        return f"/?{unquote(query)[:44]}"
    return path[:60]


def _journal(since_minutes: int) -> list[str]:
    try:
        result = subprocess.run(
            ["journalctl", "--since", f"{since_minutes} minutes ago",
             "-o", "cat", "--no-pager"],
            capture_output=True, text=True, timeout=60)
        return result.stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as e:
        print(f"  could not read the journal: {type(e).__name__}",
              file=sys.stderr)
        return []


def read_waf(lines: list[str], cache: dict,
             probes: set[str] | None = None) -> list[dict]:
    """Firewall detections, grouped per request rather than per rule.

    One request tripping four rules is one detection. The four test vectors
    run against this deployment produced thirteen log lines between them,
    so counting lines would be wrong by a factor of three.
    """
    requests: dict[str, dict] = {}

    for line in lines:
        if "http.handlers.waf" not in line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("logger") != "http.handlers.waf":
            continue
        message = entry.get("msg") or ""

        found = _UNIQUE_ID.search(message)
        if not found:
            continue
        rec = requests.setdefault(found.group(1), {
            "ts": entry.get("ts"), "score": 0, "families": set(),
            "uri": "/", "ip": ""})

        score = _SCORE.search(message)
        if score:
            rec["score"] = max(rec["score"], int(score.group(1)))
        uri = _URI.search(message)
        if uri:
            rec["uri"] = uri.group(1)
        client = _CLIENT.search(message)
        if client:
            rec["ip"] = client.group(1)
        for tag in _TAG.findall(message):
            if tag in _ATTACK_NAMES:
                rec["families"].add(_ATTACK_NAMES[tag])

    events = []
    for unique_id, rec in requests.items():
        when = datetime.fromtimestamp(rec["ts"] or 0,
                                      timezone.utc).isoformat()
        country, network = _origin(cache, rec["ip"]) if rec["ip"] \
            else ("unknown", "unknown")
        is_probe = bool(probes) and rec["ip"] in probes
        # A request tripping both SQLi and XSS counts against each family.
        for family in rec["families"] or {"uncategorised"}:
            events.append({
                "occurred_at": when, "kind": WAF, "detail": family,
                "country": country, "asn_name": network,
                "path": _label_path(rec["uri"]), "score": rec["score"],
                "probe": is_probe,
                # Unchanged on purpose. The key identifies the event, not
                # how it was classified, so re-ingesting a window that was
                # read before this flag existed stays idempotent.
                "dedup_key": dedup_key(WAF, unique_id, family),
            })
    return events


def read_refusals(lines: list[str]) -> list[dict]:
    """Searches the challenge turned away.

    A refused request never becomes a search, so this is the only record
    that it happened at all.
    """
    events = []
    for i, line in enumerate(lines):
        found = _REFUSAL.search(line)
        if not found:
            continue
        request_id = re.search(r"req=(\S+)", line)
        events.append({
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "kind": CHALLENGE_REFUSAL,
            "detail": ("presented no token" if found.group(1) == "no_token"
                       else "presented one that failed"),
            "dedup_key": dedup_key(CHALLENGE_REFUSAL,
                                   request_id.group(1) if request_id else i,
                                   found.group(1)),
        })
    return events


def read_edge_blocks(path: str, since: datetime, cache: dict,
                     probes: set[str] | None = None) -> list[dict]:
    """Requests Caddy stopped before they reached the application."""
    if not os.path.exists(path):
        return []
    events = []
    cutoff = since.timestamp()

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue

                when = entry.get("ts")
                if not isinstance(when, (int, float)) or when < cutoff:
                    continue
                status = entry.get("status")
                # Both the geo filter and the firewall refuse with 403, so
                # the status alone stopped being enough the moment the
                # engine moved to blocking. The geo handler appends
                # blocked_by="geo"; a 403 without it came from the firewall.
                if status == 403:
                    marker = ((entry.get("resp_headers") or {})
                              .get("blocked_by")
                              or entry.get("blocked_by") or "")
                    if isinstance(marker, list):
                        marker = marker[0] if marker else ""
                    detail = ("geo filter" if str(marker).lower() == "geo"
                              else "web application firewall")
                elif status == 429:
                    detail = "rate limit"
                else:
                    continue

                request = entry.get("request") or {}
                ip = (request.get("remote_ip") or "").strip()
                country, network = _origin(cache, ip) if ip \
                    else ("unknown", "unknown")
                # Here the agent is on the record itself, so the address
                # set is only a fallback - it catches the scanner vector,
                # which sends a scanner's agent by design.
                is_probe = (bool(PROBE_UA.search(_user_agent(entry)))
                            or (bool(probes) and ip in probes))
                events.append({
                    "occurred_at": datetime.fromtimestamp(
                        when, timezone.utc).isoformat(),
                    "kind": EDGE_BLOCK, "detail": detail,
                    "country": country, "asn_name": network,
                    "path": _label_path(request.get("uri") or "/"),
                    "probe": is_probe,
                    "dedup_key": dedup_key(EDGE_BLOCK, when, ip,
                                           request.get("uri")),
                })
    except OSError as e:
        print(f"  could not read {path}: {type(e).__name__}", file=sys.stderr)
    return events


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback-minutes", type=int,
                    default=DEFAULT_LOOKBACK_MINUTES)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--caddy-log", default=CADDY_LOG)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-probe-filter", action="store_true",
                    help="record the operator's own probes as ordinary "
                         "traffic, as this script did before")
    args = ap.parse_args()

    since = datetime.now(timezone.utc) - timedelta(
        minutes=args.lookback_minutes)
    cache: dict = {}

    probes = set() if args.no_probe_filter \
        else probe_addresses(args.caddy_log, since)

    lines = _journal(args.lookback_minutes)
    events = (read_waf(lines, cache, probes)
              + read_refusals(lines)
              + read_edge_blocks(args.caddy_log, since, cache, probes))

    by_kind: dict[str, int] = {}
    for e in events:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
    marked = sum(1 for e in events if e.get("probe"))

    if args.dry_run:
        print(f"  would insert {len(events)} event(s): {by_kind or 'none'}")
        print(f"  {marked} marked as the operator's own probes, from "
              f"{len(probes)} address(es)")
        return 0

    conn = sqlite3.connect(args.db)
    try:
        init_edge_events(conn)
        added = record_events(conn, events)
    finally:
        conn.close()

    print(f"  read {len(events)} event(s) {by_kind or ''}, "
          f"{added} new after deduplication")
    if marked:
        print(f"  {marked} of those were the operator's own probes, "
              f"kept but marked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
