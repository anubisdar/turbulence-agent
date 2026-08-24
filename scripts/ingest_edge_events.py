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


def _origin(cache: dict, ip: str) -> tuple[str, str]:
    """Country and network for an address, resolved once per address."""
    if ip not in cache:
        o = resolve_origin(ip)
        cache[ip] = (o.country_name or o.country or "unknown",
                     o.asn_name or "unknown")
    return cache[ip]


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


def read_waf(lines: list[str], cache: dict) -> list[dict]:
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
        # A request tripping both SQLi and XSS counts against each family.
        for family in rec["families"] or {"uncategorised"}:
            events.append({
                "occurred_at": when, "kind": WAF, "detail": family,
                "country": country, "asn_name": network,
                "path": _label_path(rec["uri"]), "score": rec["score"],
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


def read_edge_blocks(path: str, since: datetime, cache: dict) -> list[dict]:
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
                if status == 403:
                    detail = "geo filter"
                elif status == 429:
                    detail = "rate limit"
                else:
                    continue

                request = entry.get("request") or {}
                ip = (request.get("remote_ip") or "").strip()
                country, network = _origin(cache, ip) if ip \
                    else ("unknown", "unknown")
                events.append({
                    "occurred_at": datetime.fromtimestamp(
                        when, timezone.utc).isoformat(),
                    "kind": EDGE_BLOCK, "detail": detail,
                    "country": country, "asn_name": network,
                    "path": _label_path(request.get("uri") or "/"),
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
    args = ap.parse_args()

    since = datetime.now(timezone.utc) - timedelta(
        minutes=args.lookback_minutes)
    cache: dict = {}

    lines = _journal(args.lookback_minutes)
    events = (read_waf(lines, cache)
              + read_refusals(lines)
              + read_edge_blocks(args.caddy_log, since, cache))

    by_kind: dict[str, int] = {}
    for e in events:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1

    if args.dry_run:
        print(f"  would insert {len(events)} event(s): {by_kind or 'none'}")
        return 0

    conn = sqlite3.connect(args.db)
    try:
        init_edge_events(conn)
        added = record_events(conn, events)
    finally:
        conn.close()

    print(f"  read {len(events)} event(s) {by_kind or ''}, "
          f"{added} new after deduplication")
    return 0


if __name__ == "__main__":
    sys.exit(main())
