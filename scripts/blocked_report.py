#!/usr/bin/env python3
# install-to: scripts
"""
Who is knocking, and what they say they are.

Blocked requests never reach the application, so they never become rows in
`search_runs` and never appear on the status page beyond a count. Everything
known about them lives in the Caddy access log: an address, a status, and a
user agent.

This reads that log and answers three questions the count cannot:

  which countries are being turned away, which is the fact the geo filter
  itself makes invisible - the panel says "US" because everything else is
  blocked before it can be recorded

  whose network they arrive from, which distinguishes a hosting provider
  running a scanner from a person on a domestic line who happens to be
  abroad

  what they claim to be, which is the most honest signal of intent: a
  browser string, a named scanner, a library default, or nothing at all

ON ADDRESSES. Read and discarded, exactly as the application does. Countries
and networks are resolved from the local MaxMind databases at read time and
nothing here prints or stores an address. Aggregate output only.

REQUIRES `format json` on the Caddy access log. The console format drops
request headers, so a blocked request records that it was blocked and
nothing about who sent it.

Usage:
    sudo python3 scripts/blocked_report.py
    sudo python3 scripts/blocked_report.py --hours 24
    sudo python3 scripts/blocked_report.py --agents 30
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone

DEFAULT_LOG = "/var/log/caddy/access.log"
GEOIP_DIR = os.environ.get("TURBULENCE_GEOIP_DIR", "/usr/share/GeoIP")

DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
AMBER = "\033[33m"
BLUE = "\033[36m"
GREEN = "\033[32m"

#: What a user agent is claiming to be. Ordered: the first match wins, so
#: the specific patterns precede the general ones.
AGENT_KINDS = [
    ("security scanner", re.compile(
        r"nmap|masscan|nuclei|sqlmap|nikto|zgrab|censys|shodan|"
        r"paloalto|expanse|internetmeasurement|netsystems", re.I)),
    ("search crawler", re.compile(
        r"googlebot|bingbot|yandex|baiduspider|duckduckbot|applebot", re.I)),
    ("AI crawler", re.compile(
        r"gptbot|claudebot|anthropic|ccbot|perplexity|bytespider", re.I)),
    ("library default", re.compile(
        r"^(python-requests|curl|wget|go-http-client|java|okhttp|libwww|"
        r"axios|node-fetch|guzzle)", re.I)),
    ("browser", re.compile(r"mozilla|chrome|safari|firefox|edge", re.I)),
]


def classify(agent: str) -> str:
    if not agent.strip():
        return "no user agent"
    for label, pattern in AGENT_KINDS:
        if pattern.search(agent):
            return label
    return "other"


class Resolver:
    """Country, region and network from an address, using the databases the
    host already has for the edge filter."""

    def __init__(self):
        self.readers = {}
        try:
            import maxminddb
        except ImportError:
            self.maxminddb = None
            return
        self.maxminddb = maxminddb
        for key, name in (("country", "GeoLite2-Country.mmdb"),
                          ("city", "GeoLite2-City.mmdb"),
                          ("asn", "GeoLite2-ASN.mmdb")):
            path = os.path.join(GEOIP_DIR, name)
            if os.path.exists(path):
                try:
                    self.readers[key] = maxminddb.open_database(path)
                except (ValueError, OSError):
                    pass

    def available(self) -> list[str]:
        return sorted(self.readers)

    def lookup(self, ip: str) -> tuple[str, str]:
        """(country, network). Never returns the address."""
        country = network = "unknown"
        reader = self.readers.get("city") or self.readers.get("country")
        if reader:
            try:
                found = reader.get(ip)
                if isinstance(found, dict):
                    names = (found.get("country") or {}).get("names") or {}
                    country = (names.get("en")
                               or (found.get("country") or {}).get("iso_code")
                               or "unknown")
            except (ValueError, OSError):
                pass
        if "asn" in self.readers:
            try:
                found = self.readers["asn"].get(ip)
                if isinstance(found, dict):
                    org = found.get("autonomous_system_organization")
                    num = found.get("autonomous_system_number")
                    if org:
                        network = f"{org} \u00b7 AS{num}" if num else org
            except (ValueError, OSError):
                pass
        return country, network


def bar(count: int, largest: int, width: int = 22) -> str:
    filled = max(1, round(width * count / largest)) if largest else 0
    return "\u2588" * filled


def table(title: str, counts: Counter, total: int, limit: int,
          colour: str = BLUE, note: str = "") -> None:
    if not counts:
        return
    print(f"\n{colour}{BOLD}{title}{RESET}")
    if note:
        print(f"{DIM}{note}{RESET}")
    largest = counts.most_common(1)[0][1]
    for label, count in counts.most_common(limit):
        share = count / total if total else 0
        label = label if len(label) <= 46 else label[:43] + "..."
        print(f"  {count:>7,}  {share:>5.1%}  {DIM}{bar(count, largest)}{RESET}"
              f"  {label}")
    shown = sum(c for _, c in counts.most_common(limit))
    if shown < total:
        print(f"  {total - shown:>7,}  {DIM}remaining, not shown{RESET}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--hours", type=float, default=None,
                    help="only entries newer than this")
    ap.add_argument("--agents", type=int, default=12,
                    help="how many distinct user agents to list")
    ap.add_argument("--status", type=int, default=403,
                    help="which status counts as blocked")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"No access log at {args.log}.", file=sys.stderr)
        return 2

    cutoff = None
    if args.hours:
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(hours=args.hours)).timestamp()

    resolver = Resolver()
    countries, networks, agents, kinds = (Counter(), Counter(),
                                          Counter(), Counter())
    paths = Counter()
    blocked = malformed = allowed = 0
    earliest = latest = None
    console_format = False

    with open(args.log, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                console_format = True
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                malformed += 1
                continue

            when = entry.get("ts")
            if cutoff and isinstance(when, (int, float)) and when < cutoff:
                continue
            if isinstance(when, (int, float)):
                earliest = when if earliest is None else min(earliest, when)
                latest = when if latest is None else max(latest, when)

            if entry.get("status") != args.status:
                allowed += 1
                continue

            blocked += 1
            request = entry.get("request") or {}
            headers = request.get("headers") or {}
            agent = (headers.get("User-Agent") or [""])[0]
            agents[agent or "(none sent)"] += 1
            kinds[classify(agent)] += 1
            paths[request.get("uri") or "/"] += 1

            ip = request.get("remote_ip") or ""
            if ip:
                country, network = resolver.lookup(ip)
                countries[country] += 1
                networks[network] += 1

    if console_format and not blocked:
        print(f"{AMBER}The access log is in console format, which drops "
              f"request headers.{RESET}")
        print(f"{DIM}Set `format json` on the log directive in the Caddyfile "
              f"and reload Caddy.{RESET}")
        return 1

    if not blocked:
        print(f"{DIM}No requests with status {args.status} in this window.")
        print(f"That is a quiet edge, not a broken report.{RESET}")
        return 0

    span = ""
    if earliest and latest:
        span = (f"{datetime.fromtimestamp(earliest, timezone.utc):%d %b %H:%M}"
                f" to "
                f"{datetime.fromtimestamp(latest, timezone.utc):%d %b %H:%M} UTC")
    print(f"{BOLD}{blocked:,} requests blocked at the edge{RESET}"
          f"  {DIM}{span}{RESET}")
    if allowed:
        print(f"{DIM}{allowed:,} other entries in the same window were not "
              f"blocked.{RESET}")
    if not resolver.readers:
        print(f"{AMBER}No MaxMind databases found in {GEOIP_DIR} \u2014 "
              f"country and network cannot be resolved.{RESET}")

    table("Countries turned away", countries, blocked, 15, BLUE,
          "The fact the status page cannot show: it reports only what got "
          "through.")
    table("Networks", networks, blocked, 12, BLUE,
          "A hosting provider is automation. A domestic carrier is a person "
          "who is abroad.")
    table("What they claim to be", kinds, blocked, 10, AMBER)
    table("User agents", agents, blocked, args.agents, AMBER,
          "Verbatim, and worth reading: a scanner usually says so.")
    table("What they asked for", paths, blocked, 10, GREEN)

    print(f"\n{DIM}Addresses were read to resolve country and network, then "
          f"discarded. Nothing above identifies a sender.{RESET}")
    if malformed:
        print(f"{DIM}{malformed:,} line(s) could not be parsed.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
